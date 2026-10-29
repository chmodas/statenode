//! Helpers for implementing [`State::apply_step`](crate::State::apply_step).
//!
//! Each helper takes a channel's field and the writes addressed to it, in
//! task-path order, and folds the writes into the field:
//!
//! - [`replace`] accepts at most one write per step, which replaces the
//!   field;
//! - [`append`] extends a list with each write's items;
//! - [`combine`] merges each write into the field with an operator, such as a
//!   sum, a maximum, or a set union;
//! - [`fold`] applies writes of another type, such as typed operations, with
//!   an operator that may reject them.
//!
//! The two that can fail, [`replace`] and [`fold`], also take the channel's
//! name, and return a [`ReduceError`] naming it. `replace` then leaves the
//! field as it was; `fold` may have applied some writes, which is safe
//! because the engine discards a state whose `apply_step` failed.
//!
//! An `apply_step` over three channels, one per merging helper. Each task's
//! owned values are moved out of `updates` with [`std::mem::take`]:
//!
//! ```
//! use std::collections::BTreeSet;
//! use std::mem;
//!
//! use statenode::{BoxError, ChannelSpec, NameError, ReduceError, State, TaskPath, reducer};
//!
//! /// A research state that parallel branches write.
//! #[derive(Clone, Default)]
//! struct Research {
//!     findings: Vec<String>,
//!     cost: u64,
//!     tags: BTreeSet<String>,
//! }
//!
//! /// An operation on the tags.
//! enum Tag {
//!     Add(String),
//!     Remove(String),
//! }
//!
//! /// One task's update: empty fields leave their channel unwritten.
//! #[derive(Default)]
//! struct ResearchUpdate {
//!     findings: Vec<String>,
//!     cost: Option<u64>,
//!     tags: Vec<Tag>,
//! }
//!
//! /// Applies one tag operation; removing a missing tag is an error.
//! fn apply_tag(tags: &mut BTreeSet<String>, op: Tag) -> Result<(), BoxError> {
//!     match op {
//!         Tag::Add(tag) => {
//!             tags.insert(tag);
//!             Ok(())
//!         }
//!         Tag::Remove(tag) if tags.remove(&tag) => Ok(()),
//!         Tag::Remove(tag) => Err(format!("no tag `{tag}` to remove").into()),
//!     }
//! }
//!
//! impl State for Research {
//!     type Update = ResearchUpdate;
//!
//!     fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
//!         Ok(vec![
//!             ChannelSpec::new("findings")?,
//!             ChannelSpec::new("cost")?,
//!             ChannelSpec::new("tags")?,
//!         ])
//!     }
//!
//!     fn apply_step(
//!         &mut self,
//!         mut updates: Vec<(TaskPath, ResearchUpdate)>,
//!     ) -> Result<(), ReduceError> {
//!         let findings = updates
//!             .iter_mut()
//!             .map(|(p, u)| (&*p, mem::take(&mut u.findings)));
//!         reducer::append(&mut self.findings, findings);
//!
//!         let costs = updates.iter().filter_map(|(p, u)| u.cost.map(|c| (p, c)));
//!         reducer::combine(&mut self.cost, costs, |total, c| *total += c);
//!
//!         let tags = updates.iter_mut().flat_map(|(p, u)| {
//!             let p: &TaskPath = p;
//!             mem::take(&mut u.tags).into_iter().map(move |op| (p, op))
//!         });
//!         reducer::fold("tags", &mut self.tags, tags, apply_tag)
//!     }
//! }
//! ```

use std::fmt;

use crate::node::BoxError;
use crate::state::ReduceError;
use crate::task::{TaskKind, TaskPath};

/// A channel written at most once per step, whose write replaces the field.
///
/// With no writes, the field is left as it is; with one, the field takes its
/// value. The value is never cloned or compared, so applying costs time
/// proportional to the writes, not to the field's size.
///
/// # Errors
///
/// Returns a [`ReduceError`] for `channel`, caused by [`ConflictingWrites`],
/// if there is more than one write. The field is then left unchanged.
///
/// ```
/// use statenode::reducer;
/// # use statenode::{ReduceError, TaskPath};
/// # fn writes<'a>() -> Vec<(&'a TaskPath, i64)> { Vec::new() }
/// let mut count = 0;
/// // No task wrote `count` this step, so it keeps its value.
/// reducer::replace("count", &mut count, writes())?;
/// assert_eq!(count, 0);
/// # Ok::<(), ReduceError>(())
/// ```
pub fn replace<'a, T>(
    channel: &str,
    field: &mut T,
    writes: impl IntoIterator<Item = (&'a TaskPath, T)>,
) -> Result<(), ReduceError> {
    let mut writes = writes.into_iter();
    match (writes.next(), writes.next()) {
        (Some((_, value)), None) => {
            *field = value;
            Ok(())
        }
        (Some((first, _)), Some((second, _))) => Err(conflict(channel, first, second)),
        (None, _) => Ok(()),
    }
}

/// A list channel that every write extends: the field keeps its items and
/// gains each write's, in task-path order.
///
/// A write is any iterable of items: a `Vec`, an `Option` for zero or one
/// item, an array, or an iterator. Items are moved, never cloned.
///
/// ```
/// use statenode::reducer;
/// # use statenode::TaskPath;
/// # fn writes<'a>() -> Vec<(&'a TaskPath, Vec<&'static str>)> { Vec::new() }
/// let mut findings = vec!["earlier"];
/// reducer::append(&mut findings, writes());
/// // No task wrote `findings` this step, so it keeps its items.
/// assert_eq!(findings, ["earlier"]);
/// ```
pub fn append<'a, T, I>(field: &mut Vec<T>, writes: impl IntoIterator<Item = (&'a TaskPath, I)>)
where
    I: IntoIterator<Item = T>,
{
    for (_, items) in writes {
        field.extend(items);
    }
}

/// A channel that merges every write into the field with `f`, in task-path
/// order: `f(field, write)` for each write in turn.
///
/// The field's value before the step is where the merge starts, so there is
/// no special first write. The result depends on task-path order unless `f`
/// is commutative and associative, as a sum, a maximum or a set union is.
///
/// ```
/// use statenode::reducer;
/// # use statenode::TaskPath;
/// # fn writes<'a>() -> Vec<(&'a TaskPath, u64)> { Vec::new() }
/// let mut cost = 3_u64;
/// reducer::combine(&mut cost, writes(), |total, c| *total += c);
/// // No task wrote `cost` this step, so it keeps its value.
/// assert_eq!(cost, 3);
/// ```
pub fn combine<'a, T>(
    field: &mut T,
    writes: impl IntoIterator<Item = (&'a TaskPath, T)>,
    mut f: impl FnMut(&mut T, T),
) {
    for (_, value) in writes {
        f(field, value);
    }
}

/// A channel whose writes are applied to the field by `f`, in task-path
/// order; a write may be of another type than the field, such as an
/// operation on it, and `f` may reject it.
///
/// `f`'s error is a [`BoxError`]; a named function returning another error
/// type adapts with `.map_err(Into::into)`.
///
/// # Errors
///
/// Returns a [`ReduceError`] for `channel`, caused by the first error `f`
/// returns. The writes after it are not applied, and those before it are:
/// the field may be partly updated, which is safe because the engine
/// discards a state whose `apply_step` failed.
///
/// ```
/// use std::collections::BTreeSet;
///
/// use statenode::reducer;
/// # use statenode::{ReduceError, TaskPath};
/// # fn writes<'a>() -> Vec<(&'a TaskPath, &'static str)> { Vec::new() }
/// let mut tags = BTreeSet::from(["draft"]);
/// reducer::fold("tags", &mut tags, writes(), |tags, tag| {
///     tags.insert(tag);
///     Ok(())
/// })?;
/// // No task wrote `tags` this step, so it keeps its tags.
/// assert_eq!(tags, BTreeSet::from(["draft"]));
/// # Ok::<(), ReduceError>(())
/// ```
pub fn fold<'a, T, U>(
    channel: &str,
    field: &mut T,
    writes: impl IntoIterator<Item = (&'a TaskPath, U)>,
    mut f: impl FnMut(&mut T, U) -> Result<(), BoxError>,
) -> Result<(), ReduceError> {
    writes
        .into_iter()
        .try_for_each(|(_, write)| f(field, write))
        .map_err(|source| ReduceError::new(channel, source))
}

/// The error for two writes, from `first` and `second`, to the `replace`
/// channel `channel`.
fn conflict(channel: &str, first: &TaskPath, second: &TaskPath) -> ReduceError {
    ReduceError::new(
        channel,
        ConflictingWrites {
            first: first.clone(),
            second: second.clone(),
        },
    )
}

/// Two writes to a [`replace`] channel in one step, from two tasks or from the
/// same task twice.
///
/// A channel that several tasks write in one step needs a reducer that merges
/// writes: [`append`], [`combine`] or [`fold`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConflictingWrites {
    /// The task that wrote first in task-path order.
    first: TaskPath,
    /// The task that wrote second.
    second: TaskPath,
}

impl ConflictingWrites {
    /// The task that wrote first in task-path order.
    #[must_use]
    pub const fn first(&self) -> &TaskPath {
        &self.first
    }

    /// The task that wrote second.
    #[must_use]
    pub const fn second(&self) -> &TaskPath {
        &self.second
    }
}

impl fmt::Display for ConflictingWrites {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step = self.first.step();
        let rule = "but a replace channel accepts one write per step; \
            use a reducer that merges writes, such as `append` or `combine`";
        if self.first == self.second {
            write!(
                f,
                "{} wrote more than once in step {step}, {rule}",
                Task(&self.first)
            )
        } else {
            write!(
                f,
                "{} and {} both wrote in step {step}, {rule}",
                Task(&self.first),
                Task(&self.second)
            )
        }
    }
}

impl std::error::Error for ConflictingWrites {}

/// A task, described for an error message.
struct Task<'a>(&'a TaskPath);

impl fmt::Display for Task<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.kind() {
            TaskKind::Input => f.write_str("the input"),
            TaskKind::Pull { node } => write!(f, "node `{node}`"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::error::Error as _;

    use proptest::prelude::*;
    use test_strategy::{Arbitrary, proptest};

    use super::*;
    use crate::name::NodeName;

    /// The pull task path of node `name` in step `step`.
    fn pull(step: u64, name: &str) -> TaskPath {
        TaskPath::pull(step, NodeName::new(name).unwrap())
    }

    /// The writes of `paths`, in order, with the values `values`.
    fn writes<'a>(paths: &'a [TaskPath], values: &[i64]) -> Vec<(&'a TaskPath, i64)> {
        paths.iter().zip(values.iter().copied()).collect()
    }

    /// A pull task path from a few nodes, so that the same task can come up
    /// more than once. The merging reducers' tests write from `path(0, 0)`,
    /// since those reducers ignore the paths.
    fn path(step: u64, node: u8) -> TaskPath {
        pull(step, &format!("n{node}"))
    }

    // --- replace -------------------------------------------------------------

    /// Two or more writes to one channel in one step.
    #[derive(Debug, Arbitrary)]
    struct Conflict {
        /// The field's value before the step.
        initial: i64,
        /// The step all the writes belong to.
        step: u64,
        /// The writing nodes, in task-path order; repeats are the same task
        /// writing more than once.
        #[strategy(proptest::collection::vec(0..4_u8, 2..8))]
        nodes: Vec<u8>,
        /// One value per write.
        #[strategy(proptest::collection::vec(any::<i64>(), #nodes.len()))]
        values: Vec<i64>,
    }

    /// With no writes, the field keeps its value.
    #[proptest]
    fn replace_without_writes_keeps_the_field(initial: i64) {
        let mut field = initial;
        prop_assert!(replace("count", &mut field, writes(&[], &[])).is_ok());
        prop_assert_eq!(field, initial);
    }

    /// A single write replaces the field's value, whichever task wrote it.
    #[proptest]
    fn replace_takes_a_single_write(
        initial: i64,
        value: i64,
        step: u64,
        #[strategy(0..4_u8)] node: u8,
    ) {
        let mut field = initial;
        let paths = [path(step, node)];
        prop_assert!(replace("count", &mut field, writes(&paths, &[value])).is_ok());
        prop_assert_eq!(field, value);
    }

    /// Any second write in a step is an error naming the channel and the
    /// first two writing tasks, and the field keeps its value.
    #[proptest]
    fn replace_rejects_a_second_write_in_a_step(input: Conflict) {
        let paths: Vec<TaskPath> = input.nodes.iter().map(|&n| path(input.step, n)).collect();
        let mut field = input.initial;
        let error = replace("count", &mut field, writes(&paths, &input.values)).unwrap_err();
        prop_assert_eq!(error.channel(), "count");
        let conflict = error
            .source()
            .unwrap()
            .downcast_ref::<ConflictingWrites>()
            .unwrap();
        prop_assert_eq!(
            (conflict.first(), conflict.second()),
            (&paths[0], &paths[1])
        );
        prop_assert_eq!(field, input.initial);
    }

    // --- append --------------------------------------------------------------

    /// One list write per entry of `items`, all from `path`: the writes type
    /// every region-covering `append` test shares, so one instantiation
    /// covers it.
    fn list_writes(path: &TaskPath, items: Vec<Vec<i64>>) -> Vec<(&TaskPath, Vec<i64>)> {
        items.into_iter().map(|items| (path, items)).collect()
    }

    /// The field keeps its items and gains every write's, in order.
    #[proptest]
    fn append_extends_the_field_in_task_path_order(
        initial: Vec<i64>,
        /// At least one write, so the loop body runs on every case.
        #[strategy(prop::collection::vec(prop::collection::vec(any::<i64>(), 0..4), 1..=1_000))]
        items: Vec<Vec<i64>>,
    ) {
        let expected = [initial.clone(), items.concat()].concat();
        let path = path(0, 0);
        let mut field = initial;
        append(&mut field, list_writes(&path, items));
        prop_assert_eq!(field, expected);
    }

    /// With no writes, the field keeps its items.
    #[proptest]
    fn append_without_writes_keeps_the_field(initial: Vec<i64>) {
        let path = path(0, 0);
        let mut field = initial.clone();
        append(&mut field, list_writes(&path, Vec::new()));
        prop_assert_eq!(field, initial);
    }

    /// A write may be any iterable: here an `Option`, so each write adds
    /// its one item or, as `None`, nothing, and the field gains the items in
    /// order.
    #[proptest]
    fn append_takes_any_iterable_per_write(
        initial: Vec<i64>,
        /// At least one write, so the loop body runs on every case.
        #[strategy(prop::collection::vec(any::<Option<i64>>(), 1..32))]
        items: Vec<Option<i64>>,
    ) {
        let expected: Vec<i64> = initial
            .iter()
            .copied()
            .chain(items.iter().flatten().copied())
            .collect();
        let path = path(0, 0);
        let mut field = initial;
        append(&mut field, items.into_iter().map(|item| (&path, item)));
        prop_assert_eq!(field, expected);
    }

    // --- combine -------------------------------------------------------------

    /// Appends `s` to `acc`: an operator that is not commutative, so the
    /// result shows the order of the writes.
    fn concat(acc: &mut String, s: String) {
        acc.extend([s]);
    }

    /// `combine` applied with `f` to `initial` and one write per value of
    /// `writes`.
    fn combined<T: Clone>(initial: T, writes: &[T], f: impl FnMut(&mut T, T)) -> T {
        let path = path(0, 0);
        let mut field = initial;
        combine(&mut field, writes.iter().cloned().map(|w| (&path, w)), f);
        field
    }

    /// Each write is merged into the field in order, starting from the
    /// field's value.
    #[proptest]
    fn combine_folds_each_write_in_task_path_order(
        initial: String,
        /// At least one write, so the loop body runs on every case.
        #[strategy(prop::collection::vec("[a-z]{0,8}", 1..=1_000))]
        writes: Vec<String>,
    ) {
        let expected = [initial.clone(), writes.concat()].concat();
        let path = path(0, 0);
        let mut field = initial;
        let writes: Vec<(&TaskPath, String)> = writes.into_iter().map(|w| (&path, w)).collect();
        combine(&mut field, writes, concat);
        prop_assert_eq!(field, expected);
    }

    /// Writes for a commutative operator, and the same writes in another
    /// order.
    #[derive(Debug, Arbitrary)]
    struct Reordered {
        /// The field's value before the step.
        initial: i64,
        /// The writes; at least one, so the operator runs on every case.
        #[strategy(prop::collection::vec(any::<i64>(), 1..32))]
        writes: Vec<i64>,
        /// The same writes, shuffled.
        #[strategy(Just(#writes.clone()).prop_shuffle())]
        shuffled: Vec<i64>,
    }

    /// A (wrapping) sum does not depend on the order of the writes.
    #[proptest]
    fn combine_sums_whatever_the_order(input: Reordered) {
        let sum = |acc: &mut i64, x: i64| *acc = acc.wrapping_add(x);
        prop_assert_eq!(
            combined(input.initial, &input.writes, sum),
            combined(input.initial, &input.shuffled, sum)
        );
    }

    /// A maximum does not depend on the order of the writes.
    #[proptest]
    fn combine_takes_the_maximum_whatever_the_order(input: Reordered) {
        let max = |acc: &mut i64, x: i64| *acc = (*acc).max(x);
        prop_assert_eq!(
            combined(input.initial, &input.writes, max),
            combined(input.initial, &input.shuffled, max)
        );
    }

    /// A set union does not depend on the order of the writes.
    #[proptest]
    fn combine_unites_sets_whatever_the_order(input: Reordered) {
        let sets = |values: &[i64]| -> Vec<BTreeSet<i64>> {
            values.iter().map(|&v| BTreeSet::from([v])).collect()
        };
        let union = |acc: &mut BTreeSet<i64>, set: BTreeSet<i64>| acc.extend(set);
        let start = BTreeSet::from([input.initial]);
        prop_assert_eq!(
            combined(start.clone(), &sets(&input.writes), union),
            combined(start, &sets(&input.shuffled), union)
        );
    }

    // --- fold ----------------------------------------------------------------

    /// An operation on a set of numbers.
    #[derive(Debug, Clone, Copy)]
    enum Op {
        /// Adds the number.
        Add(i64),
        /// Removes the number, which must be present.
        Remove(i64),
    }

    /// Applies `op` to `set`; removing a missing number is an error.
    fn apply_op(set: &mut BTreeSet<i64>, op: Op) -> Result<(), BoxError> {
        match op {
            Op::Add(value) => {
                set.insert(value);
                Ok(())
            }
            Op::Remove(value) if set.remove(&value) => Ok(()),
            Op::Remove(value) => Err(format!("no {value} to remove").into()),
        }
    }

    /// Distinct numbers to fold: the last two are reserved as `absent` and
    /// `fresh`, the rest are added, and the first `removed` of those are
    /// then removed again.
    #[derive(Debug, Arbitrary)]
    struct Ops {
        /// The numbers, in order.
        #[strategy(prop::collection::btree_set(any::<i64>(), 3..64))]
        values: BTreeSet<i64>,
        /// How many of the added numbers are removed: at least one, so every
        /// arm of `apply_op` runs.
        #[strategy(1..=#values.len() - 2)]
        removed: usize,
    }

    impl Ops {
        /// The numbers added.
        fn added(&self) -> Vec<i64> {
            self.values
                .iter()
                .copied()
                .take(self.values.len() - 2)
                .collect()
        }

        /// A number never added, so removing it fails.
        fn absent(&self) -> i64 {
            self.values
                .iter()
                .copied()
                .nth(self.values.len() - 2)
                .unwrap()
        }

        /// A number never added, added only after the failing operation.
        fn fresh(&self) -> i64 {
            self.values.iter().copied().last().unwrap()
        }

        /// Every number added, then the first `removed` of them removed.
        fn valid(&self) -> Vec<Op> {
            let added = self.added();
            let adds = added.iter().copied().map(Op::Add);
            let removes = added.iter().copied().take(self.removed).map(Op::Remove);
            adds.chain(removes).collect()
        }

        /// The set the valid operations leave.
        fn expected(&self) -> BTreeSet<i64> {
            self.added().into_iter().skip(self.removed).collect()
        }
    }

    /// Valid operations are applied in order.
    #[proptest]
    fn fold_applies_operations_in_task_path_order(ops: Ops) {
        let path = path(0, 0);
        let writes: Vec<(&TaskPath, Op)> = ops.valid().into_iter().map(|op| (&path, op)).collect();
        let mut set = BTreeSet::new();
        prop_assert!(fold("tags", &mut set, writes, apply_op).is_ok());
        prop_assert_eq!(set, ops.expected());
    }

    /// The first rejected operation fails the fold with an error naming the
    /// channel and caused by the operator's error; the operations after it
    /// are not applied.
    #[proptest]
    fn fold_stops_at_the_first_error_naming_the_channel(ops: Ops) {
        let path = path(0, 0);
        let writes: Vec<(&TaskPath, Op)> = ops
            .valid()
            .into_iter()
            .chain([Op::Remove(ops.absent()), Op::Add(ops.fresh())])
            .map(|op| (&path, op))
            .collect();
        let mut set = BTreeSet::new();
        let error = fold("tags", &mut set, writes, apply_op).unwrap_err();
        prop_assert_eq!(error.channel(), "tags");
        prop_assert_eq!(
            error.source().unwrap().to_string(),
            format!("no {} to remove", ops.absent())
        );
        prop_assert_eq!(set, ops.expected(), "the later add did not run");
    }

    // --- ConflictingWrites ---------------------------------------------------

    /// The message names the tasks and the step; there is no source.
    #[test]
    fn error_messages_name_the_problem() {
        let message = |first, second| ConflictingWrites { first, second }.to_string();
        let tail = "but a replace channel accepts one write per step; \
            use a reducer that merges writes, such as `append` or `combine`";
        assert_eq!(
            message(pull(3, "a"), pull(3, "b")),
            format!("node `a` and node `b` both wrote in step 3, {tail}")
        );
        assert_eq!(
            message(pull(3, "a"), pull(3, "a")),
            format!("node `a` wrote more than once in step 3, {tail}")
        );
        assert_eq!(
            message(TaskPath::input(), TaskPath::input()),
            format!("the input wrote more than once in step 0, {tail}")
        );
        let error = ConflictingWrites {
            first: TaskPath::input(),
            second: TaskPath::input(),
        };
        assert!(error.source().is_none());
    }

    // --- Task ----------------------------------------------------------------

    /// The input is described as such, whatever its step.
    #[test]
    fn task_describes_the_input() {
        assert_eq!(Task(&TaskPath::input()).to_string(), "the input");
    }

    /// A pull task is described by its node, whatever its step.
    #[proptest]
    fn task_describes_a_pull_task_by_its_node(step: u64, #[strategy(0..4_u8)] node: u8) {
        prop_assert_eq!(
            Task(&path(step, node)).to_string(),
            format!("node `n{node}`")
        );
    }
}
