//! Helpers for implementing [`State::apply_step`](crate::State::apply_step).
//!
//! Each helper takes a channel's name, its field, and the writes addressed to
//! it, in task-path order, and folds the writes into the field. When the
//! writes break the channel's rule, the helper returns a [`ReduceError`]
//! naming the channel and leaves the field as it was.

use std::fmt;

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
        let rule =
            "but a replace channel accepts one write per step; use a reducer that merges writes";
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

    // --- replace -------------------------------------------------------------

    /// A pull task path from a few nodes, so that the same task can come up
    /// more than once.
    fn path(step: u64, node: u8) -> TaskPath {
        pull(step, &format!("n{node}"))
    }

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

    // --- ConflictingWrites ---------------------------------------------------

    /// The message names the tasks and the step; there is no source.
    #[test]
    fn error_messages_name_the_problem() {
        let message = |first, second| ConflictingWrites { first, second }.to_string();
        let tail =
            "but a replace channel accepts one write per step; use a reducer that merges writes";
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
