//! Parity: the engine against the naive reference interpreter.

use std::collections::BTreeSet;

use proptest::prelude::*;
use test_strategy::{Arbitrary, proptest};

use statenode::RunError;

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{GraphSpec, sorted};

/// A graph of up to 16 nodes in which every node is reachable: each node but
/// the first in `order` has an edge from a node before it. On top of those,
/// any node may fan out to more nodes or to `END`, including back to earlier
/// nodes, so cycles appear; and `START` may lead to more nodes. Each node
/// yields a random number of times, so the tasks of a step complete in
/// varying orders. No edge is listed twice, since every edge list is a set.
#[derive(Debug, Arbitrary)]
struct Case {
    /// The number of nodes.
    #[strategy(1usize..=16)]
    len: usize,
    /// The nodes, in the order they are reached: `START` leads to the first.
    #[strategy(Just((0..#len).collect::<Vec<usize>>()).prop_shuffle())]
    order: Vec<usize>,
    /// For the `i`th node in `order`, which of the nodes before it leads to
    /// it (ignored for the first).
    #[strategy(prop::collection::vec(any::<prop::sample::Index>(), #len))]
    parents: Vec<prop::sample::Index>,
    /// More edges from each node, by node index: 0 is `END`, `k` is node
    /// `k - 1`.
    #[strategy(prop::collection::vec(prop::collection::btree_set(0..=#len, 0..3), #len))]
    extra: Vec<BTreeSet<usize>>,
    /// More edges from `START`, in the same encoding.
    #[strategy(prop::collection::btree_set(0..=#len, 0..3))]
    extra_start: BTreeSet<usize>,
    /// How many times each node yields before it finishes.
    #[strategy(prop::collection::vec(0..4usize, #len))]
    yields: Vec<usize>,
    /// The recursion limit.
    #[strategy(1u32..=40)]
    limit: u32,
}

impl Case {
    /// The graph this case describes.
    fn spec(&self) -> GraphSpec {
        let end = |k: usize| k.checked_sub(1);
        let mut successors: Vec<BTreeSet<Option<usize>>> = self
            .extra
            .iter()
            .map(|ends| ends.iter().map(|&k| end(k)).collect())
            .collect();
        let tree = self
            .parents
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(i, parent)| {
                Some((*self.order.get(parent.index(i))?, *self.order.get(i)?))
            });
        for (parent, child) in tree {
            if let Some(ends) = successors.get_mut(parent) {
                ends.insert(Some(child));
            }
        }
        let start: BTreeSet<Option<usize>> = self
            .order
            .first()
            .map(|&first| Some(first))
            .into_iter()
            .chain(self.extra_start.iter().map(|&k| end(k)))
            .collect();
        GraphSpec {
            len: self.len,
            start: start.into_iter().collect(),
            successors: successors
                .into_iter()
                .map(|ends| ends.into_iter().collect())
                .collect(),
            yields: self.yields.clone(),
        }
    }
}

/// For every generated graph, limit and completion order, the engine visits
/// exactly the nodes, steps and remaining-step counts the reference visits,
/// finishes with the reference's trace, and fails with the recursion limit
/// exactly when the reference does. The reference ignores `yields`, so this
/// is also the property that results never depend on completion order.
#[proptest(async = "tokio", cases = 256)]
async fn invoke_matches_the_reference_interpreter(case: Case) {
    let spec = case.spec();
    let (result, visits) = spec.run(case.limit).await.unwrap();
    let expected = spec.interpret(case.limit);
    prop_assert_eq!(sorted(visits), expected.visits);
    match result {
        Ok(trace) => {
            prop_assert!(!expected.limited);
            prop_assert_eq!(trace.visits, expected.trace);
        }
        Err(RunError::RecursionLimit { limit, .. }) => {
            prop_assert!(expected.limited);
            prop_assert_eq!(limit.get(), case.limit);
        }
        Err(other) => prop_assert!(false, "unexpected error {:?}", other),
    }
}

/// Building the same graph twice, so with two different builder tags, gives
/// identical runs.
#[proptest(async = "tokio", cases = 64)]
async fn runs_do_not_depend_on_the_builder(case: Case) {
    let spec = case.spec();
    let (first, first_visits) = spec.run(case.limit).await.unwrap();
    let (second, second_visits) = spec.run(case.limit).await.unwrap();
    prop_assert_eq!(sorted(first_visits), sorted(second_visits));
    prop_assert_eq!(first.ok(), second.ok());
}
