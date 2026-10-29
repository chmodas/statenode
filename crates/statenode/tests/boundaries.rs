//! Boundaries: recursion limits at their edges, long chains, and
//! `remaining_steps()` at the first and last step.

use statenode::{DEFAULT_RECURSION_LIMIT, RunError};

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{GraphSpec, Visit};

/// A chain of `len` nodes ending at `END`.
fn chain(len: usize) -> GraphSpec {
    GraphSpec::chain(len, None)
}

/// The default recursion limit is LangGraph's, 10,007 supersteps.
#[test]
fn the_default_limit_is_langgraphs() {
    assert_eq!(DEFAULT_RECURSION_LIMIT.get(), 10_007);
}

/// A chain that needs exactly `limit` supersteps finishes, visiting every
/// node.
#[tokio::test]
async fn a_graph_finishing_in_exactly_limit_steps_succeeds() {
    for len in [1, 2, 7] {
        let limit = u32::try_from(len).unwrap();
        let (result, visits) = chain(len).run(limit).await.unwrap();
        assert!(result.is_ok(), "len {len}: {result:?}");
        assert_eq!(visits.len(), len);
    }
}

/// A chain that needs one superstep more than `limit` fails with the
/// recursion limit, before running its last node.
#[tokio::test]
async fn a_graph_needing_one_more_step_than_the_limit_fails() {
    for len in [2, 3, 8] {
        let limit = u32::try_from(len - 1).unwrap();
        let (result, visits) = chain(len).run(limit).await.unwrap();
        assert!(
            matches!(result, Err(RunError::RecursionLimit { .. })),
            "len {len}: {result:?}"
        );
        assert_eq!(
            visits.len(),
            len - 1,
            "the limit stops the run before the extra step"
        );
    }
}

/// With the largest limit, a short chain finishes, and its first node sees
/// all but one of the steps remaining.
#[tokio::test]
async fn the_largest_limit_lets_a_short_graph_finish() {
    let (result, visits) = chain(3).run(u32::MAX).await.unwrap();
    assert!(result.is_ok());
    assert_eq!(visits[0].remaining, u64::from(u32::MAX) - 1);
}

/// A two-node cycle runs until the limit, whether it stops after the first
/// or the second node, and runs one node per allowed step.
#[tokio::test]
async fn a_two_node_cycle_stops_at_limits_one_and_three() {
    let cycle = GraphSpec::chain(2, Some(0));
    for limit in [1, 3] {
        let (result, visits) = cycle.run(limit).await.unwrap();
        assert!(
            matches!(result, Err(RunError::RecursionLimit { limit: l, .. }) if l.get() == limit)
        );
        assert_eq!(visits.len(), usize::try_from(limit).unwrap());
    }
}

/// `remaining_steps()` counts down by one per step and is zero in the last
/// step the limit allows.
#[tokio::test]
async fn remaining_steps_counts_down_to_zero_in_the_last_allowed_step() {
    let (result, visits) = chain(3).run(3).await.unwrap();
    assert!(result.is_ok());
    let remaining: Vec<u64> = visits.iter().map(|v: &Visit| v.remaining).collect();
    assert_eq!(remaining, vec![2, 1, 0]);
    assert_eq!(
        visits.iter().map(|v| v.step).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

/// A chain of 10,000 nodes runs to its end within the default limit.
#[tokio::test]
async fn a_ten_thousand_node_chain_runs_to_the_end() {
    let len = 10_000;
    let (result, visits) = chain(len).run(DEFAULT_RECURSION_LIMIT.get()).await.unwrap();
    let trace = result.unwrap();
    assert_eq!(trace.visits.len(), len + 1, "the input and every node");
    assert_eq!(visits.last().unwrap().node, GraphSpec::name(len - 1));
}
