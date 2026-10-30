//! Joins: a join `add_edge([a, b], c)` fires, running `c`, once every source
//! has run since it last fired; each join is its own barrier; and a join
//! part-filled when a run ends is no error.

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{GraphSpec, Visit};

/// The steps in which node `node` ran, in order.
fn steps_of(visits: &[Visit], node: usize) -> Vec<u64> {
    let name = GraphSpec::name(node);
    let mut steps: Vec<u64> = visits
        .iter()
        .filter(|visit| visit.node == name)
        .map(|visit| visit.step)
        .collect();
    steps.sort_unstable();
    steps
}

/// `[a, b] → c`, with `a` running in step 0 and `b` only in step 1: `c`
/// waits for `b`, and runs once, in step 2.
#[tokio::test]
async fn a_join_waits_for_every_source() {
    // a = 0, x = 1, b = 2, c = 3: START → a, START → x → b, [a, b] → c.
    let spec = GraphSpec {
        len: 4,
        start: vec![Some(0), Some(1)],
        successors: vec![Vec::new(), vec![Some(2)], Vec::new(), Vec::new()],
        joins: vec![(vec![0, 2], 3)],
        yields: Vec::new(),
    };
    let (result, visits) = spec.run(20).await.unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(steps_of(&visits, 3), [2]);
}

/// Two joins into one node are two barriers: each runs the node once its
/// own sources have run, here in consecutive steps.
#[tokio::test]
async fn separate_joins_into_one_node_are_separate_barriers() {
    // a = 0, b = 1, c = 2, e = 3, d = 4: START → {a, b, c}, c → e,
    // [a, b] → d and [c, e] → d.
    let spec = GraphSpec {
        len: 5,
        start: vec![Some(0), Some(1), Some(2)],
        successors: vec![
            Vec::new(),
            Vec::new(),
            vec![Some(3)],
            Vec::new(),
            Vec::new(),
        ],
        joins: vec![(vec![0, 1], 4), (vec![2, 3], 4)],
        yields: Vec::new(),
    };
    let (result, visits) = spec.run(20).await.unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(steps_of(&visits, 4), [1, 2]);
}

/// A node that a plain edge and a complete join trigger in the same step
/// runs once.
#[tokio::test]
async fn an_edge_and_a_join_into_one_node_run_it_once() {
    // a = 0, b = 1, c = 2: START → {a, b}, a → c, [a, b] → c.
    let spec = GraphSpec {
        len: 3,
        start: vec![Some(0), Some(1)],
        successors: vec![vec![Some(2)], Vec::new(), Vec::new()],
        joins: vec![(vec![0, 1], 2)],
        yields: Vec::new(),
    };
    let (result, visits) = spec.run(20).await.unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(steps_of(&visits, 2), [1]);
}

/// A run can end with a join part-filled, without error: `b` runs again in
/// the step its join fires, so it counts towards a next round that never
/// comes.
#[tokio::test]
async fn a_run_can_end_with_a_partial_barrier() {
    // a = 0, x = 1, y = 2, b = 3, d = 4: START → a, START → x → y → b,
    // a → b, [a, b] → d.
    let spec = GraphSpec {
        len: 5,
        start: vec![Some(0), Some(1)],
        successors: vec![
            vec![Some(3)],
            vec![Some(2)],
            vec![Some(3)],
            Vec::new(),
            Vec::new(),
        ],
        joins: vec![(vec![0, 3], 4)],
        yields: Vec::new(),
    };
    let (result, visits) = spec.run(20).await.unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(steps_of(&visits, 3), [1, 2], "b runs twice");
    assert_eq!(steps_of(&visits, 4), [2], "d runs once");
}

/// A source that runs in the step its join fires counts towards the join's
/// next round, so the join fires again once the other source runs too.
#[tokio::test]
async fn a_source_that_reruns_as_its_join_fires_counts_for_the_next_round() {
    // a = 0, b = 1, x = 2, c = 3: START → {a, b, x}, x → b, [a, b] → c,
    // c → a.
    let spec = GraphSpec {
        len: 4,
        start: vec![Some(0), Some(1), Some(2)],
        successors: vec![Vec::new(), Vec::new(), vec![Some(1)], vec![Some(0)]],
        joins: vec![(vec![0, 1], 3)],
        yields: Vec::new(),
    };
    let (result, visits) = spec.run(20).await.unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(steps_of(&visits, 3), [1, 3]);
}
