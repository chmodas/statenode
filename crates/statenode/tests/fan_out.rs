//! Fan-out: a superstep's tasks run concurrently against one snapshot, cost
//! one step of the recursion limit together, merge in task-path order, and
//! fail with the first failing task by name.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use proptest::prelude::*;
use statenode::reducer::ConflictingWrites;
use statenode::{
    BoxError, END, GraphBuilder, NodeContext, RunConfig, RunError, START, TaskKind, TaskPath, node,
};
use test_strategy::proptest;
use tokio::sync::Barrier;

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{Counter, CounterUpdate, GraphSpec, Trace};

/// What a sibling saw of its step: the state's visits, its step, and its
/// remaining steps.
type View = (Vec<String>, u64, u64);

/// The name of the node `path` ran, if it is a pull task.
fn pulled(path: &TaskPath) -> Option<&str> {
    match path.kind() {
        TaskKind::Pull { node, .. } => Some(node.as_str()),
        _ => None,
    }
}

/// Two siblings that each wait for the other at a barrier both finish: run
/// one after the other, the first would wait forever, so they run
/// concurrently, even on a single-threaded runtime.
#[tokio::test]
async fn branches_run_concurrently() {
    let barrier = Arc::new(Barrier::new(2));
    let mut g = GraphBuilder::<Counter>::new();
    for name in ["a", "b"] {
        let barrier = Arc::clone(&barrier);
        let id = g
            .add_node(
                name,
                node::from_fn(move |_s: Arc<Counter>, _ctx| {
                    let barrier = Arc::clone(&barrier);
                    async move {
                        barrier.wait().await;
                        Ok(())
                    }
                }),
            )
            .unwrap();
        g.add_edge(START, id).unwrap();
        g.add_edge(id, END).unwrap();
    }
    let graph = g.compile().unwrap();
    let run = graph.invoke(CounterUpdate::default(), RunConfig::default());
    let result = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("run one after the other, the siblings would never pass the barrier");
    assert!(result.is_ok(), "{result:?}");
}

/// Siblings all read the state as it was before their step, never a
/// sibling's write, and see the same step and remaining steps.
#[tokio::test]
async fn siblings_share_one_snapshot() {
    let views: Arc<Mutex<Vec<View>>> = Arc::default();
    let mut g = GraphBuilder::<Trace>::new();
    let seed = g
        .add_node(
            "seed",
            node::from_fn(|_s, _ctx| async move { Ok(Some(String::from("seed"))) }),
        )
        .unwrap();
    g.add_edge(START, seed).unwrap();
    for name in ["a", "b", "c"] {
        let views = Arc::clone(&views);
        let id = g
            .add_node(
                name,
                node::from_fn(move |s: Arc<Trace>, ctx: NodeContext| {
                    let views = Arc::clone(&views);
                    async move {
                        let view = (s.visits.clone(), ctx.step(), ctx.remaining_steps());
                        views.lock().map_err(|e| e.to_string())?.push(view);
                        Ok(Some(ctx.name().as_str().to_owned()))
                    }
                }),
            )
            .unwrap();
        g.add_edge(seed, id).unwrap();
        g.add_edge(id, END).unwrap();
    }
    let limit = 10;
    let config =
        RunConfig::default().with_recursion_limit(std::num::NonZeroU32::new(limit).unwrap());
    let trace = g
        .compile()
        .unwrap()
        .invoke(String::from("input"), config)
        .await
        .unwrap();
    assert_eq!(trace.visits, ["input", "seed", "a", "b", "c"]);
    let expected = (
        vec![String::from("input"), String::from("seed")],
        1,
        u64::from(limit) - 2,
    );
    assert_eq!(*views.lock().unwrap(), vec![expected; 3]);
}

/// Three branches run in one superstep, so a run with a limit of one
/// finishes.
#[tokio::test]
async fn a_fan_out_costs_one_step_of_the_limit() {
    let spec = GraphSpec {
        len: 3,
        start: vec![Some(0), Some(1), Some(2)],
        successors: vec![vec![None]; 3],
        yields: vec![0; 3],
    };
    let (result, visits) = spec.run(1).await.unwrap();
    assert_eq!(
        result.unwrap().visits,
        ["input", "n0", "n1", "n2"],
        "every branch ran"
    );
    assert!(visits.iter().all(|visit| visit.step == 0));
}

/// Two branches that write the same `replace` channel in one-step fail the
/// run with `ConflictingWrites`, naming both writers in task-path order.
#[tokio::test]
async fn two_branches_writing_a_replace_channel_conflict() {
    let mut g = GraphBuilder::<Counter>::new();
    for name in ["b", "a"] {
        let id = g
            .add_node(
                name,
                node::from_fn(|_s, _ctx| async move { Ok(Some(CounterUpdate { count: Some(1) })) }),
            )
            .unwrap();
        g.add_edge(START, id).unwrap();
        g.add_edge(id, END).unwrap();
    }
    let error = g
        .compile()
        .unwrap()
        .invoke(CounterUpdate::default(), RunConfig::default())
        .await
        .unwrap_err();
    let RunError::Reduce(error) = error else {
        panic!("expected a reduce error, got {error:?}");
    };
    assert_eq!(error.channel(), "count");
    let conflict = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<ConflictingWrites>())
        .expect("caused by ConflictingWrites");
    assert_eq!(
        (conflict.first().step(), pulled(conflict.first())),
        (0, Some("a"))
    );
    assert_eq!(
        (conflict.second().step(), pulled(conflict.second())),
        (0, Some("b"))
    );
}

/// However, the tasks of a step complete, the run fails with the first
/// failing task by name, and only after every task has run.
#[proptest(async = "tokio")]
async fn the_first_failing_branch_by_name_is_reported(
    #[strategy(2..=12usize)] len: usize,
    /// The branches that fail.
    #[strategy(prop::collection::btree_set(0..#len, 1..=#len))]
    failing: BTreeSet<usize>,
    /// How many times each branch yields before it finishes.
    #[strategy(prop::collection::vec(0..8usize, #len))]
    yields: Vec<usize>,
) {
    let ran: Arc<Mutex<BTreeSet<String>>> = Arc::default();
    let mut g = GraphBuilder::<Counter>::new();
    for (i, &yields) in yields.iter().enumerate() {
        let ran = Arc::clone(&ran);
        let fails = failing.contains(&i);
        let id = g
            .add_node(
                &GraphSpec::name(i),
                node::from_fn(move |_s: Arc<Counter>, ctx: NodeContext| {
                    let ran = Arc::clone(&ran);
                    async move {
                        for _ in 0..yields {
                            tokio::task::yield_now().await;
                        }
                        let name = ctx.name().as_str().to_owned();
                        ran.lock().map_err(|e| e.to_string())?.insert(name.clone());
                        if fails {
                            return Err(BoxError::from(name));
                        }
                        Ok(())
                    }
                }),
            )
            .unwrap();
        g.add_edge(START, id).unwrap();
        g.add_edge(id, END).unwrap();
    }
    let error = g
        .compile()
        .unwrap()
        .invoke(CounterUpdate::default(), RunConfig::default())
        .await
        .unwrap_err();
    let first = failing.iter().map(|&i| GraphSpec::name(i)).min().unwrap();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}");
    };
    prop_assert_eq!(error.node().as_str(), first.as_str());
    prop_assert_eq!(
        ran.lock().unwrap().clone(),
        (0..len).map(GraphSpec::name).collect::<BTreeSet<_>>(),
        "every task ran to completion"
    );
}

/// Wide fan-outs, whose branches complete in reverse order, still merge
/// their updates in task-path order, either side of 30 branches, where
/// `join_all` switches from polling each future to a `FuturesOrdered`.
#[tokio::test]
async fn wide_fan_outs_merge_in_task_path_order() {
    for len in [30, 31, 1_000] {
        let spec = GraphSpec {
            len,
            start: (0..len).map(Some).collect(),
            successors: vec![vec![None]; len],
            yields: (0..len).rev().collect(),
        };
        let (result, _) = spec.run(1).await.unwrap();
        assert_eq!(result.unwrap().visits, spec.interpret(1).trace, "len {len}");
    }
}

/// When every branch of a wide fan-out fails, completing in reverse order,
/// the first by name is reported, either side of `join_all`'s switch at 30
/// branches.
#[tokio::test]
async fn wide_fan_outs_report_the_first_failure_by_name() {
    for len in [30, 31, 1_000] {
        let mut g = GraphBuilder::<Counter>::new();
        for i in 0..len {
            let yields = len - 1 - i;
            let id = g
                .add_node(
                    &GraphSpec::name(i),
                    node::from_fn(move |_s: Arc<Counter>, ctx: NodeContext| async move {
                        for _ in 0..yields {
                            tokio::task::yield_now().await;
                        }
                        Err::<(), _>(BoxError::from(ctx.name().as_str().to_owned()))
                    }),
                )
                .unwrap();
            g.add_edge(START, id).unwrap();
        }
        let error = g
            .compile()
            .unwrap()
            .invoke(CounterUpdate::default(), RunConfig::default())
            .await
            .unwrap_err();
        let RunError::Node(error) = error else {
            panic!("expected a node error, got {error:?}");
        };
        assert_eq!(error.node().as_str(), "n0", "len {len}");
    }
}
