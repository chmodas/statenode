//! The engine: compiled graphs and how they run.
//!
//! A run applies its input in its own barrier, then repeats supersteps until
//! no node is planned. Each superstep has three phases:
//!
//! 1. **Plan** the step's tasks ([`Scheduler::plan`]).
//! 2. **Execute** them concurrently, against one immutable snapshot of the
//!    state, and wait for every one to finish.
//! 3. **Apply** their updates in task-path order with
//!    [`State::apply_step`], then close the step ([`Scheduler::barrier`]),
//!    recording what it wrote, and which joins its tasks arrived at, so the
//!    next step can plan from it.
//!
//! Which tasks a step plans is decided by the scheduler in [`pregel`], the
//! one part of the engine tied to the scheduling algorithm.
//!
//! The tasks of a step are polled together on the run's own task: this is
//! concurrency, not parallelism. Nodes that wait, on a model or a tool, overlap;
//! nodes that compute run one after another. A node that needs a thread of
//! its own spawns one.
//!
//! Planning and execution change nothing, and the barrier runs only after the
//! updates were applied, so a failing step leaves the scheduling state as it
//! was.
//!
//! A failed run is reported as a [`RunError`].

mod pregel;

use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use futures_util::future::join_all;
use tower::ServiceExt as _;

use crate::builder::ValidTopology;
use crate::config::RunConfig;
use crate::node::{BoxNode, NodeContext, NodeError, NodeInput};
use crate::state::{ReduceError, State};
use crate::task::TaskPath;
use pregel::{Scheduler, Wiring};

/// A validated, immutable graph, ready to run.
///
/// Cloning is cheap and clones share the graph, so one compiled graph can
/// serve many concurrent runs.
pub struct CompiledGraph<S: State, C = ()> {
    /// The shared graph.
    inner: Arc<Inner<S, C>>,
}

/// What a [`CompiledGraph`] shares between its clones.
struct Inner<S: State, C> {
    /// The state every run starts from, before its input is applied.
    initial: S,
    /// The nodes, by node index.
    nodes: Vec<BoxNode<S, C>>,
    /// The topology, lowered for the scheduler.
    wiring: Wiring,
}

impl<S: State, C> Clone for CompiledGraph<S, C> {
    /// Another handle to the same graph.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S: State, C: Send + Sync + 'static> CompiledGraph<S, C> {
    /// A compiled graph from validated parts.
    pub(crate) fn new(initial: S, nodes: Vec<BoxNode<S, C>>, topology: ValidTopology) -> Self {
        Self {
            inner: Arc::new(Inner {
                initial,
                nodes,
                wiring: Wiring::new(topology),
            }),
        }
    }

    /// Runs the graph: applies `input` to the initial state, runs supersteps
    /// until no node is planned, and returns the final state.
    ///
    /// # Errors
    ///
    /// Returns [`RunError::Node`] if a node fails (the first by name, if
    /// several fail in one step), [`RunError::Reduce`] if the state rejects an
    /// update (including the input), and [`RunError::RecursionLimit`] if a
    /// node is planned for the superstep the recursion limit forbids.
    pub async fn invoke(&self, input: S::Update, config: RunConfig<C>) -> Result<S, RunError> {
        let inner = &*self.inner;
        let limit = config.recursion_limit;
        let mut state = Arc::new(inner.initial.clone());
        Arc::make_mut(&mut state)
            .apply_step(vec![(TaskPath::input(), input)])
            .map_err(RunError::Reduce)?;
        let mut scheduler = Scheduler::new(&inner.wiring);
        scheduler.barrier(&[], inner.wiring.start_targets.clone());

        let mut step: u64 = 0;
        loop {
            let tasks = scheduler.plan();
            if tasks.is_empty() {
                return Ok(Arc::unwrap_or_clone(state));
            }
            if step >= u64::from(limit.get()) {
                return Err(RunError::RecursionLimit { limit });
            }
            let remaining_steps = u64::from(limit.get()) - step - 1;

            // One future per task, polled together. `join_all` returns their
            // results in plan order, which is task-path order, so the first
            // error collected is the first failing task by name, and the
            // updates arrive sorted.
            let runs = tasks
                .iter()
                .filter_map(|&node| inner.nodes.get(node).zip(inner.wiring.names.get(node)))
                .map(|(node, name)| {
                    let task_path = TaskPath::pull(step, name.clone());
                    let ctx = NodeContext::new(
                        task_path.clone(),
                        name.clone(),
                        remaining_steps,
                        Arc::clone(&config.context),
                    );
                    let input = NodeInput {
                        state: Arc::clone(&state),
                        ctx,
                    };
                    async move {
                        match node.clone().oneshot(input).await {
                            Ok(output) => Ok((task_path, output)),
                            Err(source) => Err(NodeError::new(name.clone(), task_path, source)),
                        }
                    }
                });
            let outputs = join_all(runs)
                .await
                .into_iter()
                .collect::<Result<Vec<_>, NodeError>>()
                .map_err(RunError::Node)?;
            let updates = outputs
                .into_iter()
                .filter_map(|(task_path, output)| {
                    output.into_update().map(|update| (task_path, update))
                })
                .collect();

            Arc::make_mut(&mut state)
                .apply_step(updates)
                .map_err(RunError::Reduce)?;
            let written = scheduler.written_by(&tasks);
            scheduler.barrier(&tasks, written);
            step += 1;
        }
    }
}

/// A run failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunError {
    /// A task was planned for the superstep the recursion limit forbids.
    ///
    /// Non-exhaustive, so the error can gain fields without a breaking change.
    #[non_exhaustive]
    RecursionLimit {
        /// The limit the run was configured with.
        limit: NonZeroU32,
    },
    /// A node returned an error.
    ///
    /// Every task of the step ran to completion. If several failed, this is
    /// the first by task path, so by node name, whatever order they finished
    /// in.
    Node(NodeError),
    /// The state rejected an update.
    Reduce(ReduceError),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RecursionLimit { limit } => write!(
                f,
                "the run reached its recursion limit of {limit} supersteps"
            ),
            Self::Node(e) => fmt::Display::fmt(e, f),
            Self::Reduce(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RecursionLimit { .. } => None,
            Self::Node(e) => e.source(),
            Self::Reduce(e) => e.source(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for compiled graphs and run errors.

    use std::any::TypeId;
    use std::error::Error as _;

    use super::*;
    use crate::builder::{END, GraphBuilder, START};
    use crate::name::NodeName;
    use crate::node;
    use crate::testing::{Counter, noop};

    // --- CompiledGraph -------------------------------------------------------

    /// A compiled graph without a run context never has to name `C`: it
    /// defaults to `()`.
    #[test]
    fn the_context_defaults_to_unit() {
        assert_eq!(
            TypeId::of::<CompiledGraph<Counter>>(),
            TypeId::of::<CompiledGraph<Counter, ()>>()
        );
    }

    /// `invoke`'s future is `Send`, so a run that owns its graph can be
    /// spawned, and compiled graphs are `Send + Sync`.
    ///
    /// This is the only test that runs the shared [`noop`] node and
    /// `Counter::apply_step`, so it is what covers them; the builder's tests
    /// add `noop` without running it.
    #[tokio::test]
    async fn invoke_returns_a_send_future_and_graphs_are_shareable() {
        // Users spawn runs onto multi-threaded runtimes and share graphs
        // across tasks, so these bounds are part of the API.
        /// Compiles only if `T` is `Send + 'static`.
        fn assert_send<T: Send + 'static>(_: &T) {}
        /// Compiles only if `T` is `Send + Sync`.
        fn assert_send_sync<T: Send + Sync + 'static>() {}

        let mut g = GraphBuilder::<Counter>::new();
        let n = g.add_node("n", node::from_fn(noop)).unwrap();
        g.add_edge(START, n).unwrap();
        g.add_edge(n, END).unwrap();
        let graph = g.compile().unwrap();
        assert_send_sync::<CompiledGraph<Counter>>();
        let run = async move { graph.invoke(41, RunConfig::default()).await };
        assert_send(&run);
        assert_eq!(run.await.unwrap().count, 41);
    }

    // --- RunError ------------------------------------------------------------

    /// The recursion limit names the limit and has no source; node and reduce
    /// errors forward both their message and their source unchanged.
    #[test]
    fn error_messages_name_the_problem() {
        let limit = RunError::RecursionLimit {
            limit: NonZeroU32::new(25).unwrap(),
        };
        assert_eq!(
            limit.to_string(),
            "the run reached its recursion limit of 25 supersteps"
        );
        assert!(limit.source().is_none());

        let name = NodeName::new("n").unwrap();
        let node = RunError::Node(NodeError::new(
            name.clone(),
            TaskPath::pull(2, name),
            "boom".into(),
        ));
        assert_eq!(node.to_string(), "node `n` failed in step 2");
        assert_eq!(node.source().unwrap().to_string(), "boom");

        let reduce = RunError::Reduce(ReduceError::new("count", "must not be negative"));
        assert_eq!(reduce.to_string(), "channel `count` rejected an update");
        assert_eq!(reduce.source().unwrap().to_string(), "must not be negative");
    }
}
