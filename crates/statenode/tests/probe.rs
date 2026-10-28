//! The coverage harness: one state type, [`Probe`], with one context type,
//! [`ProbeCtx`], driven through every run path of every generic function.
//! The builder's error paths are covered by `src/builder.rs`'s unit tests.
//!
//! Region coverage scores each function by its best single instantiation, so
//! every path of every generic function must be exercised with the same type
//! arguments. The other test binaries' states are extra instantiations and
//! do not count towards coverage, so every generic run path is covered here,
//! in this one binary.

use std::io;
use std::num::NonZeroU32;
use std::sync::Arc;

use statenode::{
    BuildError, ChannelSpec, END, GraphBuilder, NameError, NodeContext, NodeInput, NodeOutput,
    ReduceError, RunConfig, RunError, START, State, TaskKind, TaskPath, node,
};

/// The probe state: a value and a log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Probe {
    /// The last value written.
    pub value: i64,
    /// Every message logged, in task-path order.
    pub log: Vec<String>,
}

/// An update to [`Probe`].
#[derive(Debug)]
pub enum ProbeUpdate {
    /// Sets the value.
    Set(i64),
    /// Appends a message to the log.
    Log(String),
    /// Makes `apply_step` fail.
    Reject,
}

impl State for Probe {
    type Update = ProbeUpdate;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![ChannelSpec::new("value")?, ChannelSpec::new("log")?])
    }

    fn apply_step(&mut self, updates: Vec<(TaskPath, ProbeUpdate)>) -> Result<(), ReduceError> {
        for (_, update) in updates {
            match update {
                ProbeUpdate::Set(value) => self.value = value,
                ProbeUpdate::Log(message) => self.log.push(message),
                ProbeUpdate::Reject => {
                    return Err(ReduceError::new("value", "the probe rejects this update"));
                }
            }
        }
        Ok(())
    }
}

/// The probe's run context.
#[derive(Debug, Default)]
pub struct ProbeCtx {
    /// A label nodes copy into the log.
    pub label: &'static str,
}

/// The probe's builder type.
type Builder = GraphBuilder<Probe, ProbeCtx>;

/// The error the probe's failing nodes return.
#[derive(Debug)]
pub struct Failure;

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the probe node failed")
    }
}

impl std::error::Error for Failure {}

/// A tower service node that logs the run context's label, or fails when the
/// value is negative.
fn service_node() -> impl tower::Service<
    NodeInput<Probe, ProbeCtx>,
    Response = NodeOutput<Probe>,
    Error = io::Error,
    Future: Send,
> + Clone
+ Send
+ Sync
+ 'static {
    tower::service_fn(|input: NodeInput<Probe, ProbeCtx>| async move {
        if input.state.value < 0 {
            Err(io::Error::other("negative value"))
        } else {
            Ok(NodeOutput::update(ProbeUpdate::Log(
                input.ctx.context().label.to_owned(),
            )))
        }
    })
}

/// A graph `START → set → log → service → silent → END`.
///
/// # Errors
///
/// If the builder rejects the graph.
fn happy_graph() -> Result<statenode::CompiledGraph<Probe, ProbeCtx>, BuildError> {
    let mut g = Builder::new();
    let set = g.add_node(
        "set",
        node::from_fn(|s: Arc<Probe>, _ctx| async move {
            Ok(NodeOutput::update(ProbeUpdate::Set(s.value + 1)))
        }),
    )?;
    let log = g.add_node(
        "log",
        node::from_fn(|s: Arc<Probe>, ctx: NodeContext<ProbeCtx>| async move {
            let message = format!("{} saw {} in step {}", ctx.name(), s.value, ctx.step());
            Ok(Some(ProbeUpdate::Log(message)))
        }),
    )?;
    let service = g.add_node("service", service_node())?;
    let silent = g.add_node("silent", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    g.add_edge(START, set)?;
    g.add_edge(set, log)?;
    g.add_edge(log, service)?;
    g.add_edge(service, silent)?;
    g.add_edge(silent, END)?;
    g.compile()
}

#[tokio::test]
async fn runs_closure_and_service_nodes_to_the_end() {
    let graph = happy_graph().unwrap();
    let config = RunConfig::new(ProbeCtx { label: "ctx" });
    let out = graph.invoke(ProbeUpdate::Set(10), config).await.unwrap();
    assert_eq!(
        out,
        Probe {
            value: 11,
            log: vec!["log saw 11 in step 1".to_owned(), "ctx".to_owned()],
        }
    );
}

#[tokio::test]
async fn clones_share_the_graph_and_runs_are_independent() {
    let graph = happy_graph().unwrap();
    let clone = graph.clone();
    let first = graph
        .invoke(ProbeUpdate::Set(0), RunConfig::default())
        .await
        .unwrap();
    let second = clone
        .invoke(ProbeUpdate::Set(0), RunConfig::default())
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.log,
        vec!["log saw 1 in step 1".to_owned(), String::new()]
    );
}

#[tokio::test]
async fn a_rejected_input_fails_before_any_node_runs() {
    let error = happy_graph()
        .unwrap()
        .invoke(ProbeUpdate::Reject, RunConfig::default())
        .await
        .unwrap_err();
    let RunError::Reduce(error) = error else {
        panic!("expected a reduce error, got {error:?}");
    };
    assert_eq!(error.channel(), "value");
}

#[tokio::test]
async fn a_rejected_node_update_fails_the_run() {
    let mut g = Builder::default();
    let reject = g
        .add_node(
            "reject",
            node::from_fn(|_s, _ctx| async move { Ok(Some(ProbeUpdate::Reject)) }),
        )
        .unwrap();
    g.add_edge(START, reject).unwrap();
    g.add_edge(reject, END).unwrap();
    let error = g
        .compile()
        .unwrap()
        .invoke(ProbeUpdate::Set(0), RunConfig::default())
        .await
        .unwrap_err();
    assert!(matches!(error, RunError::Reduce(_)), "{error:?}");
}

/// A function node's error fails the run, naming the node and the task, and
/// the node's own error can be recovered by type.
#[tokio::test]
async fn a_failing_function_node_fails_the_run_with_its_error() {
    let mut g = Builder::new();
    let fail = g
        .add_node(
            "fail",
            node::from_fn(|_s, _ctx| async move { Err::<(), _>(Failure.into()) }),
        )
        .unwrap();
    g.add_edge(START, fail).unwrap();
    g.add_edge(fail, END).unwrap();
    let error = g
        .compile()
        .unwrap()
        .invoke(ProbeUpdate::Set(0), RunConfig::default())
        .await
        .unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}");
    };
    assert_eq!(error.node().as_str(), "fail");
    assert_eq!(error.task_path().step(), 0);
    assert!(
        matches!(error.task_path().kind(), TaskKind::Pull { node, .. } if node.as_str() == "fail")
    );
    assert!(error.downcast_ref::<Failure>().is_some());
    assert!(error.downcast_ref::<io::Error>().is_none());
}

#[tokio::test]
async fn a_failing_service_node_fails_the_run_with_its_error() {
    let error = happy_graph()
        .unwrap()
        .invoke(ProbeUpdate::Set(-5), RunConfig::default())
        .await
        .unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}");
    };
    assert_eq!(error.node().as_str(), "service");
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().to_string(),
        "negative value"
    );
}

#[tokio::test]
async fn a_self_loop_stops_at_the_recursion_limit() {
    let mut g = Builder::new();
    let tick = g
        .add_node(
            "tick",
            node::from_fn(
                |s: Arc<Probe>, _ctx| async move { Ok(Some(ProbeUpdate::Set(s.value + 1))) },
            ),
        )
        .unwrap();
    g.add_edge(START, tick).unwrap();
    g.add_edge(tick, tick).unwrap();
    let limit = NonZeroU32::new(3).unwrap();
    let error = g
        .compile()
        .unwrap()
        .invoke(
            ProbeUpdate::Set(0),
            RunConfig::default().with_recursion_limit(limit),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, RunError::RecursionLimit { limit: l, .. } if l == limit),
        "{error:?}"
    );
}

#[tokio::test]
async fn node_contexts_report_their_task_and_clone() {
    let mut g = Builder::new();
    let inspect = g
        .add_node(
            "inspect",
            node::from_fn(|_s, ctx: NodeContext<ProbeCtx>| async move {
                let copy = ctx.clone();
                let same_task = copy.task_path() == ctx.task_path() && copy.name() == ctx.name();
                let message = format!(
                    "{} {} {} {} {}",
                    ctx.name(),
                    ctx.step(),
                    ctx.remaining_steps(),
                    copy.context().label,
                    same_task
                );
                Ok(Some(ProbeUpdate::Log(message)))
            }),
        )
        .unwrap();
    g.add_edge(START, inspect).unwrap();
    g.add_edge(inspect, END).unwrap();
    let config = RunConfig::new(ProbeCtx { label: "shared" })
        .with_recursion_limit(NonZeroU32::new(4).unwrap());
    let out = g
        .compile()
        .unwrap()
        .invoke(ProbeUpdate::Set(0), config)
        .await
        .unwrap();
    assert_eq!(out.log, vec!["inspect 0 3 shared true".to_owned()]);
}

#[tokio::test]
async fn a_cloned_config_shares_the_context_and_the_limit() {
    let config = RunConfig::new(ProbeCtx { label: "shared" })
        .with_recursion_limit(NonZeroU32::new(10).unwrap());
    let graph = happy_graph().unwrap();
    let first = graph
        .invoke(ProbeUpdate::Set(0), config.clone())
        .await
        .unwrap();
    let second = graph.invoke(ProbeUpdate::Set(0), config).await.unwrap();
    assert_eq!(first, second);
    assert_eq!(first.log[1], "shared");
}
