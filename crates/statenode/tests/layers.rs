//! Nodes wrapped in tower middleware: the stacks `add_node` accepts, and
//! what they do to a run.

use std::fmt;
use std::future::{Ready, ready};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use statenode::{
    BoxError, BuildError, CompiledGraph, END, GraphBuilder, NodeInput, NodeOutput, RunConfig,
    RunError, START, node,
};
use tower::retry::Policy;
use tower::util::BoxCloneSyncService;
use tower::{Service, ServiceBuilder};

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::{Counter, CounterUpdate};

/// The error a [`flaky`] node fails with.
#[derive(Debug)]
struct Flake;

impl fmt::Display for Flake {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the node flaked")
    }
}

impl std::error::Error for Flake {}

/// A node that fails its first `failures` calls with [`Flake`], then sets
/// `count` to the number of calls so far; `calls` counts every call.
fn flaky(
    failures: i64,
    calls: Arc<AtomicI64>,
) -> impl Service<NodeInput<Counter>, Response = NodeOutput<Counter>, Error = Flake, Future: Send + Sync>
+ Clone
+ Send
+ Sync
+ 'static {
    tower::service_fn(move |_input: NodeInput<Counter>| {
        let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
        ready(if call > failures {
            Ok(NodeOutput::update(CounterUpdate { count: Some(call) }))
        } else {
            Err(Flake)
        })
    })
}

/// A retry policy that retries every error, up to `retries` more attempts
/// per request.
#[derive(Debug, Clone)]
struct Attempts {
    /// The retries left for this request.
    retries: usize,
}

impl<Req: Clone, Res, E> Policy<Req, Res, E> for Attempts {
    type Future = Ready<()>;

    fn retry(&mut self, _req: &mut Req, result: &mut Result<Res, E>) -> Option<Self::Future> {
        (result.is_err() && self.retries > 0).then(|| {
            self.retries -= 1;
            ready(())
        })
    }

    fn clone_request(&mut self, req: &Req) -> Option<Req> {
        Some(req.clone())
    }
}

/// A node that is never ready: its `poll_ready` fails.
#[derive(Debug, Clone)]
struct NeverReady;

impl Service<NodeInput<Counter>> for NeverReady {
    type Response = NodeOutput<Counter>;
    type Error = BoxError;
    type Future = Ready<Result<Self::Response, BoxError>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        Poll::Ready(Err("the node is not ready".into()))
    }

    fn call(&mut self, _input: NodeInput<Counter>) -> Self::Future {
        ready(Ok(NodeOutput::from(())))
    }
}

/// A graph `START → node → END` around `service`.
///
/// # Errors
///
/// If the builder rejects the graph.
fn single<T>(service: T) -> Result<CompiledGraph<Counter>, BuildError>
where
    T: Service<NodeInput<Counter>, Response = NodeOutput<Counter>> + Clone + Send + Sync + 'static,
    T::Error: Into<BoxError>,
    T::Future: Send + 'static,
{
    let mut g = GraphBuilder::<Counter>::new();
    let node = g.add_node("node", service)?;
    g.add_edge(START, node)?;
    g.add_edge(node, END)?;
    g.compile()
}

/// Runs `graph` once from the default state.
async fn run(graph: &CompiledGraph<Counter>) -> Result<Counter, RunError> {
    graph
        .invoke(CounterUpdate::default(), RunConfig::default())
        .await
}

/// `Retry` resends the cloned input until the node succeeds.
#[tokio::test]
async fn retry_recovers_a_flaky_node() {
    let calls = Arc::new(AtomicI64::new(0));
    let stack = ServiceBuilder::new()
        .retry(Attempts { retries: 2 })
        .service(flaky(2, Arc::clone(&calls)));
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

/// A `node::from_fn` node takes layers like any other service: under
/// `Retry`, a function node that fails once runs to completion.
#[tokio::test]
async fn a_function_node_takes_layers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let stack = ServiceBuilder::new()
        .retry(Attempts { retries: 1 })
        .service(node::from_fn(move |_s, _ctx| {
            let call = counted.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                if call > 1 {
                    Ok(Some(CounterUpdate {
                        count: Some(i64::try_from(call)?),
                    }))
                } else {
                    Err(Flake.into())
                }
            }
        }));
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// Once its retries run out, `Retry` fails the run with the node's own
/// error.
#[tokio::test]
async fn retry_gives_up_with_the_nodes_error() {
    let calls = Arc::new(AtomicI64::new(0));
    let stack = ServiceBuilder::new()
        .retry(Attempts { retries: 1 })
        .service(flaky(5, Arc::clone(&calls)));
    let error = run(&single(stack).unwrap()).await.unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}")
    };
    assert!(error.downcast_ref::<Flake>().is_some(), "{error:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// A node outliving its `Timeout` fails the run with tower's `Elapsed`.
#[tokio::test(start_paused = true)]
async fn timeout_fails_with_elapsed() {
    let slow = tower::service_fn(|_input: NodeInput<Counter>| async {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok::<_, BoxError>(NodeOutput::from(()))
    });
    let stack = ServiceBuilder::new()
        .timeout(Duration::from_secs(1))
        .service(slow);
    let error = run(&single(stack).unwrap()).await.unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}")
    };
    assert!(
        error
            .downcast_ref::<tower::timeout::error::Elapsed>()
            .is_some(),
        "{error:?}"
    );
}

/// A concurrency limit, a retry and a timeout compose in one stack.
#[tokio::test]
async fn a_composed_stack_runs() {
    let calls = Arc::new(AtomicI64::new(0));
    let stack = ServiceBuilder::new()
        .concurrency_limit(1)
        .retry(Attempts { retries: 1 })
        .timeout(Duration::from_secs(1))
        .service(flaky(1, Arc::clone(&calls)));
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// The retry policy is cloned for every task, so a run does not use up the
/// retries of the next one.
#[tokio::test]
async fn retries_are_counted_per_task() {
    let calls = Arc::new(AtomicI64::new(0));
    let stack = ServiceBuilder::new()
        .retry(Attempts { retries: 1 })
        .service(flaky(1, Arc::clone(&calls)));
    let graph = single(stack).unwrap();
    assert_eq!(run(&graph).await.unwrap().count, 2);
    assert_eq!(run(&graph).await.unwrap().count, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

/// `RateLimit` is not `Clone`, but behind `Buffer` it is accepted and runs.
#[tokio::test]
async fn rate_limit_runs_behind_buffer() {
    let stack = ServiceBuilder::new()
        .buffer(4)
        .rate_limit(10, Duration::from_secs(1))
        .service(flaky(0, Arc::default()));
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 1);
}

/// A node's error passes through `Buffer` intact.
#[tokio::test]
async fn buffer_keeps_the_nodes_error() {
    let stack = ServiceBuilder::new()
        .buffer(1)
        .service(flaky(1, Arc::default()));
    let error = run(&single(stack).unwrap()).await.unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}")
    };
    assert!(error.downcast_ref::<Flake>().is_some(), "{error:?}");
}

/// Mapping layers let the inner service use its own request and response
/// types.
#[tokio::test]
async fn mapping_layers_adapt_an_inner_service() {
    let inner = tower::service_fn(|state: Arc<Counter>| ready(Ok::<_, BoxError>(state.count + 1)));
    let stack = ServiceBuilder::new()
        .map_request(|input: NodeInput<Counter>| input.state)
        .map_response(|count| NodeOutput::update(CounterUpdate { count: Some(count) }))
        .service(inner);
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 1);
}

/// A stack boxed with `boxed_clone_sync`, the `Sync` counterpart of
/// `boxed_clone`, is accepted and runs when its future is `Sync` too.
#[tokio::test]
async fn a_boxed_clone_sync_stack_runs() {
    let stack = ServiceBuilder::new()
        .boxed_clone_sync()
        .service(flaky(0, Arc::default()));
    assert_eq!(run(&single(stack).unwrap()).await.unwrap().count, 1);
}

/// A function node stored in a struct field, the way `node::from_fn`'s docs
/// suggest naming its type.
struct StoredNode {
    /// The boxed node.
    node: BoxCloneSyncService<NodeInput<Counter>, NodeOutput<Counter>, BoxError>,
}

/// A `node::from_fn` node boxed with `BoxCloneSyncService::new`, as its docs
/// advise, can be stored by name and still runs.
#[tokio::test]
async fn a_boxed_function_node_runs() {
    let stored = StoredNode {
        node: BoxCloneSyncService::new(node::from_fn(|_s, _ctx| async move {
            Ok(Some(CounterUpdate { count: Some(1) }))
        })),
    };
    assert_eq!(run(&single(stored.node).unwrap()).await.unwrap().count, 1);
}

/// An error from `poll_ready` fails the run like an error from `call`.
#[tokio::test]
async fn a_poll_ready_error_fails_the_run() {
    let error = run(&single(NeverReady).unwrap()).await.unwrap_err();
    let RunError::Node(error) = error else {
        panic!("expected a node error, got {error:?}")
    };
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "the node is not ready"
    );
}
