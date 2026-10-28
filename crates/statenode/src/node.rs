//! Nodes: what they receive, what they return, and how to build one from a
//! function.
//!
//! A node is a [`tower::Service`] from [`NodeInput`] to [`NodeOutput`].
//! [`from_fn`] builds one from a function or closure, which can then be
//! wrapped in layers like any other service.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use tower::ServiceExt as _;
use tower::util::BoxCloneSyncService;

use crate::name::NodeName;
use crate::state::State;
use crate::task::TaskPath;

/// A boxed error, as returned by nodes and accepted by [`ReduceError`].
///
/// [`ReduceError`]: crate::ReduceError
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A node, stored type-erased so that nodes of different types can share one
/// graph.
pub(crate) type BoxNode<S, C> = BoxCloneSyncService<NodeInput<S, C>, NodeOutput<S>, BoxError>;

/// What a node receives: a snapshot of the state, and its context.
#[non_exhaustive]
pub struct NodeInput<S, C = ()> {
    /// An immutable snapshot of the state at the start of the superstep.
    pub state: Arc<S>,
    /// The node's context for this task.
    pub ctx: NodeContext<C>,
}

impl<S, C> Clone for NodeInput<S, C> {
    /// Clones the input without requiring `S: Clone` or `C: Clone`, so
    /// middleware such as `Retry` and `Hedge` can resend a request; the clone
    /// shares the same state snapshot and run context.
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            ctx: self.ctx.clone(),
        }
    }
}

/// What a node returns: at most one update to the state.
#[non_exhaustive]
pub struct NodeOutput<S: State> {
    /// The update to apply, if any.
    update: Option<S::Update>,
}

impl<S: State> NodeOutput<S> {
    /// An output that applies `update`.
    #[must_use]
    pub fn update(update: S::Update) -> Self {
        Self {
            update: Some(update),
        }
    }

    /// The update this output carries, if any.
    pub(crate) fn into_update(self) -> Option<S::Update> {
        self.update
    }
}

impl<S: State> From<()> for NodeOutput<S> {
    /// An output that changes nothing.
    fn from((): ()) -> Self {
        Self { update: None }
    }
}

impl<S: State> From<Option<S::Update>> for NodeOutput<S> {
    /// An output that applies the update, if there is one.
    fn from(update: Option<S::Update>) -> Self {
        Self { update }
    }
}

/// The context passed to every node.
///
/// It identifies the task the node is running and carries the run's typed
/// context `C`, a read-only value such as a user id or a database pool. The
/// context is shared by every task of a run.
#[derive(Debug)]
#[non_exhaustive]
pub struct NodeContext<C = ()> {
    /// The path of the task the node is running.
    task_path: TaskPath,
    /// The node's name.
    name: NodeName,
    /// How many supersteps may still run after this one.
    remaining_steps: u64,
    /// The run's context.
    context: Arc<C>,
}

impl<C> NodeContext<C> {
    /// The context for `name`'s task at `task_path`.
    pub(crate) const fn new(
        task_path: TaskPath,
        name: NodeName,
        remaining_steps: u64,
        context: Arc<C>,
    ) -> Self {
        Self {
            task_path,
            name,
            remaining_steps,
            context,
        }
    }

    /// The name of the node being run.
    #[must_use]
    pub const fn name(&self) -> &NodeName {
        &self.name
    }

    /// The path of the task being run.
    #[must_use]
    pub const fn task_path(&self) -> &TaskPath {
        &self.task_path
    }

    /// The superstep being run, counted from 0.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.task_path.step()
    }

    /// How many supersteps may still run after this one before the run hits
    /// its recursion limit.
    ///
    /// It is 0 in the last superstep the limit allows, so a node can stop a
    /// loop gracefully instead of failing with
    /// [`RunError::RecursionLimit`](crate::RunError::RecursionLimit). This is
    /// LangGraph's `RemainingSteps`.
    #[must_use]
    pub const fn remaining_steps(&self) -> u64 {
        self.remaining_steps
    }

    /// The run's typed context.
    #[must_use]
    pub fn context(&self) -> &C {
        &self.context
    }
}

impl<C> Clone for NodeContext<C> {
    /// Clones the context without requiring `C: Clone`; the run's context is
    /// shared.
    fn clone(&self) -> Self {
        Self {
            task_path: self.task_path.clone(),
            name: self.name.clone(),
            remaining_steps: self.remaining_steps,
            context: Arc::clone(&self.context),
        }
    }
}

/// Turns a function or closure into a node service.
///
/// The function receives a snapshot of the state and the node's context, and
/// returns a future resolving to anything that converts into a
/// [`NodeOutput`]: `()` for no update, an `Option` of an update, or a
/// [`NodeOutput`]. Named `async fn`s, generic ones included, closures
/// returning an `async move` block, and capture-free `async` closures are all
/// accepted. A capturing `async` closure is not, because it does not
/// implement `Fn`; clone the captures into an `async move` block instead.
/// The function is shared behind an [`Arc`], so it need not be `Clone`.
///
/// Wrap the node in layers, such as a retry and a timeout, if it calls
/// anything that can fail, then pass it to
/// [`GraphBuilder::add_node`](crate::GraphBuilder::add_node).
///
/// # Storing the service
///
/// The returned service cannot be named. To store it, box it with
/// [`BoxCloneSyncService::new`](tower::util::BoxCloneSyncService::new), or
/// with `.layer(BoxCloneSyncService::layer())` inside a `ServiceBuilder`, at
/// the cost of one more allocation per call. `ServiceBuilder::boxed_clone_sync`
/// does not apply, because the service's future is `Send` but not `Sync`.
///
/// # Annotating closures
///
/// `from_fn` is called before `add_node`, so the graph's state and context
/// types are not yet known when a closure's body is checked:
///
/// - Give the state parameter a type, `s: Arc<State>`, when the body reads
///   through it.
/// - Give the context parameter a type, `ctx: NodeContext<C>`, when the body
///   reads through [`NodeContext::context`] and `C` is not `()`. Its other
///   methods, such as [`name`](NodeContext::name) and
///   [`step`](NodeContext::step), need no annotation.
///
/// Named functions never need annotations.
///
/// # Examples
///
/// A node that only calls the context's own methods, and one that reads the
/// state and the run's context, so both are typed:
///
/// ```
/// use std::sync::Arc;
///
/// use statenode::{GraphBuilder, NodeContext, node};
/// # use statenode::{ChannelSpec, ReduceError, State, TaskPath};
/// # #[derive(Clone, Default)]
/// # struct Counter { count: i64 }
/// # impl State for Counter {
/// #     type Update = i64;
/// #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
/// #     fn apply_step(&mut self, _: Vec<(TaskPath, i64)>) -> Result<(), ReduceError> { Ok(()) }
/// # }
/// /// The run's context.
/// struct Tax { percent: i64 }
///
/// let mut g = GraphBuilder::<Counter, Tax>::new();
/// g.add_node("log", node::from_fn(|_s, ctx| async move {
///     println!("{} in step {}", ctx.name(), ctx.step());
///     Ok(())
/// }))?;
/// g.add_node("tax", node::from_fn(|s: Arc<Counter>, ctx: NodeContext<Tax>| async move {
///     Ok(Some(s.count * ctx.context().percent / 100))
/// }))?;
/// # Ok::<(), statenode::BuildError>(())
/// ```
#[must_use = "a node does nothing until it is passed to `GraphBuilder::add_node`"]
pub fn from_fn<S, C, F, Fut, U>(
    f: F,
) -> impl tower::Service<
    NodeInput<S, C>,
    Response = NodeOutput<S>,
    Error = BoxError,
    Future: Send + 'static,
> + Clone
+ Send
+ Sync
+ 'static
where
    S: State,
    C: Send + Sync + 'static,
    F: Fn(Arc<S>, NodeContext<C>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<U, BoxError>> + Send + 'static,
    U: Into<NodeOutput<S>>,
{
    let f = Arc::new(f);
    tower::service_fn(move |input: NodeInput<S, C>| {
        let output = f(input.state, input.ctx);
        async move { output.await.map(Into::into) }
    })
}

/// Adapts a tower service into a node.
pub(crate) fn from_service<S, C, T>(service: T) -> BoxNode<S, C>
where
    S: State,
    C: Send + Sync + 'static,
    T: tower::Service<NodeInput<S, C>, Response = NodeOutput<S>> + Clone + Send + Sync + 'static,
    T::Error: Into<BoxError>,
    T::Future: Send + 'static,
{
    BoxCloneSyncService::new(service.map_err(Into::into))
}

/// A node failed.
#[derive(Debug)]
pub struct NodeError {
    /// The node that failed.
    node: NodeName,
    /// The task that failed.
    task_path: TaskPath,
    /// The error the node returned.
    source: BoxError,
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "node `{}` failed in step {}",
            self.node,
            self.task_path.step()
        )
    }
}

impl std::error::Error for NodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.source)
    }
}

impl NodeError {
    /// The error `node` returned while running the task at `task_path`.
    pub(crate) fn new(node: NodeName, task_path: TaskPath, source: BoxError) -> Self {
        Self {
            node,
            task_path,
            source,
        }
    }

    /// The node that failed.
    #[must_use]
    pub const fn node(&self) -> &NodeName {
        &self.node
    }

    /// The task that failed.
    #[must_use]
    pub const fn task_path(&self) -> &TaskPath {
        &self.task_path
    }

    /// The node's own error, if it is an `E`.
    #[must_use]
    pub fn downcast_ref<E: std::error::Error + 'static>(&self) -> Option<&E> {
        self.source.downcast_ref()
    }
}

#[cfg(test)]
mod tests {
    //! Tests for node inputs, function nodes, and node errors.

    use std::error::Error as _;

    use super::*;
    use crate::testing::Counter;

    // --- NodeInput -----------------------------------------------------------

    /// A state and run context that are not `Clone`.
    struct Opaque;

    /// A clone shares the state snapshot and the run context, and keeps the
    /// task's identity, even when neither the state nor the context is
    /// `Clone`.
    #[test]
    fn input_clones_share_the_snapshot_and_context() {
        let name = NodeName::new("n").unwrap();
        let input = NodeInput {
            state: Arc::new(Opaque),
            ctx: NodeContext::new(TaskPath::pull(3, name.clone()), name, 7, Arc::new(Opaque)),
        };
        let copy = input.clone();
        assert!(Arc::ptr_eq(&copy.state, &input.state));
        assert!(Arc::ptr_eq(&copy.ctx.context, &input.ctx.context));
        assert_eq!(copy.ctx.task_path(), input.ctx.task_path());
        assert_eq!(copy.ctx.name(), input.ctx.name());
        assert_eq!(copy.ctx.remaining_steps(), 7);
    }

    // --- from_fn -------------------------------------------------------------

    /// The input for node `name`'s task in superstep `step`: a snapshot of the
    /// default [`Counter`], and no run context.
    fn input(name: &str, step: u64) -> NodeInput<Counter> {
        let name = NodeName::new(name).unwrap();
        NodeInput {
            state: Arc::new(Counter::default()),
            ctx: NodeContext::new(TaskPath::pull(step, name.clone()), name, 0, Arc::new(())),
        }
    }

    /// Calls `node` once with `input` and returns the update it made.
    ///
    /// `node` is built by the caller, so the closure inside it is checked
    /// before `S` and `C` are known, just as it is by `add_node`.
    async fn call<T>(node: T, input: NodeInput<Counter>) -> Result<Option<i64>, BoxError>
    where
        T: tower::Service<NodeInput<Counter>, Response = NodeOutput<Counter>, Error = BoxError>,
    {
        node.oneshot(input).await.map(NodeOutput::into_update)
    }

    /// A generic function that does nothing, for any state and context.
    async fn noop<S: State, C>(_s: Arc<S>, _ctx: NodeContext<C>) -> Result<(), BoxError> {
        Ok(())
    }

    /// A closure returning an `async move` block, with parameters it does not
    /// read, infers without annotations.
    #[tokio::test]
    async fn a_closure_returning_an_async_block_needs_no_annotations() {
        let node = from_fn(|_s, _ctx| async move { Ok(()) });
        assert_eq!(call(node, input("n", 0)).await.unwrap(), None);
    }

    /// A capture-free `async` closure, with parameters it does not read,
    /// infers without annotations.
    #[tokio::test]
    async fn a_capture_free_async_closure_needs_no_annotations() {
        let node = from_fn(async |_s, _ctx| Ok(Some(3)));
        assert_eq!(call(node, input("n", 0)).await.unwrap(), Some(3));
    }

    /// `?` inside a closure returning an `async move` block converts any
    /// standard error into the node's `BoxError`.
    #[tokio::test]
    async fn question_mark_converts_standard_errors() {
        let node = from_fn(|_s, ctx| async move {
            let count: i64 = ctx.name().as_str().parse()?;
            Ok(Some(count))
        });
        assert_eq!(call(node.clone(), input("12", 0)).await.unwrap(), Some(12));
        let error = call(node, input("x", 0)).await.unwrap_err();
        assert!(error.is::<std::num::ParseIntError>(), "{error}");
    }

    /// A closure that only calls the context's own methods, not
    /// [`NodeContext::context`], needs no annotation on `ctx`; a step too
    /// large for an `i64` fails through `?` with `TryFromIntError`.
    #[tokio::test]
    async fn a_function_node_reads_its_context_methods_without_annotations() {
        let node = from_fn(|_s, ctx| async move {
            let named = i64::from(ctx.name().as_str() == "n");
            let step = i64::try_from(ctx.step())?;
            Ok(Some(named + step))
        });
        assert_eq!(call(node.clone(), input("n", 2)).await.unwrap(), Some(3));
        let error = call(node, input("n", u64::MAX)).await.unwrap_err();
        assert!(error.is::<std::num::TryFromIntError>(), "{error}");
    }

    /// `?` works inside a capture-free `async` closure too.
    #[tokio::test]
    async fn question_mark_works_in_an_async_closure() {
        let node = from_fn(async |_s, ctx| {
            let count: i64 = ctx.name().as_str().parse()?;
            Ok(Some(count))
        });
        assert_eq!(call(node.clone(), input("5", 0)).await.unwrap(), Some(5));
        let error = call(node, input("x", 0)).await.unwrap_err();
        assert!(error.is::<std::num::ParseIntError>(), "{error}");
    }

    /// A closure returning `NodeOutput::update` with no annotations infers the
    /// output type through `Into<NodeOutput<S>>`.
    #[tokio::test]
    async fn a_node_output_needs_no_annotations() {
        let node = from_fn(|_s, _ctx| async move { Ok(NodeOutput::update(4)) });
        assert_eq!(call(node, input("n", 0)).await.unwrap(), Some(4));
    }

    /// A named generic function is a node without a turbofish: the state and
    /// context are inferred from the call.
    #[tokio::test]
    async fn named_and_generic_fns_are_nodes() {
        assert_eq!(call(from_fn(noop), input("n", 0)).await.unwrap(), None);
    }

    // --- NodeError -----------------------------------------------------------

    /// The message names the node and the step; the source is the node's own
    /// error.
    #[test]
    fn error_messages_name_the_problem() {
        let name = NodeName::new("n").unwrap();
        let error = NodeError::new(name.clone(), TaskPath::pull(3, name), "boom".into());
        assert_eq!(error.to_string(), "node `n` failed in step 3");
        assert_eq!(error.source().unwrap().to_string(), "boom");
    }
}
