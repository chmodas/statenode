//! Building graphs: [`GraphBuilder`], node handles, the `START` and `END`
//! sentinels, and [`BuildError`], raised when a graph cannot be built.
//!
//! The generic builder methods are thin wrappers. Every check lives in the
//! non-generic [`Topology`], which knows only names, handles, and edges.

mod error;
mod topology;

use crate::engine::CompiledGraph;
use crate::node::{self, BoxError, BoxNode, NodeInput, NodeOutput};
use crate::state::State;

pub use error::BuildError;
/// Named only by the scheduler's tests, which build validated topologies.
#[cfg(test)]
pub(crate) use topology::Join;
use topology::Topology;
pub(crate) use topology::ValidTopology;

/// A handle to a node, returned by [`GraphBuilder::add_node`].
///
/// A handle is only valid for the builder that created it; using it with
/// another builder is rejected with [`BuildError::ForeignNode`]. Handles exist
/// only while a graph is built: everything a compiled graph records or reports
/// uses node names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    /// The node's position in its builder.
    index: usize,
    /// The tag of the builder that created the handle.
    builder: u64,
}

/// The sentinel for the start of a graph, usable only as an edge's source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Start;

/// The sentinel for the end of a graph, usable only as an edge's target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct End;

/// The start of a graph: edges from `START` lead to the first nodes to run.
pub const START: Start = Start;

/// The end of a graph: an edge to `END` marks where a branch finishes.
pub const END: End = End;

/// Where an edge starts: [`START`] or a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Source {
    /// The start of the graph.
    Start,
    /// A node.
    Node(NodeId),
}

impl From<Start> for Source {
    fn from(_: Start) -> Self {
        Self::Start
    }
}

impl From<NodeId> for Source {
    fn from(node: NodeId) -> Self {
        Self::Node(node)
    }
}

/// Where an edge ends: a node or [`END`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Target {
    /// A node.
    Node(NodeId),
    /// The end of the graph.
    End,
}

impl From<NodeId> for Target {
    fn from(node: NodeId) -> Self {
        Self::Node(node)
    }
}

impl From<End> for Target {
    fn from(_: End) -> Self {
        Self::End
    }
}

/// Where an edge starts: one source, or a join of several nodes, which leads
/// to its target once every one of them has run.
///
/// Unlike [`Source`] it can hold a list, so it is `Clone` but not `Copy`. A
/// join converts from an array of handles (`[a, b]`, passed by value), a
/// `Vec` or a slice (`&nodes[..]`); a join of one node is rejected as
/// [`SingleSourceJoin`](BuildError::SingleSourceJoin), since it is a plain
/// edge.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Sources {
    /// One source: [`START`] or a node.
    One(Source),
    /// A join of several nodes.
    Join(Vec<NodeId>),
}

impl From<Source> for Sources {
    fn from(source: Source) -> Self {
        Self::One(source)
    }
}

impl From<Start> for Sources {
    fn from(start: Start) -> Self {
        Self::One(start.into())
    }
}

impl From<NodeId> for Sources {
    fn from(node: NodeId) -> Self {
        Self::One(node.into())
    }
}

impl<const N: usize> From<[NodeId; N]> for Sources {
    fn from(nodes: [NodeId; N]) -> Self {
        Self::Join(nodes.into())
    }
}

impl From<Vec<NodeId>> for Sources {
    fn from(nodes: Vec<NodeId>) -> Self {
        Self::Join(nodes)
    }
}

impl From<&[NodeId]> for Sources {
    fn from(nodes: &[NodeId]) -> Self {
        Self::Join(nodes.to_vec())
    }
}

/// Builds a graph over the state `S`, with the run context `C`.
///
/// Add nodes with [`add_node`](Self::add_node), connect them with
/// [`add_edge`](Self::add_edge), then [`compile`](Self::compile) the graph.
pub struct GraphBuilder<S: State, C = ()> {
    /// The state every run starts from, before its input is applied.
    initial: S,
    /// The graph's names and edges.
    topology: Topology,
    /// The nodes, in the order they were added.
    nodes: Vec<BoxNode<S, C>>,
}

impl<S: State + Default, C> GraphBuilder<S, C> {
    /// A builder whose runs start from `S::default()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initial: S::default(),
            topology: Topology::new(),
            nodes: Vec::new(),
        }
    }
}

impl<S: State + Default, C> Default for GraphBuilder<S, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State, C: Send + Sync + 'static> GraphBuilder<S, C> {
    /// Adds a node implemented by a [`tower::Service`] and returns its
    /// handle.
    ///
    /// The service must take a [`NodeInput`] and return a [`NodeOutput`];
    /// services built with [`node::from_fn`] do. Its error can be anything
    /// that converts into a [`BoxError`], and an error from either
    /// `poll_ready` or `call` fails the run with
    /// [`RunError::Node`](crate::RunError::Node). Only the outermost service
    /// has to use these types: layers such as `map_request` and
    /// `map_response` let the services inside it use their own. Wrap a node
    /// in layers if it calls anything that can fail.
    ///
    /// # Layers in a graph
    ///
    /// - **One clone per task.** The engine clones the service for every task
    ///   it runs, so the service must be `Clone + Send + Sync`. State held
    ///   directly inside the service is copied per task: a retry policy's
    ///   attempt count, for example, starts afresh for each task. To share
    ///   state across tasks, keep it behind an `Arc`, or use a stateful layer
    ///   that does, such as `ConcurrencyLimit` or `Buffer`.
    /// - **Concurrent siblings.** The tasks of one superstep run concurrently,
    ///   so state shared behind an `Arc`, a `ConcurrencyLimit` or a `Buffer`
    ///   sees concurrent requests from one step. A shared limit below the
    ///   number of sibling tasks runs some of them one after another.
    /// - **Services and layers that are not `Clone` or not `Sync`**, such as
    ///   `RateLimit`, `Hedge`, `Balance`, or a stack boxed with
    ///   `ServiceBuilder::boxed_clone`, must be wrapped in `Buffer`. `Buffer`
    ///   places the service behind a multi-producer, single-consumer channel
    ///   served by one worker task: every clone of the `Buffer` is a handle
    ///   that enqueues requests on the channel, and the worker drives the
    ///   single wrapped service, so the `Buffer` is `Clone + Send + Sync`
    ///   whatever it wraps. The worker is shared by every task and every run
    ///   of the graph, so:
    ///   - It is spawned when the `Buffer` is created, which must happen
    ///     inside a Tokio runtime. Once that runtime shuts down, calls fail
    ///     with `Closed`.
    ///   - Once the wrapped service's `poll_ready` fails, every later call
    ///     fails too.
    ///   - Its bound caps the requests waiting for the worker, not those
    ///     running. To cap concurrency, put `ConcurrencyLimit` in front.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidName`] if `name` breaks the
    /// [naming rules](crate::NodeName#naming-rules), or
    /// [`BuildError::DuplicateName`] if a node already has that name.
    ///
    /// # Examples
    ///
    /// Adding a function node wrapped with concurrency limiting and a timeout:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use statenode::{END, GraphBuilder, START, node};
    /// use tower::ServiceBuilder;
    /// # use statenode::{ChannelSpec, ReduceError, State, TaskPath};
    /// # #[derive(Clone, Default)]
    /// # struct Empty;
    /// # impl State for Empty {
    /// #     type Update = ();
    /// #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
    /// #     fn apply_step(&mut self, _: Vec<(TaskPath, ())>) -> Result<(), ReduceError> { Ok(()) }
    /// # }
    ///
    /// let stack = ServiceBuilder::new()
    ///     .concurrency_limit(4)
    ///     .timeout(Duration::from_secs(30))
    ///     .service(node::from_fn(|_s, ctx| async move {
    ///         println!("{} in step {}", ctx.name(), ctx.step());
    ///         Ok(())
    ///     }));
    ///
    /// let mut g = GraphBuilder::<Empty>::new();
    /// let id = g.add_node("guarded", stack)?;
    /// g.add_edge(START, id)?;
    /// g.add_edge(id, END)?;
    /// let graph = g.compile()?;
    /// # Ok::<(), statenode::BuildError>(())
    /// ```
    pub fn add_node<T>(&mut self, name: &str, node: T) -> Result<NodeId, BuildError>
    where
        T: tower::Service<NodeInput<S, C>, Response = NodeOutput<S>>
            + Clone
            + Send
            + Sync
            + 'static,
        T::Error: Into<BoxError>,
        T::Future: Send + 'static,
    {
        let id = self.topology.add_node(name)?;
        self.nodes.push(node::from_service(node));
        Ok(id)
    }

    /// Adds an edge: when `from` completes, `to` runs in the next superstep.
    ///
    /// A node, or [`START`], may have any number of edges. Their targets run
    /// concurrently in the next superstep, against the same state, and a node
    /// that several edges lead to runs once.
    ///
    /// # Joins
    ///
    /// With several nodes as `from`, such as `[a, b]`, the edge is a join: it
    /// fires, and `to` runs, in the superstep after every one of them has
    /// completed since the join last fired. Each call declares its own join. A
    /// node with several joins, or with joins and plain edges, runs when any
    /// one of them is ready, and once per superstep however many are; a join
    /// keeps what it has heard while its target runs through another of them.
    /// Then:
    ///
    /// - a source that runs more than once before the join fires counts once;
    /// - a run can end with a join part-filled, some of its sources having run
    ///   and others not since it last fired, without error;
    /// - in a cycle, a run can reach its recursion limit before a join fires.
    ///
    /// # Errors
    ///
    /// Checks, in this order, and returns the first problem found:
    /// [`BuildError::ForeignNode`] if a source, then the target, is a handle
    /// from a different builder; for a join,
    /// [`EmptyJoin`](BuildError::EmptyJoin),
    /// [`SingleSourceJoin`](BuildError::SingleSourceJoin),
    /// [`DuplicateJoinSource`](BuildError::DuplicateJoinSource) and
    /// [`JoinToEnd`](BuildError::JoinToEnd); then
    /// [`BuildError::DuplicateEdge`] if the same edge was already added, a
    /// join counting as the same whatever the order of its sources. A rejected
    /// edge is not added.
    ///
    /// # Examples
    ///
    /// A single node between [`START`] and [`END`]:
    ///
    /// ```
    /// use statenode::{END, GraphBuilder, START, node};
    /// # use statenode::{ChannelSpec, ReduceError, State, TaskPath};
    /// # #[derive(Clone, Default)]
    /// # struct Empty;
    /// # impl State for Empty {
    /// #     type Update = ();
    /// #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
    /// #     fn apply_step(&mut self, _: Vec<(TaskPath, ())>) -> Result<(), ReduceError> { Ok(()) }
    /// # }
    ///
    /// let mut g = GraphBuilder::<Empty>::new();
    /// let greet = g.add_node("greet", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// g.add_edge(START, greet)?;
    /// g.add_edge(greet, END)?;
    /// let graph = g.compile()?;
    /// # Ok::<(), statenode::BuildError>(())
    /// ```
    ///
    /// Two nodes that run concurrently, in the same superstep:
    ///
    /// ```
    /// use statenode::{BuildError, END, GraphBuilder, START, node};
    /// # use statenode::{ChannelSpec, ReduceError, State, TaskPath};
    /// # #[derive(Clone, Default)]
    /// # struct Empty;
    /// # impl State for Empty {
    /// #     type Update = ();
    /// #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
    /// #     fn apply_step(&mut self, _: Vec<(TaskPath, ())>) -> Result<(), ReduceError> { Ok(()) }
    /// # }
    ///
    /// let mut g = GraphBuilder::<Empty>::new();
    /// let search = g.add_node("search", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// let lookup = g.add_node("lookup", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// g.add_edge(START, search)?;
    /// g.add_edge(START, lookup)?;
    /// g.add_edge(search, END)?;
    /// g.add_edge(lookup, END)?;
    /// assert_eq!(
    ///     g.add_edge(START, search),
    ///     Err(BuildError::DuplicateEdge { edge: "START → search".to_owned() })
    /// );
    /// let graph = g.compile()?;
    /// # Ok::<(), BuildError>(())
    /// ```
    ///
    /// A join: `write` runs once both `search` and `lookup` have run.
    ///
    /// ```
    /// use statenode::{BuildError, END, GraphBuilder, START, node};
    /// # use statenode::{ChannelSpec, ReduceError, State, TaskPath};
    /// # #[derive(Clone, Default)]
    /// # struct Empty;
    /// # impl State for Empty {
    /// #     type Update = ();
    /// #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
    /// #     fn apply_step(&mut self, _: Vec<(TaskPath, ())>) -> Result<(), ReduceError> { Ok(()) }
    /// # }
    ///
    /// let mut g = GraphBuilder::<Empty>::new();
    /// let search = g.add_node("search", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// let lookup = g.add_node("lookup", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// let write = g.add_node("write", node::from_fn(|_s, _ctx| async move { Ok(()) }))?;
    /// g.add_edge(START, search)?;
    /// g.add_edge(START, lookup)?;
    /// g.add_edge([search, lookup], write)?;
    /// g.add_edge(write, END)?;
    /// assert_eq!(
    ///     g.add_edge([lookup, search], write),
    ///     Err(BuildError::DuplicateEdge { edge: "join:lookup+search:write".to_owned() })
    /// );
    /// let graph = g.compile()?;
    /// # Ok::<(), BuildError>(())
    /// ```
    pub fn add_edge(
        &mut self,
        from: impl Into<Sources>,
        to: impl Into<Target>,
    ) -> Result<(), BuildError> {
        self.topology.add_edge(from.into(), to.into())
    }

    /// Validates the graph and compiles it into a [`CompiledGraph`].
    ///
    /// # Errors
    ///
    /// Returns the first structural problem found, checking in this order:
    /// [`Empty`](BuildError::Empty), [`NoStartEdge`](BuildError::NoStartEdge),
    /// [`Unreachable`](BuildError::Unreachable) (a join's target is reached
    /// once every one of its sources is), then the state's channels:
    /// [`InvalidChannelName`](BuildError::InvalidChannelName) for the name
    /// error [`State::channels`] returned, if any, wherever that name falls
    /// among the channels, then
    /// [`DuplicateChannel`](BuildError::DuplicateChannel) for the first name
    /// repeated. Duplicates are only checked once every name is valid.
    pub fn compile(self) -> Result<CompiledGraph<S, C>, BuildError> {
        let wiring = self.topology.validate(self.initial.channels())?;
        Ok(CompiledGraph::new(self.initial, self.nodes, wiring))
    }
}

#[cfg(test)]
mod tests {
    //! Tests for edge ends and the builder.

    use std::any::TypeId;

    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;
    use crate::MAX_NAME_LEN;
    use crate::name::{NodeName, validate_name};
    use crate::testing::{Counter, INVALID_NAME, invalid_name, noop, valid_name};

    // --- Source and Target ---------------------------------------------------

    /// The sentinels and node handles convert into the matching edge ends.
    #[test]
    fn sentinels_and_handles_convert_into_edge_ends() {
        let id = Topology::new().add_node("a").unwrap();
        let ids = [id];
        assert_eq!(Source::from(START), Source::Start);
        assert_eq!(Source::from(ids[0]), Source::Node(ids[0]));
        assert_eq!(Target::from(ids[0]), Target::Node(ids[0]));
        assert_eq!(Target::from(END), Target::End);
    }

    // --- Sources -------------------------------------------------------------

    /// `START`, a handle or a `Source` converts to a single source, and an
    /// array of handles to a join of them, in the order given.
    #[test]
    fn one_source_converts_to_a_single_source() {
        let mut g = GraphBuilder::<Counter>::new();
        let a = g.add_node("a", node::from_fn(noop)).unwrap();
        let b = g.add_node("b", node::from_fn(noop)).unwrap();
        assert_eq!(Sources::from(START), Sources::One(Source::Start));
        assert_eq!(Sources::from(a), Sources::One(Source::Node(a)));
        assert_eq!(
            Sources::from(Source::Node(b)),
            Sources::One(Source::Node(b))
        );
        assert_eq!(Sources::from([b, a]), Sources::Join(vec![b, a]));
    }

    /// A `Vec` or a slice of handles converts to a join of exactly those
    /// handles, in the order given, whatever their number, order or repeats:
    /// the conversion checks nothing.
    #[proptest]
    fn a_list_of_handles_converts_to_a_join_in_order(
        #[strategy(1..8usize)] nodes: usize,
        /// The nodes listed, by index.
        #[strategy(prop::collection::vec(0..#nodes, 0..8))]
        picks: Vec<usize>,
    ) {
        let mut g = GraphBuilder::<Counter>::new();
        let ids: Vec<NodeId> = (0..nodes)
            .map(|i| g.add_node(&format!("n{i}"), node::from_fn(noop)).unwrap())
            .collect();
        let listed: Vec<NodeId> = picks.iter().map(|&i| ids[i]).collect();
        prop_assert_eq!(
            Sources::from(listed.as_slice()),
            Sources::Join(listed.clone())
        );
        prop_assert_eq!(Sources::from(listed.clone()), Sources::Join(listed));
    }

    // --- GraphBuilder --------------------------------------------------------

    /// `add_node` rejects an invalid name and a name already taken and only
    /// registers the node in between.
    ///
    /// Every call passes `node::from_fn` of the same function, so one
    /// `add_node` instantiation covers both its error and its success path on
    /// every run.
    #[test]
    fn add_node_rejects_invalid_and_duplicate_names() {
        let mut g = GraphBuilder::<Counter>::new();
        assert_eq!(
            g.add_node(INVALID_NAME, node::from_fn(noop)).unwrap_err(),
            BuildError::InvalidName(validate_name(INVALID_NAME).unwrap_err())
        );
        g.add_node("taken", node::from_fn(noop)).unwrap();
        assert_eq!(
            g.add_node("taken", node::from_fn(noop)).unwrap_err(),
            BuildError::DuplicateName {
                name: "taken".to_owned()
            }
        );
        assert_eq!(g.nodes.len(), 1, "rejected nodes are not stored");
    }

    /// `add_node` accepts every valid name.
    #[proptest]
    fn add_node_accepts_valid_names(#[strategy(valid_name(MAX_NAME_LEN))] name: String) {
        let mut g = GraphBuilder::<Counter>::new();
        prop_assert!(g.add_node(&name, node::from_fn(noop)).is_ok());
    }

    /// `add_node` rejects every invalid name with the error `validate_name`
    /// reports for it.
    #[proptest]
    fn add_node_rejects_invalid_names(#[strategy(invalid_name())] name: String) {
        let mut g = GraphBuilder::<Counter>::new();
        prop_assert_eq!(
            g.add_node(&name, node::from_fn(noop)).err(),
            Some(BuildError::InvalidName(validate_name(&name).unwrap_err()))
        );
    }

    /// `add_edge` takes `START`, `END` and handles at the ends they are
    /// allowed and records each edge in order and direction. It rejects a
    /// handle from another builder at either end, and an edge added twice,
    /// without recording either.
    #[test]
    fn add_edge_records_edges_and_rejects_foreign_handles() {
        let mut g = GraphBuilder::<Counter>::new();
        let a = g.add_node("a", node::from_fn(noop)).unwrap();
        let foreign = GraphBuilder::<Counter>::new()
            .add_node("f", node::from_fn(noop))
            .unwrap();
        g.add_edge(START, a).unwrap();
        g.add_edge(a, a).unwrap();
        g.add_edge(a, END).unwrap();
        assert_eq!(g.add_edge(foreign, a), Err(BuildError::ForeignNode));
        assert_eq!(g.add_edge(a, foreign), Err(BuildError::ForeignNode));
        assert_eq!(
            g.add_edge(a, END),
            Err(BuildError::DuplicateEdge {
                edge: "a → END".to_owned()
            })
        );
        let edge = |from, to| topology::Edge { from, to };
        assert_eq!(
            g.topology.edges,
            vec![
                edge(None, Some(0)),
                edge(Some(0), Some(0)),
                edge(Some(0), None)
            ],
            "edges are recorded in order, rejected ones are not"
        );
    }

    /// `add_edge` records a join, with its sources in name order, apart from
    /// the plain edges.
    #[test]
    fn add_edge_records_joins() {
        let mut g = GraphBuilder::<Counter>::new();
        let b = g.add_node("b", node::from_fn(noop)).unwrap();
        let a = g.add_node("a", node::from_fn(noop)).unwrap();
        let c = g.add_node("c", node::from_fn(noop)).unwrap();
        g.add_edge([b, a], c).unwrap();
        assert_eq!(
            g.topology.joins,
            [Join {
                sources: vec![1, 0],
                target: 2
            }]
        );
        assert_eq!(g.topology.edges, Vec::new());
    }

    /// `compile` reports a join's target as unreachable while one of its
    /// sources is: the target, `a`, sorts before the unreached source.
    #[test]
    fn compile_reports_an_unreachable_join_target() {
        let mut g = GraphBuilder::<Counter>::new();
        let z = g.add_node("z", node::from_fn(noop)).unwrap();
        let y = g.add_node("y", node::from_fn(noop)).unwrap();
        let a = g.add_node("a", node::from_fn(noop)).unwrap();
        g.add_edge(START, z).unwrap();
        g.add_edge([z, y], a).unwrap();
        assert_eq!(
            g.compile().err(),
            Some(BuildError::Unreachable {
                node: NodeName::new("a").unwrap()
            })
        );
    }

    /// `compile` reports the topology's errors, here `Empty`.
    #[test]
    fn compile_reports_structural_errors() {
        assert_eq!(
            GraphBuilder::<Counter>::new().compile().err(),
            Some(BuildError::Empty)
        );
    }

    /// `compile` checks the channels of the builder's initial state: one
    /// breaking the naming rules fails the build with the error
    /// `validate_name` reports.
    #[test]
    fn compile_checks_the_states_channels() {
        let mut g = GraphBuilder::<Counter>::new();
        g.initial.channels = vec![INVALID_NAME];
        let id = g.add_node("only", node::from_fn(noop)).unwrap();
        g.add_edge(START, id).unwrap();
        g.add_edge(id, END).unwrap();
        assert_eq!(
            g.compile().err(),
            Some(BuildError::InvalidChannelName(
                validate_name(INVALID_NAME).unwrap_err()
            ))
        );
    }

    /// A builder without a run context never has to name `C`: it defaults to
    /// `()`.
    #[test]
    fn the_context_defaults_to_unit() {
        assert_eq!(
            TypeId::of::<GraphBuilder<Counter>>(),
            TypeId::of::<GraphBuilder<Counter, ()>>()
        );
    }

    /// Builders are `Send + Sync`, so one can be built on one task and
    /// compiled on another.
    #[test]
    fn builders_are_send_and_sync() {
        /// Compiles only if `T` is `Send + Sync`.
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<GraphBuilder<Counter>>();
    }
}
