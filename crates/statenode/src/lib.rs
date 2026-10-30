//! A typed graph runtime for LLM agent workflows, using a Bulk Synchronous
//! Parallel (BSP) execution model: a run proceeds in supersteps.
//!
//! A graph is a typed [`State`], a set of nodes that read a snapshot of the
//! state and return updates, and edges between them. In each superstep, the
//! nodes an edge leads to from the nodes that completed in the previous one,
//! and the nodes whose joins (`add_edge([a, b], c)`) have heard from every
//! source, run against the same snapshot, and their updates are applied
//! together, in [`TaskPath`] order.
//!
//! ```
//! use std::sync::Arc;
//!
//! use statenode::{
//!     ChannelSpec, END, GraphBuilder, NameError, ReduceError, RunConfig, START, State, TaskPath,
//!     node, reducer,
//! };
//!
//! /// A state with one channel, `count`.
//! #[derive(Clone, Default)]
//! struct Counter {
//!     count: i64,
//! }
//!
//! /// An update to [`Counter`]: `None` leaves `count` unwritten.
//! #[derive(Default)]
//! struct CounterUpdate {
//!     count: Option<i64>,
//! }
//!
//! impl State for Counter {
//!     type Update = CounterUpdate;
//!
//!     fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
//!         Ok(vec![ChannelSpec::new("count")?])
//!     }
//!
//!     fn apply_step(
//!         &mut self,
//!         updates: Vec<(TaskPath, CounterUpdate)>,
//!     ) -> Result<(), ReduceError> {
//!         let writes = updates
//!             .iter()
//!             .filter_map(|(path, u)| u.count.map(|c| (path, c)));
//!         reducer::replace("count", &mut self.count, writes)
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut g = GraphBuilder::<Counter>::new();
//! let increment = g.add_node(
//!     "increment",
//!     node::from_fn(async |s: Arc<Counter>, _ctx| {
//!         Ok(Some(CounterUpdate {
//!             count: Some(s.count + 1),
//!         }))
//!     }),
//! )?;
//! g.add_edge(START, increment)?;
//! g.add_edge(increment, END)?;
//! let graph = g.compile()?;
//!
//! let out = graph
//!     .invoke(CounterUpdate { count: Some(41) }, RunConfig::default())
//!     .await?;
//! assert_eq!(out.count, 42);
//! # Ok(())
//! # }
//! ```
//!
//! A capturing `async` closure does not implement `Fn`, so it cannot be a
//! node; clone the captures into an `async move` block instead:
//!
//! ```compile_fail
//! # use std::sync::Arc;
//! # use statenode::{ChannelSpec, GraphBuilder, ReduceError, State, TaskPath, node};
//! # #[derive(Clone, Default)]
//! # struct Empty;
//! # impl State for Empty {
//! #     type Update = ();
//! #     fn channels(&self) -> Result<Vec<ChannelSpec>, statenode::NameError> { Ok(Vec::new()) }
//! #     fn apply_step(&mut self, _: Vec<(TaskPath, ())>) -> Result<(), ReduceError> { Ok(()) }
//! # }
//! let mut g = GraphBuilder::<Empty>::new();
//! let greeting = String::from("hello");
//! g.add_node("greet", node::from_fn(async move |_s: Arc<Empty>, _ctx| {
//!     println!("{greeting}");
//!     Ok(())
//! }));
//! ```

mod builder;
mod config;
mod engine;
mod name;
pub mod node;
pub mod reducer;
mod state;
mod task;
#[cfg(test)]
mod testing;

pub use builder::{
    BuildError, END, End, GraphBuilder, NodeId, START, Source, Sources, Start, Target,
};
pub use config::{DEFAULT_RECURSION_LIMIT, RunConfig};
pub use engine::{CompiledGraph, RunError};
pub use name::{MAX_NAME_LEN, NameError, NodeName};
#[doc(inline)]
pub use node::{BoxError, NodeContext, NodeError, NodeInput, NodeOutput};
pub use state::{ChannelSpec, ReduceError, State};
pub use task::{TaskKind, TaskPath};
