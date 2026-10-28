//! Shared test states, and the naive reference interpreter.
//!
//! Each test binary compiles this module separately and uses only part of
//! it, so the items some binaries do not use carry `#[allow(dead_code)]`.

use std::sync::{Arc, Mutex};

use statenode::{
    BoxError, ChannelSpec, CompiledGraph, END, GraphBuilder, NameError, NodeContext, NodeId,
    ReduceError, RunConfig, RunError, START, Source, State, Target, TaskPath, node, reducer,
};

/// A state with one replace channel, `count`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Counter {
    /// The current count.
    pub count: i64,
}

/// An update to [`Counter`]: `None` leaves `count` unwritten.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct CounterUpdate {
    /// The new count, if any.
    pub count: Option<i64>,
}

impl State for Counter {
    type Update = CounterUpdate;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![ChannelSpec::new("count")?])
    }

    fn apply_step(&mut self, updates: Vec<(TaskPath, CounterUpdate)>) -> Result<(), ReduceError> {
        let writes = updates
            .iter()
            .filter_map(|(path, u)| u.count.map(|c| (path, c)));
        reducer::replace("count", &mut self.count, writes)
    }
}

/// One visit to a node, as recorded by a [`Trace`] graph's run context.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Visit {
    /// The node visited.
    pub node: String,
    /// The superstep it ran in.
    pub step: u64,
    /// The node's `remaining_steps()`.
    pub remaining: u64,
}

/// A run context that records every visit, so tests can observe a run that
/// fails.
#[allow(dead_code)]
pub type Recorder = Arc<Mutex<Vec<Visit>>>;

/// A state that appends the name of every node that runs to `visits`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Trace {
    /// The nodes, which ran, in order.
    pub visits: Vec<String>,
}

impl State for Trace {
    /// The name of the node that ran.
    type Update = String;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        Ok(vec![ChannelSpec::new("visits")?])
    }

    fn apply_step(&mut self, updates: Vec<(TaskPath, String)>) -> Result<(), ReduceError> {
        self.visits
            .extend(updates.into_iter().map(|(_, name)| name));
        Ok(())
    }
}

/// A graph that is a chain of `len` nodes, `n0 → n1 → … → n{len-1}`, ending
/// either at `END` or with an edge back to node `cycle_to`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChainSpec {
    /// The number of nodes.
    pub len: usize,
    /// Where the last node's edge leads: `None` for `END`.
    pub cycle_to: Option<usize>,
}

#[allow(dead_code)]
impl ChainSpec {
    /// The name of node `i`.
    pub fn name(i: usize) -> String {
        format!("n{i}")
    }

    /// Builds the graph. Every node records its visit in the run context and
    /// appends its name to the state.
    ///
    /// # Errors
    ///
    /// If `cycle_to` is not a node, or the builder rejects the graph.
    pub fn build(&self) -> Result<CompiledGraph<Trace, Recorder>, BoxError> {
        let mut g = GraphBuilder::<Trace, Recorder>::new();
        let ids = (0..self.len)
            .map(|i| {
                g.add_node(
                    &Self::name(i),
                    node::from_fn(|_s, ctx: NodeContext<Recorder>| async move {
                        let node = ctx.name().as_str().to_owned();
                        ctx.context()
                            .lock()
                            .map_err(|e| e.to_string())?
                            .push(Visit {
                                node: node.clone(),
                                step: ctx.step(),
                                remaining: ctx.remaining_steps(),
                            });
                        Ok(Some(node))
                    }),
                )
            })
            .collect::<Result<Vec<NodeId>, _>>()?;
        let mut from = Source::from(START);
        for &id in &ids {
            g.add_edge(from, id)?;
            from = Source::from(id);
        }
        let last = match self.cycle_to {
            Some(target) => Target::from(*ids.get(target).ok_or("cycle_to is not a node")?),
            None => Target::from(END),
        };
        g.add_edge(from, last)?;
        Ok(g.compile()?)
    }

    /// Runs the graph with `limit`, returning the final state (or the error)
    /// and every visit recorded.
    ///
    /// # Errors
    ///
    /// If `limit` is zero, the graph cannot be built, or the recorder's lock
    /// is poisoned.
    pub async fn run(&self, limit: u32) -> Result<(Result<Trace, RunError>, Vec<Visit>), BoxError> {
        let recorder = Recorder::default();
        let limit = std::num::NonZeroU32::new(limit).ok_or("the recursion limit is zero")?;
        let config = RunConfig::new(Arc::clone(&recorder)).with_recursion_limit(limit);
        let result = self.build()?.invoke(String::from("input"), config).await;
        let visits = recorder.lock().map_err(|e| e.to_string())?.clone();
        Ok((result, visits))
    }

    /// The naive reference interpreter: walks the edges one node at a time.
    ///
    /// Returns the visits, and whether the run hit the recursion limit.
    pub fn interpret(&self, limit: u32) -> (Vec<Visit>, bool) {
        let mut visits = Vec::new();
        let mut current = Some(0);
        let mut step = 0;
        while let Some(node) = current {
            if step >= u64::from(limit) {
                return (visits, true);
            }
            visits.push(Visit {
                node: Self::name(node),
                step,
                remaining: u64::from(limit) - step - 1,
            });
            current = if node + 1 < self.len {
                Some(node + 1)
            } else {
                self.cycle_to
            };
            step += 1;
        }
        (visits, false)
    }
}
