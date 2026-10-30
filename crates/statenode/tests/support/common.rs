//! Shared test states, and the naive reference interpreter.
//!
//! Each test binary compiles this module separately and uses only part of
//! it, so the items some binaries do not use carry `#[allow(dead_code)]`.

use std::collections::BTreeSet;
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

/// A graph of `len` nodes, `n0` to `n{len-1}`, wired by node index.
///
/// The spec does not remove repeats: an edge or join listed twice makes
/// [`build`] fail with `DuplicateEdge`, so generators draw `start`, each
/// node's `successors`, and the joins from sets.
///
/// [`build`]: Self::build
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GraphSpec {
    /// The number of nodes.
    pub len: usize,
    /// Where `START` leads: `Some(node)`, or `None` for `END`.
    pub start: Vec<Option<usize>>,
    /// Where each node leads, by node index: `Some(node)`, or `None` for
    /// `END`.
    pub successors: Vec<Vec<Option<usize>>>,
    /// The joins, each its distinct sources and its target, by node index.
    pub joins: Vec<(Vec<usize>, usize)>,
    /// How many times each node yields to the runtime before it finishes, so
    /// tests can vary the order in which a step's tasks complete. A node
    /// with no entry does not yield.
    pub yields: Vec<usize>,
}

/// What the reference interpreter expects of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Reference {
    /// Every visit, by step, then by name.
    pub visits: Vec<Visit>,
    /// The final [`Trace`]'s visits, `"input"` first, if the run finishes.
    pub trace: Vec<String>,
    /// Whether the run reaches its recursion limit.
    pub limited: bool,
}

#[allow(dead_code)]
impl GraphSpec {
    /// The name of node `i`.
    pub fn name(i: usize) -> String {
        format!("n{i}")
    }

    /// A chain of `len` nodes, `n0 → n1 → … → n{len-1}`, ending either at
    /// `END` or with an edge back to node `cycle_to`.
    pub fn chain(len: usize, cycle_to: Option<usize>) -> Self {
        Self {
            len,
            start: vec![Some(0)],
            successors: (0..len)
                .map(|i| vec![if i + 1 < len { Some(i + 1) } else { cycle_to }])
                .collect(),
            joins: Vec::new(),
            yields: vec![0; len],
        }
    }

    /// Builds the graph. Every node yields as often as `yields` says, then
    /// records its visit in the run context and appends its name to the
    /// state.
    ///
    /// # Errors
    ///
    /// If an edge names a node that does not exist, or the builder rejects
    /// the graph.
    pub fn build(&self) -> Result<CompiledGraph<Trace, Recorder>, BoxError> {
        let mut g = GraphBuilder::<Trace, Recorder>::new();
        let ids = (0..self.len)
            .map(|i| {
                let yields = self.yields.get(i).copied().unwrap_or_default();
                g.add_node(
                    &Self::name(i),
                    node::from_fn(move |_s, ctx: NodeContext<Recorder>| async move {
                        for _ in 0..yields {
                            tokio::task::yield_now().await;
                        }
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
        let node = |i: usize| ids.get(i).copied().ok_or("an edge names a missing node");
        let target =
            |end: Option<usize>| end.map_or(Ok(Target::from(END)), |i| node(i).map(Target::from));
        for &end in &self.start {
            g.add_edge(START, target(end)?)?;
        }
        for (i, ends) in self.successors.iter().enumerate() {
            let from = Source::from(node(i)?);
            for &end in ends {
                g.add_edge(from, target(end)?)?;
            }
        }
        for (sources, to) in &self.joins {
            let sources = sources
                .iter()
                .map(|&i| node(i))
                .collect::<Result<Vec<NodeId>, _>>()?;
            g.add_edge(sources, node(*to)?)?;
        }
        Ok(g.compile()?)
    }

    /// Runs the graph with `limit`, returning the final state (or the error)
    /// and every visit recorded, in the order the tasks completed.
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

    /// The naive reference interpreter. Each step runs, in name order, every
    /// node an edge leads to from the nodes of the step before, and every
    /// node whose join has heard from all its sources, starting from
    /// `START`'s targets, until no node is left or the limit is reached.
    ///
    /// After each step, in this order: a complete join whose target ran is
    /// reset, then the sources that ran arrive at their joins. A join
    /// complete at the start of a step always had its target planned in that
    /// step, so every complete join is reset before new arrivals, and the
    /// joins complete afterwards are exactly those that fire next.
    pub fn interpret(&self, limit: u32) -> Reference {
        let limit = u64::from(limit);
        let mut planned: BTreeSet<(String, usize)> = self
            .start
            .iter()
            .flatten()
            .map(|&i| (Self::name(i), i))
            .collect();
        let mut arrived: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); self.joins.len()];
        let mut visits = Vec::new();
        let mut trace = vec![String::from("input")];
        let mut step = 0;
        while !planned.is_empty() {
            if step >= limit {
                return Reference {
                    visits,
                    trace,
                    limited: true,
                };
            }
            for (node, _) in &planned {
                visits.push(Visit {
                    node: node.clone(),
                    step,
                    remaining: limit - step - 1,
                });
                trace.push(node.clone());
            }
            let ran: BTreeSet<usize> = planned.iter().map(|&(_, i)| i).collect();
            for ((sources, target), heard) in self.joins.iter().zip(&mut arrived) {
                if ran.contains(target) && heard.len() == sources.len() {
                    heard.clear();
                }
                heard.extend(sources.iter().filter(|source| ran.contains(source)));
            }
            let fired = self
                .joins
                .iter()
                .zip(&arrived)
                .filter(|((sources, _), heard)| heard.len() == sources.len())
                .map(|((_, target), _)| target);
            planned = ran
                .iter()
                .flat_map(|&i| self.successors.get(i).into_iter().flatten().flatten())
                .chain(fired)
                .map(|&j| (Self::name(j), j))
                .collect();
            step += 1;
        }
        Reference {
            visits,
            trace,
            limited: false,
        }
    }
}

/// `visits` by step, then by name: the order the reference interpreter
/// expects, since a run records visits in the order its tasks complete.
#[allow(dead_code)]
pub fn sorted(mut visits: Vec<Visit>) -> Vec<Visit> {
    visits.sort_by(|a, b| (a.step, &a.node).cmp(&(b.step, &b.node)));
    visits
}
