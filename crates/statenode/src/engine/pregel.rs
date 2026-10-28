//! The Pregel scheduler: a validated topology lowered to trigger channels,
//! and the scheduling state of one run, channel versions and what each node
//! has seen.
//!
//! A node is planned when its trigger channel's version is newer than the
//! version it saw when it last ran. The barrier gives every channel the step
//! wrote one new version, and records what each node that ran has seen.
//!
//! This is the only module tied to the scheduling algorithm. Non-generic, so
//! every state type shares one implementation of the scheduling rules.

use crate::builder::ValidTopology;
use crate::name::NodeName;

/// The version of a channel.
///
/// Versions only grow. At each barrier one next version is computed from the
/// largest current version and given to every channel written in that step,
/// so a version depends only on the channel map, never on a counter that
/// restarts with each run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(crate) struct ChannelVersion(u64);

impl ChannelVersion {
    /// The version after this one.
    ///
    /// A run is bounded by its recursion limit, a `u32`, so a `u64` version
    /// cannot overflow.
    pub(crate) const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// A validated topology, lowered for the scheduler.
///
/// Each node has one trigger channel, `branch:to:<node>`, with the same index
/// as the node. An edge `a → b` means "when `a` completes, write `b`'s trigger
/// channel"; an edge to `END` writes nothing.
#[derive(Debug)]
pub(crate) struct Wiring {
    /// Node names, by node index.
    pub(crate) names: Vec<NodeName>,
    /// Each node's position in name order, which is task-path order.
    pub(crate) rank: Vec<usize>,
    /// The trigger channels each node writes when it completes.
    pub(crate) successors: Vec<Vec<usize>>,
    /// The trigger channels the input writes.
    pub(crate) start_targets: Vec<usize>,
}

impl Wiring {
    /// Lowers `topology`: each node's successors become the trigger channels
    /// it writes, and nodes are ranked by name for task-path order.
    pub(crate) fn new(topology: ValidTopology) -> Self {
        let mut by_name: Vec<(&NodeName, usize)> = topology.names.iter().zip(0..).collect();
        by_name.sort_unstable();
        let mut rank = vec![0; topology.names.len()];
        for (position, &(_, node)) in by_name.iter().enumerate() {
            rank.get_mut(node).into_iter().for_each(|r| *r = position);
        }
        Self {
            rank,
            successors: topology.successors,
            start_targets: topology.start,
            names: topology.names,
        }
    }
}

/// The scheduling state of one run.
///
/// Planning reads it and never changes it; only [`Scheduler::barrier`],
/// called after a step's updates have been applied successfully, changes it.
/// A failing step therefore leaves versions and `versions_seen` exactly as
/// they were at its start.
#[derive(Debug)]
pub(crate) struct Scheduler<'w> {
    /// The graph's lowered edges.
    wiring: &'w Wiring,
    /// The version of each trigger channel, by node index.
    versions: Vec<ChannelVersion>,
    /// The version of its trigger channel each node last ran against.
    versions_seen: Vec<ChannelVersion>,
    /// The largest version of any channel.
    max_version: ChannelVersion,
    /// The trigger channels written at the last barrier: the only candidates
    /// for the next step, which keeps planning proportional to the writes.
    updated: Vec<usize>,
}

impl<'w> Scheduler<'w> {
    /// The scheduling state before the input is applied: every version is 0.
    pub(crate) fn new(wiring: &'w Wiring) -> Self {
        let len = wiring.names.len();
        Self {
            wiring,
            versions: vec![ChannelVersion::default(); len],
            versions_seen: vec![ChannelVersion::default(); len],
            max_version: ChannelVersion::default(),
            updated: Vec::new(),
        }
    }

    /// The nodes to run in the next step, sorted by task path.
    ///
    /// A node runs if its trigger channel has a newer version than the one it
    /// last ran against.
    ///
    /// With only edge triggers, every candidate was just written and so always
    /// passes the comparison. It stays because it *is* the scheduling rule:
    /// it starts to reject candidates when a node has several triggers, and
    /// when a run resumes from stored versions.
    pub(crate) fn plan(&self) -> Vec<usize> {
        let mut tasks: Vec<usize> = self
            .updated
            .iter()
            .copied()
            .filter(|&node| self.versions.get(node) > self.versions_seen.get(node))
            .collect();
        // `updated` holds each channel once (the barrier deduplicates it), and
        // each channel triggers one node, so the tasks are already unique.
        tasks.sort_unstable_by_key(|&node| self.wiring.rank.get(node).copied());
        tasks
    }

    /// The trigger channels the completed `tasks` write.
    pub(crate) fn written_by(&self, tasks: &[usize]) -> Vec<usize> {
        tasks
            .iter()
            .filter_map(|&node| self.wiring.successors.get(node))
            .flatten()
            .copied()
            .collect()
    }

    /// Commits a step: `ran` saw the versions they were planned against, and
    /// `written` get the step's version.
    pub(crate) fn barrier(&mut self, ran: &[usize], mut written: Vec<usize>) {
        for &node in ran {
            let version = self.versions.get(node).copied();
            self.versions_seen
                .get_mut(node)
                .into_iter()
                .zip(version)
                .for_each(|(seen, version)| *seen = version);
        }
        written.sort_unstable();
        written.dedup();
        if !written.is_empty() {
            let next = self.max_version.next();
            for &channel in &written {
                self.versions
                    .get_mut(channel)
                    .into_iter()
                    .for_each(|version| *version = next);
            }
            self.max_version = next;
        }
        self.updated = written;
    }
}

#[cfg(test)]
mod tests {
    //! Tests for channel versions, lowering, and the scheduling rules.

    use proptest::prelude::*;
    use test_strategy::{Arbitrary, proptest};

    use std::iter::once;

    use super::*;
    use crate::testing::valid_name;

    // --- ChannelVersion ------------------------------------------------------

    /// A version's successor is one more than it, whatever the version.
    ///
    /// `u64::MAX` is left out: a run's recursion limit keeps versions far
    /// below it.
    #[proptest]
    fn versions_grow_by_one(#[strategy(0..u64::MAX)] version: u64) {
        prop_assert_eq!(ChannelVersion(version).next(), ChannelVersion(version + 1));
    }

    /// Versions order as the numbers they hold. Half the pairs are equal, so
    /// equality is drawn as often as the two inequalities together.
    #[proptest]
    fn versions_order_by_value(a: u64, #[strategy(prop_oneof![Just(#a), any::<u64>()])] b: u64) {
        prop_assert_eq!(ChannelVersion(a).cmp(&ChannelVersion(b)), a.cmp(&b));
    }

    // --- Wiring --------------------------------------------------------------

    /// The parts of a topology to lower: distinct valid names, as the builder
    /// guarantees, and arbitrary edges between them. Lowering does not
    /// validate, so the edges need not form a valid graph.
    #[derive(Debug, Arbitrary)]
    struct Parts {
        /// The node names, by node index, shuffled by proptest's own RNG so a
        /// recorded failure replays in the same declaration order.
        #[strategy(prop::collection::btree_set(valid_name(8), 1..8).prop_map(Vec::from_iter).prop_shuffle())]
        names: Vec<String>,
        /// The nodes `START` leads to.
        #[strategy(prop::collection::vec(0..#names.len(), 1..3))]
        start: Vec<usize>,
        /// The nodes each node leads to.
        #[strategy(prop::collection::vec(prop::collection::vec(0..#names.len(), 0..3), #names.len()))]
        successors: Vec<Vec<usize>>,
    }

    impl Parts {
        /// The names as node names.
        fn names(&self) -> Vec<NodeName> {
            self.names
                .iter()
                .map(|n| NodeName::new(n).unwrap())
                .collect()
        }

        /// The validated topology made of these parts.
        fn topology(&self) -> ValidTopology {
            ValidTopology {
                names: self.names(),
                start: self.start.clone(),
                successors: self.successors.clone(),
            }
        }
    }

    /// Each node's rank is its position in name order: the number of nodes
    /// whose names sort before its own, whatever the declaration order.
    #[proptest]
    fn lowering_ranks_nodes_by_name(parts: Parts) {
        let w = Wiring::new(parts.topology());
        for (node, name) in w.names.iter().enumerate() {
            let before = w.names.iter().filter(|other| *other < name).count();
            prop_assert_eq!(w.rank[node], before);
        }
    }

    /// Lowering keeps the names in declaration order, and the edges as the
    /// trigger channels each node and the input write.
    #[proptest]
    fn lowering_keeps_names_and_edges(parts: Parts) {
        let w = Wiring::new(parts.topology());
        prop_assert_eq!(
            (w.names, w.successors, w.start_targets),
            (parts.names(), parts.successors, parts.start)
        );
    }

    // --- Scheduler -----------------------------------------------------------

    /// Wiring for `len` nodes named `n0`, `n1`, …, with the given successors and
    /// start targets. Names are zero-padded so name order is index order.
    fn wiring(successors: Vec<Vec<usize>>, start_targets: Vec<usize>) -> Wiring {
        let len = successors.len();
        Wiring {
            names: (0..len)
                .map(|i| NodeName::new(&format!("n{i:03}")).unwrap())
                .collect(),
            rank: (0..len).collect(),
            successors,
            start_targets,
        }
    }

    /// The planning rule applied to every node, ignoring the candidate index.
    fn plan_by_full_scan(s: &Scheduler<'_>) -> Vec<usize> {
        (0..s.versions.len())
            .filter(|&n| s.versions[n] > s.versions_seen[n])
            .collect()
    }

    /// A random functional graph: each node writes at most one trigger.
    #[derive(Debug, Arbitrary)]
    struct Graph {
        /// The number of nodes.
        #[strategy(1usize..40)]
        len: usize,
        /// For each node, the index (modulo `len`) of its successor, or `None` for
        /// `END`.
        #[strategy(prop::collection::vec(prop::option::of(0usize..1000), #len))]
        successors: Vec<Option<usize>>,
        /// The index (modulo `len`) of the start target.
        start: usize,
        /// How many steps to simulate: at least one, so the step loop always
        /// runs.
        #[strategy(1usize..60)]
        steps: usize,
    }

    impl Graph {
        /// The graph's wiring.
        fn wiring(&self) -> Wiring {
            wiring(
                self.successors
                    .iter()
                    .map(|s| s.map(|t| t % self.len).into_iter().collect())
                    .collect(),
                vec![self.start % self.len],
            )
        }
    }

    /// Wiring for a path through every node, from `START` to `path[0]`: each
    /// node leads to the next, and the last one to `END` or, if `closed`,
    /// back to `path[0]`.
    fn path_wiring(path: &[usize], closed: bool) -> Wiring {
        let last = closed.then_some(path[0]);
        let next = path.iter().skip(1).copied().map(Some).chain(once(last));
        let mut successors = vec![Vec::new(); path.len()];
        for (&node, next) in path.iter().zip(next) {
            successors[node] = next.into_iter().collect();
        }
        wiring(successors, vec![path[0]])
    }

    /// The scheduling state once the input's barrier has written the start
    /// targets.
    fn started(w: &Wiring) -> Scheduler<'_> {
        let mut s = Scheduler::new(w);
        s.barrier(&[], w.start_targets.clone());
        s
    }

    /// Runs one superstep, writing the trigger channels its tasks lead to,
    /// and returns what it planned.
    fn step(s: &mut Scheduler<'_>) -> Vec<usize> {
        let tasks = s.plan();
        let written = s.written_by(&tasks);
        s.barrier(&tasks, written);
        tasks
    }

    /// The scheduling state after the input barrier and `steps` supersteps.
    fn after(w: &Wiring, steps: usize) -> Scheduler<'_> {
        let mut s = started(w);
        for _ in 0..steps {
            step(&mut s);
        }
        s
    }

    /// Runs `steps` supersteps after the input barrier and returns what each
    /// one planned.
    fn run(w: &Wiring, steps: usize) -> Vec<Vec<usize>> {
        let mut s = started(w);
        (0..steps).map(|_| step(&mut s)).collect()
    }

    /// The input's barrier plans exactly the start targets, in rank order,
    /// whatever order the input wrote them in.
    #[proptest]
    fn the_input_barrier_plans_exactly_the_start_targets(
        #[strategy(1..8usize)] _nodes: usize,
        /// The nodes in rank order.
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())]
        order: Vec<usize>,
        /// The start targets, in rank order.
        #[strategy(prop::sample::subsequence(#order.clone(), 1..=#order.len()))]
        targets: Vec<usize>,
        /// The start targets, in the order the input writes them.
        #[strategy(Just(#targets.clone()).prop_shuffle())]
        written: Vec<usize>,
    ) {
        let mut w = wiring(vec![Vec::new(); order.len()], written);
        for (position, &node) in order.iter().enumerate() {
            w.rank[node] = position;
        }
        prop_assert_eq!(started(&w).plan(), targets);
    }

    /// Nothing is planned before the input's barrier, whatever the graph.
    #[proptest]
    fn nothing_is_planned_before_the_input(graph: Graph) {
        prop_assert_eq!(Scheduler::new(&graph.wiring()).plan(), Vec::<usize>::new());
    }

    /// A node whose trigger channel is written several times in one step is
    /// planned once.
    #[proptest]
    fn a_node_written_several_times_is_planned_once(
        #[strategy(1..8usize)] nodes: usize,
        /// The written nodes, in index order, which is rank order here.
        #[strategy(prop::sample::subsequence((0..#nodes).collect::<Vec<usize>>(), 1..=#nodes))]
        targets: Vec<usize>,
        /// The written nodes that are written again.
        #[strategy(prop::sample::subsequence(#targets.clone(), 1..=#targets.len()))]
        _repeated: Vec<usize>,
        /// Every write, in the order they are made.
        #[strategy(Just([#targets.clone(), #_repeated.clone()].concat()).prop_shuffle())]
        written: Vec<usize>,
    ) {
        let w = wiring(vec![Vec::new(); nodes], written);
        prop_assert_eq!(started(&w).plan(), targets);
    }

    /// Planning reads the scheduling state and never changes it, at any
    /// point of any run.
    #[proptest]
    fn planning_changes_nothing(graph: Graph) {
        let w = graph.wiring();
        let s = after(&w, graph.steps);
        let before = (s.versions.clone(), s.versions_seen.clone(), s.max_version);
        prop_assert_eq!(s.plan(), s.plan());
        prop_assert_eq!((s.versions, s.versions_seen, s.max_version), before);
    }

    /// A barrier that writes nothing leaves every version, and the largest
    /// one, as they were, at any point of any run.
    #[proptest]
    fn a_barrier_without_writes_keeps_every_version(graph: Graph) {
        let w = graph.wiring();
        let mut s = after(&w, graph.steps);
        let before = (s.versions.clone(), s.max_version);
        let tasks = s.plan();
        s.barrier(&tasks, Vec::new());
        prop_assert_eq!((s.versions, s.max_version), before);
    }

    /// A barrier that writes nothing still marks the nodes that ran as
    /// having seen their channels' current versions, at any point of any
    /// run. Every node is marked, because the run may have ended and planned
    /// none.
    #[proptest]
    fn a_barrier_without_writes_marks_what_ran_as_seen(graph: Graph) {
        let w = graph.wiring();
        let mut s = after(&w, graph.steps);
        let ran: Vec<usize> = (0..graph.len).collect();
        s.barrier(&ran, Vec::new());
        prop_assert_eq!(s.versions_seen, s.versions);
    }

    /// A chain of any length, through the nodes in any order, runs one node
    /// per step in chain order, then plans nothing.
    #[proptest]
    fn a_chain_runs_in_order_then_stops(
        #[strategy(1..8usize)] _nodes: usize,
        /// The chain, `START` to its first node.
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())]
        path: Vec<usize>,
    ) {
        let planned = run(&path_wiring(&path, false), path.len() + 1);
        let expected: Vec<Vec<usize>> = path
            .iter()
            .map(|&n| vec![n])
            .chain(once(Vec::new()))
            .collect();
        prop_assert_eq!(planned, expected);
    }

    /// A cycle of any length, a self-loop included, through the nodes in any
    /// order, runs one node per step in cycle order, round after round.
    #[proptest]
    fn a_cycle_repeats_in_order(
        #[strategy(1..8usize)] _nodes: usize,
        /// The cycle, `START` to its first node.
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())]
        path: Vec<usize>,
        /// How many times to go round.
        #[strategy(1..4usize)]
        rounds: usize,
    ) {
        // One step past the last round, so the closing edge is always seen.
        let steps = path.len() * rounds + 1;
        let planned = run(&path_wiring(&path, true), steps);
        let expected: Vec<Vec<usize>> = path.iter().cycle().take(steps).map(|&n| vec![n]).collect();
        prop_assert_eq!(planned, expected);
    }

    /// The indexed planner selects exactly what a full scan selects, at every
    /// step, and versions behave as specified.
    #[proptest(cases = 512)]
    fn planning_matches_a_full_scan_and_versions_grow_only_on_writes(graph: Graph) {
        let w = graph.wiring();
        let mut s = started(&w);
        for _ in 0..graph.steps {
            let tasks = s.plan();
            prop_assert_eq!(&tasks, &plan_by_full_scan(&s));
            let before = s.versions.clone();
            let max_before = s.max_version;
            let written = s.written_by(&tasks);
            s.barrier(&tasks, written.clone());
            for (channel, (old, new)) in before.iter().zip(&s.versions).enumerate() {
                if written.contains(&channel) {
                    // Written channels share the step's version, one above the old maximum.
                    prop_assert_eq!(*new, max_before.next());
                    prop_assert!(new > old);
                } else {
                    prop_assert_eq!(new, old);
                }
            }
            prop_assert_eq!(s.max_version, *s.versions.iter().max().unwrap());
            for &n in &tasks {
                prop_assert_eq!(
                    s.versions_seen[n],
                    before[n],
                    "a task sees the version it was planned for"
                );
            }
        }
    }
}
