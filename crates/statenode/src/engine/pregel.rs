//! The Pregel scheduler: a validated topology lowered to channels, and the
//! scheduling state of one run: channel versions, what each channel's node
//! has seen, and which sources each join has heard from.
//!
//! Every node has a trigger channel, `branch:to:<node>`, and every join a
//! barrier channel, `join:<sources>:<target>`. Each channel triggers exactly
//! one node. A node is planned when one of its channels is *available* and
//! has a newer version than the node saw when it last ran: a trigger channel
//! is always available, a barrier only once every source has arrived. The
//! barrier at the end of a step consumes the barriers that fired, records the
//! step's writes and arrivals, gives every channel that changed one new
//! version, and records what each node that ran has seen.
//!
//! This is the only module tied to the scheduling algorithm. Non-generic, so
//! every state type shares one implementation of the scheduling rules.

use crate::builder::ValidTopology;
use crate::name::NodeName;

/// The version of a channel.
///
/// Versions only grow. At each barrier one next version is computed from the
/// largest current version and given to every channel that changed in that
/// step, written, arrived at or consumed, so a version depends only on the
/// channel map, never on a counter that restarts with each run.
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
/// Channels `0..n` are the nodes' trigger channels, with the same index as
/// their node; channels from `n` on are the joins' barriers, in declaration
/// order. An edge `a → b` means "when `a` completes, write `b`'s trigger
/// channel" (an edge to `END` writes nothing); a join `[a, b] → c` means
/// "when `a` or `b` completes, it arrives at the join's barrier", which
/// triggers `c` once both have.
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
    /// The node each channel triggers, by channel.
    pub(crate) subscriber: Vec<usize>,
    /// The channels that trigger each node by node: its trigger channel,
    /// then the barriers of the joins into it.
    pub(crate) triggers: Vec<Vec<usize>>,
    /// Where each node arrives when it completes, by node: a barrier channel
    /// and the node's position among that join's sources.
    pub(crate) arrivals: Vec<Vec<(usize, usize)>>,
    /// How many sources each channel waits for, by channel: none for a
    /// trigger channel, every source of the join for a barrier.
    pub(crate) required: Vec<usize>,
}

impl Wiring {
    /// Lowers `topology`: each node's successors become the trigger channels
    /// it writes, each joins a barrier channel, and nodes are ranked by name
    /// for task-path order.
    pub(crate) fn new(topology: ValidTopology) -> Self {
        let len = topology.names.len();
        let mut by_name: Vec<(&NodeName, usize)> = topology.names.iter().zip(0..).collect();
        by_name.sort_unstable();
        let mut rank = vec![0; len];
        for (position, &(_, node)) in by_name.iter().enumerate() {
            rank.get_mut(node).into_iter().for_each(|r| *r = position);
        }
        let mut subscriber: Vec<usize> = (0..len).collect();
        let mut triggers: Vec<Vec<usize>> = (0..len).map(|node| vec![node]).collect();
        let mut arrivals: Vec<Vec<(usize, usize)>> = vec![Vec::new(); len];
        let mut required = vec![0; len];
        for (barrier, join) in (len..).zip(&topology.joins) {
            subscriber.push(join.target);
            required.push(join.sources.len());
            triggers
                .get_mut(join.target)
                .into_iter()
                .for_each(|of| of.push(barrier));
            for (position, &source) in join.sources.iter().enumerate() {
                arrivals
                    .get_mut(source)
                    .into_iter()
                    .for_each(|of| of.push((barrier, position)));
            }
        }
        Self {
            rank,
            successors: topology.successors,
            start_targets: topology.start,
            names: topology.names,
            subscriber,
            triggers,
            arrivals,
            required,
        }
    }
}

/// The scheduling state of one run.
///
/// Planning reads it and never changes it; only [`Scheduler::barrier`],
/// called after a step's updates have been applied successfully, changes it.
/// A failing step therefore leaves the scheduling state exactly as it was at
/// its start.
#[derive(Debug)]
pub(crate) struct Scheduler<'w> {
    /// The graph's lowered edges and joins.
    wiring: &'w Wiring,
    /// The version of each channel, by channel.
    versions: Vec<ChannelVersion>,
    /// The version of each channel its node last ran against, by channel:
    /// each channel triggers one node, so one entry per channel suffices.
    versions_seen: Vec<ChannelVersion>,
    /// The largest version of any channel.
    max_version: ChannelVersion,
    /// The channels written, or barriers completed, at the last barrier: the
    /// only candidates for the next step, which keeps planning proportional
    /// to the writes.
    updated: Vec<usize>,
    /// Which sources have arrived at each barrier since it last fired, by
    /// channel and source position; empty for trigger channels.
    arrived: Vec<Vec<bool>>,
    /// How many sources have arrived at each barrier, by channel.
    count: Vec<usize>,
}

impl<'w> Scheduler<'w> {
    /// The scheduling state before the input is applied: every version is 0,
    /// and no source has arrived anywhere.
    pub(crate) fn new(wiring: &'w Wiring) -> Self {
        let channels = wiring.required.len();
        Self {
            wiring,
            versions: vec![ChannelVersion::default(); channels],
            versions_seen: vec![ChannelVersion::default(); channels],
            max_version: ChannelVersion::default(),
            updated: Vec::new(),
            arrived: wiring.required.iter().map(|&n| vec![false; n]).collect(),
            count: vec![0; channels],
        }
    }

    /// Whether `channel` can trigger its node: a trigger channel always can,
    /// a barrier once every source has arrived.
    fn available(&self, channel: usize) -> bool {
        self.count.get(channel) == self.wiring.required.get(channel)
    }

    /// Whether `channel` is a barrier every source has arrived at.
    fn complete(&self, channel: usize) -> bool {
        self.wiring.required.get(channel).is_some_and(|&n| n > 0) && self.available(channel)
    }

    /// The nodes to run in the next step, sorted by task path, each once.
    ///
    /// A node runs if one of its channels is available and has a newer
    /// version than the one it last ran against.
    ///
    /// The candidates are the nodes that the channels updated at the last
    /// barrier trigger. A candidate planned from the last barrier's state
    /// always passes the comparison: its channel was written, or its barrier
    /// completed, after the node last ran. The comparison still decides, so
    /// planning stays correct from versions that did not come from the last
    /// barrier, such as stored ones.
    pub(crate) fn plan(&self) -> Vec<usize> {
        let mut tasks: Vec<usize> = self
            .updated
            .iter()
            .filter_map(|&channel| self.wiring.subscriber.get(channel).copied())
            .collect();
        tasks.sort_unstable_by_key(|&node| (self.wiring.rank.get(node).copied(), node));
        tasks.dedup();
        tasks.retain(|&node| {
            self.wiring
                .triggers
                .get(node)
                .into_iter()
                .flatten()
                .any(|&channel| {
                    self.available(channel)
                        && self.versions.get(channel) > self.versions_seen.get(channel)
                })
        });
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

    /// Commits a step: first `ran` saw the versions they were planned
    /// against, and the barriers that fired for them are consumed; then
    /// `written` and the arrivals of `ran` are recorded, so a source that ran
    /// as its join fired counts towards the join's next round. Every channel
    /// that changed gets the step's version; the written channels and the
    /// barriers that completed are the next step's candidates.
    pub(crate) fn barrier(&mut self, ran: &[usize], written: Vec<usize>) {
        let wiring = self.wiring;
        let mut changed = Vec::new();
        for &node in ran {
            for &channel in wiring.triggers.get(node).into_iter().flatten() {
                let version = self.versions.get(channel).copied();
                self.versions_seen
                    .get_mut(channel)
                    .into_iter()
                    .zip(version)
                    .for_each(|(seen, version)| *seen = version);
                if self.complete(channel) {
                    self.arrived
                        .get_mut(channel)
                        .into_iter()
                        .flatten()
                        .for_each(|arrived| *arrived = false);
                    self.count
                        .get_mut(channel)
                        .into_iter()
                        .for_each(|count| *count = 0);
                    changed.push(channel);
                }
            }
        }
        let mut updated = written;
        changed.extend(&updated);
        for &node in ran {
            for &(channel, position) in wiring.arrivals.get(node).into_iter().flatten() {
                let new = self
                    .arrived
                    .get_mut(channel)
                    .and_then(|arrived| arrived.get_mut(position))
                    .is_some_and(|arrived| !std::mem::replace(arrived, true));
                if new {
                    self.count
                        .get_mut(channel)
                        .into_iter()
                        .for_each(|count| *count += 1);
                    changed.push(channel);
                    if self.complete(channel) {
                        updated.push(channel);
                    }
                }
            }
        }
        changed.sort_unstable();
        changed.dedup();
        updated.sort_unstable();
        updated.dedup();
        if !changed.is_empty() {
            let next = self.max_version.next();
            for &channel in &changed {
                self.versions
                    .get_mut(channel)
                    .into_iter()
                    .for_each(|version| *version = next);
            }
            self.max_version = next;
        }
        self.updated = updated;
    }
}

#[cfg(test)]
mod tests {
    //! Tests for channel versions, lowering, and the scheduling rules.

    use proptest::prelude::*;
    use test_strategy::{Arbitrary, proptest};

    use std::collections::{BTreeMap, BTreeSet};
    use std::iter::once;

    use super::*;
    use crate::builder::Join;
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
        #[strategy(prop::collection::btree_set(valid_name(8), 2..8).prop_map(Vec::from_iter).prop_shuffle())]
        names: Vec<String>,
        /// The nodes `START` leads to.
        #[strategy(prop::collection::vec(0..#names.len(), 1..3))]
        start: Vec<usize>,
        /// The nodes each node leads to.
        #[strategy(prop::collection::vec(prop::collection::vec(0..#names.len(), 0..3), #names.len()))]
        successors: Vec<Vec<usize>>,
        /// The joins: at least one, so the join tests' loops run on every
        /// case, each with distinct sources and a target.
        #[strategy(prop::collection::vec(
            (
                prop::sample::subsequence(
                    (0..#names.len()).collect::<Vec<usize>>(),
                    2..=#names.len().min(3)
                ),
                0..#names.len(),
            ),
            1..3
        ))]
        joins: Vec<(Vec<usize>, usize)>,
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
                joins: self
                    .joins
                    .iter()
                    .map(|(sources, target)| Join {
                        sources: sources.clone(),
                        target: *target,
                    })
                    .collect(),
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

    /// Each node's trigger channel shares its index and triggers it alone;
    /// each join's barrier channel, after them in declaration order,
    /// triggers the join's target, waits for every source, and is where each
    /// source arrives, at its position in the join.
    #[proptest]
    fn lowering_wires_each_join(parts: Parts) {
        let w = Wiring::new(parts.topology());
        let len = parts.names.len();
        let barriers = (len..).zip(&parts.joins);
        let mut triggers: Vec<Vec<usize>> = (0..len).map(|node| vec![node]).collect();
        let mut arrivals: Vec<Vec<(usize, usize)>> = vec![Vec::new(); len];
        for (barrier, (sources, target)) in barriers.clone() {
            triggers[*target].push(barrier);
            for (position, &source) in sources.iter().enumerate() {
                arrivals[source].push((barrier, position));
            }
        }
        let subscriber: Vec<usize> = (0..len)
            .chain(parts.joins.iter().map(|(_, target)| *target))
            .collect();
        let required: Vec<usize> = std::iter::repeat_n(0, len)
            .chain(parts.joins.iter().map(|(sources, _)| sources.len()))
            .collect();
        prop_assert_eq!(
            (w.subscriber, w.required, w.triggers, w.arrivals),
            (subscriber, required, triggers, arrivals)
        );
    }

    // --- Scheduler -----------------------------------------------------------

    /// Wiring lowered from one node per entry of `successors`, named `n000`,
    /// `n001`, …, so name order is index order, with the given start targets
    /// and joins, each its sources and target.
    fn join_wiring(
        successors: Vec<Vec<usize>>,
        start_targets: Vec<usize>,
        joins: Vec<(Vec<usize>, usize)>,
    ) -> Wiring {
        Wiring::new(ValidTopology {
            names: (0..successors.len())
                .map(|i| NodeName::new(&format!("n{i:03}")).unwrap())
                .collect(),
            start: start_targets,
            successors,
            joins: joins
                .into_iter()
                .map(|(sources, target)| Join { sources, target })
                .collect(),
        })
    }

    /// Wiring without joins: see [`join_wiring`].
    fn wiring(successors: Vec<Vec<usize>>, start_targets: Vec<usize>) -> Wiring {
        join_wiring(successors, start_targets, Vec::new())
    }

    /// The wiring of one join of the nodes `0..sources` into node `sources`,
    /// without edges.
    fn one_join(sources: usize) -> Wiring {
        join_wiring(
            vec![Vec::new(); sources + 1],
            Vec::new(),
            vec![((0..sources).collect(), sources)],
        )
    }

    /// `nodes` grouped into steps, `nodes[i]` running in step `steps[i]`:
    /// the steps that run a node, in ascending order, each listing its nodes
    /// in input order. `steps` has one entry per node.
    fn in_steps(nodes: &[usize], steps: &[usize]) -> Vec<Vec<usize>> {
        let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
        for (&node, &step) in nodes.iter().zip(steps) {
            grouped.entry(step).or_default().push(node);
        }
        grouped.into_values().collect()
    }

    /// The nodes in `0..sources`, a join's sources in these tests, that are
    /// not in `nodes`, in index order.
    fn others(sources: usize, nodes: &[usize]) -> Vec<usize> {
        (0..sources).filter(|i| !nodes.contains(i)).collect()
    }

    /// The planning rule applied to every node, ignoring the candidates: the
    /// nodes, in index order, with an available channel newer than they saw.
    fn plan_by_full_scan(s: &Scheduler<'_>) -> Vec<usize> {
        (0..s.wiring.names.len())
            .filter(|&n| {
                s.wiring.triggers[n]
                    .iter()
                    .any(|&ch| s.available(ch) && s.versions[ch] > s.versions_seen[ch])
            })
            .collect()
    }

    /// A random graph: `START` leads to up to three random nodes, each node
    /// to up to two, and up to two joins of two or three nodes each lead to
    /// another.
    ///
    /// Its wiring appends a fixed gadget of four nodes, `START → a`,
    /// `START → x → b` and the join `[a, b] → c`, so whatever was drawn the
    /// first four steps leave a barrier partial, plan with it unavailable and
    /// complete it, consume it, and leave it alone: the properties' test code
    /// takes every arm on every run.
    #[derive(Debug, Arbitrary)]
    struct Graph {
        /// The number of random nodes.
        #[strategy(2usize..40)]
        len: usize,
        /// For each node, the indices (modulo `len`) of its successors; `None`
        /// is `END`.
        #[strategy(prop::collection::vec(
            prop::collection::btree_set(prop::option::of(0usize..1000), 0..=2),
            #len
        ))]
        successors: Vec<BTreeSet<Option<usize>>>,
        /// The indices (modulo `len`) of the start targets.
        #[strategy(prop::collection::btree_set(0usize..1000, 1..=3))]
        start: BTreeSet<usize>,
        /// The joins, each with distinct sources, by node index.
        #[strategy(prop::collection::vec(
            (
                prop::sample::subsequence((0..#len).collect::<Vec<usize>>(), 2..=#len.min(3)),
                0..#len,
            ),
            0..=2
        ))]
        joins: Vec<(Vec<usize>, usize)>,
        /// How many steps to simulate: at least four, so the gadget's barrier
        /// goes through every phase.
        #[strategy(4usize..60)]
        steps: usize,
    }

    impl Graph {
        /// The graph's wiring, with the gadget's `a`, `x`, `b` and `c` after
        /// the random nodes. Each node's triggers are distinct, as a validated
        /// topology's are.
        fn wiring(&self) -> Wiring {
            let (a, x, b, c) = (self.len, self.len + 1, self.len + 2, self.len + 3);
            // One closure for both uses, so the start targets, which are never
            // empty, cover it.
            let wrap = |t: usize| t % self.len;
            let distinct = |targets: BTreeSet<usize>| targets.into_iter().collect::<Vec<_>>();
            let successors = self
                .successors
                .iter()
                .map(|targets| distinct(targets.iter().flatten().copied().map(wrap).collect()))
                .chain([Vec::new(), vec![b], Vec::new(), Vec::new()])
                .collect();
            let start = distinct(self.start.iter().copied().map(wrap).collect())
                .into_iter()
                .chain([a, x])
                .collect();
            let joins = self
                .joins
                .iter()
                .cloned()
                .chain(once((vec![a, b], c)))
                .collect();
            join_wiring(successors, start, joins)
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
        let snapshot = |s: &Scheduler<'_>| {
            (
                s.versions.clone(),
                s.versions_seen.clone(),
                s.max_version,
                s.updated.clone(),
                s.arrived.clone(),
                s.count.clone(),
            )
        };
        let before = snapshot(&s);
        prop_assert_eq!(s.plan(), s.plan());
        prop_assert_eq!(snapshot(&s), before);
    }

    /// A barrier with nothing run and nothing written leaves every version,
    /// and the largest one, as they were, at any point of any run, and plans
    /// nothing.
    #[proptest]
    fn a_barrier_without_writes_keeps_every_version(graph: Graph) {
        let w = graph.wiring();
        let mut s = after(&w, graph.steps);
        let before = (s.versions.clone(), s.max_version);
        s.barrier(&[], Vec::new());
        prop_assert_eq!((s.versions.clone(), s.max_version), before);
        prop_assert!(s.plan().is_empty());
    }

    /// A barrier marks the nodes that ran as having seen the version each of
    /// their channels had before it, at any point of any run, even when it
    /// writes nothing. Every node is marked, because the run may have ended
    /// and planned none.
    #[proptest]
    fn a_barrier_without_writes_marks_what_ran_as_seen(graph: Graph) {
        let w = graph.wiring();
        let mut s = after(&w, graph.steps);
        let before = s.versions.clone();
        let ran: Vec<usize> = (0..w.names.len()).collect();
        s.barrier(&ran, Vec::new());
        prop_assert_eq!(s.versions_seen, before);
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
    /// step, and versions change as specified: a channel changes exactly when
    /// it is written, a source arrives at it anew, or it is a complete barrier
    /// whose node ran; every channel that changes gets the step's one new
    /// version; and each task's channels are marked seen at their versions
    /// before the barrier.
    #[proptest(cases = 512)]
    fn planning_matches_a_full_scan_and_versions_change_only_on_writes_arrivals_and_consumes(
        graph: Graph,
    ) {
        let w = graph.wiring();
        let mut s = started(&w);
        for _ in 0..graph.steps {
            let tasks = s.plan();
            prop_assert_eq!(&tasks, &plan_by_full_scan(&s));
            let before = s.versions.clone();
            let max_before = s.max_version;
            let written = s.written_by(&tasks);
            let consumed = tasks
                .iter()
                .flat_map(|&n| w.triggers[n].iter().copied())
                .filter(|&ch| s.complete(ch));
            let arrived = tasks
                .iter()
                .flat_map(|&n| w.arrivals[n].iter().copied())
                .filter(|&(ch, position)| !s.arrived[ch][position])
                .map(|(ch, _)| ch);
            let changes: BTreeSet<usize> = written
                .iter()
                .copied()
                .chain(consumed)
                .chain(arrived)
                .collect();
            s.barrier(&tasks, written);
            for (channel, (old, new)) in before.iter().zip(&s.versions).enumerate() {
                if changes.contains(&channel) {
                    prop_assert_eq!(*new, max_before.next());
                } else {
                    prop_assert_eq!(new, old);
                }
            }
            prop_assert_eq!(s.max_version, *s.versions.iter().max().unwrap());
            for &n in &tasks {
                for &ch in &w.triggers[n] {
                    prop_assert_eq!(
                        s.versions_seen[ch],
                        before[ch],
                        "a task sees the versions it was planned against"
                    );
                }
            }
        }
    }

    /// A join's target is planned exactly once its last source has arrived,
    /// whatever order the sources arrive in, one per step.
    #[proptest]
    fn a_barrier_fires_when_its_last_source_arrives(
        #[strategy(2..=5usize)] sources: usize,
        /// The order the sources arrive in.
        #[strategy(Just((0..#sources).collect::<Vec<usize>>()).prop_shuffle())]
        order: Vec<usize>,
    ) {
        let w = one_join(sources);
        let mut s = Scheduler::new(&w);
        for (i, &source) in order.iter().enumerate() {
            s.barrier(&[source], Vec::new());
            let expected = if i + 1 == sources {
                vec![sources]
            } else {
                Vec::new()
            };
            prop_assert_eq!(s.plan(), expected);
        }
    }

    /// A barrier that fired, however its sources' arrivals were grouped into
    /// steps, resets once its target runs, its version bumping so a later
    /// round fires it again, and is then no longer a candidate.
    #[proptest]
    fn a_fired_barrier_resets_and_bumps_its_version(
        #[strategy(2..=5usize)] sources: usize,
        /// The step each source arrives in.
        #[strategy(prop::collection::vec(0..#sources, #sources))]
        steps: Vec<usize>,
    ) {
        let (target, barrier) = (sources, sources + 1);
        let w = one_join(sources);
        let mut s = Scheduler::new(&w);
        for step in in_steps(&(0..sources).collect::<Vec<_>>(), &steps) {
            s.barrier(&step, Vec::new());
        }
        prop_assert_eq!(s.plan(), vec![target]);
        let fired = s.versions[barrier];
        s.barrier(&[target], Vec::new());
        prop_assert_eq!(&s.arrived[barrier], &vec![false; sources]);
        prop_assert_eq!(s.count[barrier], 0);
        prop_assert!(s.versions[barrier] > fired);
        prop_assert!(s.plan().is_empty());
    }

    /// A source that arrives again before its join's target runs, whether
    /// the barrier is partial or complete, leaves the barrier's arrivals, its
    /// count and every version unchanged.
    #[proptest]
    fn a_repeated_arrival_changes_nothing(
        #[strategy(2..=5usize)] sources: usize,
        /// The sources that arrive first.
        #[strategy(prop::sample::subsequence((0..#sources).collect::<Vec<usize>>(), 1..=#sources))]
        arrived: Vec<usize>,
        /// The sources that arrive again.
        #[strategy(prop::sample::subsequence(#arrived.clone(), 1..=#arrived.len()))]
        again: Vec<usize>,
    ) {
        let w = one_join(sources);
        let mut s = Scheduler::new(&w);
        s.barrier(&arrived, Vec::new());
        // `updated` is left out: the second step completes nothing, so it
        // updates nothing, even when the first completed the barrier.
        let before = (s.versions.clone(), s.arrived.clone(), s.count.clone());
        s.barrier(&again, Vec::new());
        prop_assert_eq!((s.versions, s.arrived, s.count), before);
    }

    /// A node triggered in one step by both an edge and a join is planned
    /// once, whichever source has the edge.
    #[proptest]
    fn an_edge_and_a_join_into_one_node_plan_it_once(
        #[strategy(2..=5usize)] sources: usize,
        /// The source with an edge to the target as well.
        #[strategy(0..#sources)]
        edge: usize,
    ) {
        let (target, barrier) = (sources, sources + 1);
        let mut successors = vec![Vec::new(); sources + 1];
        successors[edge] = vec![target];
        let w = join_wiring(
            successors,
            Vec::new(),
            vec![((0..sources).collect(), target)],
        );
        let mut s = Scheduler::new(&w);
        s.barrier(&others(sources, &[edge]), Vec::new());
        s.barrier(&[edge], s.written_by(&[edge]));
        prop_assert_eq!(
            &s.updated,
            &vec![target, barrier],
            "the edge's channel and the barrier"
        );
        prop_assert_eq!(s.plan(), vec![target]);
    }

    /// A barrier some sources have arrived at, but not all, however their
    /// arrivals are grouped into steps, is no candidate.
    #[proptest]
    fn a_partial_barrier_is_not_a_candidate(
        #[strategy(2..=5usize)] sources: usize,
        /// The sources that arrive: some, never all.
        #[strategy(prop::sample::subsequence((0..#sources).collect::<Vec<usize>>(), 1..#sources))]
        partial: Vec<usize>,
        /// The step each of them arrives in.
        #[strategy(prop::collection::vec(0..#sources, #partial.len()))]
        steps: Vec<usize>,
    ) {
        let w = one_join(sources);
        let mut s = Scheduler::new(&w);
        for step in in_steps(&partial, &steps) {
            s.barrier(&step, Vec::new());
            prop_assert_eq!(&s.updated, &Vec::<usize>::new());
            prop_assert!(s.plan().is_empty());
        }
    }

    /// A barrier keeps its arrivals when its target runs through another
    /// channel while it is partial, whichever sources have arrived: only a
    /// complete barrier is consumed.
    #[proptest]
    fn a_partial_barrier_survives_its_target_running_through_another_channel(
        #[strategy(2..=5usize)] sources: usize,
        /// The sources that arrive: some, never all.
        #[strategy(prop::sample::subsequence((0..#sources).collect::<Vec<usize>>(), 1..#sources))]
        partial: Vec<usize>,
        /// The arrived source that also has an edge to the target.
        #[strategy(prop::sample::select(#partial.clone()))]
        edge: usize,
    ) {
        let (target, barrier) = (sources, sources + 1);
        let mut successors = vec![Vec::new(); sources + 1];
        successors[edge] = vec![target];
        let w = join_wiring(
            successors,
            Vec::new(),
            vec![((0..sources).collect(), target)],
        );
        let mut s = Scheduler::new(&w);
        s.barrier(&partial, s.written_by(&partial));
        prop_assert_eq!(s.plan(), vec![target], "through the edge");
        s.barrier(&[target], Vec::new());
        prop_assert_eq!(
            &s.arrived[barrier],
            &(0..sources)
                .map(|i| partial.contains(&i))
                .collect::<Vec<bool>>()
        );
        s.barrier(&others(sources, &partial), Vec::new());
        prop_assert_eq!(
            s.plan(),
            vec![target],
            "through the join, which kept the earlier arrivals"
        );
    }

    /// A source that runs in the step its join fires counts towards the
    /// join's next round, whichever sources run again and in whatever order
    /// the step lists its tasks: the barrier is consumed before the step's
    /// arrivals are recorded.
    #[proptest]
    fn a_source_that_runs_as_its_join_fires_counts_for_the_next_round(
        #[strategy(2..=5usize)] sources: usize,
        /// The sources that run again alongside the target: some, never all.
        #[strategy(prop::sample::subsequence((0..#sources).collect::<Vec<usize>>(), 1..#sources))]
        rerun: Vec<usize>,
        /// The step's tasks: the rerun sources and the target, in any order.
        #[strategy(Just(#rerun.iter().copied().chain(once(#sources)).collect::<Vec<usize>>()).prop_shuffle())]
        ran: Vec<usize>,
    ) {
        let target = sources;
        let w = one_join(sources);
        let mut s = Scheduler::new(&w);
        s.barrier(&(0..sources).collect::<Vec<_>>(), Vec::new());
        prop_assert_eq!(s.plan(), vec![target]);
        s.barrier(&ran, Vec::new());
        prop_assert!(
            s.plan().is_empty(),
            "only the rerun sources have arrived since the target ran"
        );
        s.barrier(&others(sources, &rerun), Vec::new());
        prop_assert_eq!(s.plan(), vec![target]);
    }
}
