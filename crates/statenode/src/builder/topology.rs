//! [`Topology`]: a graph's names, edges and joins, and every check on them,
//! and the [`ValidTopology`] it produces.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{BuildError, NodeId, Source, Sources, Target};
use crate::name::{NameError, NodeName};
use crate::state::ChannelSpec;

/// The tag source for graph builders.
///
/// This is the one documented exception to "no process-global mutable state":
/// the rule exists to stop shared state from changing behaviour, and a
/// uniqueness counter cannot. A tag only lets a builder recognise its own
/// [`NodeId`]s; it never reaches anything persisted or observable, so the
/// counter restarting with the process is harmless. At 10⁹ builders a second
/// it would take over 500 years to wrap.
static NEXT_BUILDER: AtomicU64 = AtomicU64::new(0);

/// A graph's names and edges, and every check on them.
///
/// Non-generic, so every state type shares one implementation of the checks.
#[derive(Debug)]
pub(crate) struct Topology {
    /// This builder's tag, carried by every [`NodeId`] it creates.
    tag: u64,
    /// Node names, by node index.
    names: Vec<NodeName>,
    /// The same names, for duplicate detection.
    known: HashSet<NodeName>,
    /// Edges, in the order they were added. Visible to the builder, whose
    /// tests check what its `add_edge` records.
    pub(super) edges: Vec<Edge>,
    /// The same edges, for duplicate detection.
    declared: HashSet<Edge>,
    /// Joins, in the order they were added. Visible to the builder, whose
    /// tests check what its `add_edge` records.
    pub(super) joins: Vec<Join>,
    /// The same joins for duplicate detection: their sources are in name
    /// order, so a join repeated in another order is the same.
    declared_joins: HashSet<Join>,
}

/// An edge, by node index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Edge {
    /// The source node; `None` is `START`.
    pub(super) from: Option<usize>,
    /// The target node; `None` is `END`.
    pub(super) to: Option<usize>,
}

impl Edge {
    /// The edge as `from → to`, naming nodes by `names` and the boundaries
    /// `START` and `END`.
    fn describe(self, names: &[NodeName]) -> String {
        format!(
            "{} → {}",
            end_name(names, self.from, "START"),
            end_name(names, self.to, "END")
        )
    }
}

/// The name of the node at `index`, or `boundary` for `None` (or an index
/// with no name, which a resolved edge never has).
fn end_name<'a>(names: &'a [NodeName], index: Option<usize>, boundary: &'a str) -> &'a str {
    index
        .and_then(|i| names.get(i))
        .map_or(boundary, NodeName::as_str)
}

/// A join, by node index: it fires, and `target` runs, once every source has
/// run since it last fired.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Join {
    /// The source nodes, in name order.
    pub(crate) sources: Vec<usize>,
    /// The target node.
    pub(crate) target: usize,
}

impl Join {
    /// The join as `join:a+b:c`, the name of its barrier channel: its sources
    /// in name order, joined by `+`, then its target, all named from `names`.
    fn describe(&self, names: &[NodeName]) -> String {
        let sources: Vec<&str> = self
            .sources
            .iter()
            .filter_map(|&source| names.get(source))
            .map(NodeName::as_str)
            .collect();
        let target = names.get(self.target).map_or("", NodeName::as_str);
        format!("join:{}:{target}", sources.join("+"))
    }
}

impl Topology {
    /// An empty topology with a fresh tag.
    pub(crate) fn new() -> Self {
        Self {
            tag: NEXT_BUILDER.fetch_add(1, Ordering::Relaxed),
            names: Vec::new(),
            known: HashSet::new(),
            edges: Vec::new(),
            declared: HashSet::new(),
            joins: Vec::new(),
            declared_joins: HashSet::new(),
        }
    }

    /// Registers a node named `name`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidName`] or [`BuildError::DuplicateName`].
    pub(crate) fn add_node(&mut self, name: &str) -> Result<NodeId, BuildError> {
        let name = NodeName::new(name).map_err(BuildError::InvalidName)?;
        if !self.known.insert(name.clone()) {
            return Err(BuildError::DuplicateName {
                name: name.as_str().to_owned(),
            });
        }
        let index = self.names.len();
        self.names.push(name);
        Ok(NodeId {
            index,
            builder: self.tag,
        })
    }

    /// The index of `node`, if it belongs to this builder.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::ForeignNode`] for a handle from another builder.
    fn resolve(&self, node: NodeId) -> Result<usize, BuildError> {
        if node.builder == self.tag {
            Ok(node.index)
        } else {
            Err(BuildError::ForeignNode)
        }
    }

    /// Records the edge `from → to`, a plain edge or a join.
    ///
    /// # Errors
    ///
    /// Returns the first problem found, in the order documented on
    /// [`GraphBuilder::add_edge`](crate::GraphBuilder::add_edge). A rejected
    /// edge is not recorded.
    pub(crate) fn add_edge(&mut self, from: Sources, to: Target) -> Result<(), BuildError> {
        match from {
            Sources::One(source) => self.add_plain_edge(source, to),
            Sources::Join(sources) => self.add_join(&sources, to),
        }
    }

    /// Records the plain edge `from → to`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::ForeignNode`] if either end is a handle from
    /// another builder, then [`BuildError::DuplicateEdge`] if the edge was
    /// already added.
    fn add_plain_edge(&mut self, from: Source, to: Target) -> Result<(), BuildError> {
        let from = match from {
            Source::Start => None,
            Source::Node(node) => Some(self.resolve(node)?),
        };
        let to = match to {
            Target::Node(node) => Some(self.resolve(node)?),
            Target::End => None,
        };
        let edge = Edge { from, to };
        if !self.declared.insert(edge) {
            return Err(BuildError::DuplicateEdge {
                edge: edge.describe(&self.names),
            });
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Records the join of `sources` into `to`.
    ///
    /// # Errors
    ///
    /// In this order: [`BuildError::ForeignNode`] for a source, in list
    /// order, then for the target; [`BuildError::EmptyJoin`];
    /// [`BuildError::SingleSourceJoin`]; [`BuildError::DuplicateJoinSource`]
    /// for the first node to appear a second time; [`BuildError::JoinToEnd`];
    /// then [`BuildError::DuplicateEdge`] if the same sources already join
    /// into the same target.
    fn add_join(&mut self, sources: &[NodeId], to: Target) -> Result<(), BuildError> {
        let mut sources = sources
            .iter()
            .map(|&node| self.resolve(node))
            .collect::<Result<Vec<usize>, _>>()?;
        let target = match to {
            Target::Node(node) => Some(self.resolve(node)?),
            Target::End => None,
        };
        let named: Vec<(usize, &NodeName)> = sources
            .iter()
            .filter_map(|&source| self.names.get(source).map(|name| (source, name)))
            .collect();
        match named.as_slice() {
            [] => return Err(BuildError::EmptyJoin),
            [(_, name)] => {
                return Err(BuildError::SingleSourceJoin {
                    node: (*name).clone(),
                });
            }
            _ => {}
        }
        let mut seen = HashSet::new();
        if let Some(&(_, name)) = named.iter().find(|&&(source, _)| !seen.insert(source)) {
            return Err(BuildError::DuplicateJoinSource { node: name.clone() });
        }
        let target = target.ok_or(BuildError::JoinToEnd)?;
        sources.sort_unstable_by_key(|&source| self.names.get(source));
        let join = Join { sources, target };
        if !self.declared_joins.insert(join.clone()) {
            return Err(BuildError::DuplicateEdge {
                edge: join.describe(&self.names),
            });
        }
        self.joins.push(join);
        Ok(())
    }

    /// Checks the graph and the state's `channels`, and returns the validated
    /// topology.
    ///
    /// # Errors
    ///
    /// Returns the first problem found, in the order documented on
    /// [`GraphBuilder::compile`](crate::GraphBuilder::compile).
    pub(crate) fn validate(
        self,
        channels: Result<Vec<ChannelSpec>, NameError>,
    ) -> Result<ValidTopology, BuildError> {
        if self.names.is_empty() {
            return Err(BuildError::Empty);
        }

        let mut start_targets = Vec::new();
        let mut outgoing: Vec<Vec<Option<usize>>> = vec![Vec::new(); self.names.len()];
        for &Edge { from, to } in &self.edges {
            match from {
                None => start_targets.push(to),
                Some(node) => outgoing
                    .get_mut(node)
                    .into_iter()
                    .for_each(|edges| edges.push(to)),
            }
        }
        if start_targets.is_empty() {
            return Err(BuildError::NoStartEdge);
        }

        let mut by_name: Vec<(&NodeName, usize)> = self.names.iter().zip(0..).collect();
        by_name.sort_unstable();

        let reached = reachable(&start_targets, &outgoing, &self.joins, self.names.len());
        let unreachable = by_name
            .iter()
            .find(|&&(_, node)| reached.get(node).is_some_and(|&r| !r));
        if let Some(&(name, _)) = unreachable {
            return Err(BuildError::Unreachable { node: name.clone() });
        }

        validate_channels(channels)?;

        Ok(ValidTopology {
            start: start_targets.into_iter().flatten().collect(),
            successors: outgoing
                .into_iter()
                .map(|edges| edges.into_iter().flatten().collect())
                .collect(),
            joins: self.joins,
            names: self.names,
        })
    }
}

/// Which nodes can be reached from `START`, by node index.
///
/// A join's target is reached once every one of its sources is: each join
/// counts down as its sources are first reached, and at zero reaches its
/// target. An iterative walk, so a long chain cannot overflow the stack, and
/// linear in the nodes, edges and join sources.
fn reachable(
    start_targets: &[Option<usize>],
    outgoing: &[Vec<Option<usize>>],
    joins: &[Join],
    len: usize,
) -> Vec<bool> {
    let mut reached = vec![false; len];
    // Joins list each source once (`add_edge` rejects repeats), and a node
    // counts its joins down only on its first visit, so no count goes below
    // zero.
    let mut waiting: Vec<usize> = joins.iter().map(|join| join.sources.len()).collect();
    let mut joins_of: Vec<Vec<usize>> = vec![Vec::new(); len];
    for (index, join) in joins.iter().enumerate() {
        for &source in &join.sources {
            joins_of
                .get_mut(source)
                .into_iter()
                .for_each(|of| of.push(index));
        }
    }
    let mut pending: Vec<usize> = start_targets.iter().flatten().copied().collect();
    while let Some(node) = pending.pop() {
        let first_visit = reached
            .get_mut(node)
            .is_some_and(|seen| !std::mem::replace(seen, true));
        if first_visit {
            pending.extend(outgoing.get(node).into_iter().flatten().flatten().copied());
            for &index in joins_of.get(node).into_iter().flatten() {
                let complete = waiting.get_mut(index).is_some_and(|left| {
                    *left -= 1;
                    *left == 0
                });
                pending.extend(
                    joins
                        .get(index)
                        .filter(|_| complete)
                        .map(|join| join.target),
                );
            }
        }
    }
    reached
}

/// Checks the state's channels as its `channels()` returned them: first the
/// name error it reported, if any, then that every name is unique.
///
/// # Errors
///
/// Returns [`BuildError::InvalidChannelName`] for the error `channels()`
/// returned, or [`BuildError::DuplicateChannel`] for the first name repeated,
/// in declaration order.
fn validate_channels(channels: Result<Vec<ChannelSpec>, NameError>) -> Result<(), BuildError> {
    let channels = channels.map_err(BuildError::InvalidChannelName)?;
    let mut seen = HashSet::new();
    for channel in &channels {
        if !seen.insert(channel.name()) {
            return Err(BuildError::DuplicateChannel {
                name: channel.name().to_owned(),
            });
        }
    }
    Ok(())
}

/// A validated topology: the nodes and the edges between them, by node index,
/// in the order they were declared.
///
/// It says nothing about how a run is scheduled; the engine lowers it to its
/// own representation.
#[derive(Debug)]
pub(crate) struct ValidTopology {
    /// Node names, by node index.
    pub(crate) names: Vec<NodeName>,
    /// The nodes `START` leads to.
    pub(crate) start: Vec<usize>,
    /// The nodes each node leads to, by node index; edges to `END` are left
    /// out.
    pub(crate) successors: Vec<Vec<usize>>,
    /// The joins, in the order they were declared, each with its sources in
    /// name order.
    pub(crate) joins: Vec<Join>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_NAME_LEN;
    use crate::name::validate_name;
    use crate::testing::{INVALID_NAME, distinct_tags, invalid_name, valid_name};
    use proptest::prelude::*;
    use test_strategy::proptest;

    /// A topology with nodes named `names`.
    fn topology(names: &[&str]) -> (Topology, Vec<NodeId>) {
        let mut t = Topology::new();
        let ids = names.iter().map(|n| t.add_node(n).unwrap()).collect();
        (t, ids)
    }

    /// The name of a node, for comparing errors.
    fn name(n: &str) -> NodeName {
        NodeName::new(n).unwrap()
    }

    /// A minimal valid topology: `START → a → END`.
    fn valid() -> Topology {
        let (mut t, ids) = topology(&["a"]);
        t.add_edge(Source::Start.into(), Target::Node(ids[0]))
            .unwrap();
        t.add_edge(Source::Node(ids[0]).into(), Target::End)
            .unwrap();
        t
    }

    // --- new ----------------------------------------------------------------

    /// Each builder draws a fresh tag.
    #[test]
    fn every_builder_gets_its_own_tag() {
        assert_ne!(Topology::new().tag, Topology::new().tag);
    }

    // --- add_node ------------------------------------------------------------

    /// `add_node` accepts every valid name.
    #[proptest]
    fn add_node_accepts_valid_names(#[strategy(valid_name(MAX_NAME_LEN))] name: String) {
        let mut t = Topology::new();
        prop_assert_eq!(
            t.add_node(&name),
            Ok(NodeId {
                index: 0,
                builder: t.tag
            })
        );
        prop_assert_eq!(t.names, vec![NodeName::new(&name).unwrap()]);
        prop_assert_eq!(t.known, HashSet::from([NodeName::new(&name).unwrap()]));
        prop_assert!(t.edges.is_empty());
    }

    /// `add_node` rejects every invalid name with the error `validate_name`
    /// reports for it.
    #[proptest]
    fn add_node_rejects_invalid_names(#[strategy(invalid_name())] name: String) {
        let mut t = Topology::new();
        prop_assert_eq!(
            t.add_node(&name).err(),
            Some(BuildError::InvalidName(validate_name(&name).unwrap_err()))
        );
        prop_assert!(t.names.is_empty());
        prop_assert!(t.known.is_empty());
        prop_assert!(t.edges.is_empty());
    }

    /// Invalid and duplicate names are rejected as soon as they are added, and
    /// a rejected name is not registered.
    #[proptest]
    fn add_node_rejects_duplicate_names(#[strategy(valid_name(MAX_NAME_LEN))] name: String) {
        let mut t = Topology::new();
        t.add_node(&name).unwrap();
        prop_assert_eq!(
            t.add_node(&name),
            Err(BuildError::DuplicateName { name: name.clone() })
        );
        prop_assert_eq!(t.names, vec![NodeName::new(&name).unwrap()]);
        prop_assert_eq!(t.known, HashSet::from([NodeName::new(&name).unwrap()]));
        prop_assert!(t.edges.is_empty());
    }

    // --- add_edge ------------------------------------------------------------

    /// A handle from another builder is rejected at either end of an edge,
    /// opposite a node, `START` or `END`, and the rejected edge is not
    /// recorded.
    #[proptest]
    fn rejects_foreign_handles_at_either_end(#[strategy(distinct_tags())] tags: (u64, u64)) {
        let (tag, foreign_tag) = tags;
        let names = vec![NodeName::new("a").unwrap()];
        let mut t = Topology {
            tag,
            known: HashSet::from_iter(names.clone()),
            names,
            edges: Vec::new(),
            declared: HashSet::new(),
            joins: Vec::new(),
            declared_joins: HashSet::new(),
        };
        // Node 0 exists in both builders, so only the tag tells them apart.
        let own = NodeId {
            index: 0,
            builder: tag,
        };
        let foreign = NodeId {
            index: 0,
            builder: foreign_tag,
        };
        for (from, to) in [
            (Source::Node(foreign), Target::Node(own)),
            (Source::Node(own), Target::Node(foreign)),
            (Source::Start, Target::Node(foreign)),
            (Source::Node(foreign), Target::End),
        ] {
            prop_assert_eq!(t.add_edge(from.into(), to), Err(BuildError::ForeignNode));
        }
        prop_assert!(t.edges.is_empty() && t.declared.is_empty());
        prop_assert!(t.joins.is_empty() && t.declared_joins.is_empty());
    }

    /// Every combination of this builder's ends is accepted, `START` or a
    /// node as the source and a node or `END` as the target, and each edge is
    /// recorded in order, by node index, with `None` for `START` and `END`.
    #[test]
    fn add_edge_accepts_valid_sources_and_targets() {
        let (mut t, ids) = topology(&["a", "b"]);
        let (a, b) = (Source::Node(ids[0]), Source::Node(ids[1]));
        let (to_a, to_b) = (Target::Node(ids[0]), Target::Node(ids[1]));
        let edges = [
            (Source::Start, to_a),
            (a, to_b),
            (b, Target::End),
            (Source::Start, Target::End),
            (a, to_a),
        ];
        for (from, to) in edges {
            assert_eq!(t.add_edge(from.into(), to), Ok(()));
        }
        let edge = |from, to| Edge { from, to };
        assert_eq!(
            t.edges,
            vec![
                edge(None, Some(0)),
                edge(Some(0), Some(1)),
                edge(Some(1), None),
                edge(None, None),
                edge(Some(0), Some(0)),
            ]
        );
    }

    /// An edge added a second time, wherever it falls among the others, is
    /// rejected with its canonical form and not recorded again; every other
    /// edge is accepted.
    #[proptest]
    fn rejects_a_repeated_edge(
        #[strategy(1..=4usize)] nodes: usize,
        /// Distinct edges, in the encoding of `validate`'s `build`: 0 is
        /// `START` as a source and `END` as a target, and `k` is the node
        /// `k - 1`.
        #[strategy(
            prop::collection::btree_set((0..=#nodes, 0..=#nodes), 1..8)
                .prop_map(Vec::from_iter)
                .prop_shuffle()
        )]
        edges: Vec<(usize, usize)>,
        /// The edge added again.
        #[strategy(0..#edges.len())]
        repeated: usize,
        /// Where the repeat is added, after the edge's first addition.
        #[strategy(#repeated + 1..=#edges.len())]
        at: usize,
    ) {
        let names: Vec<String> = (0..nodes).map(|i| format!("n{i}")).collect();
        let (mut t, ids) = topology(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let sources: Vec<Source> = std::iter::once(Source::Start)
            .chain(ids.iter().copied().map(Source::Node))
            .collect();
        let targets: Vec<Target> = std::iter::once(Target::End)
            .chain(ids.iter().copied().map(Target::Node))
            .collect();
        let recorded: Vec<Edge> = edges
            .iter()
            .map(|&(from, to)| Edge {
                from: from.checked_sub(1),
                to: to.checked_sub(1),
            })
            .collect();
        // The canonical form is pinned by `describes_edges_by_their_ends`.
        let duplicate = BuildError::DuplicateEdge {
            edge: recorded[repeated].describe(&t.names),
        };
        let mut calls = edges.clone();
        calls.insert(at, edges[repeated]);
        for (i, &(from, to)) in calls.iter().enumerate() {
            let expected = if i == at {
                Err(duplicate.clone())
            } else {
                Ok(())
            };
            prop_assert_eq!(t.add_edge(sources[from].into(), targets[to]), expected);
        }
        prop_assert_eq!(
            &t.declared,
            &recorded.iter().copied().collect::<HashSet<_>>()
        );
        prop_assert_eq!(t.edges, recorded);
    }

    /// An edge reads `from → to`, with nodes by name and the boundaries as
    /// `START` and `END`, whatever the names and whichever nodes it joins.
    /// Every case checks all four shapes of edge.
    #[proptest]
    fn describes_edges_by_their_ends(
        #[strategy(
            prop::collection::hash_set(valid_name(MAX_NAME_LEN), 1..4).prop_map(Vec::from_iter)
        )]
        names: Vec<String>,
        /// The source node, when the edge has one.
        #[strategy(0..#names.len())]
        from: usize,
        /// The target node, when the edge has one.
        #[strategy(0..#names.len())]
        to: usize,
    ) {
        let (t, _) = topology(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let describe = |from, to| Edge { from, to }.describe(&t.names);
        let (a, b) = (&names[from], &names[to]);
        prop_assert_eq!(describe(None, Some(to)), format!("START → {b}"));
        prop_assert_eq!(describe(Some(from), None), format!("{a} → END"));
        prop_assert_eq!(describe(Some(from), Some(to)), format!("{a} → {b}"));
        prop_assert_eq!(describe(None, None), "START → END");
    }

    // --- add_join ------------------------------------------------------------

    /// Four nodes, `n0` to `n3`, and their handles.
    fn four() -> (Topology, Vec<NodeId>) {
        topology(&["n0", "n1", "n2", "n3"])
    }

    /// A join of `sources` into `target`.
    fn join(t: &mut Topology, sources: &[NodeId], target: Target) -> Result<(), BuildError> {
        t.add_edge(Sources::Join(sources.to_vec()), target)
    }

    /// A join reads `join:a+b:c`, its sources in name order and joined by
    /// `+`, whatever order they were given in and whatever the names.
    #[proptest]
    fn describes_joins_by_their_sources_in_name_order(
        #[strategy(
            prop::collection::btree_set(valid_name(MAX_NAME_LEN), 3..6)
                .prop_map(Vec::from_iter)
                .prop_shuffle()
        )]
        names: Vec<String>,
        /// The join's sources, as given: all but the last node, shuffled.
        #[strategy(Just((0..#names.len() - 1).collect::<Vec<usize>>()).prop_shuffle())]
        sources: Vec<usize>,
    ) {
        let (mut t, ids) = topology(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let target = names.len() - 1;
        let given: Vec<NodeId> = sources.iter().map(|&i| ids[i]).collect();
        join(&mut t, &given, Target::Node(ids[target])).unwrap();
        let mut by_name: Vec<&str> = sources.iter().map(|&i| names[i].as_str()).collect();
        by_name.sort_unstable();
        prop_assert_eq!(
            t.joins[0].describe(&t.names),
            format!("join:{}:{}", by_name.join("+"), names[target])
        );
    }

    /// A handle from another builder is rejected anywhere among a join's
    /// sources, or as its target, and nothing is recorded.
    #[proptest]
    fn rejects_a_foreign_handle_in_a_join(
        #[strategy(distinct_tags())] tags: (u64, u64),
        /// Where the foreign source goes among two of this builder's.
        #[strategy(0..=2usize)]
        at: usize,
    ) {
        let (mut t, ids) = four();
        t.tag = tags.0;
        let own: Vec<NodeId> = ids
            .iter()
            .map(|id| NodeId {
                index: id.index,
                builder: tags.0,
            })
            .collect();
        let foreign = NodeId {
            index: 0,
            builder: tags.1,
        };
        let mut sources = vec![own[0], own[1]];
        sources.insert(at, foreign);
        prop_assert_eq!(
            join(&mut t, &sources, Target::Node(own[3])),
            Err(BuildError::ForeignNode)
        );
        prop_assert_eq!(
            join(&mut t, &own[..2], Target::Node(foreign)),
            Err(BuildError::ForeignNode)
        );
        prop_assert!(t.joins.is_empty());
    }

    /// A join without sources is rejected, whatever its target.
    #[test]
    fn rejects_an_empty_join() {
        let (mut t, ids) = four();
        assert_eq!(
            join(&mut t, &[], Target::Node(ids[0])),
            Err(BuildError::EmptyJoin)
        );
        assert_eq!(join(&mut t, &[], Target::End), Err(BuildError::EmptyJoin));
    }

    /// A join of one source is rejected, naming it, whether it leads to a
    /// node or to `END`.
    #[proptest]
    fn rejects_a_single_source_join(
        /// The one source.
        #[strategy(0..4usize)]
        source: usize,
        /// The node target, tried alongside `END`.
        #[strategy(0..4usize)]
        target: usize,
    ) {
        let (mut t, ids) = four();
        let expected = Err(BuildError::SingleSourceJoin {
            node: name(&format!("n{source}")),
        });
        prop_assert_eq!(
            join(&mut t, &[ids[source]], Target::Node(ids[target])),
            expected.clone()
        );
        prop_assert_eq!(join(&mut t, &[ids[source]], Target::End), expected);
    }

    /// A join that lists a node twice is rejected, naming the first node to
    /// appear a second time: of the two repeats appended to distinct
    /// sources, the first.
    #[proptest]
    fn rejects_a_repeated_join_source(
        /// The distinct sources, in the order given.
        #[strategy(prop::sample::subsequence((0..4usize).collect::<Vec<_>>(), 2..=4).prop_shuffle())]
        sources: Vec<usize>,
        /// The source repeated first.
        #[strategy(0..#sources.len())]
        first: usize,
        /// How far after `first`, wrapping, the source repeated second is.
        #[strategy(1..#sources.len())]
        offset: usize,
    ) {
        let (mut t, ids) = four();
        let second = (first + offset) % sources.len();
        let given: Vec<NodeId> = sources
            .iter()
            .chain([&sources[first], &sources[second]])
            .map(|&i| ids[i])
            .collect();
        prop_assert_eq!(
            join(&mut t, &given, Target::End),
            Err(BuildError::DuplicateJoinSource {
                node: name(&format!("n{}", sources[first]))
            })
        );
    }

    /// A join into `END` is rejected.
    #[proptest]
    fn rejects_a_join_to_end(
        #[strategy(prop::sample::subsequence((0..4usize).collect::<Vec<_>>(), 2..=4).prop_shuffle())]
        sources: Vec<usize>,
    ) {
        let (mut t, ids) = four();
        let given: Vec<NodeId> = sources.iter().map(|&i| ids[i]).collect();
        prop_assert_eq!(
            join(&mut t, &given, Target::End),
            Err(BuildError::JoinToEnd)
        );
    }

    /// The same sources joining into the same target a second time, in any
    /// order, are a duplicate edge, named by the join's form, and not
    /// recorded again.
    #[proptest]
    fn rejects_a_repeated_join_whatever_its_order(
        /// The join's sources, as first given.
        #[strategy(prop::sample::subsequence((0..3usize).collect::<Vec<_>>(), 2..=3).prop_shuffle())]
        sources: Vec<usize>,
        /// The same sources, as given again.
        #[strategy(Just(#sources.clone()).prop_shuffle())]
        again: Vec<usize>,
    ) {
        let (mut t, ids) = four();
        let given = |order: &[usize]| order.iter().map(|&i| ids[i]).collect::<Vec<_>>();
        join(&mut t, &given(&sources), Target::Node(ids[3])).unwrap();
        // The form is pinned by `describes_joins_by_their_sources_in_name_order`.
        let edge = t.joins[0].describe(&t.names);
        prop_assert_eq!(
            join(&mut t, &given(&again), Target::Node(ids[3])),
            Err(BuildError::DuplicateEdge { edge })
        );
        prop_assert_eq!(t.joins.len(), 1);
    }

    /// A join and a plain edge into the same node are different edges.
    #[test]
    fn a_join_beside_a_plain_edge_is_not_a_duplicate() {
        let (mut t, ids) = four();
        t.add_edge(Source::Node(ids[0]).into(), Target::Node(ids[2]))
            .unwrap();
        assert_eq!(join(&mut t, &ids[..2], Target::Node(ids[2])), Ok(()));
        assert_eq!((t.edges.len(), t.joins.len()), (1, 1));
    }

    /// When a join fails several checks, the first in the documented order is
    /// reported. Each case below also fails every later check it can.
    #[test]
    fn reports_the_first_join_check_that_fails() {
        let (mut t, ids) = four();
        let foreign = NodeId {
            index: 0,
            builder: t.tag.wrapping_add(1),
        };
        let (a, b, c) = (ids[0], ids[1], ids[2]);
        // A foreign target, and no sources at all.
        assert_eq!(
            join(&mut t, &[], Target::Node(foreign)),
            Err(BuildError::ForeignNode)
        );
        // No sources, into END.
        assert_eq!(join(&mut t, &[], Target::End), Err(BuildError::EmptyJoin));
        // One source, into END.
        assert_eq!(
            join(&mut t, &[a], Target::End),
            Err(BuildError::SingleSourceJoin { node: name("n0") })
        );
        // A repeated source, into END.
        assert_eq!(
            join(&mut t, &[a, a], Target::End),
            Err(BuildError::DuplicateJoinSource { node: name("n0") })
        );
        // Distinct sources, into END.
        assert_eq!(
            join(&mut t, &[a, b], Target::End),
            Err(BuildError::JoinToEnd)
        );
        // A valid join, then the same again.
        join(&mut t, &[a, b], Target::Node(c)).unwrap();
        assert_eq!(
            join(&mut t, &[b, a], Target::Node(c)),
            Err(BuildError::DuplicateEdge {
                edge: "join:n0+n1:n2".to_owned()
            })
        );
    }

    // --- validate ------------------------------------------------------------

    /// Channel names for the properties, valid and invalid; `validate_name`
    /// decides which are which.
    const CHANNEL_NAMES: &[&str] = &["count", "total", INVALID_NAME, "__x", "a:b", "\u{7}"];

    /// The names `n{order[i]}` for node `i`, so name order can differ from
    /// declaration order.
    fn names(order: &[usize]) -> Vec<String> {
        order.iter().map(|i| format!("n{i}")).collect()
    }

    /// A topology with nodes named `names` and the given edges, built through
    /// `add_node` and `add_edge`. Each end is an index into the boundary and
    /// the nodes: 0 is `START` as a source and `END` as a target, and `k` is
    /// node `k - 1`.
    fn build(names: &[String], edges: &[(usize, usize)]) -> Topology {
        let (mut t, ids) = topology(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let sources: Vec<Source> = std::iter::once(Source::Start)
            .chain(ids.iter().copied().map(Source::Node))
            .collect();
        let targets: Vec<Target> = std::iter::once(Target::End)
            .chain(ids.iter().copied().map(Target::Node))
            .collect();
        for &(from, to) in edges {
            t.add_edge(sources[from].into(), targets[to]).unwrap();
        }
        t
    }

    /// The edges of a chain from `START` through the nodes of `path`, in
    /// order, to `END`, in [`build`]'s encoding.
    fn chain(path: &[usize]) -> Vec<(usize, usize)> {
        let stops: Vec<usize> = std::iter::once(0)
            .chain(path.iter().map(|&node| node + 1))
            .chain(std::iter::once(0))
            .collect();
        stops.windows(2).map(|pair| (pair[0], pair[1])).collect()
    }

    /// The channels named `names`, as a state's `channels()` returns them:
    /// the error of the first name the naming rules reject, or the specs.
    fn specs(names: &[&'static str]) -> Result<Vec<ChannelSpec>, NameError> {
        names.iter().copied().map(ChannelSpec::new).collect()
    }

    /// [`build`]'s topology, plus the join of `sources` into `target`, by
    /// node index.
    fn build_with_join(
        names: &[String],
        edges: &[(usize, usize)],
        sources: &[usize],
        target: usize,
    ) -> Topology {
        let mut t = build(names, edges);
        let id = |index| NodeId {
            index,
            builder: t.tag,
        };
        let join = Sources::Join(sources.iter().map(|&i| id(i)).collect());
        t.add_edge(join, Target::Node(id(target))).unwrap();
        t
    }

    /// Nodes `n0` to `n{sources-1}`, which joins draw on, then a target named
    /// `a`, which sorts before them all.
    fn join_names(sources: usize) -> Vec<String> {
        (0..sources)
            .map(|i| format!("n{i}"))
            .chain(std::iter::once(String::from("a")))
            .collect()
    }

    /// A graph without nodes is rejected as empty.
    #[test]
    fn rejects_an_empty_graph() {
        let mut t = Topology::new();
        t.add_edge(Source::Start.into(), Target::End).unwrap();
        assert_eq!(
            t.validate(specs(&["value"])).unwrap_err(),
            BuildError::Empty
        );
    }

    /// A graph with nodes but no edge from `START` reports `NoStartEdge`,
    /// whatever its other edges and channels.
    #[proptest]
    fn reports_a_missing_start_edge(
        #[strategy(1..=6usize)] nodes: usize,
        #[strategy(
            prop::collection::btree_set((1..=#nodes, 0..=#nodes), 0..10).prop_map(Vec::from_iter)
        )]
        edges: Vec<(usize, usize)>,
        #[strategy(prop::collection::vec(prop::sample::select(CHANNEL_NAMES), 0..6))] channels: Vec<
            &'static str,
        >,
    ) {
        let order: Vec<usize> = (0..nodes).collect();
        prop_assert_eq!(
            build(&names(&order), &edges)
                .validate(specs(&channels))
                .err(),
            Some(BuildError::NoStartEdge)
        );
    }

    /// A chain from `START` through some of the nodes, with every other node
    /// given exactly one edge of its own, reports the first node left off the
    /// chain by name, whatever its channels. An empty chain is `START → END`.
    #[proptest]
    fn reports_the_first_unreachable_node_by_name(
        // Read only by the other inputs' strategies, which it sizes.
        #[strategy(1..=6usize)] _nodes: usize,
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())] order: Vec<usize>,
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())] path: Vec<usize>,
        #[strategy(0..#_nodes)] reached: usize,
        #[strategy(prop::collection::vec(0..=#_nodes, #_nodes))] targets: Vec<usize>,
        #[strategy(prop::collection::vec(prop::sample::select(CHANNEL_NAMES), 0..6))] channels: Vec<
            &'static str,
        >,
    ) {
        // Edges from unreached nodes, even into the chain, reach nothing new.
        let edges: Vec<(usize, usize)> = chain(&path[..reached])
            .into_iter()
            .chain(
                path[reached..]
                    .iter()
                    .map(|&node| (node + 1, targets[node])),
            )
            .collect();
        let names = names(&order);
        let first = path[reached..]
            .iter()
            .map(|&node| &names[node])
            .min()
            .unwrap();
        prop_assert_eq!(
            build(&names, &edges).validate(specs(&channels)).err(),
            Some(BuildError::Unreachable { node: name(first) })
        );
    }

    /// When a graph fails several checks, the first in the documented order is
    /// reported: no nodes, no edge from `START`, an unreachable node, then the
    /// channels. Each case below also fails every check after the one it
    /// expects.
    #[test]
    fn reports_the_first_check_that_fails() {
        let bad = || specs(&[INVALID_NAME]);
        let two = names(&[0, 1]);
        // No nodes, so no edge from START either, and bad channels: empty.
        assert_eq!(
            build(&[], &[]).validate(bad()).unwrap_err(),
            BuildError::Empty
        );
        // n0 → END and n1 → n0, so both are unreachable, but nothing leaves
        // START.
        assert_eq!(
            build(&two, &[(1, 0), (2, 1)]).validate(bad()).unwrap_err(),
            BuildError::NoStartEdge
        );
        // START → n0 → END, with n1 never reached.
        assert_eq!(
            build(&two, &[(0, 1), (1, 0)]).validate(bad()).unwrap_err(),
            BuildError::Unreachable { node: name("n1") }
        );
        // A valid structure: the channels decide.
        assert_eq!(
            build(&two, &[(0, 1), (1, 2), (2, 0)])
                .validate(bad())
                .unwrap_err(),
            BuildError::InvalidChannelName(validate_name(INVALID_NAME).unwrap_err())
        );
    }

    /// Any number of edges may leave `START` and each node: with every node
    /// reached, the graph is valid, and the validated topology keeps each
    /// node's targets, and `START`'s, in declaration order, without `END`.
    #[proptest]
    fn accepts_fan_out_from_start_and_nodes(
        #[strategy(1..=6usize)] nodes: usize,
        /// A chain through every node, so each is reached.
        #[strategy(Just((0..#nodes).collect::<Vec<usize>>()).prop_shuffle())]
        path: Vec<usize>,
        /// More edges, in `build`'s encoding and in random order; those the
        /// chain already has are dropped. Never empty, so the filter below
        /// always runs.
        #[strategy(
            prop::collection::btree_set((0..=#nodes, 0..=#nodes), 1..12)
                .prop_map(Vec::from_iter)
                .prop_shuffle()
        )]
        extra: Vec<(usize, usize)>,
    ) {
        let chained = chain(&path);
        let edges: Vec<(usize, usize)> = chained
            .iter()
            .copied()
            .chain(extra.iter().copied().filter(|edge| !chained.contains(edge)))
            .collect();
        let order: Vec<usize> = (0..nodes).collect();
        let valid = build(&names(&order), &edges)
            .validate(specs(&["count"]))
            .unwrap();
        let targets = |source: usize| -> Vec<usize> {
            edges
                .iter()
                .filter(|&&(from, to)| from == source && to != 0)
                .map(|&(_, to)| to - 1)
                .collect()
        };
        prop_assert_eq!(valid.start, targets(0));
        prop_assert_eq!(
            valid.successors,
            (1..=nodes).map(targets).collect::<Vec<_>>()
        );
    }

    /// A reached node without an outgoing edge is a valid dead end: the run
    /// simply stops after it.
    #[test]
    fn accepts_a_reached_node_without_an_outgoing_edge() {
        let (mut t, ids) = topology(&["a"]);
        t.add_edge(Source::Start.into(), Target::Node(ids[0]))
            .unwrap();
        assert!(t.validate(specs(&["value"])).is_ok());
    }

    /// Validation keeps the edges by node index, in declaration order, with
    /// cycles kept.
    #[test]
    fn keeps_edges_by_node_index() {
        // Declared out of name order: c, a, b. START → c → a → b → a (a cycle).
        let (mut t, ids) = topology(&["c", "a", "b"]);
        let (c, a, b) = (ids[0], ids[1], ids[2]);
        t.add_edge(Source::Start.into(), Target::Node(c)).unwrap();
        t.add_edge(Source::Node(c).into(), Target::Node(a)).unwrap();
        t.add_edge(Source::Node(a).into(), Target::Node(b)).unwrap();
        t.add_edge(Source::Node(b).into(), Target::Node(a)).unwrap();
        let topology = t.validate(specs(&["value"])).unwrap();
        assert_eq!(topology.start, vec![0]);
        assert_eq!(topology.successors, vec![vec![1], vec![2], vec![1]]);
        assert_eq!(topology.names, vec![name("c"), name("a"), name("b")]);
    }

    /// An edge to `END` leaves no successor.
    #[test]
    fn edges_to_end_leave_no_successor() {
        let topology = valid().validate(specs(&["value"])).unwrap();
        assert_eq!(topology.successors, vec![Vec::<usize>::new()]);
    }

    /// On a structurally valid graph, a chain through every node, `validate`
    /// reports exactly what `validate_channels` reports for the channels.
    /// That verdict comes from `validate_channels` on purpose: its rules are
    /// pinned in its own group.
    #[proptest]
    fn a_valid_structure_reports_its_channels_verdict(
        // Read only by the other inputs' strategies, which it sizes.
        #[strategy(1..=6usize)] _nodes: usize,
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())] order: Vec<usize>,
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())] path: Vec<usize>,
        #[strategy(prop::collection::vec(prop::sample::select(CHANNEL_NAMES), 0..6))] channels: Vec<
            &'static str,
        >,
    ) {
        let specs = specs(&channels);
        prop_assert_eq!(
            build(&names(&order), &chain(&path))
                .validate(specs.clone())
                .err(),
            validate_channels(specs).err()
        );
    }

    /// A node reached only through a join is reached once every one of the
    /// join's sources is.
    #[proptest]
    fn a_join_target_is_reached_once_every_source_is(
        #[strategy(2..=5usize)] _sources: usize,
        /// A chain through every source.
        #[strategy(Just((0..#_sources).collect::<Vec<usize>>()).prop_shuffle())]
        path: Vec<usize>,
        /// The join's sources.
        #[strategy(prop::sample::subsequence((0..#_sources).collect::<Vec<usize>>(), 2..=#_sources))]
        joined: Vec<usize>,
    ) {
        let target = path.len();
        let t = build_with_join(&join_names(target), &chain(&path), &joined, target);
        prop_assert!(t.validate(specs(&["count"])).is_ok());
    }

    /// A node reached only through a join is unreachable while one of the
    /// join's sources is: the target, named to sort first, is reported.
    #[proptest]
    fn a_join_target_is_unreachable_unless_every_source_is(
        #[strategy(2..=5usize)] _sources: usize,
        /// A chain through every source, the last of which is left off it.
        #[strategy(Just((0..#_sources).collect::<Vec<usize>>()).prop_shuffle())]
        path: Vec<usize>,
    ) {
        let target = path.len();
        let (missing, reached) = path.split_last().unwrap();
        let mut edges = chain(reached);
        edges.push((missing + 1, 0));
        let t = build_with_join(&join_names(target), &edges, &path, target);
        prop_assert_eq!(
            t.validate(specs(&["count"])).err(),
            Some(BuildError::Unreachable { node: name("a") })
        );
    }

    /// A validated topology keeps each join with its sources in name order,
    /// whatever order they were given in.
    #[proptest]
    fn keeps_joins_with_sources_in_name_order(
        #[strategy(2..=5usize)] _sources: usize,
        /// The node names `n{order[i]}`, so name order is not index order.
        #[strategy(Just((0..=#_sources).collect::<Vec<usize>>()).prop_shuffle())]
        order: Vec<usize>,
        /// The join's sources, as given: every node but the last, shuffled.
        #[strategy(Just((0..#_sources).collect::<Vec<usize>>()).prop_shuffle())]
        joined: Vec<usize>,
    ) {
        let target = joined.len();
        let names = names(&order);
        let t = build_with_join(&names, &chain(&joined), &joined, target);
        let valid = t.validate(specs(&["count"])).unwrap();
        let mut by_name = joined.clone();
        by_name.sort_unstable_by_key(|&i| names[i].clone());
        prop_assert_eq!(&valid.joins[0].sources, &by_name);
    }

    // --- reachable -----------------------------------------------------------

    /// Reachability marks exactly the nodes `START` leads to, through cycles,
    /// self-loops and fan-out, and terminates; every other node stays
    /// unreached, even when its own edges lead into the reached part.
    #[proptest]
    fn reachability_follows_cycles_without_revisiting(
        // Read only by the other inputs' strategies, which it sizes.
        #[strategy(2..=8usize)] _nodes: usize,
        #[strategy(Just((0..#_nodes).collect::<Vec<usize>>()).prop_shuffle())] path: Vec<usize>,
        #[strategy(1..#_nodes)] reached: usize,
        #[strategy(prop::collection::vec((0..#_nodes, 0..=#_nodes), 1..12))] extra: Vec<(
            usize,
            usize,
        )>,
        #[strategy(prop::collection::vec((0..#_nodes, prop::option::of(0..#_nodes)), 1..8))]
        unreached_edges: Vec<(usize, Option<usize>)>,
    ) {
        let n = path.len();
        let (inside, outside) = path.split_at(reached);
        let mut outgoing: Vec<Vec<Option<usize>>> = vec![Vec::new(); n];
        // A chain START → inside[0] → … → END makes every inside node reached.
        let nexts = inside[1..]
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None));
        for (&node, next) in inside.iter().zip(nexts) {
            outgoing[node].push(next);
        }
        // Extra edges stay inside or lead to END: cycles, self-loops, fan-out.
        let ends: Vec<Option<usize>> = std::iter::once(None)
            .chain(inside.iter().copied().map(Some))
            .collect();
        for &(from, to) in &extra {
            outgoing[inside[from % reached]].push(ends[to % ends.len()]);
        }
        // Outside nodes may lead anywhere, but nothing inside leads to them.
        for &(from, to) in &unreached_edges {
            outgoing[outside[from % outside.len()]].push(to);
        }
        let mut expected = vec![false; n];
        for &node in inside {
            expected[node] = true;
        }
        prop_assert_eq!(
            reachable(&[Some(inside[0]), None], &outgoing, &[], n),
            expected
        );
    }

    /// Reachability walks a 100 thousand-node chain without overflowing the stack.
    #[test]
    fn reachability_handles_a_long_chain_without_recursion() {
        let len = 100_000;
        let outgoing: Vec<Vec<Option<usize>>> = (0..len)
            .map(|i| vec![if i + 1 < len { Some(i + 1) } else { None }])
            .collect();
        assert!(
            reachable(&[Some(0)], &outgoing, &[], len)
                .into_iter()
                .all(|r| r)
        );
    }

    /// Each join counts down once per source, on the source's first visit,
    /// however often a cycle revisits it: the join of two sources in a cycle
    /// reaches its target, and the join of a reached and an unreached source
    /// does not.
    #[test]
    fn reachability_counts_each_join_down_once() {
        // 0 ⇄ 1, and 3 is never reached.
        let outgoing = vec![
            vec![Some(1)],
            vec![Some(0)],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let joins = [
            Join {
                sources: vec![0, 1],
                target: 2,
            },
            Join {
                sources: vec![0, 3],
                target: 4,
            },
        ];
        assert_eq!(
            reachable(&[Some(0)], &outgoing, &joins, 5),
            vec![true, true, true, false, false]
        );
    }

    // --- validate_channels ---------------------------------------------------

    /// The error of the first invalid channel name is passed on as
    /// `InvalidChannelName`, whatever valid names come before it and
    /// whatever names follow it.
    #[proptest]
    fn rejects_invalid_channel_names(
        #[strategy(prop::collection::vec(valid_name(MAX_NAME_LEN), 0..4))] before: Vec<String>,
        #[strategy(invalid_name())] invalid: String,
        #[strategy(prop::collection::vec(prop_oneof![valid_name(MAX_NAME_LEN), invalid_name()], 0..4))]
        after: Vec<String>,
    ) {
        let expected = BuildError::InvalidChannelName(validate_name(&invalid).unwrap_err());
        let channels = before
            .into_iter()
            .chain(std::iter::once(invalid))
            .chain(after)
            .map(ChannelSpec::new)
            .collect();
        prop_assert_eq!(valid().validate(channels).err(), Some(expected));
    }

    /// A valid name declared a second time, anywhere after its first, is
    /// reported by name.
    #[proptest]
    fn rejects_duplicate_channels(
        #[strategy(prop::collection::btree_set(valid_name(MAX_NAME_LEN), 1..5).prop_map(Vec::from_iter).prop_shuffle())]
        names: Vec<String>,
        #[strategy(0..#names.len())] repeated: usize,
        #[strategy(#repeated + 1..=#names.len())] at: usize,
    ) {
        let mut channels = names.clone();
        channels.insert(at, names[repeated].clone());
        let channels = channels.into_iter().map(ChannelSpec::new).collect();
        prop_assert_eq!(
            valid().validate(channels).err(),
            Some(BuildError::DuplicateChannel {
                name: names[repeated].clone()
            })
        );
    }

    /// With several names repeated, the one whose repeat comes first in
    /// declaration order is reported, even when a name declared before it is
    /// repeated later.
    #[proptest]
    fn reports_the_first_duplicate_in_declaration_order(
        #[strategy(prop::collection::btree_set(valid_name(MAX_NAME_LEN), 2..6).prop_map(Vec::from_iter).prop_shuffle())]
        names: Vec<String>,
        #[strategy(0..#names.len())] repeated: usize,
        #[strategy(#repeated + 1..=#names.len())] at: usize,
        #[strategy(prop::collection::vec(0..#names.len(), 1..4))] later: Vec<usize>,
    ) {
        // Everything before `at` is distinct; every repeat after it comes
        // later than the first.
        let mut channels = names.clone();
        channels.insert(at, names[repeated].clone());
        channels.extend(later.iter().map(|&i| names[i].clone()));
        let channels = channels.into_iter().map(ChannelSpec::new).collect();
        prop_assert_eq!(
            validate_channels(channels),
            Err(BuildError::DuplicateChannel {
                name: names[repeated].clone()
            })
        );
    }

    /// An invalid name is reported even when a duplicate comes before it:
    /// `channels()` stops at the first invalid name, so duplicates are only
    /// checked once every name is valid.
    #[test]
    fn an_invalid_name_wins_over_an_earlier_duplicate() {
        assert_eq!(
            validate_channels(specs(&["count", "count", INVALID_NAME])),
            Err(BuildError::InvalidChannelName(
                validate_name(INVALID_NAME).unwrap_err()
            ))
        );
    }

    /// A state without channels is valid.
    #[test]
    fn accepts_a_state_without_channels() {
        assert!(valid().validate(Ok(Vec::new())).is_ok());
    }
}
