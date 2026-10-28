//! [`BuildError`], raised when a graph cannot be built.

use std::fmt;

use crate::name::{NameError, NodeName};

/// A graph could not be built.
///
/// Invalid names and handles from another builder are rejected when they are
/// passed in. The structural checks run in
/// [`compile`](crate::GraphBuilder::compile), in the order the variants below
/// are declared, and the first failure is reported.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildError {
    /// A node name breaks the [naming rules](NodeName#naming-rules).
    InvalidName(NameError),
    /// Two nodes have the same name.
    DuplicateName {
        /// The repeated name.
        name: String,
    },
    /// A node handle from a different builder was used.
    ForeignNode,
    /// The graph has no nodes.
    Empty,
    /// No edge leaves `START`.
    NoStartEdge,
    /// More than one edge leaves `START`.
    MultipleStartEdges,
    /// More than one edge leaves a node.
    MultipleOutgoing {
        /// The node, the first by name with this problem.
        node: NodeName,
    },
    /// A node cannot be reached from `START`.
    Unreachable {
        /// The node, the first by name with this problem.
        node: NodeName,
    },
    /// A state channel's name breaks the [naming rules](NodeName#naming-rules),
    /// as reported by [`State::channels`](crate::State::channels).
    InvalidChannelName(NameError),
    /// Two state channels have the same name.
    DuplicateChannel {
        /// The repeated name.
        name: String,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName(e) => write!(f, "invalid node name: {e}"),
            Self::DuplicateName { name } => write!(f, "a node named `{name}` already exists"),
            Self::ForeignNode => {
                f.write_str("the node handle belongs to a different graph builder")
            }
            Self::Empty => f.write_str("the graph has no nodes"),
            Self::NoStartEdge => f.write_str("no edge leaves START"),
            Self::MultipleStartEdges => {
                f.write_str("more than one edge leaves START, which is not supported yet")
            }
            Self::MultipleOutgoing { node } => write!(
                f,
                "more than one edge leaves node `{node}`, which is not supported yet"
            ),
            Self::Unreachable { node } => write!(f, "node `{node}` cannot be reached from START"),
            Self::InvalidChannelName(e) => write!(f, "invalid channel name: {e}"),
            Self::DuplicateChannel { name } => {
                write!(f, "the state declares channel `{name}` more than once")
            }
        }
    }
}

impl std::error::Error for BuildError {}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    /// Each message names the problem and, where there is one, the offending
    /// node or channel.
    #[test]
    fn error_messages_name_the_problem() {
        let cases = [
            (
                BuildError::InvalidName(NameError::Empty),
                "invalid node name: names must not be empty",
            ),
            (
                BuildError::DuplicateName {
                    name: "plan".to_owned(),
                },
                "a node named `plan` already exists",
            ),
            (
                BuildError::ForeignNode,
                "the node handle belongs to a different graph builder",
            ),
            (BuildError::Empty, "the graph has no nodes"),
            (BuildError::NoStartEdge, "no edge leaves START"),
            (
                BuildError::MultipleStartEdges,
                "more than one edge leaves START, which is not supported yet",
            ),
            (
                BuildError::MultipleOutgoing {
                    node: NodeName::new("plan").unwrap(),
                },
                "more than one edge leaves node `plan`, which is not supported yet",
            ),
            (
                BuildError::Unreachable {
                    node: NodeName::new("plan").unwrap(),
                },
                "node `plan` cannot be reached from START",
            ),
            (
                BuildError::InvalidChannelName(NameError::Empty),
                "invalid channel name: names must not be empty",
            ),
            (
                BuildError::DuplicateChannel {
                    name: "count".to_owned(),
                },
                "the state declares channel `count` more than once",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
            assert!(error.source().is_none(), "{error:?}");
        }
    }
}
