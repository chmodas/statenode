//! Task paths: the identity and merge order of every task in a run.

use crate::name::NodeName;

/// The position of a task in a run.
///
/// A task is one execution of one node within one superstep, or the input
/// applied before the first superstep. Task paths are unique within a run and
/// totally ordered: by step, then by kind (the input first), then by node
/// name. Updates are merged in this order, never in completion order, so the
/// result of a superstep does not depend on which task finishes first.
///
/// Ordering by name rather than by declaration order means reordering
/// `add_node` calls changes neither the merge order nor, once runs are
/// checkpointed, the identity of a task.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub struct TaskPath {
    /// The superstep the task belongs to. The input uses step 0, and is
    /// ordered before the step's node tasks by its kind.
    step: u64,
    /// What kind of task this is.
    kind: TaskKind,
}

/// What kind of task a [`TaskPath`] identifies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TaskKind {
    /// The run's input, applied in its own barrier before the first
    /// superstep.
    Input,
    /// A node triggered by a newer version of one of its trigger channels.
    ///
    /// Non-exhaustive so the task can gain fields, such as its durable id,
    /// without a breaking change.
    #[non_exhaustive]
    Pull {
        /// The node the task runs.
        node: NodeName,
    },
}

impl TaskPath {
    /// The path of the input task.
    pub(crate) const fn input() -> Self {
        Self {
            step: 0,
            kind: TaskKind::Input,
        }
    }

    /// The path of `node`'s task in superstep `step`.
    pub(crate) const fn pull(step: u64, node: NodeName) -> Self {
        Self {
            step,
            kind: TaskKind::Pull { node },
        }
    }

    /// The superstep the task belongs to.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// What kind of task this is.
    #[must_use]
    pub const fn kind(&self) -> &TaskKind {
        &self.kind
    }
}
