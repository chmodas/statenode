//! The [`State`] trait: a graph's channels and how updates apply to them.

use std::borrow::Cow;
use std::fmt;

use crate::name::{NameError, validate_name};
use crate::node::BoxError;
use crate::task::TaskPath;

/// The state a graph runs on: its channels, and the rules for applying node
/// updates to them.
///
/// Nodes never change the state directly. Every node in a superstep receives
/// the same snapshot, shared as an [`Arc`](std::sync::Arc), and returns an
/// [`Update`](State::Update). Once they have all finished, the engine passes
/// their updates to [`apply_step`](State::apply_step) in one call, sorted by
/// [`TaskPath`], so the result does not depend on which node finished first.
///
/// An implementation lists the channels and applies each channel's writes
/// with a helper from [`reducer`](crate::reducer):
///
/// ```
/// use statenode::{ChannelSpec, NameError, ReduceError, State, TaskPath, reducer};
///
/// /// A state with one channel, `count`.
/// #[derive(Clone, Default)]
/// struct Counter {
///     count: i64,
/// }
///
/// /// An update to [`Counter`]: `None` leaves `count` unwritten.
/// struct CounterUpdate {
///     count: Option<i64>,
/// }
///
/// impl State for Counter {
///     type Update = CounterUpdate;
///
///     fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
///         Ok(vec![ChannelSpec::new("count")?])
///     }
///
///     fn apply_step(
///         &mut self,
///         updates: Vec<(TaskPath, CounterUpdate)>,
///     ) -> Result<(), ReduceError> {
///         let writes = updates
///             .iter()
///             .filter_map(|(path, u)| u.count.map(|c| (path, c)));
///         reducer::replace("count", &mut self.count, writes)
///     }
/// }
/// ```
pub trait State: Clone + Send + Sync + 'static {
    /// What a node returns: a partial description of a state change.
    type Update: Send + 'static;

    /// The state's channels, in declaration order.
    ///
    /// The channels come from the instance rather than from the type, so a
    /// state whose schema is only known at run time can implement this trait.
    /// Every instance used by one graph must return the same channels; the
    /// graph reads them once when it is compiled.
    ///
    /// # Errors
    ///
    /// Returns the [`NameError`] of a channel whose name breaks the naming
    /// rules, as [`ChannelSpec::new`] reports it.
    /// [`GraphBuilder::compile`](crate::GraphBuilder::compile) reports it as
    /// [`BuildError::InvalidChannelName`](crate::BuildError::InvalidChannelName).
    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError>;

    /// Applies one superstep's updates, sorted by [`TaskPath`].
    ///
    /// Updates from one task keep the order the task produced them in.
    ///
    /// # Errors
    ///
    /// Returns a [`ReduceError`] if an update breaks one of the state's
    /// invariants, such as a second write in one step to a channel applied
    /// with [`reducer::replace`](crate::reducer::replace), or an operation
    /// that [`reducer::fold`](crate::reducer::fold)'s operator rejects. The
    /// run then fails with [`RunError::Reduce`](crate::RunError::Reduce), and
    /// the partially updated state is discarded.
    fn apply_step(&mut self, updates: Vec<(TaskPath, Self::Update)>) -> Result<(), ReduceError>;
}

/// The description of one state channel.
///
/// Its name follows the [naming rules](crate::NodeName#naming-rules), which
/// [`new`](Self::new) checks. Names must also be unique within a state, which
/// [`GraphBuilder::compile`](crate::GraphBuilder::compile) checks, failing
/// with [`BuildError::DuplicateChannel`](crate::BuildError::DuplicateChannel).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChannelSpec {
    /// The channel's name, which follows the
    /// [naming rules](crate::NodeName#naming-rules).
    name: Cow<'static, str>,
}

impl ChannelSpec {
    /// A channel named `name`, which may be a literal or a name only known at
    /// run time.
    ///
    /// # Errors
    ///
    /// Returns the [`NameError`] for the first
    /// [naming rule](crate::NodeName#naming-rules) `name` breaks.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Result<Self, NameError> {
        let name = name.into();
        validate_name(&name).map(|()| Self { name })
    }

    /// The channel's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// An error raised by [`State::apply_step`] when an update cannot be applied.
#[derive(Debug)]
pub struct ReduceError {
    /// The channel that rejected the update.
    channel: String,
    /// Why the update was rejected.
    source: BoxError,
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "channel `{}` rejected an update", self.channel)
    }
}

impl std::error::Error for ReduceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.source)
    }
}

impl ReduceError {
    /// An error for `channel`, caused by `source`.
    pub fn new(channel: impl Into<String>, source: impl Into<BoxError>) -> Self {
        Self {
            channel: channel.into(),
            source: source.into(),
        }
    }

    /// The channel that rejected the update.
    #[must_use]
    pub fn channel(&self) -> &str {
        &self.channel
    }
}

#[cfg(test)]
mod tests {
    //! Tests for channel descriptions and reduce errors.

    use std::error::Error as _;

    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;
    use crate::MAX_NAME_LEN;
    use crate::testing::{invalid_name, valid_name};

    // --- ChannelSpec ---------------------------------------------------------

    /// Every valid name makes a channel, which reports the name unchanged.
    #[proptest]
    fn new_accepts_valid_names(#[strategy(valid_name(MAX_NAME_LEN))] name: String) {
        let spec = ChannelSpec::new(name.clone());
        prop_assert_eq!(spec.as_ref().map(ChannelSpec::name), Ok(name.as_str()));
    }

    /// Every invalid name is rejected with the error `validate_name` reports
    /// for it.
    #[proptest]
    fn new_rejects_invalid_names(#[strategy(invalid_name())] name: String) {
        prop_assert_eq!(
            ChannelSpec::new(name.clone()).err(),
            Some(validate_name(&name).unwrap_err())
        );
    }

    // --- ReduceError ---------------------------------------------------------

    /// The message names the channel; the source is why the update was
    /// rejected.
    #[test]
    fn error_messages_name_the_problem() {
        let error = ReduceError::new("count", "must not be negative");
        assert_eq!(error.to_string(), "channel `count` rejected an update");
        assert_eq!(error.channel(), "count");
        assert_eq!(error.source().unwrap().to_string(), "must not be negative");
    }
}
