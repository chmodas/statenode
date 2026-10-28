//! Fixtures shared by several modules' unit tests.
//!
//! Compiled only for tests. Every function here must be run by the unit tests
//! that use it, because test code counts toward coverage.

use std::borrow::Cow;
use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::prelude::*;
use test_strategy::Arbitrary;

use crate::MAX_NAME_LEN;
use crate::name::{NameError, RESERVED_CHARACTERS, RESERVED_PREFIX};
use crate::node::{BoxError, NodeContext};
use crate::reducer;
use crate::state::{ChannelSpec, ReduceError, State};
use crate::task::TaskPath;

/// A state with one replace channel, `count`.
///
/// Its channels are a field, so a test can give one builder a state whose
/// channels are invalid without a second `State` impl, whose `apply_step`
/// would never run.
#[derive(Clone)]
pub(crate) struct Counter {
    /// The current count.
    pub(crate) count: i64,
    /// The names of the channels [`State::channels`] returns: `count`
    /// unless a test says otherwise.
    pub(crate) channels: Vec<&'static str>,
}

impl Default for Counter {
    /// A count of zero, with the one valid channel.
    fn default() -> Self {
        Self {
            count: 0,
            channels: vec!["count"],
        }
    }
}

impl State for Counter {
    /// The new count.
    type Update = i64;

    fn channels(&self) -> Result<Vec<ChannelSpec>, NameError> {
        self.channels
            .iter()
            .copied()
            .map(ChannelSpec::new)
            .collect()
    }

    fn apply_step(&mut self, updates: Vec<(TaskPath, i64)>) -> Result<(), ReduceError> {
        let writes = updates.iter().map(|(path, count)| (path, *count));
        reducer::replace("count", &mut self.count, writes)
    }
}

/// A node function that changes nothing.
///
/// A named function rather than a closure, so every test that only needs
/// *a* node shares one instantiation, which one run covers.
pub(crate) async fn noop(_s: Arc<Counter>, _ctx: NodeContext) -> Result<(), BoxError> {
    Ok(())
}

/// A name the naming rules reject.
///
/// Tests outside `name.rs` use it as a name the rules reject, taking any
/// expected error from `validate_name`, so they do not depend on which
/// rule it breaks; only `name.rs` tests the rules themselves.
pub(crate) const INVALID_NAME: &str = "";

/// Printable ASCII without the reserved characters `+`, `:` and `|`.
const ASCII: &[RangeInclusive<char>] = &[' '..='*', ','..='9', ';'..='{', '}'..='~'];

/// A character a name may contain anywhere.
///
/// The four arms are 1-, 2-, 3- and 4-byte UTF-8, equally weighted, so
/// multibyte characters are common. The ranges skip both blocks of control
/// characters, and `prop::char` never yields surrogates.
pub(crate) fn name_char() -> impl Strategy<Value = char> {
    prop_oneof![
        prop::char::ranges(Cow::Borrowed(ASCII)),
        prop::char::range('\u{a0}', '\u{7ff}'),
        prop::char::range('\u{800}', '\u{ffff}'),
        prop::char::range('\u{10000}', char::MAX),
    ]
}

/// The longest prefix of `chars` that fits in `max_bytes` bytes of UTF-8.
pub(crate) fn truncate(chars: Vec<char>, max_bytes: usize) -> String {
    chars
        .into_iter()
        .scan(0, |len, c| {
            *len += c.len_utf8();
            (*len <= max_bytes).then_some(c)
        })
        .collect()
}

/// A valid name of at most `max_bytes` bytes.
///
/// `max_bytes` must be at least 4, so the first character always fits and
/// the name is never empty.
pub(crate) fn valid_name(max_bytes: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(name_char(), 1..=max_bytes)
        .prop_map(move |chars| truncate(chars, max_bytes))
        .prop_filter("reserved prefix", |name| !name.starts_with("__"))
}

/// A control character: the Unicode `Cc` category, which never changes.
///
/// The draws are not uniform: `prop::char` favours a few C0 controls (`\0`,
/// `\t`, `\n`, …), while the C1 controls (`\u{80}` to `\u{9f}`) make up
/// about half of all draws.
pub(crate) fn control_char() -> impl Strategy<Value = char> {
    prop_oneof![
        prop::char::range('\0', '\x1f'),
        prop::char::range('\x7f', '\u{9f}'),
    ]
}

/// One way for a name to break the naming rules: one variant per
/// [`NameError`].
#[derive(Debug, Clone, Copy, Arbitrary)]
pub(crate) enum BrokenRule {
    /// The empty name.
    Empty,
    /// A name longer than [`MAX_NAME_LEN`].
    TooLong,
    /// A name starting with the reserved prefix.
    ReservedPrefix,
    /// A name containing this reserved character.
    ReservedCharacter(#[strategy(prop::sample::select(&RESERVED_CHARACTERS[..]))] char),
    /// A name containing this control character.
    ControlCharacter(#[strategy(control_char())] char),
}

impl BrokenRule {
    /// Turns `base` into a name that breaks this rule and no rule checked
    /// before it.
    ///
    /// `base` must be a valid name of at most `(MAX_NAME_LEN - 8) / 2` bytes,
    /// so two copies of it around one character stay within the length limit.
    pub(crate) fn apply(self, base: &str) -> String {
        match self {
            Self::Empty => String::new(),
            Self::TooLong => base.repeat(MAX_NAME_LEN / base.len() + 1),
            Self::ReservedPrefix => format!("{RESERVED_PREFIX}{base}"),
            Self::ReservedCharacter(c) | Self::ControlCharacter(c) => format!("{base}{c}{base}"),
        }
    }

    /// The error `validate_name` must report for a `name` that
    /// [`apply`](Self::apply) built from this rule.
    pub(crate) fn error(self, name: &str) -> NameError {
        let name = name.to_owned();
        match self {
            Self::Empty => NameError::Empty,
            Self::TooLong => NameError::TooLong {
                len: name.len(),
                name,
            },
            Self::ReservedPrefix => NameError::ReservedPrefix { name },
            Self::ReservedCharacter(character) => NameError::ReservedCharacter { name, character },
            Self::ControlCharacter(_) => NameError::ControlCharacter { name },
        }
    }
}

/// A broken rule and a name breaking exactly that rule, built from a random
/// valid name.
pub(crate) fn broken_name() -> impl Strategy<Value = (BrokenRule, String)> {
    (any::<BrokenRule>(), valid_name((MAX_NAME_LEN - 8) / 2))
        .prop_map(|(rule, base)| (rule, rule.apply(&base)))
}

/// A name that breaks the naming rules: every kind of [`NameError`] can be
/// drawn, and [`BrokenRule::error`] gives the error each must produce.
pub(crate) fn invalid_name() -> impl Strategy<Value = String> {
    broken_name().prop_map(|(_, name)| name)
}

/// Two different builder tags, the first arbitrary and the second the first
/// plus a nonzero offset, wrapping, so they can never be equal and no draw is
/// rejected.
pub(crate) fn distinct_tags() -> impl Strategy<Value = (u64, u64)> {
    (any::<u64>(), 1..=u64::MAX).prop_map(|(tag, offset)| (tag, tag.wrapping_add(offset)))
}
