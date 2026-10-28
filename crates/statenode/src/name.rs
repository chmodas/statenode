//! Node names and the checks behind the [naming rules](NodeName#naming-rules).

use std::fmt;
use std::sync::Arc;

/// The longest name accepted, in bytes of UTF-8.
pub const MAX_NAME_LEN: usize = 128;

/// The prefix reserved for names the engine creates.
pub(crate) const RESERVED_PREFIX: &str = "__";

/// Characters a name may not contain, because the engine uses them as
/// separators.
pub(crate) const RESERVED_CHARACTERS: [char; 3] = ['|', ':', '+'];

/// Why a name breaks the [naming rules](NodeName#naming-rules).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NameError {
    /// The name is the empty string.
    Empty,
    /// The name is longer than [`MAX_NAME_LEN`] bytes.
    TooLong {
        /// The rejected name.
        name: String,
        /// Its length in bytes.
        len: usize,
    },
    /// The name starts with the reserved prefix `__`.
    ReservedPrefix {
        /// The rejected name.
        name: String,
    },
    /// The name contains one of the reserved separators `|`, `:` or `+`.
    ReservedCharacter {
        /// The rejected name.
        name: String,
        /// The first reserved character found.
        character: char,
    },
    /// The name contains a control character.
    ControlCharacter {
        /// The rejected name.
        name: String,
    },
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("names must not be empty"),
            Self::TooLong { name, len } => write!(
                f,
                "name `{name}` is {len} bytes long, but names are limited to {MAX_NAME_LEN} bytes"
            ),
            Self::ReservedPrefix { name } => {
                write!(f, "name `{name}` starts with `__`, which is reserved")
            }
            Self::ReservedCharacter { name, character } => {
                write!(f, "name `{name}` contains `{character}`, which is reserved")
            }
            Self::ControlCharacter { name } => {
                write!(f, "name {name:?} contains a control character")
            }
        }
    }
}

impl std::error::Error for NameError {}

/// Checks `name` against the [naming rules](NodeName#naming-rules).
///
/// The checks run in this order, and the first failure is reported:
/// [`Empty`](NameError::Empty), [`TooLong`](NameError::TooLong),
/// [`ReservedPrefix`](NameError::ReservedPrefix), then
/// [`ReservedCharacter`](NameError::ReservedCharacter) or
/// [`ControlCharacter`](NameError::ControlCharacter), for whichever such
/// character comes first in the name.
///
/// # Errors
///
/// Returns the [`NameError`] describing the first rule `name` breaks.
pub(crate) fn validate_name(name: &str) -> Result<(), NameError> {
    if name.is_empty() {
        return Err(NameError::Empty);
    }
    if name.len() > MAX_NAME_LEN {
        return Err(NameError::TooLong {
            name: name.to_owned(),
            len: name.len(),
        });
    }
    if name.starts_with(RESERVED_PREFIX) {
        return Err(NameError::ReservedPrefix {
            name: name.to_owned(),
        });
    }
    match name
        .chars()
        .find(|c| RESERVED_CHARACTERS.contains(c) || c.is_control())
    {
        None => Ok(()),
        Some(c) if c.is_control() => Err(NameError::ControlCharacter {
            name: name.to_owned(),
        }),
        Some(character) => Err(NameError::ReservedCharacter {
            name: name.to_owned(),
            character,
        }),
    }
}

/// The validated name of a node.
///
/// Cheap to clone: tasks, contexts, and errors all carry the name of the node
/// they belong to.
///
/// # Naming rules
///
/// Every name given to a graph, node names and state channel names alike,
/// follows these rules:
///
/// - A name is non-empty and at most [`MAX_NAME_LEN`] bytes of UTF-8.
/// - It does not start with `__`, which is reserved for the engine.
/// - It contains none of `|`, `:` and `+`, which separate namespace segments,
///   internal channel name parts, and join sources.
/// - It contains no control characters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeName(Arc<str>);

impl NodeName {
    /// Validates `name` and wraps it.
    ///
    /// # Errors
    ///
    /// Returns a [`NameError`] if `name` breaks the
    /// [naming rules](Self#naming-rules).
    pub(crate) fn new(name: &str) -> Result<Self, NameError> {
        validate_name(name).map(|()| Self(Arc::from(name)))
    }

    /// The name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NodeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the naming rules.

    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;
    use crate::testing::{BrokenRule, broken_name, control_char, truncate, valid_name};

    // --- NameError -----------------------------------------------------------

    /// Each message names the rule that was broken and, where there is one,
    /// the rejected name.
    #[test]
    fn error_messages_name_the_problem() {
        assert_eq!(NameError::Empty.to_string(), "names must not be empty");
        assert_eq!(
            NameError::TooLong {
                name: "x".to_owned(),
                len: 129
            }
            .to_string(),
            "name `x` is 129 bytes long, but names are limited to 128 bytes"
        );
        assert_eq!(
            NameError::ReservedPrefix {
                name: "__x".to_owned()
            }
            .to_string(),
            "name `__x` starts with `__`, which is reserved"
        );
        assert_eq!(
            NameError::ReservedCharacter {
                name: "a:b".to_owned(),
                character: ':'
            }
            .to_string(),
            "name `a:b` contains `:`, which is reserved"
        );
        assert_eq!(
            NameError::ControlCharacter {
                name: "a\nb".to_owned()
            }
            .to_string(),
            "name \"a\\nb\" contains a control character"
        );
    }

    // --- validate_name -------------------------------------------------------

    /// The empty string is rejected as empty.
    #[test]
    fn rejects_the_empty_name() {
        assert_eq!(validate_name(""), Err(NameError::Empty));
    }

    /// Padded to exactly the limit, a valid name is accepted; any byte more
    /// is rejected. Names are mostly multibyte, so this also pins that length
    /// is measured in bytes, not characters.
    #[proptest]
    fn accepts_names_up_to_the_limit_and_rejects_longer_ones(
        #[strategy(valid_name(MAX_NAME_LEN))] name: String,
        #[strategy(1..=MAX_NAME_LEN)] extra: usize,
    ) {
        let at_limit = name.clone() + &"x".repeat(MAX_NAME_LEN - name.len());
        prop_assert_eq!(validate_name(&at_limit), Ok(()));

        let over = at_limit + &"x".repeat(extra);
        prop_assert_eq!(
            validate_name(&over),
            Err(NameError::TooLong {
                name: over.clone(),
                len: MAX_NAME_LEN + extra
            })
        );
    }

    /// The reserved prefix is reported whatever follows it, forbidden
    /// characters included.
    #[proptest]
    fn rejects_the_reserved_prefix_before_any_character(
        #[strategy(prop::collection::vec(any::<char>(), 0..=MAX_NAME_LEN))] tail: Vec<char>,
    ) {
        let name = "__".to_owned() + &truncate(tail, MAX_NAME_LEN - 2);
        prop_assert_eq!(
            validate_name(&name),
            Err(NameError::ReservedPrefix { name: name.clone() })
        );
    }

    /// Every control character is rejected, even after an allowed `_`.
    #[proptest]
    fn rejects_control_characters(#[strategy(control_char())] c: char) {
        let name = format!("_{c}");
        prop_assert_eq!(
            validate_name(&name),
            Err(NameError::ControlCharacter { name: name.clone() })
        );
    }

    /// Each reserved separator is rejected inside a name and reported as
    /// the offending character.
    #[test]
    fn rejects_each_reserved_character() {
        for character in ['|', ':', '+'] {
            let name = format!("a{character}b");
            assert_eq!(
                validate_name(&name),
                Err(NameError::ReservedCharacter {
                    name: name.clone(),
                    character
                })
            );
        }
    }

    /// When a name breaks several rules, the first in check order is reported:
    /// the length, then the prefix, then whichever forbidden character
    /// comes first in the name.
    #[test]
    fn reports_the_first_rule_broken() {
        // Too long and reserved: the length is reported.
        let long_reserved = format!("__{}", ":".repeat(MAX_NAME_LEN));
        assert!(matches!(
            validate_name(&long_reserved),
            Err(NameError::TooLong { .. })
        ));
        // Reserved prefix and a reserved character: the prefix is reported.
        assert!(matches!(
            validate_name("__a:b"),
            Err(NameError::ReservedPrefix { .. })
        ));
        // A control character before a reserved one, and the other way round:
        // the first in the name is reported.
        assert!(matches!(
            validate_name("\u{7}:"),
            Err(NameError::ControlCharacter { .. })
        ));
        assert!(matches!(
            validate_name(":\u{7}"),
            Err(NameError::ReservedCharacter { character: ':', .. })
        ));
    }

    /// Every rule the shared invalid-name strategy can break yields exactly
    /// its own error, so the strategy reaches every kind of [`NameError`].
    #[test]
    fn each_broken_rule_yields_its_own_error() {
        let rules = [
            BrokenRule::Empty,
            BrokenRule::TooLong,
            BrokenRule::ReservedPrefix,
        ]
        .into_iter()
        .chain(RESERVED_CHARACTERS.map(BrokenRule::ReservedCharacter))
        .chain(['\0', '\x7f'].map(BrokenRule::ControlCharacter));
        for rule in rules {
            let name = rule.apply("plan");
            assert_eq!(validate_name(&name), Err(rule.error(&name)), "{rule:?}");
        }
    }

    /// Every name the shared invalid-name strategy draws breaks exactly the
    /// rule it was drawn for, whatever valid name it starts from.
    #[proptest]
    fn broken_names_break_exactly_their_rule(
        #[strategy(broken_name())] broken: (BrokenRule, String),
    ) {
        let (rule, name) = broken;
        prop_assert_eq!(validate_name(&name), Err(rule.error(&name)));
    }

    /// The shared valid-name strategy only ever draws names the rules accept.
    #[proptest]
    fn valid_names_are_accepted(#[strategy(valid_name(MAX_NAME_LEN))] name: String) {
        prop_assert_eq!(validate_name(&name), Ok(()));
    }

    /// Validation never panics, on arbitrary strings of up to
    /// `2 * MAX_NAME_LEN` characters, control characters included.
    #[proptest]
    fn never_panics(
        #[strategy(prop::collection::vec(any::<char>(), 0..=2 * MAX_NAME_LEN))] chars: Vec<char>,
    ) {
        let _ = validate_name(&chars.into_iter().collect::<String>());
    }

    // --- NodeName ------------------------------------------------------------

    /// A node name wraps a valid name unchanged and refuses an invalid one
    /// with the error `validate_name` reports.
    #[test]
    fn node_names_wrap_valid_names() {
        let name = NodeName::new("plan").unwrap();
        assert_eq!(name.as_str(), "plan");
        assert_eq!(name.to_string(), "plan");
        assert_eq!(NodeName::new(":"), Err(validate_name(":").unwrap_err()));
    }

    /// Node names compare and order exactly as their text does: two names
    /// from the same text are equal, and a clone equals its original.
    #[proptest]
    fn node_names_order_by_their_text(
        #[strategy(valid_name(MAX_NAME_LEN))] a: String,
        #[strategy(valid_name(MAX_NAME_LEN))] b: String,
    ) {
        let (x, y) = (NodeName::new(&a).unwrap(), NodeName::new(&b).unwrap());
        prop_assert_eq!(x.cmp(&y), a.cmp(&b));
        prop_assert_eq!(x == y, a == b);
        prop_assert_eq!(NodeName::new(&a).unwrap(), x.clone());
        prop_assert_eq!(x.clone(), x);
    }
}
