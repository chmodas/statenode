//! Parity: the engine against the naive reference interpreter.

use proptest::prelude::*;
use test_strategy::{Arbitrary, proptest};

use statenode::RunError;

/// Helpers shared with the other test binaries.
mod support {
    pub mod common;
}

use crate::support::common::ChainSpec;

/// A chain of up to 64 nodes, optionally closing into a cycle, run with a
/// recursion limit that sometimes stops it and sometimes does not.
#[derive(Debug, Arbitrary)]
struct Case {
    /// The number of nodes.
    #[strategy(1usize..=64)]
    len: usize,
    /// Where the last node's edge leads, modulo `len`: `None` for `END`.
    cycle_to: Option<usize>,
    /// The recursion limit.
    #[strategy(1u32..=150)]
    limit: u32,
}

/// For every generated chain and limit, the engine visits exactly the nodes,
/// steps and remaining-step counts the reference visits, and fails with the
/// recursion limit exactly when the reference does.
#[proptest(async = "tokio", cases = 256)]
async fn invoke_matches_the_reference_interpreter(case: Case) {
    let spec = ChainSpec {
        len: case.len,
        cycle_to: case.cycle_to.map(|t| t % case.len),
    };
    let (result, visits) = spec.run(case.limit).await.unwrap();
    let (expected, limited) = spec.interpret(case.limit);
    prop_assert_eq!(&visits, &expected);
    match result {
        Ok(trace) => {
            prop_assert!(!limited);
            let names: Vec<String> = expected.into_iter().map(|v| v.node).collect();
            prop_assert_eq!(&trace.visits[0], "input");
            prop_assert_eq!(&trace.visits[1..], &names[..]);
        }
        Err(RunError::RecursionLimit { limit, .. }) => {
            prop_assert!(limited);
            prop_assert_eq!(limit.get(), case.limit);
        }
        Err(other) => prop_assert!(false, "unexpected error {:?}", other),
    }
}

/// Building the same graph twice, so with two different builder tags, gives
/// identical runs.
#[proptest(async = "tokio", cases = 64)]
async fn runs_do_not_depend_on_the_builder(case: Case) {
    let spec = ChainSpec {
        len: case.len,
        cycle_to: case.cycle_to.map(|t| t % case.len),
    };
    let (first, first_visits) = spec.run(case.limit).await.unwrap();
    let (second, second_visits) = spec.run(case.limit).await.unwrap();
    prop_assert_eq!(first_visits, second_visits);
    prop_assert_eq!(first.ok(), second.ok());
}
