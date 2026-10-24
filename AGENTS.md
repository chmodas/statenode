# Development guidelines

This is the operational manual for contributors.

## Linting and formatting

The project maintains strict code quality and consistency through automated enforcement. Our CI pipeline will reject any contributions that do not meet these formatting and linting requirements. We use standard Rust tooling with custom configurations:

- **Formatting:** [rustfmt](https://github.com/rust-lang/rustfmt/blob/main/Configurations.md) (configured via `rustfmt.toml`).
- **Linting:** [clippy](https://github.com/rust-lang/rust-clippy) (configured via `clippy.toml`).

Before considering implementation complete, ensure your code is compliant:

- **Format code:** `cargo +nightly fmt` (pass `--check` if you only want to verify)
- **Check for lints:** `cargo clippy --all-targets --all-features -- -D warnings`, and again with `--no-default-features` in place of `--all-features`, since some code is compiled only without the default features

## Quality and testing

Processing errors are unacceptable given the sensitive nature of our data. We treat robust, well-tested code as a non-negotiable requirement, maintaining a minimum code coverage of 95% across the repository; line coverage is the floor, not the ceiling. 

- **Philosophy:** Prioritise exhaustive edge-case handling over ‘happy path’ validation; logic must hold even under hostility.
- **Verification:** Use `cargo llvm-cov` to generate coverage reports and ensure the 95% threshold is met before merging.

### Property-based testing

We use `proptest` extensively. Prefer property-based tests (PBT) over example-based tests whenever the input space is large, edge cases are critical, or state invariants must hold. When writing or modifying code, adhere to the following strategies:

- **Roundtrip (isomorphism):** For any serialisation or transformation: `decode(encode(data))==data`.
- **Invariants (sanity):** Regardless of input, certain rules must never be broken (e.g. ‘Total balance never negative’ or ‘Output always sanitized’).
- **Commutativity and idempotency:**
    - **Commutative:** Result of `Merge(A,B)` must equal `Merge(B,A)`.
    - **Idempotent:** `Process(data)` must equal `Process(Process(data))`.
- **No-panic fuzzing:** Pure "negative testing" to ensure untrusted or malformed input returns Err, never a crash.
- **Algorithmic parity:** Compare a high-performance/complex implementation against a ‘naive’ but obviously correct reference. Results must match exactly.
- **Boundary enforcement:** Generate inputs specifically at the edges of constraints (i.e. `MAX_INT`, empty strings, max-depth recursion) to verify graceful handling.

### Writing property tests

Property tests use [`test-strategy`](https://github.com/frozenlib/test-strategy) on top of `proptest`, not the `proptest!` macro, so test bodies stay visible to rustfmt and the IDE:

- **Tests:** `#[proptest]` on a plain `fn`, with strategies on the parameters (`#[strategy(..)]`, `#[filter(..)]`). Configure with `#[proptest(cases = 512)]`; async tests use `#[proptest(async = "tokio")]` instead of building a runtime by hand (it creates a multi-thread runtime per case).
- **Structured inputs:** `#[derive(Arbitrary)]` on a documented struct, where one field can depend on another (`#[strategy(f(#other))]`), instead of `prop_flat_map` and tuple plumbing. Every field must be read somewhere, or `dead_code` fires.
- **Imports:** `use proptest::prelude::*;` plus `use test_strategy::{Arbitrary, proptest};` (the explicit import shadows the prelude's `proptest!`).
- **Gotchas:** a predicate taking a slice must use the expression form, `#[filter(pred(&#param))]`; format strings inside attributes use positional `{}`. `#[filter]` rejections are *local* (up to 65 536 per test) rather than *global* like `prop_assume!` (1 024), so a generator that degrades into mostly rejected inputs retries quietly instead of failing fast – keep filters for rare rejections and shape the strategy instead.
- **Documentation:** document test code like any other code. The lints do not check `#[cfg(test)]` code or `tests/`, so reviewers must.

### Generator standards

Invest time in structured custom generators rather than relying on chaotic, fully random bytes. Your generators should produce syntactically valid but extreme data – such as maximum depth configurations or bizarre Unicode paths – to ensure `proptest` penetrates deep business logic rather than failing at the validation layer.

### Regression workflow

When a property-based test discovers a failure:

- Allow `proptest` to shrink the failing input to its minimal reproducible case.
- Before fixing the bug, hardcode that exact minimal input as a permanent, example-based regression test.
- Fix the logic to satisfy both the new regression test and the PBT suite.

### Intent-based examples

While PBT verifies logic, example-based tests define intent and boundaries. Use them for:

- **Documentation:** Every public API must have at least one "happy path" example test demonstrating standard usage.
- **Bootstrapping:** Use example tests during TDD to define the initial implementation before the input space is well-understood enough to build a generator.
- **IO and integration:** Use example tests for operations involving the file system, network, or external crates where structured generators are prohibitively complex or slow.
- **The "Golden Case":** Maintain a set of representative real-world data samples to ensure the most common use cases never regress in performance or accuracy.

## Documentation

We take the documentation seriously. All APIs – both public and private – must be fully documented. A developer should never have to guess the intent, context, or edge cases of a function or module.

## Commit convention

Use [conventional commits](https://www.conventionalcommits.org/) with a capital letter after the colon. No scope.

Format: `type: Message`

Types:
- `ci:` – Changes to CI configuration files and scripts
- `docs:` – Documentation only changes
- `feat:` – A new feature
- `fix:` – A bug fix
- `perf:` – A code change that improves performance
- `refactor:` – A code change that neither fixes a bug nor adds a feature
- `test:` Adding missing tests or correcting existing tests

Choose the type from what the change does. New behaviour is `feat:`, even when it completes something an earlier change documented as missing; `fix:` is only for correcting behaviour that is wrong.
