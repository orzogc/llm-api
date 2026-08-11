//! Drives the downstream README fixture
//! (`tests/fixtures/downstream_readme/`): a minimal crate whose
//! `Cargo.toml` mirrors the README installation section, `cargo check`ed
//! the way a downstream user would build it. This catches README snippets
//! relying on dependencies the installation section does not declare —
//! `tests/readme_examples.rs` cannot, because inside this crate optional
//! and dev dependencies are always available.

use std::path::Path;
use std::process::Command;

/// `#[ignore]`: shells out to a nested cargo and may hit the network to
/// resolve registry dependencies. CI runs it explicitly via
/// `cargo test --test downstream_readme -- --ignored`.
#[test]
#[ignore = "nested cargo invocation; may download dependencies (CI runs it explicitly)"]
fn downstream_readme_crate_compiles() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--manifest-path")
        .arg(root.join("tests/fixtures/downstream_readme/Cargo.toml"))
        // A dedicated target dir under the parent's ignored `target/`:
        // reused across runs, never fighting the parent build's lock.
        .env("CARGO_TARGET_DIR", root.join("target/downstream_readme"))
        .output()
        .expect("spawn cargo check");
    assert!(
        output.status.success(),
        "the downstream README crate failed to compile — README installation \
         section and examples are out of sync:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
