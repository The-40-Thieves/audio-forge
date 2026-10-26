//! Keeps the limiter timing quoted in the README and the architecture guide in sync with
//! the constants the limiters actually use.

use forge_dsp::{LIMITER_LOOKAHEAD_MS, LIMITER_RELEASE_MS};
use std::path::Path;

fn repo_doc(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn assert_mentions(doc: &str, name: &str, needle: &str) {
    assert!(
        doc.contains(needle),
        "{name} should say \"{needle}\"; update it to match the limiter constants in \
         crates/forge-dsp/src/loudness_limiter.rs"
    );
}

#[test]
fn readme_quotes_the_limiter_lookahead() {
    let readme = repo_doc("README.md");
    assert_mentions(
        &readme,
        "README.md",
        &format!("{LIMITER_LOOKAHEAD_MS} ms lookahead"),
    );
}

#[test]
fn architecture_guide_quotes_the_limiter_timing() {
    let guide = repo_doc("docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md");
    let name = "docs/ARCHITECTURE_AND_ENGINEERING_GUIDE.md";
    assert_mentions(
        &guide,
        name,
        &format!("`{LIMITER_LOOKAHEAD_MS} ms` attack window"),
    );
    assert_mentions(&guide, name, &format!("`{LIMITER_RELEASE_MS} ms` release"));
}
