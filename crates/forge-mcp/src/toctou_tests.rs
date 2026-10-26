//! Swap-after-check tests (THE-1600, unix only).
//!
//! Each test resolves a path normally, then uses the `anchored::test_hook` seam to swap a
//! component for a symlink that leaves the root between the check and the open. The tool
//! must fail or stay inside the root; it must never read or write outside it. With the
//! anchored I/O every case fails, which the tests assert.

use crate::anchored::test_hook;
use crate::handle_request;
use crate::sandbox::PathSandbox;
use forge_core::AudioBuffer;
use serde_json::{Value, json};
use std::cell::Cell;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Clears the hook even when an assertion fails, so it cannot leak into another test that
/// reuses this thread.
struct HookGuard;

impl Drop for HookGuard {
    fn drop(&mut self) {
        test_hook::clear();
    }
}

fn call_tool(sandbox: &PathSandbox, name: &str, arguments: Value) -> (bool, String) {
    let resp = handle_request(
        json!(1),
        "tools/call",
        &json!({ "name": name, "arguments": arguments }),
        sandbox,
    );
    let result = resp.result.expect("tool result");
    let text = result["content"][0]["text"].as_str().unwrap_or_default();
    (
        result["isError"].as_bool().expect("isError"),
        text.to_string(),
    )
}

/// Two seconds of a stereo tone, enough for every analysis stage.
fn write_tone(path: &Path) {
    let ch: Vec<f64> = (0..96_000)
        .map(|i| 0.1 * (f64::from(i) * 2.0 * std::f64::consts::PI * 220.0 / 48_000.0).sin())
        .collect();
    AudioBuffer::from_channels(vec![ch.clone(), ch], 48_000)
        .expect("buffer")
        .save_wav(path, 24)
        .expect("write fixture");
}

/// Installs a hook that, the first time a root-relative path starting with `prefix` is
/// about to be opened, runs `swap`. Returns a flag that records whether it fired.
fn swap_once(prefix: &'static str, swap: impl FnOnce() + 'static) -> (HookGuard, Rc<Cell<bool>>) {
    let fired = Rc::new(Cell::new(false));
    let flag = Rc::clone(&fired);
    let mut swap = Some(swap);
    test_hook::set(move |rel: &Path| {
        if rel.starts_with(prefix)
            && let Some(swap) = swap.take()
        {
            swap();
            flag.set(true);
        }
    });
    (HookGuard, fired)
}

/// Replaces the directory `root/sub` with a symlink to `outside`, keeping the original
/// contents at `root/sub.orig`.
fn swap_dir_for_link(root: &Path, outside: &Path) -> impl FnOnce() + 'static {
    let (sub, moved, outside) = (
        root.join("sub"),
        root.join("sub.orig"),
        outside.to_path_buf(),
    );
    move || {
        std::fs::rename(&sub, &moved).expect("move sub aside");
        symlink(&outside, &sub).expect("symlink sub to outside");
    }
}

/// Asserts that nothing named in `names` was created outside the root.
fn assert_nothing_outside(outside: &Path, names: &[&str], text: &str) {
    for name in names {
        let escaped: PathBuf = outside.join(name);
        assert!(
            !escaped.exists(),
            "write escaped the root: {escaped:?} ({text})"
        );
    }
}

#[test]
fn swapped_output_directory_cannot_redirect_a_midi_write() {
    let root = tempfile::tempdir().expect("root");
    let outside = tempfile::tempdir().expect("outside");
    std::fs::create_dir(root.path().join("sub")).expect("mkdir sub");
    let sb = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
    let (_guard, fired) = swap_once("sub", swap_dir_for_link(root.path(), outside.path()));

    let (is_error, text) = call_tool(
        &sb,
        "forge_music_theory_midi",
        json!({ "chords": "Dm9 G13", "output_midi": "sub/chords.mid" }),
    );
    assert!(fired.get(), "the swap hook never ran");
    assert_nothing_outside(outside.path(), &["chords.mid"], &text);
    assert!(
        is_error,
        "write through a swapped directory reported success: {text}"
    );
}

#[test]
fn swapped_output_directory_cannot_redirect_a_wav_and_report_write() {
    let root = tempfile::tempdir().expect("root");
    let outside = tempfile::tempdir().expect("outside");
    std::fs::create_dir(root.path().join("sub")).expect("mkdir sub");
    write_tone(&root.path().join("mix.wav"));
    let sb = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
    let (_guard, fired) = swap_once("sub", swap_dir_for_link(root.path(), outside.path()));

    let (is_error, text) = call_tool(
        &sb,
        "forge_deliver",
        json!({ "input": "mix.wav", "output": "sub/master.wav" }),
    );
    assert!(fired.get(), "the swap hook never ran");
    assert_nothing_outside(
        outside.path(),
        &["master.wav", "master.delivery.json"],
        &text,
    );
    assert!(
        is_error,
        "write through a swapped directory reported success: {text}"
    );
}

#[test]
fn swapped_input_file_cannot_redirect_a_read() {
    let root = tempfile::tempdir().expect("root");
    let outside = tempfile::tempdir().expect("outside");
    write_tone(&root.path().join("in.wav"));
    let secret = outside.path().join("secret.wav");
    write_tone(&secret);
    let sb = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
    let input = root.path().join("in.wav");
    let (_guard, fired) = swap_once("in.wav", move || {
        std::fs::remove_file(&input).expect("remove input");
        symlink(&secret, &input).expect("symlink input to outside");
    });

    let (is_error, text) = call_tool(&sb, "forge_audit_audio", json!({ "path": "in.wav" }));
    assert!(fired.get(), "the swap hook never ran");
    assert!(is_error, "read through a swapped symlink succeeded: {text}");
    assert!(text.contains("sandbox root handle"), "{text}");
}

#[test]
fn stem_swapped_after_the_worker_ran_fails_verification() {
    let root = tempfile::tempdir().expect("root");
    let outside = tempfile::tempdir().expect("outside");
    write_tone(&root.path().join("mix.wav"));
    let decoy = outside.path().join("decoy.wav");
    write_tone(&decoy);
    let sb = PathSandbox::new([root.path().to_path_buf()]).expect("sandbox");
    let stems = root.path().join("session").join("stems");
    let (_guard, fired) = swap_once("session/stems/03_Bass_Stem.wav", move || {
        let stem = stems.join("03_Bass_Stem.wav");
        std::fs::remove_file(&stem).expect("remove stem");
        symlink(&decoy, &stem).expect("symlink stem to outside");
    });

    let (is_error, text) = call_tool(
        &sb,
        "forge_ingest_to_reaper",
        json!({ "input": "mix.wav", "output_dir": "session" }),
    );
    // The bass stem is not read again after separation, so only the post-run check sees it.
    assert!(fired.get(), "the swap hook never ran");
    assert!(
        is_error,
        "a stem swapped for an outside link was accepted: {text}"
    );
    assert!(text.contains("sandbox root handle"), "{text}");
    assert!(!root.path().join("session/ForgeAudio_Session.rpp").exists());
}
