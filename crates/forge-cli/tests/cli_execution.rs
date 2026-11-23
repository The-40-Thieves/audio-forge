//! End-to-end execution integration tests for the `forge` CLI binary.

use std::f64::consts::PI;
use std::process::Command;

use forge_core::AudioBuffer;

fn write_test_wav(path: &std::path::Path, freq_hz: f64, sr: u32, secs: f64) {
    let frames = (f64::from(sr) * secs).round() as usize;
    let mut l = Vec::with_capacity(frames);
    let mut r = Vec::with_capacity(frames);
    for i in 0..frames {
        let s = 0.35 * (2.0 * PI * freq_hz * (i as f64) / f64::from(sr)).sin();
        l.push(s);
        r.push(s);
    }
    let buf = AudioBuffer::from_channels(vec![l, r], sr).expect("valid test buffer");
    buf.save_wav(path, 24).expect("save test wav");
}

#[test]
fn cli_audit_restore_and_phase_align_execute_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src_wav = dir.path().join("input.wav");
    let ref_wav = dir.path().join("ref.wav");
    let restored_wav = dir.path().join("restored.wav");
    let aligned_wav = dir.path().join("aligned.wav");
    write_test_wav(&src_wav, 440.0, 48_000, 0.5);
    write_test_wav(&ref_wav, 440.0, 48_000, 0.5);

    let bin = env!("CARGO_BIN_EXE_forge");

    // 1. `forge audit --json`
    let audit_out = Command::new(bin)
        .args(["audit", "--json"])
        .arg(&src_wav)
        .output()
        .expect("run forge audit");
    assert!(
        audit_out.status.success(),
        "forge audit failed: {}",
        String::from_utf8_lossy(&audit_out.stderr)
    );
    let audit_json: serde_json::Value =
        serde_json::from_slice(&audit_out.stdout).expect("valid audit JSON");
    assert!(audit_json.get("integrated_lufs").is_some());
    assert!(audit_json.get("true_peak_dbtp").is_some());

    // 2. `forge restore`
    let restore_out = Command::new(bin)
        .args(["restore"])
        .arg(&src_wav)
        .arg("-o")
        .arg(&restored_wav)
        .output()
        .expect("run forge restore");
    assert!(
        restore_out.status.success(),
        "forge restore failed: {}",
        String::from_utf8_lossy(&restore_out.stderr)
    );
    assert!(restored_wav.is_file());

    // 3. `forge phase-align`
    let align_out = Command::new(bin)
        .args(["phase-align"])
        .arg(&ref_wav)
        .arg(&src_wav)
        .arg("-o")
        .arg(&aligned_wav)
        .output()
        .expect("run forge phase-align");
    assert!(
        align_out.status.success(),
        "forge phase-align failed: {}",
        String::from_utf8_lossy(&align_out.stderr)
    );
    assert!(aligned_wav.is_file());
}
