//! Pure-Rust Mid/Side + Linkwitz-Riley + transient-mask 4-stem splitter (0 VRAM, no Python).

use std::path::Path;

use forge_core::{AudioBuffer, Result, StemBundle};
use forge_dsp::{Biquad, sanitize_sample_f64};

/// Per-sample filter and envelope state for the deterministic splitter.
struct DspSplitter {
    bass_lp_l: Biquad,
    bass_lp_r: Biquad,
    voc_hp: Biquad,
    voc_lp: Biquad,
    env_fast: f64,
    env_slow: f64,
}

/// One output frame: `[bass, vocals, drums, other]`, each as `(left, right)`.
type SplitFrame = [(f64, f64); 4];

impl DspSplitter {
    fn new(sr: f64) -> Self {
        // Filters for Bass (<155 Hz), Vocal Center (220 Hz - 5.5 kHz Mid), and Instruments (Side + Highs)
        let q = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            bass_lp_l: Biquad::lowpass(sr, 155.0, q),
            bass_lp_r: Biquad::lowpass(sr, 155.0, q),
            voc_hp: Biquad::highpass(sr, 220.0, q),
            voc_lp: Biquad::lowpass(sr, 5800.0, q),
            env_fast: 0.0,
            env_slow: 0.0,
        }
    }

    fn process(&mut self, l: f64, r: f64) -> SplitFrame {
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r);

        // 1. Sub & Bass stem (<155 Hz)
        let b_l = self.bass_lp_l.process_sample(l);
        let b_r = self.bass_lp_r.process_sample(r);
        let rem_l = l - b_l;
        let rem_r = r - b_r;

        // 2. Transient envelope detector for Percussive/Drum separation
        let abs_m = mid.abs();
        self.env_fast = 0.90 * self.env_fast + 0.10 * abs_m;
        self.env_slow = 0.996 * self.env_slow + 0.004 * abs_m + 1e-7;
        let transient_mask = ((self.env_fast / self.env_slow - 1.15) * 1.4).clamp(0.0, 1.0);

        // 3. Center-panned harmonic vocal band
        let v_mid = self.voc_lp.process_sample(self.voc_hp.process_sample(mid))
            * (1.0 - transient_mask * 0.7);

        // 4. Percussive Drum stem
        let d_l = rem_l * transient_mask;
        let d_r = rem_r * transient_mask;

        // 5. Residual Harmonic Instrumentation (Guitars/Keys/Synths + Stereo Side)
        let o_l = (rem_l - v_mid - d_l) + side * 0.35;
        let o_r = (rem_r - v_mid - d_r) - side * 0.35;
        [(b_l, b_r), (v_mid, v_mid), (d_l, d_r), (o_l, o_r)]
    }
}

/// Splits `input_path` into four stems in `output_dir` with the deterministic Rust DSP.
pub(crate) fn separate_stems_dsp(input_path: &Path, output_dir: &Path) -> Result<StemBundle> {
    let buf = AudioBuffer::load(input_path)?.to_stereo();
    let frames = buf.num_frames();
    let mut splitter = DspSplitter::new(f64::from(buf.sample_rate.max(1)));
    // Order matches `SplitFrame`: bass, vocals, drums, other.
    let mut stems = [
        AudioBuffer::new(2, frames, buf.sample_rate)?,
        AudioBuffer::new(2, frames, buf.sample_rate)?,
        AudioBuffer::new(2, frames, buf.sample_rate)?,
        AudioBuffer::new(2, frames, buf.sample_rate)?,
    ];

    for i in 0..frames {
        let l = sanitize_sample_f64(buf.channels[0][i]);
        let r = sanitize_sample_f64(buf.channels[1][i]);
        for (stem, (sl, sr)) in stems.iter_mut().zip(splitter.process(l, r)) {
            stem.channels[0][i] = sl;
            stem.channels[1][i] = sr;
        }
    }

    let [bass, vocals, drums, other] = stems;
    let vocals_path = output_dir.join("01_Vocals_Clean.wav");
    let drums_path = output_dir.join("02_Drums_Stem.wav");
    let bass_path = output_dir.join("03_Bass_Stem.wav");
    let other_path = output_dir.join("04_Instruments_Other.wav");

    vocals.save_wav(&vocals_path, 24)?;
    drums.save_wav(&drums_path, 24)?;
    bass.save_wav(&bass_path, 24)?;
    other.save_wav(&other_path, 24)?;

    Ok(StemBundle {
        vocals: Some(vocals_path),
        drums: Some(drums_path),
        bass: Some(bass_path),
        other: Some(other_path),
        ..StemBundle::default()
    })
}
