//! Impulse-delay tests: the delay each real-time processor actually applies must equal the
//! `latency_samples()` it reports, since that value drives the host's delay compensation.

use forge_dsp::{RealtimeMasterLimiter, RealtimeOpenSoothe, RealtimeOpenSpacer};

const SAMPLE_RATES: [f64; 3] = [44_100.0, 48_000.0, 96_000.0];
/// Impulse position, past the start so the overlap-add has settled.
const IMPULSE_AT: usize = 4096;
const FRAMES: usize = 12_288;
/// Below the limiter's -1 dBTP ceiling, so the impulse passes without gain reduction.
const IMPULSE: f32 = 0.5;

/// Feeds an impulse through `process` and returns how many samples later the output peaks.
fn measured_delay(mut process: impl FnMut(f32) -> f32) -> usize {
    let mut peak = (0_usize, 0.0_f32);
    for i in 0..FRAMES {
        let x = if i == IMPULSE_AT { IMPULSE } else { 0.0 };
        let y = process(x).abs();
        if y > peak.1 {
            peak = (i, y);
        }
    }
    assert!(peak.1 > 0.1 * IMPULSE, "impulse did not reach the output");
    peak.0 - IMPULSE_AT
}

fn soothe_delay(dsp: &mut RealtimeOpenSoothe) -> usize {
    // A lone impulse has a flat spectrum, so no bin is a resonance and no gain is applied.
    measured_delay(|x| dsp.process_stereo_sample(x, x, 24.0, 0.5, 8, false).0)
}

fn spacer_delay(dsp: &mut RealtimeOpenSpacer) -> usize {
    // A silent sidechain asks for no carving.
    measured_delay(|x| {
        dsp.process_sample(x, x, 0.0, 0.0, 180.0, 8_000.0, 0.5, 6.0)
            .0
    })
}

#[test]
fn open_soothe_reports_its_real_delay() {
    for sr in SAMPLE_RATES {
        let mut dsp = RealtimeOpenSoothe::new(sr);
        assert_eq!(
            soothe_delay(&mut dsp),
            dsp.latency_samples() as usize,
            "new @ {sr}"
        );
        dsp.reset(sr);
        assert_eq!(
            soothe_delay(&mut dsp),
            dsp.latency_samples() as usize,
            "reset @ {sr}"
        );
    }
}

#[test]
fn open_spacer_reports_its_real_delay() {
    for sr in SAMPLE_RATES {
        let mut dsp = RealtimeOpenSpacer::new(sr);
        assert_eq!(
            spacer_delay(&mut dsp),
            dsp.latency_samples() as usize,
            "new @ {sr}"
        );
        dsp.reset(sr);
        assert_eq!(
            spacer_delay(&mut dsp),
            dsp.latency_samples() as usize,
            "reset @ {sr}"
        );
    }
}

#[test]
fn master_limiter_delay_matches_when_active_or_bypassed() {
    for sr in SAMPLE_RATES {
        // Start at a different rate so `reset` has to change the lookahead length.
        let mut dsp = RealtimeMasterLimiter::new(if sr == 48_000.0 { 96_000.0 } else { 48_000.0 });
        dsp.reset(sr);
        let latency = dsp.latency_samples() as usize;
        assert_eq!(
            measured_delay(|x| dsp.process_stereo_sample(x, x, 0.0, -1.0, 85.0).0),
            latency,
            "active @ {sr}"
        );
        dsp.reset(sr);
        assert_eq!(
            measured_delay(|x| dsp.process_delay_only(x, x).0),
            latency,
            "bypassed @ {sr}"
        );
    }
}
