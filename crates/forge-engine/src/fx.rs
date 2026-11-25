//! Zero-allocation runtime FX processor instances hosting `forge-dsp` and `forge-vocal`
//! engines per-object, per-track, and on the master bus.

use forge_dsp::{
    AnalogHardwareModel, IspOversampling, ParamSmoother, RealtimeDensityAndClipper,
    RealtimeDrumTrigger, RealtimeFdnReverb, RealtimeMasterLimiter, RealtimeMasteringEq18,
    RealtimeNeuralPreamp, RealtimeOpenSoothe, RealtimeOpenSpacer, RealtimePhaseAligner,
    RealtimeSubAndTapeBus, RealtimeTransientTonalSplit, RealtimeVocalSurgical,
};
use forge_vocal::{RealtimeVocalTune, VocalTuneScaleMode};

use crate::project::FxSlotConfig;

fn hardware_model_from_idx(idx: u8) -> AnalogHardwareModel {
    match idx {
        0 => AnalogHardwareModel::Neve1073ClassA,
        1 => AnalogHardwareModel::Fet1176AllButtons,
        2 => AnalogHardwareModel::TubeV76Triode,
        _ => AnalogHardwareModel::StuderA800Tape,
    }
}

fn apply_three_band_curve(eq: &mut RealtimeMasteringEq18, low_db: f32, mid_db: f32, air_db: f32) {
    let mut gains = [0.0_f64; 18];
    let lo = f64::from(low_db);
    let mid = f64::from(mid_db);
    let air = f64::from(air_db);
    for slot in &mut gains[0..3] {
        *slot = lo;
    }
    for slot in &mut gains[7..13] {
        *slot = mid;
    }
    for slot in &mut gains[14..18] {
        *slot = air;
    }
    eq.set_gains(&gains, 1.0);
}

/// Returns `true` iff `cfg` is a non-bypassed `OpenSpacer` whose configured
/// `sidechain_track_id` resolves to a non-self track index in `track_ids`.
#[inline]
pub fn open_spacer_sidechain_active(
    cfg: &FxSlotConfig,
    self_track_idx: Option<usize>,
    track_ids: &[u64],
) -> bool {
    matches!(
        cfg,
        FxSlotConfig::OpenSpacer {
            bypass: false,
            sidechain_track_id: Some(sc_id),
            ..
        } if track_ids
            .iter()
            .enumerate()
            .any(|(idx, &id)| self_track_idx != Some(idx) && id == *sc_id)
    )
}

/// Zero-allocation 3-voice percussive reinforcement synthesizer driven by [`RealtimeDrumTrigger`].
pub struct DrumPunchSynth {
    sample_rate: f64,
    kick_env: f64,
    kick_phase: f64,
    snare_env: f64,
    snare_phase: f64,
    hat_env: f64,
    rng_state: u32,
}

impl DrumPunchSynth {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            kick_env: 0.0,
            kick_phase: 0.0,
            snare_env: 0.0,
            snare_phase: 0.0,
            hat_env: 0.0,
            rng_state: 0x6D2B_79F5,
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8_000.0);
        self.kick_env = 0.0;
        self.kick_phase = 0.0;
        self.snare_env = 0.0;
        self.snare_phase = 0.0;
        self.hat_env = 0.0;
    }

    #[inline]
    fn trigger(&mut self, note: u8, velocity: f32) {
        let v = f64::from(velocity.clamp(0.05, 1.0));
        match note {
            36 => {
                self.kick_env = self.kick_env.max(v);
                self.kick_phase = 0.0;
            }
            38 => {
                self.snare_env = self.snare_env.max(v);
                self.snare_phase = 0.0;
            }
            _ => {
                self.hat_env = self.hat_env.max(v * 0.7);
            }
        }
    }

    #[inline]
    fn next_noise(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        ((x as f64) / f64::from(u32::MAX)) * 2.0 - 1.0
    }

    #[inline]
    fn next_sample(&mut self) -> f32 {
        if self.kick_env < 1e-4 && self.snare_env < 1e-4 && self.hat_env < 1e-4 {
            return 0.0;
        }
        let sr = self.sample_rate;
        let mut sum = 0.0_f64;
        if self.kick_env >= 1e-4 {
            let hz = 48.0 + 72.0 * self.kick_env;
            self.kick_phase = (self.kick_phase + hz / sr).fract();
            sum += (2.0 * std::f64::consts::PI * self.kick_phase).sin() * self.kick_env * 0.85;
            self.kick_env *= 1.0 - (14.0 / sr).min(0.1);
        }
        if self.snare_env >= 1e-4 {
            self.snare_phase = (self.snare_phase + 190.0 / sr).fract();
            let noise = self.next_noise();
            let tone = (2.0 * std::f64::consts::PI * self.snare_phase).sin();
            sum += (0.45 * tone + 0.55 * noise) * self.snare_env * 0.65;
            self.snare_env *= 1.0 - (24.0 / sr).min(0.15);
        }
        if self.hat_env >= 1e-4 {
            let noise = self.next_noise();
            sum += noise * self.hat_env * 0.4;
            self.hat_env *= 1.0 - (60.0 / sr).min(0.25);
        }
        sum.clamp(-1.0, 1.0) as f32
    }
}

/// Live zero-alloc DSP runtime corresponding to one [`FxSlotConfig`].
pub enum FxProcessor {
    GainTrim(ParamSmoother),
    OpenSoothe(Box<RealtimeOpenSoothe>),
    VocalSurgical(Box<RealtimeVocalSurgical>),
    SubMonoTape(Box<RealtimeSubAndTapeBus>),
    MasterLimiter(Box<RealtimeMasterLimiter>),
    SplitEq(Box<RealtimeTransientTonalSplit>),
    DensityClipper(Box<RealtimeDensityAndClipper>),
    SpaceVerb(Box<RealtimeFdnReverb>),
    VocalTune(Box<RealtimeVocalTune>),
    OpenSpacer(Box<RealtimeOpenSpacer>),
    PhaseMatchEq {
        aligner: Box<RealtimePhaseAligner>,
        eq: Box<RealtimeMasteringEq18>,
        cached_curve: [f32; 3],
    },
    NeuralPreamp(Box<RealtimeNeuralPreamp>),
    DrumTrigger {
        detector: Box<RealtimeDrumTrigger>,
        punch: Box<DrumPunchSynth>,
    },
}

impl FxProcessor {
    /// Instantiates the DSP runtime for `cfg` at `sample_rate` (off-audio-thread).
    pub fn from_config(cfg: &FxSlotConfig, sample_rate: f64) -> Self {
        let sr = sample_rate.max(8_000.0);
        match cfg {
            FxSlotConfig::GainTrim { gain_db, .. } => {
                let g = 10.0_f64.powf(f64::from(gain_db.clamp(-48.0, 24.0)) / 20.0);
                Self::GainTrim(ParamSmoother::new(g, sr, 5.0))
            }
            FxSlotConfig::OpenSoothe {
                low_latency,
                ultrasonic_lpf_hz,
                mid_side_balance,
                stereo_link,
                hard_mode,
                ..
            } => {
                let mut d = RealtimeOpenSoothe::new(sr);
                d.set_low_latency(*low_latency);
                d.set_ultrasonic_lpf_hz(f64::from(*ultrasonic_lpf_hz));
                d.set_mid_side_balance(f64::from(*mid_side_balance));
                d.set_stereo_link(f64::from(*stereo_link));
                d.set_hard_mode(*hard_mode);
                Self::OpenSoothe(Box::new(d))
            }
            FxSlotConfig::VocalSurgical { .. } => {
                Self::VocalSurgical(Box::new(RealtimeVocalSurgical::new(sr)))
            }
            FxSlotConfig::SubMonoTape { mono_cutoff_hz, .. } => Self::SubMonoTape(Box::new(
                RealtimeSubAndTapeBus::new(sr, f64::from(*mono_cutoff_hz)),
            )),
            FxSlotConfig::MasterLimiter { isp_8x, .. } => {
                let mut d = RealtimeMasterLimiter::new(sr);
                d.set_isp_oversampling(if *isp_8x {
                    IspOversampling::X8
                } else {
                    IspOversampling::X4
                });
                Self::MasterLimiter(Box::new(d))
            }
            FxSlotConfig::SplitEq { .. } => {
                Self::SplitEq(Box::new(RealtimeTransientTonalSplit::new(sr)))
            }
            FxSlotConfig::DensityClipper { .. } => {
                Self::DensityClipper(Box::new(RealtimeDensityAndClipper::new(sr)))
            }
            FxSlotConfig::SpaceVerb { .. } => Self::SpaceVerb(Box::new(RealtimeFdnReverb::new(sr))),
            FxSlotConfig::VocalTune { .. } => Self::VocalTune(Box::new(RealtimeVocalTune::new(sr))),
            FxSlotConfig::OpenSpacer {
                mid_side_balance,
                delta_listen,
                ..
            } => {
                let mut d = RealtimeOpenSpacer::new(sr);
                d.set_mid_side_balance(f64::from(*mid_side_balance));
                d.set_delta_listen(*delta_listen);
                Self::OpenSpacer(Box::new(d))
            }
            FxSlotConfig::PhaseMatchEq {
                delay_samples,
                invert_polarity,
                low_shelf_db,
                mid_dip_db,
                air_shelf_db,
                ..
            } => {
                let mut aligner = Box::new(RealtimePhaseAligner::new(sr));
                aligner.set_alignment(*delay_samples, *invert_polarity);
                let mut eq = Box::new(RealtimeMasteringEq18::new(sr));
                apply_three_band_curve(&mut eq, *low_shelf_db, *mid_dip_db, *air_shelf_db);
                Self::PhaseMatchEq {
                    aligner,
                    eq,
                    cached_curve: [*low_shelf_db, *mid_dip_db, *air_shelf_db],
                }
            }
            FxSlotConfig::NeuralPreamp { model, .. } => {
                let mut d = RealtimeNeuralPreamp::new(hardware_model_from_idx(*model));
                d.set_sample_rate(sr);
                Self::NeuralPreamp(Box::new(d))
            }
            FxSlotConfig::DrumTrigger { .. } => Self::DrumTrigger {
                detector: Box::new(RealtimeDrumTrigger::new(sr)),
                punch: Box::new(DrumPunchSynth::new(sr)),
            },
        }
    }

    /// Algorithmic latency in samples reported for Plugin Delay Compensation (PDC).
    #[inline]
    pub fn latency_samples(&self, cfg: &FxSlotConfig) -> usize {
        if cfg.is_bypassed() {
            return 0;
        }
        match self {
            Self::OpenSoothe(d) => d.latency_samples() as usize,
            Self::MasterLimiter(d) => d.latency_samples() as usize,
            Self::VocalTune(d) => d.latency_samples() as usize,
            Self::OpenSpacer(d) => match cfg {
                FxSlotConfig::OpenSpacer {
                    sidechain_track_id: None,
                    ..
                } => 0,
                _ => d.latency_samples() as usize,
            },
            _ => 0,
        }
    }

    /// Algorithmic latency in samples reported for Plugin Delay Compensation (PDC),
    /// validating any external `OpenSpacer` sidechain route against `track_ids`.
    #[inline]
    pub fn latency_samples_for(
        &self,
        cfg: &FxSlotConfig,
        self_track_idx: Option<usize>,
        track_ids: &[u64],
    ) -> usize {
        if cfg.is_bypassed() {
            return 0;
        }
        match self {
            Self::OpenSpacer(d) => {
                if open_spacer_sidechain_active(cfg, self_track_idx, track_ids) {
                    d.latency_samples() as usize
                } else {
                    0
                }
            }
            _ => self.latency_samples(cfg),
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(8_000.0);
        match self {
            Self::GainTrim(s) => s.set_sample_rate(sr, 5.0),
            Self::OpenSoothe(d) => d.reset(sr),
            Self::VocalSurgical(d) => d.reset(sr),
            Self::SubMonoTape(d) => d.reset(sr),
            Self::MasterLimiter(d) => d.reset(sr),
            Self::SplitEq(d) => d.reset(sr),
            Self::DensityClipper(d) => d.reset(sr),
            Self::SpaceVerb(d) => d.reset(sr),
            Self::VocalTune(d) => d.reset(sr),
            Self::OpenSpacer(d) => d.reset(sr),
            Self::PhaseMatchEq { aligner, eq, .. } => {
                aligner.reset();
                eq.reset();
            }
            Self::NeuralPreamp(d) => {
                d.set_sample_rate(sr);
                d.reset();
            }
            Self::DrumTrigger { detector, punch } => {
                detector.reset(sr);
                punch.reset(sr);
            }
        }
    }

    /// Processes one stereo sample with zero heap allocations.
    #[inline]
    pub fn process_stereo(&mut self, cfg: &FxSlotConfig, in_l: f32, in_r: f32) -> (f32, f32) {
        self.process_stereo_with_sidechain(cfg, in_l, in_r, None)
    }

    /// Processes one stereo sample with an optional external stereo sidechain feed (zero allocs).
    #[inline]
    pub fn process_stereo_with_sidechain(
        &mut self,
        cfg: &FxSlotConfig,
        in_l: f32,
        in_r: f32,
        sidechain: Option<(f32, f32)>,
    ) -> (f32, f32) {
        let _ = sidechain;
        if cfg.is_bypassed() {
            return (in_l, in_r);
        }
        match (self, cfg) {
            (Self::GainTrim(smoother), FxSlotConfig::GainTrim { gain_db, .. }) => {
                let target = 10.0_f64.powf(f64::from(gain_db.clamp(-48.0, 24.0)) / 20.0);
                smoother.set_target(target);
                let g = smoother.next_val() as f32;
                (in_l * g, in_r * g)
            }
            (
                Self::OpenSoothe(d),
                FxSlotConfig::OpenSoothe {
                    sensitivity_db,
                    max_depth_db,
                    sharpness_bins,
                    low_latency,
                    tilt_db_per_oct,
                    ultrasonic_lpf_hz,
                    mid_side_balance,
                    stereo_link,
                    hard_mode,
                    ..
                },
            ) => {
                d.set_low_latency(*low_latency);
                d.set_tilt_db_per_oct(f64::from(*tilt_db_per_oct));
                d.set_ultrasonic_lpf_hz(f64::from(*ultrasonic_lpf_hz));
                d.set_mid_side_balance(f64::from(*mid_side_balance));
                d.set_stereo_link(f64::from(*stereo_link));
                d.set_hard_mode(*hard_mode);
                d.process_stereo_sample(
                    in_l,
                    in_r,
                    f64::from(*sensitivity_db),
                    f64::from(*max_depth_db),
                    *sharpness_bins,
                    false,
                )
            }
            (
                Self::VocalSurgical(d),
                FxSlotConfig::VocalSurgical {
                    declick,
                    deplosive,
                    deess_thresh_db,
                    gate_thresh_db,
                    ..
                },
            ) => d.process_stereo_sample(
                in_l,
                in_r,
                if *declick { 0.55 } else { 0.0 },
                if *deplosive { 0.6 } else { 0.0 },
                f64::from(*deess_thresh_db),
                f64::from(*gate_thresh_db),
            ),
            (
                Self::SubMonoTape(d),
                FxSlotConfig::SubMonoTape {
                    mono_cutoff_hz,
                    sub_reinforce,
                    stereo_width,
                    tape_drive,
                    ..
                },
            ) => {
                d.set_mono_cutoff(f64::from(*mono_cutoff_hz));
                d.process_stereo_sample(
                    in_l,
                    in_r,
                    f64::from(*sub_reinforce),
                    f64::from(*stereo_width),
                    f64::from(*tape_drive),
                )
            }
            (
                Self::MasterLimiter(d),
                FxSlotConfig::MasterLimiter {
                    drive_db,
                    ceiling_dbtp,
                    release_ms,
                    isp_8x,
                    ..
                },
            ) => {
                d.set_isp_oversampling(if *isp_8x {
                    IspOversampling::X8
                } else {
                    IspOversampling::X4
                });
                d.process_stereo_sample(
                    in_l,
                    in_r,
                    f64::from(*drive_db),
                    f64::from(*ceiling_dbtp),
                    f64::from(*release_ms),
                )
            }
            (
                Self::SplitEq(d),
                FxSlotConfig::SplitEq {
                    tr_low_db,
                    tr_mid_db,
                    tr_high_db,
                    tn_low_db,
                    tn_mid_db,
                    tn_high_db,
                    ..
                },
            ) => d.process_stereo_sample(
                in_l,
                in_r,
                [
                    f64::from(*tr_low_db),
                    f64::from(*tr_mid_db),
                    f64::from(*tr_high_db),
                ],
                [
                    f64::from(*tn_low_db),
                    f64::from(*tn_mid_db),
                    f64::from(*tn_high_db),
                ],
                1.5,
                0,
            ),
            (
                Self::DensityClipper(d),
                FxSlotConfig::DensityClipper {
                    density,
                    clip_drive_db,
                    ceiling_db,
                    ..
                },
            ) => d.process_stereo_sample(
                in_l,
                in_r,
                f64::from(*density),
                f64::from(*clip_drive_db),
                f64::from(*ceiling_db),
            ),
            (
                Self::SpaceVerb(d),
                FxSlotConfig::SpaceVerb {
                    decay_rt60_secs,
                    damping,
                    shimmer,
                    ducking,
                    wet_mix,
                    ..
                },
            ) => d.process_stereo_sample(
                in_l,
                in_r,
                f64::from(*decay_rt60_secs),
                f64::from(*damping),
                f64::from(*shimmer),
                f64::from(*ducking),
                f64::from(*wet_mix),
            ),
            (
                Self::VocalTune(d),
                FxSlotConfig::VocalTune {
                    key_root,
                    scale_mode,
                    retune_speed_ms,
                    retune_amount,
                    humanize,
                    formant_shift_st,
                    wet_mix,
                    ..
                },
            ) => {
                let mode = match *scale_mode {
                    1 => VocalTuneScaleMode::Major,
                    2 => VocalTuneScaleMode::Minor,
                    3 => VocalTuneScaleMode::PentatonicMinor,
                    4 => VocalTuneScaleMode::PentatonicMajor,
                    _ => VocalTuneScaleMode::Chromatic,
                };
                d.set_key_and_scale(*key_root, mode);
                d.set_retune_speed_ms(f64::from(*retune_speed_ms));
                d.set_retune_amount(f64::from(*retune_amount));
                d.set_humanize(f64::from(*humanize));
                d.set_formant_shift_st(f64::from(*formant_shift_st));
                d.set_wet_mix(f64::from(*wet_mix));
                d.process_stereo_sample(in_l, in_r)
            }
            (
                Self::OpenSpacer(d),
                FxSlotConfig::OpenSpacer {
                    amount,
                    max_cut_db,
                    low_hz,
                    high_hz,
                    mid_side_balance,
                    delta_listen,
                    ..
                },
            ) => {
                let Some((sc_l, sc_r)) = sidechain else {
                    return (in_l, in_r);
                };
                d.set_mid_side_balance(f64::from(*mid_side_balance));
                d.set_delta_listen(*delta_listen);
                d.process_sample(
                    in_l,
                    in_r,
                    sc_l,
                    sc_r,
                    f64::from(*low_hz),
                    f64::from(*high_hz),
                    f64::from(*amount),
                    f64::from(*max_cut_db),
                )
            }
            (
                Self::PhaseMatchEq {
                    aligner,
                    eq,
                    cached_curve,
                },
                FxSlotConfig::PhaseMatchEq {
                    delay_samples,
                    invert_polarity,
                    low_shelf_db,
                    mid_dip_db,
                    air_shelf_db,
                    ..
                },
            ) => {
                aligner.set_alignment(*delay_samples, *invert_polarity);
                let target_curve = [*low_shelf_db, *mid_dip_db, *air_shelf_db];
                if *cached_curve != target_curve {
                    *cached_curve = target_curve;
                    apply_three_band_curve(eq, *low_shelf_db, *mid_dip_db, *air_shelf_db);
                }
                let (al, ar) = aligner.process_stereo_sample(in_l, in_r);
                eq.process_stereo_sample(al, ar)
            }
            (
                Self::NeuralPreamp(d),
                FxSlotConfig::NeuralPreamp {
                    model,
                    input_drive_db,
                    output_trim_db,
                    ..
                },
            ) => {
                d.set_model(hardware_model_from_idx(*model));
                d.process_stereo_sample(
                    in_l,
                    in_r,
                    f64::from(*input_drive_db),
                    f64::from(*output_trim_db),
                )
            }
            (
                Self::DrumTrigger { detector, punch },
                FxSlotConfig::DrumTrigger {
                    threshold_db, mix, ..
                },
            ) => {
                for (note, vel) in detector
                    .process_stereo_sample(in_l, in_r, f64::from(*threshold_db))
                    .into_iter()
                    .flatten()
                {
                    punch.trigger(note, vel);
                }
                let hit = punch.next_sample();
                let wet = mix.clamp(0.0, 1.0);
                let out_l = (in_l + hit * wet).clamp(-1.5, 1.5);
                let out_r = (in_r + hit * wet).clamp(-1.5, 1.5);
                (out_l, out_r)
            }
            _ => (in_l, in_r),
        }
    }
}
