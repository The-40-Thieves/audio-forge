//! Shared lock-free audio-to-GUI telemetry and dark-mode studio metering/spectrum canvas
//! rendered across all 10 `forge-plug` VST3/CLAP editors.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use forge_dsp::{Biquad, flush_denormal_f64, read_shared_spectrum, sanitize_sample_f32};
use nih_plug::prelude::{BoolParam, FloatParam, IntParam, Param, ParamSetter};
use nih_plug_egui::egui;

/// Sentinel stored in `GuiTelemetry::lufs_momentary` / `true_peak_dbtp` when a plugin does not
/// publish K-weighted loudness telemetry.
pub const UNSET_LOUDNESS_DB: f32 = -120.0;

/// Safe, zero-`unsafe` handle to any `nih_plug` parameter for A/B state recall and factory presets.
#[derive(Clone, Copy)]
pub enum ParamRef<'a> {
    Float(&'a FloatParam),
    Int(&'a IntParam),
    Bool(&'a BoolParam),
}

impl ParamRef<'_> {
    #[inline]
    pub fn normalized(&self) -> f32 {
        match self {
            Self::Float(p) => p.unmodulated_normalized_value(),
            Self::Int(p) => p.unmodulated_normalized_value(),
            Self::Bool(p) => p.unmodulated_normalized_value(),
        }
    }

    #[inline]
    pub fn set_normalized(&self, setter: &ParamSetter, norm: f32) {
        let clamped = if norm.is_finite() {
            norm.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Self::Float(p) => {
                setter.begin_set_parameter(*p);
                setter.set_parameter_normalized(*p, clamped);
                setter.end_set_parameter(*p);
            }
            Self::Int(p) => {
                setter.begin_set_parameter(*p);
                setter.set_parameter_normalized(*p, clamped);
                setter.end_set_parameter(*p);
            }
            Self::Bool(p) => {
                setter.begin_set_parameter(*p);
                setter.set_parameter_normalized(*p, clamped);
                setter.end_set_parameter(*p);
            }
        }
    }
}

/// One curated studio factory preset defined as `(param_id, normalized_value)` pairs.
#[derive(Debug, Clone, Copy)]
pub struct FactoryPreset {
    pub name: &'static str,
    pub values: &'static [(&'static str, f32)],
}

/// GUI-thread A/B parameter-bank state holding two full normalized snapshots so switching `A <-> B`
/// updates both the host automation state and DSP via `ParamSetter`.
#[derive(Debug, Clone, Default)]
pub struct AbBankState {
    pub active_slot: usize,
    pub slot_a: Vec<(String, f32)>,
    pub slot_b: Vec<(String, f32)>,
}

impl AbBankState {
    /// Captures current normalized values from `params` into both A and B if uninitialized.
    pub fn ensure_initialized(&mut self, params: &[(&'static str, ParamRef<'_>)]) {
        if self.slot_a.is_empty() {
            let snap = snapshot_params(params);
            self.slot_a = snap.clone();
            self.slot_b = snap;
        }
    }

    /// Switches to `target_slot` (`0` = A, `1` = B): saves current values into the outgoing slot
    /// and applies the target slot's normalized values through `setter`.
    pub fn switch_to(
        &mut self,
        target_slot: usize,
        params: &[(&'static str, ParamRef<'_>)],
        setter: &ParamSetter,
    ) {
        self.switch_with(target_slot, params, |pref, norm| {
            pref.set_normalized(setter, norm);
        });
    }

    /// Core bank-switching transition parameterized by `apply_fn` so unit tests can verify
    /// full A/B snapshot recall across Float/Int/Bool parameters without a live DAW host.
    pub fn switch_with(
        &mut self,
        target_slot: usize,
        params: &[(&'static str, ParamRef<'_>)],
        mut apply_fn: impl FnMut(ParamRef<'_>, f32),
    ) {
        let target = usize::from(target_slot != 0);
        self.ensure_initialized(params);
        if target == self.active_slot {
            return;
        }
        let current = snapshot_params(params);
        let recalled = if self.active_slot == 0 {
            self.slot_a = current;
            &self.slot_b
        } else {
            self.slot_b = current;
            &self.slot_a
        };
        for (target_id, norm) in recalled {
            if let Some((_, pref)) = params.iter().find(|(id, _)| *id == target_id.as_str()) {
                apply_fn(*pref, *norm);
            }
        }
        self.active_slot = target;
    }

    /// Copies the active slot's live values into the inactive slot (`A -> B` or `B -> A`).
    pub fn copy_active_to_other(&mut self, params: &[(&'static str, ParamRef<'_>)]) {
        let current = snapshot_params(params);
        if self.active_slot == 0 {
            self.slot_a = current.clone();
            self.slot_b = current;
        } else {
            self.slot_b = current.clone();
            self.slot_a = current;
        }
    }
}

/// Captures `(param_id, normalized_value)` for every parameter in `params`.
pub fn snapshot_params(params: &[(&'static str, ParamRef<'_>)]) -> Vec<(String, f32)> {
    params
        .iter()
        .map(|&(id, pref)| (id.to_string(), pref.normalized()))
        .collect()
}

/// Applies a curated [`FactoryPreset`] through `ParamSetter`.
pub fn apply_factory_preset(
    params: &[(&'static str, ParamRef<'_>)],
    setter: &ParamSetter,
    preset: &FactoryPreset,
) {
    for &(target_id, norm) in preset.values {
        if let Some((_, pref)) = params.iter().find(|(id, _)| *id == target_id) {
            pref.set_normalized(setter, norm);
        }
    }
}

/// Renders the studio A/B parameter-bank toolbar (`A`, `B`, `A->B`) and curated factory preset pills.
pub(crate) fn draw_ab_and_presets_bar(
    ui: &mut egui::Ui,
    setter: &ParamSetter,
    params: &[(&'static str, ParamRef<'_>)],
    ab: &mut AbBankState,
    presets: &[FactoryPreset],
) {
    ab.ensure_initialized(params);
    ui.horizontal_wrapped(|ui| {
        let a_active = ab.active_slot == 0;
        let active_col = egui::Color32::from_rgb(74, 222, 128);
        let muted_col = egui::Color32::from_rgb(148, 163, 184);

        if ui
            .button(
                egui::RichText::new("A")
                    .monospace()
                    .strong()
                    .color(if a_active { active_col } else { muted_col }),
            )
            .on_hover_text("Recall parameter bank A")
            .clicked()
        {
            ab.switch_to(0, params, setter);
        }
        if ui
            .button(
                egui::RichText::new("B")
                    .monospace()
                    .strong()
                    .color(if !a_active { active_col } else { muted_col }),
            )
            .on_hover_text("Recall parameter bank B")
            .clicked()
        {
            ab.switch_to(1, params, setter);
        }
        if ui
            .button(
                egui::RichText::new(if a_active { "A->B" } else { "B->A" })
                    .monospace()
                    .small()
                    .color(egui::Color32::from_rgb(90, 210, 255)),
            )
            .on_hover_text("Copy active bank into inactive bank")
            .clicked()
        {
            ab.copy_active_to_other(params);
        }

        if !presets.is_empty() {
            ui.separator();
            ui.label(
                egui::RichText::new("Presets:")
                    .small()
                    .color(egui::Color32::from_rgb(165, 175, 195)),
            );
            for preset in presets {
                if ui
                    .button(egui::RichText::new(preset.name).small())
                    .clicked()
                {
                    apply_factory_preset(params, setter, preset);
                }
            }
        }
    });
    ui.add_space(4.0);
}

/// Number of scrolling Gain-Reduction history points rendered across the studio canvas.
pub const GR_HISTORY_LEN: usize = 64;

/// Lock-free 32-band telemetry, dynamic Gain-Reduction / Bode curve, scrolling GR history ribbon,
/// spectrum freeze, and peak-hold state shared between the real-time audio thread and the 60 FPS
/// `nih_plug_egui` studio GUI.
pub struct GuiTelemetry {
    pub spectrum_db: [AtomicU32; 32],
    pub band_gr_db: [AtomicU32; 32],
    pub frozen_shared_db: [AtomicU32; 32],
    pub gr_history_db: [AtomicU32; GR_HISTORY_LEN],
    pub gr_history_head: AtomicUsize,
    gr_history_decim: AtomicU32,
    pub peak_in_db: AtomicU32,
    pub peak_out_db: AtomicU32,
    pub peak_hold_out_db: AtomicU32,
    pub gr_db: AtomicU32,
    pub lufs_momentary: AtomicU32,
    pub lufs_short_term: AtomicU32,
    pub true_peak_dbtp: AtomicU32,
    pub ab_slot: AtomicU32,
    pub spectrum_frozen: AtomicBool,
}

impl Default for GuiTelemetry {
    fn default() -> Self {
        Self {
            spectrum_db: std::array::from_fn(|_| AtomicU32::new((-90.0_f32).to_bits())),
            band_gr_db: std::array::from_fn(|_| AtomicU32::new(0.0_f32.to_bits())),
            frozen_shared_db: std::array::from_fn(|_| AtomicU32::new((-90.0_f32).to_bits())),
            gr_history_db: std::array::from_fn(|_| AtomicU32::new(0.0_f32.to_bits())),
            gr_history_head: AtomicUsize::new(0),
            gr_history_decim: AtomicU32::new(0),
            peak_in_db: AtomicU32::new((-90.0_f32).to_bits()),
            peak_out_db: AtomicU32::new((-90.0_f32).to_bits()),
            peak_hold_out_db: AtomicU32::new((-90.0_f32).to_bits()),
            gr_db: AtomicU32::new(0.0_f32.to_bits()),
            lufs_momentary: AtomicU32::new(UNSET_LOUDNESS_DB.to_bits()),
            lufs_short_term: AtomicU32::new(UNSET_LOUDNESS_DB.to_bits()),
            true_peak_dbtp: AtomicU32::new(UNSET_LOUDNESS_DB.to_bits()),
            ab_slot: AtomicU32::new(0),
            spectrum_frozen: AtomicBool::new(false),
        }
    }
}

/// Zero-allocation audio-thread ITU-R BS.1770-4 sliding-window loudness & 4x output true-peak meter.
///
/// Tracks exact 100 ms K-weighted energy sub-blocks in a 30-slot ring:
/// - **Momentary (400 ms)** = mean energy over the latest 4 blocks (`0.4 s`).
/// - **Short-Term (3.0 s)** = mean energy over all 30 blocks (`3.0 s`).
/// - **True Peak (dBTP)** = 4x Catmull-Rom cubic intersample peak measured directly from the
///   emitted gain-applied output samples.
pub struct OutputLoudnessMeter {
    k_shelf_l: Biquad,
    k_rlb_l: Biquad,
    k_shelf_r: Biquad,
    k_rlb_r: Biquad,
    samples_per_block: usize,
    block_accum: f64,
    block_count: usize,
    blocks: [f64; 30],
    block_head: usize,
    blocks_filled: usize,
    hist_l: [f64; 4],
    hist_r: [f64; 4],
    tp_lin: f64,
    tp_decay: f64,
}

impl OutputLoudnessMeter {
    pub fn new(sample_rate: f64) -> Self {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 768_000.0)
        } else {
            48_000.0
        };
        let samples_per_block = ((sr * 0.100).round() as usize).max(64);
        Self {
            k_shelf_l: Biquad::k_weight_pre_shelf(sr),
            k_rlb_l: Biquad::k_weight_rlb_highpass(sr),
            k_shelf_r: Biquad::k_weight_pre_shelf(sr),
            k_rlb_r: Biquad::k_weight_rlb_highpass(sr),
            samples_per_block,
            block_accum: 0.0,
            block_count: 0,
            blocks: [0.0; 30],
            block_head: 0,
            blocks_filled: 0,
            hist_l: [0.0; 4],
            hist_r: [0.0; 4],
            tp_lin: 1e-6,
            tp_decay: (-1.0 / (0.350 * sr)).exp(),
        }
    }

    pub fn reset(&mut self, sample_rate: f64) {
        *self = Self::new(sample_rate);
    }

    #[inline]
    fn cubic_4x_peak(h: &mut [f64; 4], x: f64) -> f64 {
        h[0] = h[1];
        h[1] = h[2];
        h[2] = h[3];
        h[3] = x;
        let (p0, p1, p2, p3) = (h[0], h[1], h[2], h[3]);
        let mut peak = p1.abs().max(p2.abs()).max(x.abs());
        for t in [0.25_f64, 0.50, 0.75] {
            let t2 = t * t;
            let t3 = t2 * t;
            let y = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
            peak = peak.max(y.abs());
        }
        peak
    }

    /// Feeds one emitted stereo output sample into the K-weighted loudness windows and 4x TP meter.
    #[inline]
    pub fn push_output_sample(&mut self, out_l: f32, out_r: f32) {
        let ol = sanitize_sample_f32(out_l);
        let or_ = sanitize_sample_f32(out_r);

        let kw_l = self
            .k_rlb_l
            .process_sample(self.k_shelf_l.process_sample(ol));
        let kw_r = self
            .k_rlb_r
            .process_sample(self.k_shelf_r.process_sample(or_));
        self.block_accum += kw_l * kw_l + kw_r * kw_r;
        self.block_count += 1;
        if self.block_count >= self.samples_per_block {
            self.blocks[self.block_head] =
                flush_denormal_f64(self.block_accum / (self.block_count as f64));
            self.block_head = (self.block_head + 1) % self.blocks.len();
            self.blocks_filled = (self.blocks_filled + 1).min(self.blocks.len());
            self.block_accum = 0.0;
            self.block_count = 0;
        }

        let tp = Self::cubic_4x_peak(&mut self.hist_l, ol)
            .max(Self::cubic_4x_peak(&mut self.hist_r, or_));
        self.tp_lin = flush_denormal_f64((self.tp_lin * self.tp_decay).max(tp)).max(1e-6);
    }

    /// Returns `(momentary_400ms_lufs, short_term_3s_lufs, output_true_peak_dbtp)`.
    pub fn snapshot(&self) -> (f32, f32, f32) {
        let partial_mean = if self.block_count > 0 {
            Some(self.block_accum / (self.block_count as f64))
        } else {
            None
        };
        let window_lufs = |max_blocks: usize| -> f32 {
            let n = self.blocks_filled.min(max_blocks);
            let mut sum = 0.0_f64;
            for k in 0..n {
                let idx = (self.block_head + self.blocks.len() - 1 - k) % self.blocks.len();
                sum += self.blocks[idx];
            }
            let mean = if n > 0 {
                sum / (n as f64)
            } else {
                partial_mean.unwrap_or(0.0)
            };
            if mean > 1e-10 {
                ((-0.691 + 10.0 * mean.log10()) as f32).clamp(-90.0, 6.0)
            } else {
                -90.0
            }
        };
        let m_lufs = window_lufs(4);
        let s_lufs = window_lufs(30);
        let tp_dbtp = (20.0 * self.tp_lin.max(1e-6).log10() as f32).clamp(-90.0, 12.0);
        (m_lufs, s_lufs, tp_dbtp)
    }
}

impl GuiTelemetry {
    /// Clears the latched output peak-hold readout back to `-90.0 dBFS`.
    #[inline]
    pub fn reset_peak_hold(&self) {
        self.peak_hold_out_db
            .store((-90.0_f32).to_bits(), Ordering::Relaxed);
    }

    /// Publishes lock-free ITU-R BS.1770-4 Momentary (400 ms), Short-Term (3 s) LUFS and
    /// 4x Output True-Peak dBTP telemetry. Non-finite inputs preserve `UNSET_LOUDNESS_DB`.
    #[inline]
    pub fn update_loudness_and_tp(&self, lufs_m: f32, lufs_s: f32, tp_dbtp: f32) {
        let clean_m = if lufs_m.is_finite() {
            lufs_m.clamp(-90.0, 6.0)
        } else {
            UNSET_LOUDNESS_DB
        };
        let clean_s = if lufs_s.is_finite() {
            lufs_s.clamp(-90.0, 6.0)
        } else {
            UNSET_LOUDNESS_DB
        };
        let clean_tp = if tp_dbtp.is_finite() {
            tp_dbtp.clamp(-90.0, 12.0)
        } else {
            UNSET_LOUDNESS_DB
        };
        self.lufs_momentary
            .store(clean_m.to_bits(), Ordering::Relaxed);
        self.lufs_short_term
            .store(clean_s.to_bits(), Ordering::Relaxed);
        self.true_peak_dbtp
            .store(clean_tp.to_bits(), Ordering::Relaxed);
    }

    /// Toggles live spectrum, shared-bus ghost, and GR history freeze and returns the new state.
    #[inline]
    pub fn toggle_spectrum_freeze(&self, shared_slot: Option<usize>) -> bool {
        let next = !self.spectrum_frozen.load(Ordering::Relaxed);
        if next && let Some(slot) = shared_slot {
            let snap = read_shared_spectrum(slot);
            for (dst, &v) in self.frozen_shared_db.iter().zip(&snap) {
                let clean = if v.is_finite() {
                    v.clamp(-90.0, 6.0)
                } else {
                    -90.0
                };
                dst.store(clean.to_bits(), Ordering::Relaxed);
            }
        }
        self.spectrum_frozen.store(next, Ordering::Release);
        next
    }

    /// Appends one Gain-Reduction sample (in dB, `<= 0.0`) into the lock-free scrolling history ring.
    ///
    /// Writes the slot value *before* publishing the updated head index with `Release` ordering so
    /// the GUI reader (`Acquire`) never observes an uninitialized/stale newest sample.
    #[inline]
    pub fn push_gr_history(&self, gr_db: f32) {
        let clean = if gr_db.is_finite() {
            gr_db.clamp(-36.0, 0.0)
        } else {
            0.0
        };
        let head = self.gr_history_head.load(Ordering::Relaxed);
        let idx = head % GR_HISTORY_LEN;
        self.gr_history_db[idx].store(clean.to_bits(), Ordering::Relaxed);
        self.gr_history_head
            .store(head.wrapping_add(1), Ordering::Release);
    }

    /// Reads the scrolling Gain-Reduction history ordered oldest -> newest.
    #[inline]
    pub fn read_gr_history_ordered(&self) -> [f32; GR_HISTORY_LEN] {
        let head = self.gr_history_head.load(Ordering::Acquire) % GR_HISTORY_LEN;
        let mut out = [0.0_f32; GR_HISTORY_LEN];
        for (i, slot) in out.iter_mut().enumerate() {
            let src = (head + i) % GR_HISTORY_LEN;
            let v = f32::from_bits(self.gr_history_db[src].load(Ordering::Relaxed));
            *slot = if v.is_finite() {
                v.clamp(-36.0, 0.0)
            } else {
                0.0
            };
        }
        out
    }

    #[inline]
    pub fn update_peaks(&self, in_l: f32, in_r: f32, out_l: f32, out_r: f32, gr_db: f32) {
        let il = if in_l.is_finite() {
            in_l.abs().min(100.0)
        } else {
            0.0
        };
        let ir = if in_r.is_finite() {
            in_r.abs().min(100.0)
        } else {
            0.0
        };
        let ol = if out_l.is_finite() {
            out_l.abs().min(100.0)
        } else {
            0.0
        };
        let or_ = if out_r.is_finite() {
            out_r.abs().min(100.0)
        } else {
            0.0
        };
        let gr = if gr_db.is_finite() {
            gr_db.clamp(-90.0, 24.0)
        } else {
            0.0
        };

        let pk_in = 20.0 * il.max(ir).max(1e-5).log10();
        let pk_out = 20.0 * ol.max(or_).max(1e-5).log10();
        let raw_prev_in = f32::from_bits(self.peak_in_db.load(Ordering::Relaxed));
        let raw_prev_out = f32::from_bits(self.peak_out_db.load(Ordering::Relaxed));
        let prev_in = if raw_prev_in.is_finite() {
            raw_prev_in.clamp(-100.0, 40.0)
        } else {
            -90.0
        };
        let prev_out = if raw_prev_out.is_finite() {
            raw_prev_out.clamp(-100.0, 40.0)
        } else {
            -90.0
        };
        let sm_in = if pk_in > prev_in {
            pk_in
        } else {
            prev_in * 0.9995 + pk_in * 0.0005
        };
        let sm_out = if pk_out > prev_out {
            pk_out
        } else {
            prev_out * 0.9995 + pk_out * 0.0005
        };
        self.peak_in_db.store(sm_in.to_bits(), Ordering::Relaxed);
        self.peak_out_db.store(sm_out.to_bits(), Ordering::Relaxed);

        let raw_hold = f32::from_bits(self.peak_hold_out_db.load(Ordering::Relaxed));
        let prev_hold = if raw_hold.is_finite() {
            raw_hold.clamp(-100.0, 40.0)
        } else {
            -90.0
        };
        if pk_out > prev_hold {
            self.peak_hold_out_db
                .store(pk_out.to_bits(), Ordering::Relaxed);
        }
        self.gr_db.store(gr.to_bits(), Ordering::Relaxed);
        // Push one scrolling GR-history point roughly every ~20 ms (~46 Hz at 48 kHz),
        // unless the user has frozen the spectrum & GR display.
        if !self.spectrum_frozen.load(Ordering::Acquire) {
            let decim = self.gr_history_decim.fetch_add(1, Ordering::Relaxed);
            if decim & 1023 == 0 {
                self.push_gr_history(gr.min(0.0));
            }
        }
    }

    #[inline]
    pub fn update_spectrum_and_gr(&self, spec_32: &[f32; 32], gr_32: &[f32; 32]) {
        if self.spectrum_frozen.load(Ordering::Relaxed) {
            return;
        }
        for i in 0..32 {
            let s = if spec_32[i].is_finite() {
                spec_32[i].clamp(-90.0, 6.0)
            } else {
                -90.0
            };
            let g = if gr_32[i].is_finite() {
                gr_32[i].clamp(-24.0, 24.0)
            } else {
                0.0
            };
            self.spectrum_db[i].store(s.to_bits(), Ordering::Relaxed);
            self.band_gr_db[i].store(g.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Maps a dBFS level (`-60..=0`) to a color-coded studio meter fill color.
fn meter_color_for_db(db: f32) -> egui::Color32 {
    if db >= -0.5 {
        egui::Color32::from_rgb(248, 113, 113) // Red clip zone
    } else if db >= -6.0 {
        egui::Color32::from_rgb(250, 204, 21) // Amber hot zone
    } else {
        egui::Color32::from_rgb(74, 222, 128) // Green nominal zone
    }
}

/// Draws a compact horizontal studio level bar (`norm` in `0.0..=1.0`).
fn draw_compact_meter_bar(ui: &mut egui::Ui, label: &str, norm: f32, fill: egui::Color32) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .monospace()
                .small()
                .color(egui::Color32::from_rgb(165, 175, 195)),
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(72.0, 8.0), egui::Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 3.0, egui::Color32::from_rgb(24, 30, 42));
        let fill_w = (rect.width() * norm.clamp(0.0, 1.0)).max(0.0);
        if fill_w > 0.5 {
            let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height()));
            p.rect_filled(fill_rect, 3.0, fill);
        }
    });
}

/// Renders the shared ForgeAudio dark-mode studio header, clickable Peak-Hold / Clip indicator,
/// visual I/O + Gain Reduction bars, and 32-band Real-Time Spectrum analyzer with frequency/dB
/// graticule, distinct shared-bus masking overlay, and Dynamic Gain-Reduction / Bode curve.
pub(crate) fn draw_forge_studio_header_and_spectrum(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    telemetry: &GuiTelemetry,
    shared_slot: Option<usize>,
) {
    let pk_in = f32::from_bits(telemetry.peak_in_db.load(Ordering::Relaxed));
    let pk_out = f32::from_bits(telemetry.peak_out_db.load(Ordering::Relaxed));
    let pk_hold = f32::from_bits(telemetry.peak_hold_out_db.load(Ordering::Relaxed));
    let gr = f32::from_bits(telemetry.gr_db.load(Ordering::Relaxed));

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.heading(
                egui::RichText::new(title)
                    .color(egui::Color32::from_rgb(90, 210, 255))
                    .strong(),
            );
            ui.label(
                egui::RichText::new(subtitle)
                    .color(egui::Color32::from_rgb(165, 175, 195))
                    .small(),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let clipped = pk_hold >= -0.1;
            let hold_color = if clipped {
                egui::Color32::from_rgb(248, 113, 113)
            } else {
                egui::Color32::from_rgb(250, 204, 21)
            };
            let hold_text = if clipped {
                format!("CLIP {:+4.1} dB", pk_hold)
            } else {
                format!("PK {:+5.1} dB", pk_hold)
            };
            if ui
                .button(egui::RichText::new(hold_text).monospace().color(hold_color))
                .on_hover_text("Click to reset latched output peak-hold")
                .clicked()
            {
                telemetry.reset_peak_hold();
            }
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "IN {:+5.1}  |  OUT {:+5.1}  |  GR {:5.1} dB",
                    pk_in, pk_out, gr
                ))
                .monospace()
                .color(egui::Color32::from_rgb(120, 240, 170)),
            );
        });
    });

    // Visual I/O & Gain Reduction strip + Freeze control
    ui.horizontal(|ui| {
        let norm_in = ((pk_in + 60.0) / 60.0).clamp(0.0, 1.0);
        let norm_out = ((pk_out + 60.0) / 60.0).clamp(0.0, 1.0);
        let norm_gr = (gr.abs() / 18.0).clamp(0.0, 1.0);
        draw_compact_meter_bar(ui, "IN", norm_in, meter_color_for_db(pk_in));
        ui.add_space(10.0);
        draw_compact_meter_bar(ui, "OUT", norm_out, meter_color_for_db(pk_out));
        ui.add_space(10.0);
        draw_compact_meter_bar(ui, "GR", norm_gr, egui::Color32::from_rgb(251, 146, 60));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let frozen = telemetry.spectrum_frozen.load(Ordering::Acquire);
            let freeze_color = if frozen {
                egui::Color32::from_rgb(56, 189, 248)
            } else {
                egui::Color32::from_rgb(148, 163, 184)
            };
            if ui
                .button(
                    egui::RichText::new(if frozen { "FROZEN" } else { "FREEZE" })
                        .monospace()
                        .small()
                        .color(freeze_color),
                )
                .on_hover_text("Freeze/unfreeze the live spectrum, shared-bus ghost & GR history")
                .clicked()
            {
                telemetry.toggle_spectrum_freeze(shared_slot);
            }

            let lufs_m = f32::from_bits(telemetry.lufs_momentary.load(Ordering::Relaxed));
            let lufs_s = f32::from_bits(telemetry.lufs_short_term.load(Ordering::Relaxed));
            let tp_dbtp = f32::from_bits(telemetry.true_peak_dbtp.load(Ordering::Relaxed));
            if lufs_m > UNSET_LOUDNESS_DB + 1.0
                && lufs_s > UNSET_LOUDNESS_DB + 1.0
                && tp_dbtp > UNSET_LOUDNESS_DB + 1.0
            {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!(
                        "M {:+5.1} | S {:+5.1} LUFS | TP {:+5.1} dBTP",
                        lufs_m, lufs_s, tp_dbtp
                    ))
                    .monospace()
                    .small()
                    .color(egui::Color32::from_rgb(192, 132, 252)),
                );
            }
        });
    });
    ui.add_space(4.0);

    // 32-Band Real-Time Spectral + Shared-Bus Ghost Overlay + Dynamic GR / Bode Curve Canvas
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 104.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(16, 20, 28));
    painter.rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(42, 54, 72)),
        egui::StrokeKind::Inside,
    );

    let plot_top = rect.top() + 6.0;
    let plot_bottom = rect.bottom() - 16.0;
    let plot_h = (plot_bottom - plot_top).max(20.0);
    let mid_y = plot_top + plot_h * 0.5;

    // Subtle horizontal dB graticule (-18, -36, -54 dB) + 0 dB Bode centerline
    let grid_color = egui::Color32::from_rgba_unmultiplied(100, 116, 139, 45);
    for frac in [0.25_f32, 0.50, 0.75] {
        let y = plot_top + plot_h * frac;
        painter.line_segment(
            [
                egui::pos2(rect.left() + 6.0, y),
                egui::pos2(rect.right() - 6.0, y),
            ],
            egui::Stroke::new(1.0_f32, grid_color),
        );
    }

    let frozen = telemetry.spectrum_frozen.load(Ordering::Acquire);
    let shared_vals = shared_slot.map(|slot| {
        if frozen {
            std::array::from_fn(|i| {
                f32::from_bits(telemetry.frozen_shared_db[i].load(Ordering::Relaxed))
            })
        } else {
            read_shared_spectrum(slot)
        }
    });
    let n_bars = 32_usize;
    let bar_w = (rect.width() - 16.0) / (n_bars as f32);
    let mut curve_pts = Vec::with_capacity(n_bars);

    for b in 0..n_bars {
        let x0 = rect.left() + 8.0 + (b as f32) * bar_w;

        // Shared-bus reference ghost spectrum (translucent magenta for masking visibility)
        if let Some(ref sv) = shared_vals {
            let shared_norm = ((sv[b] + 72.0) / 72.0).clamp(0.0, 1.0);
            if shared_norm > 0.02 {
                let sh = shared_norm * plot_h;
                let ghost_rect = egui::Rect::from_min_max(
                    egui::pos2(x0 + 1.0, plot_bottom - sh),
                    egui::pos2(x0 + bar_w - 2.0, plot_bottom),
                );
                painter.rect_filled(
                    ghost_rect,
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(232, 121, 249, 85),
                );
            }
        }

        // Local 32-band spectrum bar (cyan)
        let local_db = f32::from_bits(telemetry.spectrum_db[b].load(Ordering::Relaxed));
        let norm = ((local_db + 72.0) / 72.0).clamp(0.03, 1.0);
        let h = norm * plot_h;
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(x0 + 1.5, plot_bottom - h),
            egui::pos2(x0 + bar_w - 2.5, plot_bottom),
        );
        painter.rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(56, 189, 248));

        // 32-band Dynamic Gain Reduction / EQ Bode response curve (-18 dB .. +18 dB)
        let gr_b =
            f32::from_bits(telemetry.band_gr_db[b].load(Ordering::Relaxed)).clamp(-18.0, 18.0);
        let y_pt =
            (mid_y - (gr_b / 18.0) * (plot_h * 0.44)).clamp(plot_top + 2.0, plot_bottom - 2.0);
        curve_pts.push(egui::pos2(x0 + bar_w * 0.5, y_pt));
    }

    if curve_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            curve_pts,
            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(251, 146, 60)),
        ));
    }

    // Scrolling Gain-Reduction history ribbon (Pro-L 2 / soothe2 style) anchored from plot_top
    let gr_hist = telemetry.read_gr_history_ordered();
    if gr_hist.iter().any(|&v| v < -0.1) {
        let step_x = (rect.width() - 16.0) / ((GR_HISTORY_LEN - 1) as f32);
        let mut hist_pts = Vec::with_capacity(GR_HISTORY_LEN);
        for (i, &gr_v) in gr_hist.iter().enumerate() {
            let x = rect.left() + 8.0 + (i as f32) * step_x;
            let drop_h = (gr_v.abs() / 18.0).clamp(0.0, 1.0) * (plot_h * 0.45);
            hist_pts.push(egui::pos2(x, plot_top + drop_h));
        }
        painter.add(egui::Shape::line(
            hist_pts,
            egui::Stroke::new(
                1.5_f32,
                egui::Color32::from_rgba_unmultiplied(248, 113, 113, 195),
            ),
        ));
    }

    // Frequency axis labels placed at exact log-frequency fractions matching the 32-band analyzer
    // (25 Hz .. 18.5 kHz).
    let axis_color = egui::Color32::from_rgb(120, 134, 155);
    let log_span = (18_500.0_f32 / 25.0_f32).ln();
    for &(freq_hz, label) in &[
        (50.0_f32, "50Hz"),
        (200.0, "200Hz"),
        (1_000.0, "1kHz"),
        (5_000.0, "5kHz"),
        (16_000.0, "16kHz"),
    ] {
        let frac = ((freq_hz / 25.0_f32).ln() / log_span).clamp(0.0, 1.0);
        let x = rect.left() + 8.0 + frac * (rect.width() - 16.0);
        painter.text(
            egui::pos2(x, rect.bottom() - 8.0),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::monospace(9.5),
            axis_color,
        );
    }

    ui.add_space(6.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_tracks_peak_hold_and_resets_cleanly() {
        let tel = GuiTelemetry::default();
        tel.update_peaks(0.5, 0.25, 0.8, 0.2, -3.5);
        let hold_before = f32::from_bits(tel.peak_hold_out_db.load(Ordering::Relaxed));
        assert!(
            (hold_before - 20.0 * 0.8_f32.log10()).abs() < 0.05,
            "expected peak hold near -1.94 dB, got {hold_before}"
        );

        // Quieter sample must not drop latched peak hold
        tel.update_peaks(0.01, 0.01, 0.01, 0.01, 0.0);
        let hold_after_quiet = f32::from_bits(tel.peak_hold_out_db.load(Ordering::Relaxed));
        assert_eq!(hold_before, hold_after_quiet);

        // Reset returns peak hold to -90 dB
        tel.reset_peak_hold();
        let hold_reset = f32::from_bits(tel.peak_hold_out_db.load(Ordering::Relaxed));
        assert!((hold_reset - (-90.0)).abs() < 1e-6);
    }

    #[test]
    fn telemetry_sanitizes_non_finite_peaks_and_spectrum() {
        let tel = GuiTelemetry::default();
        tel.update_peaks(
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::NAN,
        );
        assert!(f32::from_bits(tel.peak_in_db.load(Ordering::Relaxed)).is_finite());
        assert!(f32::from_bits(tel.peak_out_db.load(Ordering::Relaxed)).is_finite());
        assert!(f32::from_bits(tel.peak_hold_out_db.load(Ordering::Relaxed)).is_finite());
        assert!(f32::from_bits(tel.gr_db.load(Ordering::Relaxed)).is_finite());

        let bad_spec = [f32::NAN; 32];
        let bad_gr = [f32::INFINITY; 32];
        tel.update_spectrum_and_gr(&bad_spec, &bad_gr);
        for i in 0..32 {
            assert_eq!(
                f32::from_bits(tel.spectrum_db[i].load(Ordering::Relaxed)),
                -90.0
            );
            assert_eq!(
                f32::from_bits(tel.band_gr_db[i].load(Ordering::Relaxed)),
                0.0
            );
        }
    }

    #[test]
    fn gr_history_ring_orders_oldest_to_newest_and_sanitizes() {
        let tel = GuiTelemetry::default();
        for i in 0..(GR_HISTORY_LEN + 10) {
            tel.push_gr_history(-(i as f32) * 0.25);
        }
        tel.push_gr_history(f32::NAN);
        let hist = tel.read_gr_history_ordered();
        assert_eq!(hist.len(), GR_HISTORY_LEN);
        assert!(hist.iter().all(|v| v.is_finite() && *v <= 0.0));
        // Most recent sample was NaN -> sanitized to 0.0
        assert_eq!(hist[GR_HISTORY_LEN - 1], 0.0);
        // Preceding sample was -(73 * 0.25) = -18.25 dB
        assert!((hist[GR_HISTORY_LEN - 2] - (-18.25)).abs() < 1e-4);
    }

    #[test]
    fn spectrum_freeze_holds_spectrum_and_gr_history() {
        let tel = GuiTelemetry::default();
        let initial = [-24.0_f32; 32];
        let initial_gr = [-3.0_f32; 32];
        tel.update_spectrum_and_gr(&initial, &initial_gr);
        tel.push_gr_history(-4.5);

        assert!(tel.toggle_spectrum_freeze(None));
        tel.update_spectrum_and_gr(&[-6.0_f32; 32], &[-12.0_f32; 32]);
        for _ in 0..2048 {
            tel.update_peaks(0.5, 0.5, 0.5, 0.5, -15.0);
        }
        // Frozen snapshot must remain untouched
        assert_eq!(
            f32::from_bits(tel.spectrum_db[0].load(Ordering::Relaxed)),
            -24.0
        );
        assert_eq!(
            f32::from_bits(tel.band_gr_db[0].load(Ordering::Relaxed)),
            -3.0
        );
        assert_eq!(tel.read_gr_history_ordered()[GR_HISTORY_LEN - 1], -4.5);

        assert!(!tel.toggle_spectrum_freeze(None));
        tel.update_spectrum_and_gr(&[-6.0_f32; 32], &[-12.0_f32; 32]);
        assert_eq!(
            f32::from_bits(tel.spectrum_db[0].load(Ordering::Relaxed)),
            -6.0
        );
    }

    #[test]
    fn loudness_and_tp_telemetry_updates_and_preserves_unset_sentinel() {
        let tel = GuiTelemetry::default();
        tel.update_loudness_and_tp(-14.2, -15.6, -1.05);
        assert!(
            (f32::from_bits(tel.lufs_momentary.load(Ordering::Relaxed)) - (-14.2)).abs() < 1e-4
        );
        assert!(
            (f32::from_bits(tel.lufs_short_term.load(Ordering::Relaxed)) - (-15.6)).abs() < 1e-4
        );
        assert!(
            (f32::from_bits(tel.true_peak_dbtp.load(Ordering::Relaxed)) - (-1.05)).abs() < 1e-4
        );

        // Non-finite telemetry must restore UNSET_LOUDNESS_DB, never bogus -90.0 dB.
        tel.update_loudness_and_tp(f32::NAN, f32::NEG_INFINITY, f32::INFINITY);
        assert_eq!(
            f32::from_bits(tel.lufs_momentary.load(Ordering::Relaxed)),
            UNSET_LOUDNESS_DB
        );
        assert_eq!(
            f32::from_bits(tel.lufs_short_term.load(Ordering::Relaxed)),
            UNSET_LOUDNESS_DB
        );
        assert_eq!(
            f32::from_bits(tel.true_peak_dbtp.load(Ordering::Relaxed)),
            UNSET_LOUDNESS_DB
        );
    }

    #[test]
    fn output_loudness_meter_tracks_400ms_momentary_3s_short_term_and_true_peak() {
        let sr = 48_000.0;
        let mut meter = OutputLoudnessMeter::new(sr);
        // Feed 1.5 s of 1 kHz stereo sine at -12 dBFS peak (~0.2512)
        let amp = 10.0_f64.powf(-12.0 / 20.0);
        for i in 0..(sr as usize * 3 / 2) {
            let s = (amp * (2.0 * std::f64::consts::PI * 1000.0 * (i as f64) / sr).sin()) as f32;
            meter.push_output_sample(s, s);
        }
        let (m1, s1, tp1) = meter.snapshot();
        assert!(m1 > -16.0 && m1 < -8.0, "momentary LUFS out of range: {m1}");
        assert!(
            s1 > -16.0 && s1 < -8.0,
            "short-term LUFS out of range: {s1}"
        );
        assert!(
            (tp1 - (-12.0)).abs() < 0.5,
            "true peak should track -12 dBTP, got {tp1}"
        );

        // Feed 0.5 s of silence: Momentary (400 ms) must drop to floor while Short-Term (3 s) retains energy.
        for _ in 0..(sr as usize / 2) {
            meter.push_output_sample(0.0, 0.0);
        }
        let (m2, s2, _) = meter.snapshot();
        assert!(
            m2 < -60.0,
            "400 ms momentary should drop after 500 ms silence, got {m2}"
        );
        assert!(
            s2 > -20.0,
            "3 s short-term should still hold recent 1.5 s tone, got {s2}"
        );
    }

    #[test]
    fn ab_bank_state_round_trips_and_copies_in_both_directions() {
        use nih_plug::prelude::{FloatRange, IntRange};

        let p_float = FloatParam::new("Gain", 0.25, FloatRange::Linear { min: 0.0, max: 1.0 });
        let p_int = IntParam::new("Mode", 1, IntRange::Linear { min: 0, max: 4 });
        let p_bool = BoolParam::new("Bypass", false);
        let params = [
            ("gain", ParamRef::Float(&p_float)),
            ("mode", ParamRef::Int(&p_int)),
            ("bypass", ParamRef::Bool(&p_bool)),
        ];

        let mut ab = AbBankState::default();
        ab.ensure_initialized(&params);
        assert_eq!(ab.active_slot, 0);
        assert_eq!(ab.slot_a.len(), 3);
        assert_eq!(ab.slot_b.len(), 3);

        // Switch A -> B: records applied values for all 3 param types.
        ab.slot_b = vec![
            ("gain".to_string(), 0.75),
            ("mode".to_string(), 1.0),
            ("bypass".to_string(), 1.0),
        ];
        let mut applied = Vec::new();
        ab.switch_with(1, &params, |_pref, v| applied.push(v));
        assert_eq!(ab.active_slot, 1);
        assert_eq!(applied, vec![0.75, 1.0, 1.0]);

        // Copy B -> A while on slot B: overwrites slot_a with current live values.
        ab.copy_active_to_other(&params);
        assert_eq!(ab.slot_a, ab.slot_b);

        // Switch B -> A: recalls slot_a cleanly.
        let mut applied_back = Vec::new();
        ab.switch_with(0, &params, |_pref, v| applied_back.push(v));
        assert_eq!(ab.active_slot, 0);
        assert_eq!(applied_back.len(), 3);

        // Copy A -> B while on slot A: keeps both slots synchronized.
        ab.copy_active_to_other(&params);
        assert_eq!(ab.slot_a, ab.slot_b);
    }
}
