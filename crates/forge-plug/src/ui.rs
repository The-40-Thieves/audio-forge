//! Shared lock-free audio-to-GUI telemetry and dark-mode studio metering/spectrum canvas
//! rendered across all 10 `forge-plug` VST3/CLAP editors.

use std::sync::atomic::{AtomicU32, Ordering};

use forge_dsp::read_shared_spectrum;
use nih_plug_egui::egui;

/// Lock-free 32-band telemetry, dynamic Gain-Reduction / Bode curve, and peak-hold state
/// shared between the real-time audio thread and the 60 FPS `nih_plug_egui` studio GUI.
pub struct GuiTelemetry {
    pub spectrum_db: [AtomicU32; 32],
    pub band_gr_db: [AtomicU32; 32],
    pub peak_in_db: AtomicU32,
    pub peak_out_db: AtomicU32,
    pub peak_hold_out_db: AtomicU32,
    pub gr_db: AtomicU32,
    pub ab_slot: AtomicU32,
}

impl Default for GuiTelemetry {
    fn default() -> Self {
        Self {
            spectrum_db: std::array::from_fn(|_| AtomicU32::new((-90.0_f32).to_bits())),
            band_gr_db: std::array::from_fn(|_| AtomicU32::new(0.0_f32.to_bits())),
            peak_in_db: AtomicU32::new((-90.0_f32).to_bits()),
            peak_out_db: AtomicU32::new((-90.0_f32).to_bits()),
            peak_hold_out_db: AtomicU32::new((-90.0_f32).to_bits()),
            gr_db: AtomicU32::new(0.0_f32.to_bits()),
            ab_slot: AtomicU32::new(0),
        }
    }
}

impl GuiTelemetry {
    /// Clears the latched output peak-hold readout back to `-90.0 dBFS`.
    #[inline]
    pub fn reset_peak_hold(&self) {
        self.peak_hold_out_db
            .store((-90.0_f32).to_bits(), Ordering::Relaxed);
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
    }

    #[inline]
    pub fn update_spectrum_and_gr(&self, spec_32: &[f32; 32], gr_32: &[f32; 32]) {
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

    // Visual I/O & Gain Reduction strip
    ui.horizontal(|ui| {
        let norm_in = ((pk_in + 60.0) / 60.0).clamp(0.0, 1.0);
        let norm_out = ((pk_out + 60.0) / 60.0).clamp(0.0, 1.0);
        let norm_gr = (gr.abs() / 18.0).clamp(0.0, 1.0);
        draw_compact_meter_bar(ui, "IN", norm_in, meter_color_for_db(pk_in));
        ui.add_space(10.0);
        draw_compact_meter_bar(ui, "OUT", norm_out, meter_color_for_db(pk_out));
        ui.add_space(10.0);
        draw_compact_meter_bar(ui, "GR", norm_gr, egui::Color32::from_rgb(251, 146, 60));
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

    let shared_vals = shared_slot.map(read_shared_spectrum);
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

    // Frequency axis labels aligned to log-spaced 32-band analyzer (25 Hz .. 18.5 kHz)
    let axis_color = egui::Color32::from_rgb(120, 134, 155);
    for &(frac, label) in &[
        (0.08_f32, "50Hz"),
        (0.28, "200Hz"),
        (0.52, "1kHz"),
        (0.76, "5kHz"),
        (0.94, "16kHz"),
    ] {
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
}
