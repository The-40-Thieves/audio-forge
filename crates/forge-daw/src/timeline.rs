//! Arrange Timeline canvas: ruler, visual loop brace, audio object waveforms/fades/drag,
//! movable/trimmable MIDI clips, and interactive Volume/Pan automation lanes.

use eframe::egui;
use forge_engine::{AudioObject, DawProject};

use crate::{DragMode, ForgeDawApp, InspectorTab};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DragTarget {
    AudioObject(u64),
    MidiClip(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum AutoLaneMode {
    #[default]
    Volume,
    Pan,
}

fn draw_loop_brace_band(
    p: &egui::Painter,
    rect: egui::Rect,
    proj: &DawProject,
    scroll_secs: f32,
    pps: f32,
    sr: f64,
    is_ruler: bool,
) {
    if !proj.loop_enabled || proj.loop_end_samples <= proj.loop_start_samples {
        return;
    }
    let lx0 = rect.left() + (((proj.loop_start_samples as f64 / sr) as f32) - scroll_secs) * pps;
    let lx1 = rect.left() + (((proj.loop_end_samples as f64 / sr) as f32) - scroll_secs) * pps;
    if lx1 <= rect.left() || lx0 >= rect.right() {
        return;
    }
    let clamped = egui::Rect::from_min_max(
        egui::pos2(lx0.max(rect.left()), rect.top()),
        egui::pos2(
            lx1.min(rect.right()),
            if is_ruler {
                rect.top() + 5.0
            } else {
                rect.bottom()
            },
        ),
    );
    let fill = if is_ruler {
        egui::Color32::from_rgba_unmultiplied(56, 189, 248, 180)
    } else {
        egui::Color32::from_rgba_unmultiplied(56, 189, 248, 16)
    };
    p.rect_filled(clamped, 1.0, fill);
}

fn draw_ruler_seconds_and_tempo(
    rp: &egui::Painter,
    ruler_rect: egui::Rect,
    proj: &DawProject,
    scroll_secs: f32,
    pps: f32,
) {
    let start_sec = scroll_secs.floor() as i32;
    let vis_secs = (ruler_rect.width() / pps).ceil() as i32 + 2;
    for s in start_sec..(start_sec + vis_secs) {
        if s < 0 {
            continue;
        }
        let x = ruler_rect.left() + ((s as f32) - scroll_secs) * pps;
        if x >= ruler_rect.left() && x <= ruler_rect.right() {
            rp.line_segment(
                [
                    egui::pos2(x, ruler_rect.top()),
                    egui::pos2(x, ruler_rect.bottom()),
                ],
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(60, 74, 96)),
            );
            rp.text(
                egui::pos2(x + 3.0, ruler_rect.center().y),
                egui::Align2::LEFT_CENTER,
                format!("{s}s"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(165, 175, 195),
            );
        }
    }
    if proj.tempo_map.is_empty() {
        return;
    }
    let vis_end_s = f64::from(scroll_secs) + f64::from(ruler_rect.width() / pps) + 2.0;
    let b_start = proj.secs_to_beat(f64::from(scroll_secs.max(0.0))).floor() as u64;
    let b_end = (proj.secs_to_beat(vis_end_s).ceil() as u64).min(b_start.saturating_add(512));
    let bar_len = u64::from(proj.time_sig_num.max(1));
    for b in b_start..=b_end {
        if !b.is_multiple_of(bar_len) {
            continue;
        }
        let bx = ruler_rect.left() + (proj.beat_to_secs(b as f64) as f32 - scroll_secs) * pps;
        if bx >= ruler_rect.left() && bx <= ruler_rect.right() {
            rp.line_segment(
                [
                    egui::pos2(bx, ruler_rect.center().y),
                    egui::pos2(bx, ruler_rect.bottom()),
                ],
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(250, 204, 21)),
            );
        }
    }
    for tp in proj.tempo_map.iter().take(64) {
        let tx = ruler_rect.left() + (tp.time_secs as f32 - scroll_secs) * pps;
        if tx >= ruler_rect.left() && tx <= ruler_rect.right() {
            rp.text(
                egui::pos2(tx + 2.0, ruler_rect.top() + 2.0),
                egui::Align2::LEFT_TOP,
                format!("{:.0}", tp.bpm),
                egui::FontId::monospace(8.5),
                egui::Color32::from_rgb(250, 204, 21),
            );
        }
    }
}

fn trim_object_left(obj: &mut AudioObject, delta_smp: i64) {
    let max_shift = obj.length_samples.saturating_sub(256).min(i64::MAX as u64) as i64;
    let min_lim = obj
        .source_offset_samples
        .min(obj.timeline_start_samples)
        .min(i64::MAX as u64) as i64;
    let shift = delta_smp.clamp((-min_lim).min(max_shift), max_shift);
    if shift >= 0 {
        let s = shift as u64;
        obj.timeline_start_samples = obj.timeline_start_samples.saturating_add(s);
        obj.source_offset_samples = obj.source_offset_samples.saturating_add(s);
        obj.length_samples = obj.length_samples.saturating_sub(s);
    } else {
        let s = (-shift) as u64;
        obj.timeline_start_samples = obj.timeline_start_samples.saturating_sub(s);
        obj.source_offset_samples = obj.source_offset_samples.saturating_sub(s);
        obj.length_samples = obj.length_samples.saturating_add(s);
    }
}

fn draw_crossfade_overlaps(
    p: &egui::Painter,
    lane_rect: egui::Rect,
    overlaps: &[(u64, u64)],
    scroll_secs: f32,
    pps: f32,
    sr: f64,
) {
    for &(ov_start, ov_end) in overlaps {
        let ox0 = lane_rect.left() + (((ov_start as f64 / sr) as f32) - scroll_secs) * pps;
        let ox1 = lane_rect.left() + (((ov_end as f64 / sr) as f32) - scroll_secs) * pps;
        if ox1 > lane_rect.left() && ox0 < lane_rect.right() {
            let xrect = egui::Rect::from_min_max(
                egui::pos2(ox0.max(lane_rect.left()), lane_rect.top() + 4.0),
                egui::pos2(ox1.min(lane_rect.right()), lane_rect.bottom() - 4.0),
            );
            p.rect_filled(
                xrect,
                2.0,
                egui::Color32::from_rgba_unmultiplied(250, 204, 21, 45),
            );
            let stroke = egui::Stroke::new(1.25_f32, egui::Color32::from_rgb(250, 204, 21));
            p.line_segment([xrect.left_top(), xrect.right_bottom()], stroke);
            p.line_segment([xrect.left_bottom(), xrect.right_top()], stroke);
        }
    }
}

impl ForgeDawApp {
    fn draw_time_ruler(&mut self, ui: &mut egui::Ui, header_w: f32, sr: f64) {
        ui.horizontal(|ui| {
            ui.allocate_exact_size(egui::vec2(header_w, 22.0), egui::Sense::hover());
            let (ruler_rect, ruler_resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(200.0), 22.0),
                egui::Sense::click_and_drag(),
            );
            let rp = ui.painter_at(ruler_rect);
            rp.rect_filled(ruler_rect, 3.0, egui::Color32::from_rgb(22, 28, 40));
            draw_loop_brace_band(
                &rp,
                ruler_rect,
                &self.project,
                self.scroll_secs,
                self.pixels_per_sec,
                sr,
                true,
            );
            if (ruler_resp.clicked() || ruler_resp.dragged())
                && let Some(pos) = ruler_resp.interact_pointer_pos()
            {
                let rel_x = (pos.x - ruler_rect.left()).max(0.0);
                let target_secs = f64::from(self.scroll_secs + rel_x / self.pixels_per_sec);
                self.set_playhead_samples((target_secs * sr).round() as u64);
            }
            draw_ruler_seconds_and_tempo(
                &rp,
                ruler_rect,
                &self.project,
                self.scroll_secs,
                self.pixels_per_sec,
            );
            for mk in &self.project.markers {
                let mk_secs = (mk.sample_pos as f64 / sr) as f32;
                let mx = ruler_rect.left() + (mk_secs - self.scroll_secs) * self.pixels_per_sec;
                if mx >= ruler_rect.left() && mx <= ruler_rect.right() {
                    let flag = egui::Rect::from_min_size(
                        egui::pos2(mx, ruler_rect.top() + 1.0),
                        egui::vec2(38.0, 14.0),
                    );
                    rp.rect_filled(flag, 2.0, egui::Color32::from_rgb(14, 116, 144));
                    rp.text(
                        egui::pos2(mx + 3.0, ruler_rect.top() + 8.0),
                        egui::Align2::LEFT_CENTER,
                        &mk.name,
                        egui::FontId::monospace(9.5),
                        egui::Color32::WHITE,
                    );
                    if ruler_resp.clicked()
                        && let Some(pt) = ruler_resp.interact_pointer_pos()
                        && flag.contains(pt)
                    {
                        self.set_playhead_samples(mk.sample_pos);
                    }
                }
            }
        });
    }

    fn draw_track_header(
        &mut self,
        ui: &mut egui::Ui,
        ti: usize,
        header_w: f32,
        lane_h: f32,
    ) -> bool {
        let mut changed = false;
        let selected_tr = self.selected_track_idx == ti;
        let tr_color = egui::Color32::from_rgb(
            self.project.tracks[ti].color_rgb[0],
            self.project.tracks[ti].color_rgb[1],
            self.project.tracks[ti].color_rgb[2],
        );
        egui::Frame::group(ui.style())
            .fill(if selected_tr {
                egui::Color32::from_rgb(30, 41, 59)
            } else {
                egui::Color32::from_rgb(20, 26, 37)
            })
            .show(ui, |ui| {
                ui.set_min_size(egui::vec2(header_w - 14.0, lane_h - 10.0));
                ui.set_max_size(egui::vec2(header_w - 14.0, lane_h - 10.0));
                ui.vertical(|ui| {
                    if ui
                        .selectable_label(
                            selected_tr,
                            egui::RichText::new(&self.project.tracks[ti].name)
                                .strong()
                                .color(tr_color),
                        )
                        .clicked()
                    {
                        self.selected_track_idx = ti;
                    }
                    ui.horizontal(|ui| {
                        let tr = &mut self.project.tracks[ti];
                        if ui.selectable_label(tr.mute, "M").clicked() {
                            tr.mute = !tr.mute;
                            changed = true;
                        }
                        if ui.selectable_label(tr.solo, "S").clicked() {
                            tr.solo = !tr.solo;
                            changed = true;
                        }
                        if ui
                            .selectable_label(
                                tr.record_armed,
                                egui::RichText::new("R").color(if tr.record_armed {
                                    egui::Color32::from_rgb(248, 113, 113)
                                } else {
                                    egui::Color32::from_rgb(160, 170, 188)
                                }),
                            )
                            .clicked()
                        {
                            tr.record_armed = !tr.record_armed;
                        }
                        if ui
                            .selectable_label(
                                tr.show_automation,
                                egui::RichText::new("A").color(if tr.show_automation {
                                    egui::Color32::from_rgb(250, 204, 21)
                                } else {
                                    egui::Color32::from_rgb(160, 170, 188)
                                }),
                            )
                            .clicked()
                        {
                            tr.show_automation = !tr.show_automation;
                            if tr.show_automation {
                                tr.volume_auto.enabled = true;
                                changed = true;
                            }
                        }
                        if tr.show_automation {
                            let lbl = match self.auto_lane_mode {
                                AutoLaneMode::Volume => "Vol",
                                AutoLaneMode::Pan => "Pan",
                            };
                            if ui.small_button(lbl).clicked() {
                                self.auto_lane_mode = match self.auto_lane_mode {
                                    AutoLaneMode::Volume => AutoLaneMode::Pan,
                                    AutoLaneMode::Pan => AutoLaneMode::Volume,
                                };
                            }
                        }
                    });
                    ui.label(
                        egui::RichText::new(format!("{:+.1} dB", self.project.tracks[ti].gain_db))
                            .monospace()
                            .small(),
                    );
                });
            });
        changed
    }

    fn draw_object_waveform_and_fades(
        &self,
        p: &egui::Painter,
        lane_rect: egui::Rect,
        obj_rect: egui::Rect,
        obj: &AudioObject,
        tr_color: egui::Color32,
        sr: f64,
    ) {
        if let Some(peaks) = self.cache.get_peaks(&obj.source_path) {
            let mid_y = obj_rect.center().y;
            let half_h = (obj_rect.height() * 0.42).max(4.0);
            let step_px = 2.0_f32;
            let n_cols = ((obj_rect.width() / step_px) as usize).max(1);
            let c_start = (((lane_rect.left() - obj_rect.left()) / step_px)
                .floor()
                .max(0.0) as usize)
                .min(n_cols);
            let c_end = (((lane_rect.right() - obj_rect.left()) / step_px)
                .ceil()
                .max(0.0) as usize)
                .min(n_cols);
            let stretch = f64::from(obj.time_stretch.clamp(0.5, 2.0));
            let obj_len = obj.length_samples.max(1) as f64;
            for c in c_start..c_end {
                let frac = (c as f64) / (n_cols as f64);
                let smp = obj
                    .source_offset_samples
                    .saturating_add((frac * obj_len * stretch) as u64);
                if let Some(&(mn, mx)) =
                    peaks.min_max.get((smp as usize) / peaks.samples_per_bucket)
                {
                    let cx = obj_rect.left() + (c as f32) * step_px;
                    p.line_segment(
                        [
                            egui::pos2(cx, mid_y - mx * half_h),
                            egui::pos2(cx, mid_y - mn * half_h),
                        ],
                        egui::Stroke::new(1.0_f32, tr_color),
                    );
                }
            }
        }
        let fade_stroke = egui::Stroke::new(
            1.5_f32,
            egui::Color32::from_rgba_unmultiplied(250, 204, 21, 200),
        );
        let rem_fi = obj
            .fade_in_samples
            .saturating_sub(obj.fade_in_offset_samples);
        if rem_fi > 0 {
            let fi_w = (((rem_fi as f64 / sr) as f32) * self.pixels_per_sec).min(obj_rect.width());
            let start_g = obj.fade_gain_at(0) as f32;
            let y0 = obj_rect.bottom() - start_g * obj_rect.height();
            p.line_segment(
                [
                    egui::pos2(obj_rect.left(), y0),
                    egui::pos2(obj_rect.left() + fi_w, obj_rect.top()),
                ],
                fade_stroke,
            );
        }
        let rem_fo = obj
            .fade_out_samples
            .saturating_sub(obj.fade_out_offset_samples);
        if rem_fo > 0 {
            let fo_w = (((rem_fo as f64 / sr) as f32) * self.pixels_per_sec).min(obj_rect.width());
            let end_g = obj.fade_gain_at(obj.length_samples) as f32;
            let y1 = obj_rect.bottom() - end_g * obj_rect.height();
            p.line_segment(
                [
                    egui::pos2(obj_rect.right() - fo_w, obj_rect.top()),
                    egui::pos2(obj_rect.right(), y1),
                ],
                fade_stroke,
            );
        }
    }

    fn handle_object_drag(
        &mut self,
        lane_resp: &egui::Response,
        ti: usize,
        oi: usize,
        obj_rect: egui::Rect,
        sr: f64,
    ) -> bool {
        let obj_id = self.project.tracks[ti].objects[oi].id;
        if lane_resp.drag_started()
            && !self.project.tracks[ti].show_automation
            && let Some(pt) = lane_resp.interact_pointer_pos()
            && obj_rect.contains(pt)
        {
            self.selected_track_idx = ti;
            self.selected_object_id = Some(obj_id);
            self.selected_midi_clip_id = None;
            let mode = if (pt.x - obj_rect.left()).abs() <= 8.0 {
                DragMode::TrimLeft
            } else if (obj_rect.right() - pt.x).abs() <= 8.0 {
                DragMode::TrimRight
            } else {
                DragMode::Move
            };
            self.active_drag = Some((ti, DragTarget::AudioObject(obj_id), mode));
            self.drag_anchor_samples = self.project.tracks[ti].objects[oi].timeline_start_samples;
            self.drag_anchor_x = pt.x;
        }
        if lane_resp.clicked()
            && !self.project.tracks[ti].show_automation
            && let Some(pt) = lane_resp.interact_pointer_pos()
            && obj_rect.contains(pt)
        {
            self.selected_track_idx = ti;
            self.selected_object_id = Some(obj_id);
            self.selected_midi_clip_id = None;
        }
        let Some((drag_ti, DragTarget::AudioObject(drag_id), drag_mode)) = self.active_drag else {
            return false;
        };
        if drag_ti != ti || drag_id != obj_id || !lane_resp.dragged() {
            return false;
        }
        let dx = lane_resp.drag_delta().x;
        let cum_dx = lane_resp
            .interact_pointer_pos()
            .map_or(dx, |pt| pt.x - self.drag_anchor_x);
        if dx.abs() <= 0.1 && !matches!(drag_mode, DragMode::Move) {
            return false;
        }
        let delta_smp = (f64::from(dx / self.pixels_per_sec) * sr).round() as i64;
        let snapped_move = matches!(drag_mode, DragMode::Move).then(|| {
            let cum_smp = (f64::from(cum_dx / self.pixels_per_sec) * sr).round() as i64;
            let raw = (self.drag_anchor_samples as i64 + cum_smp).max(0) as u64;
            self.snap_grid.snap_samples_for_project(raw, &self.project)
        });
        let obj = &mut self.project.tracks[ti].objects[oi];
        match drag_mode {
            DragMode::Move => {
                if let Some(snapped) = snapped_move
                    && snapped != obj.timeline_start_samples
                {
                    obj.timeline_start_samples = snapped;
                    return true;
                }
                false
            }
            DragMode::TrimLeft => {
                trim_object_left(obj, delta_smp);
                true
            }
            DragMode::TrimRight => {
                obj.length_samples = (obj.length_samples as i64 + delta_smp).max(256) as u64;
                true
            }
        }
    }

    fn draw_track_midi_clips(
        &mut self,
        p: &egui::Painter,
        lane_rect: egui::Rect,
        lane_resp: &egui::Response,
        ti: usize,
        sr: f64,
    ) -> bool {
        let mut changed = false;
        for mi in 0..self.project.tracks[ti].midi_clips.len() {
            let (mc_id, ms, ml) = {
                let mc = &self.project.tracks[ti].midi_clips[mi];
                (
                    mc.id,
                    (mc.timeline_start_samples as f64 / sr) as f32,
                    (mc.length_samples as f64 / sr) as f32,
                )
            };
            let mx0 = lane_rect.left() + (ms - self.scroll_secs) * self.pixels_per_sec;
            let mw = (ml * self.pixels_per_sec).max(20.0);
            let mrect = egui::Rect::from_min_size(
                egui::pos2(mx0, lane_rect.top() + 4.0),
                egui::vec2(mw, lane_rect.height() - 8.0),
            );
            let is_sel = self.selected_midi_clip_id == Some(mc_id);
            p.rect_filled(
                mrect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(168, 85, 247, if is_sel { 90 } else { 55 }),
            );
            p.rect_stroke(
                mrect,
                4.0,
                egui::Stroke::new(
                    if is_sel { 2.0 } else { 1.25 },
                    if is_sel {
                        egui::Color32::from_rgb(250, 204, 21)
                    } else {
                        egui::Color32::from_rgb(192, 132, 252)
                    },
                ),
                egui::StrokeKind::Inside,
            );
            let mc = &self.project.tracks[ti].midi_clips[mi];
            p.text(
                egui::pos2(mrect.left() + 6.0, mrect.top() + 10.0),
                egui::Align2::LEFT_CENTER,
                format!("MIDI: {}", mc.name),
                egui::FontId::proportional(10.5),
                egui::Color32::from_rgb(233, 213, 255),
            );
            for note in &mc.notes {
                let nx0 =
                    mrect.left() + ((note.start_samples as f64 / sr) as f32) * self.pixels_per_sec;
                let nw =
                    (((note.length_samples as f64 / sr) as f32) * self.pixels_per_sec).max(3.0);
                let pitch_norm = ((f32::from(note.pitch) - 44.0) / 32.0).clamp(0.0, 1.0);
                let ny = mrect.bottom() - 6.0 - pitch_norm * (mrect.height() - 22.0);
                if nx0 + nw >= lane_rect.left() && nx0 <= lane_rect.right() {
                    p.rect_filled(
                        egui::Rect::from_min_size(egui::pos2(nx0, ny), egui::vec2(nw, 3.5)),
                        1.0,
                        egui::Color32::from_rgb(216, 180, 254),
                    );
                }
            }
            if lane_resp.drag_started()
                && !self.project.tracks[ti].show_automation
                && let Some(pt) = lane_resp.interact_pointer_pos()
                && mrect.contains(pt)
            {
                self.selected_track_idx = ti;
                self.selected_object_id = None;
                self.selected_midi_clip_id = Some(mc_id);
                self.inspector_tab = InspectorTab::PianoRoll;
                let mode = if (mrect.right() - pt.x).abs() <= 8.0 {
                    DragMode::TrimRight
                } else {
                    DragMode::Move
                };
                self.active_drag = Some((ti, DragTarget::MidiClip(mc_id), mode));
                self.drag_anchor_samples =
                    self.project.tracks[ti].midi_clips[mi].timeline_start_samples;
                self.drag_anchor_x = pt.x;
            }
            if lane_resp.clicked()
                && let Some(pt) = lane_resp.interact_pointer_pos()
                && mrect.contains(pt)
            {
                self.selected_track_idx = ti;
                self.selected_object_id = None;
                self.selected_midi_clip_id = Some(mc_id);
                self.inspector_tab = InspectorTab::PianoRoll;
            }
            if let Some((drag_ti, DragTarget::MidiClip(drag_id), drag_mode)) = self.active_drag
                && drag_ti == ti
                && drag_id == mc_id
                && lane_resp.dragged()
            {
                let dx = lane_resp.drag_delta().x;
                let cum_dx = lane_resp
                    .interact_pointer_pos()
                    .map_or(dx, |pt| pt.x - self.drag_anchor_x);
                if matches!(drag_mode, DragMode::Move) {
                    let cum_smp = (f64::from(cum_dx / self.pixels_per_sec) * sr).round() as i64;
                    let raw = (self.drag_anchor_samples as i64 + cum_smp).max(0) as u64;
                    let snapped = self.snap_grid.snap_samples_for_project(raw, &self.project);
                    let snapped_beat = self.project.secs_to_beat((snapped as f64) / sr);
                    let clip = &mut self.project.tracks[ti].midi_clips[mi];
                    if snapped != clip.timeline_start_samples {
                        clip.timeline_start_samples = snapped;
                        if clip.timeline_start_beats.is_some() {
                            clip.timeline_start_beats = Some(snapped_beat);
                        }
                        changed = true;
                    }
                } else if matches!(drag_mode, DragMode::TrimRight) && dx.abs() > 0.1 {
                    let delta_smp = (f64::from(dx / self.pixels_per_sec) * sr).round() as i64;
                    let clip = &mut self.project.tracks[ti].midi_clips[mi];
                    let new_len = (clip.length_samples as i64 + delta_smp).max(1024) as u64;
                    clip.length_samples = new_len;
                    if clip.length_beats.is_some() {
                        let beat_smp = (60.0 / self.project.bpm.max(20.0)) * sr;
                        clip.length_beats = Some((new_len as f64) / beat_smp.max(1.0));
                    }
                    changed = true;
                }
            }
        }
        changed
    }

    fn draw_track_automation_lane(
        &mut self,
        p: &egui::Painter,
        lane_rect: egui::Rect,
        lane_resp: &egui::Response,
        ti: usize,
        sr: f64,
    ) -> bool {
        if !self.project.tracks[ti].show_automation {
            return false;
        }
        let mut changed = false;
        let is_pan = matches!(self.auto_lane_mode, AutoLaneMode::Pan);
        let (min_v, max_v, stroke_col) = if is_pan {
            (-1.0_f32, 1.0_f32, egui::Color32::from_rgb(56, 189, 248))
        } else {
            (-48.0_f32, 12.0_f32, egui::Color32::from_rgb(250, 204, 21))
        };
        let span = max_v - min_v;
        let val_to_y = |v: f32| -> f32 {
            let norm = ((v.clamp(min_v, max_v) - min_v) / span).clamp(0.0, 1.0);
            lane_rect.bottom() - 6.0 - norm * (lane_rect.height() - 12.0)
        };
        let y_to_val = |y: f32| -> f32 {
            let norm = ((lane_rect.bottom() - 6.0 - y) / (lane_rect.height() - 12.0).max(1.0))
                .clamp(0.0, 1.0);
            norm * span + min_v
        };
        if (lane_resp.clicked() || lane_resp.dragged())
            && let Some(pt) = lane_resp.interact_pointer_pos()
            && lane_rect.contains(pt)
        {
            let secs = ((pt.x - lane_rect.left()) / self.pixels_per_sec) + self.scroll_secs;
            let raw_smp = (f64::from(secs.max(0.0)) * sr).round() as u64;
            let smp = self
                .snap_grid
                .snap_samples_for_project(raw_smp, &self.project);
            let val = y_to_val(pt.y);
            let lane = if is_pan {
                &mut self.project.tracks[ti].pan_auto
            } else {
                &mut self.project.tracks[ti].volume_auto
            };
            lane.enabled = true;
            lane.upsert_point(smp, val);
            changed = true;
        }
        let lane = if is_pan {
            &self.project.tracks[ti].pan_auto
        } else {
            &self.project.tracks[ti].volume_auto
        };
        let screen_pts: Vec<egui::Pos2> = lane
            .points
            .iter()
            .map(|ap| {
                egui::pos2(
                    lane_rect.left()
                        + (((ap.sample_pos as f64 / sr) as f32) - self.scroll_secs)
                            * self.pixels_per_sec,
                    val_to_y(ap.value),
                )
            })
            .collect();
        if screen_pts.len() >= 2 {
            p.add(egui::Shape::line(
                screen_pts.clone(),
                egui::Stroke::new(2.0_f32, stroke_col),
            ));
        }
        for sp in screen_pts {
            if sp.x >= lane_rect.left() && sp.x <= lane_rect.right() {
                p.circle_filled(sp, 4.0, stroke_col);
            }
        }
        changed
    }

    pub(crate) fn draw_arrange_timeline(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("ARRANGE TIMELINE").strong());
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.pixels_per_sec, 30.0..=400.0).text("Zoom (px/s)"),
                );
                let sr = f64::from(self.project.sample_rate.max(1));
                let max_scroll_secs =
                    ((self.project.total_length_samples() as f64 / sr) as f32 + 30.0).max(120.0);
                ui.add(
                    egui::Slider::new(&mut self.scroll_secs, 0.0..=max_scroll_secs)
                        .text("Scroll (s)"),
                );
            });
            ui.separator();
            let sr = f64::from(self.project.sample_rate.max(1));
            let playhead_secs = (self.playhead_samples() as f64 / sr) as f32;
            let (header_w, lane_h) = (175.0_f32, 84.0_f32);
            self.draw_time_ruler(ui, header_w, sr);

            let prev_snapshot = self.project.clone();
            let mut changed = false;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for ti in 0..self.project.tracks.len() {
                    let tr_color = egui::Color32::from_rgb(
                        self.project.tracks[ti].color_rgb[0],
                        self.project.tracks[ti].color_rgb[1],
                        self.project.tracks[ti].color_rgb[2],
                    );
                    ui.horizontal(|ui| {
                        changed |= self.draw_track_header(ui, ti, header_w, lane_h);
                        let (lane_rect, lane_resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().max(200.0), lane_h),
                            egui::Sense::click_and_drag(),
                        );
                        let p = ui.painter_at(lane_rect);
                        p.rect_filled(lane_rect, 4.0, egui::Color32::from_rgb(14, 18, 26));
                        p.rect_stroke(
                            lane_rect,
                            4.0,
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(35, 45, 62)),
                            egui::StrokeKind::Inside,
                        );
                        draw_loop_brace_band(
                            &p,
                            lane_rect,
                            &self.project,
                            self.scroll_secs,
                            self.pixels_per_sec,
                            sr,
                            false,
                        );

                        for oi in 0..self.project.tracks[ti].objects.len() {
                            let (obj_id, start_s, len_s, label) = {
                                let o = &self.project.tracks[ti].objects[oi];
                                let lbl = if o.fx_chain.is_empty() {
                                    o.name.clone()
                                } else {
                                    format!("{}  [FX:{}]", o.name, o.fx_chain.len())
                                };
                                (
                                    o.id,
                                    (o.timeline_start_samples as f64 / sr) as f32,
                                    (o.length_samples as f64 / sr) as f32,
                                    lbl,
                                )
                            };
                            let obj_rect = egui::Rect::from_min_size(
                                egui::pos2(
                                    lane_rect.left()
                                        + (start_s - self.scroll_secs) * self.pixels_per_sec,
                                    lane_rect.top() + 4.0,
                                ),
                                egui::vec2(
                                    (len_s * self.pixels_per_sec).max(16.0),
                                    lane_rect.height() - 8.0,
                                ),
                            );
                            let is_sel = self.selected_object_id == Some(obj_id);
                            p.rect_filled(
                                obj_rect,
                                4.0,
                                egui::Color32::from_rgba_unmultiplied(
                                    tr_color.r(),
                                    tr_color.g(),
                                    tr_color.b(),
                                    if is_sel { 80 } else { 42 },
                                ),
                            );
                            p.rect_stroke(
                                obj_rect,
                                4.0,
                                egui::Stroke::new(
                                    if is_sel { 2.0 } else { 1.0 },
                                    if is_sel {
                                        egui::Color32::from_rgb(250, 204, 21)
                                    } else {
                                        tr_color
                                    },
                                ),
                                egui::StrokeKind::Inside,
                            );
                            self.draw_object_waveform_and_fades(
                                &p,
                                lane_rect,
                                obj_rect,
                                &self.project.tracks[ti].objects[oi],
                                tr_color,
                                sr,
                            );
                            p.text(
                                egui::pos2(obj_rect.left() + 6.0, obj_rect.top() + 10.0),
                                egui::Align2::LEFT_CENTER,
                                label,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            changed |= self.handle_object_drag(&lane_resp, ti, oi, obj_rect, sr);
                        }
                        if lane_resp.drag_stopped() {
                            self.active_drag = None;
                        }
                        changed |= self.draw_track_midi_clips(&p, lane_rect, &lane_resp, ti, sr);
                        let overlaps = self.project.tracks[ti].overlap_regions();
                        draw_crossfade_overlaps(
                            &p,
                            lane_rect,
                            &overlaps,
                            self.scroll_secs,
                            self.pixels_per_sec,
                            sr,
                        );
                        changed |=
                            self.draw_track_automation_lane(&p, lane_rect, &lane_resp, ti, sr);

                        let ph_x = lane_rect.left()
                            + (playhead_secs - self.scroll_secs) * self.pixels_per_sec;
                        if ph_x >= lane_rect.left() && ph_x <= lane_rect.right() {
                            p.line_segment(
                                [
                                    egui::pos2(ph_x, lane_rect.top()),
                                    egui::pos2(ph_x, lane_rect.bottom()),
                                ],
                                egui::Stroke::new(1.75_f32, egui::Color32::from_rgb(248, 113, 113)),
                            );
                        }
                    });
                    ui.add_space(4.0);
                }
            });
            if changed {
                if self.auto_crossfade {
                    for tr in &mut self.project.tracks {
                        tr.auto_crossfade_overlaps();
                    }
                }
                self.commit_project_change(prev_snapshot);
            }
        });
    }
}
