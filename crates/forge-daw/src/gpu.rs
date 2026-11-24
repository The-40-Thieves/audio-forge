//! Native GPU renderer setup (`wgpu` DX12/Metal/Vulkan + `glow` fallback) and studio theme.

use eframe::egui;

pub(crate) fn build_native_options(renderer: eframe::Renderer) -> eframe::NativeOptions {
    let mut wgpu_setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.power_preference = eframe::wgpu::PowerPreference::HighPerformance;
    wgpu_setup.instance_descriptor.backends = eframe::wgpu::Backends::PRIMARY;

    let wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        surface: eframe::egui_wgpu::SurfaceConfig {
            present_mode: eframe::wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: Some(2),
        },
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
        ..Default::default()
    };

    let glow_options = eframe::egui_glow::GlowConfiguration {
        vsync: true,
        hardware_acceleration: eframe::egui_glow::HardwareAcceleration::Preferred,
        ..Default::default()
    };

    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 840.0])
            .with_min_inner_size([960.0, 600.0])
            .with_title("ForgeStudio DAW — Object-Oriented Audio & Mastering"),
        renderer,
        glow_options,
        multisampling: 4,
        wgpu_options,
        ..Default::default()
    }
}

pub(crate) fn detect_gpu_label(cc: &eframe::CreationContext<'_>) -> String {
    if let Some(ref rs) = cc.wgpu_render_state {
        let info = rs.adapter.get_info();
        return format!(
            "{} [{:?} | {:?}]",
            info.name, info.backend, info.device_type
        );
    }
    "OpenGL (glow fallback)".to_string()
}

pub(crate) fn configure_studio_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgb(15, 18, 25);
    visuals.panel_fill = egui::Color32::from_rgb(18, 22, 31);
    visuals.faint_bg_color = egui::Color32::from_rgb(24, 30, 42);
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(26, 32, 44);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(33, 41, 56);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(45, 58, 79);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(56, 189, 248);
    ctx.set_visuals(visuals);
}
