//! `forge` — Unified CLI for ForgeAudio (`audio-forge`).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use forge_core::AudioBuffer;
use forge_dsp::{
    DeFizzConfig, DeliveryOptions, DeliveryTarget, DitherMode, GenreTonalTarget, LimiterStyle,
    RestorationConfig, ToneMatch, UnmaskConfig, align_phase_gcc_phat, align_vocal_double_dtw,
    align_vocal_double_dtw_elastic, analyze_beat_grid, apply_18_band_eq, apply_spectral_defizz,
    apply_spectral_unmask, audit_audio, audit_spatial_adm_bwf, clean_vocal_surgical,
    compute_genre_tonal_balance_eq, compute_reference_match_eq, deliver, normalize_and_limit,
    quantize_to_constant_bpm, run_full_restoration_chain,
};
use forge_engine::{ClipCache, DawProject, ObjectOrientedEngine};
use forge_reaper::{
    ReaperProjectBuilder, ReaperTrackSpec, ReleaseMetadata, TempFileGuard, install_jsfx_plugins,
    send_reaper_osc, validate_isrc, write_distrokid_metadata,
};
use forge_theory::{
    GrooveStyle, NoteEvent, VoicingStyle, export_abc_score, extract_drum_midi_triggers,
    parse_abc_score, reharmonize_progression, solve_voice_leading, validate_bpm, write_midi_file,
};
use forge_vocal::{
    VocalScale, generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved,
    track_vocal_pitch,
};
use forge_worker::{
    DrumHit, F0TrackFrame, F0TrackResult, SeparationConfig, denoise_hybrid,
    evaluate_quality_hybrid, generate_music_lyria, index_drum_sample_library,
    neural_restore_hybrid, parse_backend, parse_quality, render_drum_sample_layer,
    resolve_default_local_worker, run_local_f0_track, separate_stems, track_beats_hybrid,
    transcribe_midi_hybrid, vocal_rescue_hybrid,
};

#[derive(Parser)]
#[command(
    name = "forge",
    version,
    about = "ForgeAudio (audio-forge): Open-source Rust Studio DSP, AI De-Fizz, Vocal/MIDI Surgery, Music Theory & REAPER 7.82 Engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform an ITU-R BS.1770-4 Loudness, True-Peak, Phase, Hum, and AI Fizz telemetry audit
    Audit {
        /// Path to audio file (WAV, FLAC, MP3, OGG, M4A)
        input: PathBuf,
        /// Emit raw JSON output
        #[arg(long)]
        json: bool,
    },

    /// Run full corrective studio restoration + OpenSoothe Dynamic Spectral De-Fizz (2-6 kHz & 5-14 kHz)
    Restore {
        /// Input audio file
        input: PathBuf,
        /// Output restored WAV path
        #[arg(short, long)]
        output: PathBuf,
        /// Sensitivity in dB for OpenSoothe spectral resonance suppression (default: 3.5, or adaptive with --ai-mode)
        #[arg(long)]
        defizz_sensitivity: Option<f64>,
        /// Maximum suppression depth in dB (default: 9.0, or adaptive with --ai-mode)
        #[arg(long)]
        defizz_depth: Option<f64>,
        /// Output only the removed harshness/fizz (Delta listen mode)
        #[arg(long)]
        delta_listen: bool,
        /// Spectral sensitivity tilt in dB/octave pivoted at 1 kHz (-6.0 to +6.0)
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true, value_parser = parse_tilt)]
        tilt: f64,
        /// Stereo gain linking (0.0 = dual-mono, 1.0 = linked stereo)
        #[arg(long, default_value_t = 1.0, value_parser = parse_unit_interval)]
        stereo_link: f64,
        /// Mid/Side focus (-1.0 = Mid only, 0.0 = balanced, +1.0 = Side only; default: auto with --ai-mode, else 0.0)
        #[arg(long, allow_negative_numbers = true, value_parser = parse_mid_side)]
        mid_side: Option<f64>,
        /// Fundamental hum frequency (50 or 60 Hz; 0 to disable; default: 60 or auto with --ai-mode)
        #[arg(long, value_parser = parse_hum_hz)]
        hum_hz: Option<f64>,
        /// Airwindows Console8 / ChowTape analog warmth drive (0.0 to 1.0; default: auto with --ai-mode, else 0.15)
        #[arg(long, value_parser = parse_unit_interval)]
        tape_drive: Option<f64>,
        /// Enable steeper OpenSoothe suppression slope (0.95) for stubborn AI codec buzz
        #[arg(long)]
        hard_mode: bool,
        /// HPSS erratic frame shimmer suppression boost (0.0..=1.0)
        #[arg(long, default_value_t = 0.0, value_parser = parse_unit_interval)]
        hpss_shimmer: f64,
        /// Optional gentle ultrasonic LPF cutoff in Hz (e.g. 16000 for AI vocals)
        #[arg(long, value_parser = parse_lpf_hz)]
        ultrasonic_lpf_hz: Option<f64>,
        /// Derive telemetry-driven restoration and de-fizz settings automatically via audit_audio
        #[arg(long)]
        ai_mode: bool,
    },

    /// Carve a dynamic sidechain spectral pocket in a target stem (Trackspacer / Neutron Unmask equivalent)
    Unmask {
        /// Target audio stem (e.g. Instrument Bus or Sub-Bass)
        target: PathBuf,
        /// Priority sidechain audio stem (e.g. Lead Vocal or Kick Drum)
        #[arg(short, long)]
        sidechain: PathBuf,
        /// Output unmasked WAV path
        #[arg(short, long)]
        output: PathBuf,
        /// Mode: `vocal` (200Hz-7.5kHz) or `kick` (30Hz-130Hz)
        #[arg(long, default_value = "vocal")]
        mode: String,
        /// Solo only the carved frequencies removed by the sidechain (Delta listen)
        #[arg(long = "delta", alias = "delta-listen")]
        delta_listen: bool,
        /// Mid/Side carving focus (-1.0 = Mid only, 0.0 = balanced, +1.0 = Side only)
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true, value_parser = parse_mid_side)]
        mid_side: f64,
    },

    /// Align multi-mic phase & polarity via GCC-PHAT or vocal doubles via FastDTW stretch warping
    PhaseAlign {
        /// Reference microphone track (e.g. Snare Top or Bass DI)
        reference: PathBuf,
        /// Target microphone track to align (e.g. Overhead or Bass Amp)
        target: PathBuf,
        /// Output phase-aligned WAV path
        #[arg(short, long)]
        output: PathBuf,
        /// Align vocal double timing via FastDTW (VocAlign equivalent) instead of GCC-PHAT
        #[arg(long)]
        vocal_dtw: bool,
        /// Use WSOLA pitch-preserving elastic stretch warping when --vocal-dtw is active
        #[arg(long)]
        elastic: bool,
    },

    /// Surgical vocal cleanup, scale-locked formant-preserving pitch correction, harmonies & FastDTW VocAlign
    Vocal {
        /// Input vocal WAV file
        input: PathBuf,
        /// Output directory for processed vocal & harmony tracks
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Musical scale for pitch correction & diatonic harmonies (e.g. "D minor", "C major")
        #[arg(long, default_value = "D minor")]
        scale: String,
        /// Retune strength (0.0 = natural, 1.0 = hard snap)
        #[arg(long, default_value_t = 0.85, value_parser = parse_unit_interval)]
        retune: f64,
        /// Generate 3-part chord-aware vocal harmony stack (3rd up, 3rd down, formant double)
        #[arg(long)]
        harmonize: bool,
        /// Optional guide vocal path to time-align against via FastDTW (VocAlign equivalent)
        #[arg(long)]
        align_to: Option<PathBuf>,
    },

    /// Music theory voice-leading solver, modal/Neo-Soul reharmonizer, groove humanizer & MIDI/ABC compiler
    Theory {
        /// Space-separated chord progression (e.g. "Dm7 G7 Cmaj7 A7")
        #[arg(long, default_value = "Dm7 G7 Cmaj7 A7")]
        chords: String,
        /// Voicing style: `neo-soul`, `drop2`, `close`, `cinematic`
        #[arg(long, default_value = "neo-soul")]
        style: String,
        /// Groove micro-timing feel: `dilla`, `mpc58`, `strum`, `straight`
        #[arg(long, default_value = "dilla")]
        groove: String,
        /// Tempo in BPM (30 to 300)
        #[arg(long, default_value_t = 92.0, value_parser = parse_bpm)]
        bpm: f64,
        /// Output multi-track MIDI file (.mid)
        #[arg(short, long)]
        output_midi: PathBuf,
        /// Optional output ABC notation score file (.abc)
        #[arg(long)]
        output_abc: Option<PathBuf>,
        /// Optional input ABC notation score file (.abc) to compile alongside voiced chords
        #[arg(long)]
        abc_input: Option<PathBuf>,
    },

    /// Deliver a mix to a platform target: repair, de-fizz, master, then verify with pass/fail checks
    Deliver {
        /// Input mix or stem WAV
        input: PathBuf,
        /// Output master WAV path (a `.delivery.json` report is written next to it)
        #[arg(short, long)]
        output: PathBuf,
        /// Target: spotify, apple_music, youtube, amazon_music, tidal, deezer, podcast, ebu_r128, atsc_a85
        #[arg(short, long, default_value = "spotify", conflicts_with = "lufs")]
        target: String,
        /// Custom integrated loudness in LUFS (requires --ceiling-dbtp)
        #[arg(long, requires = "ceiling_dbtp", allow_negative_numbers = true, value_parser = parse_target_lufs)]
        lufs: Option<f64>,
        /// Custom true-peak ceiling in dBTP (requires --lufs)
        #[arg(long, requires = "lufs", allow_negative_numbers = true, value_parser = parse_ceiling_dbtp)]
        ceiling_dbtp: Option<f64>,
        /// Skip DC removal, de-clip and hum notch
        #[arg(long)]
        no_repair: bool,
        /// Skip spectral de-fizz
        #[arg(long)]
        no_defizz: bool,
        /// Apply vocal clean-up (vocal stems only)
        #[arg(long)]
        vocal: bool,
        /// Reference track WAV for 18-band tone match
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Genre tonal-balance target (`pop`, `hiphop`, `neosoul`, `edm`, `rock`, `jazz`)
        #[arg(long, conflicts_with = "reference")]
        genre: Option<String>,
        /// Limiter topology (`transparent`, `punchy`, `dynamic`, `aggressive`)
        #[arg(long, default_value = "transparent")]
        limiter_style: String,
        /// Stereo channel link (0.0 = dual-mono, 1.0 = linked stereo)
        #[arg(long, default_value_t = 1.0, value_parser = parse_unit_interval)]
        channel_link: f64,
        /// Mastering output dither (`none`, `tpdf`, `noise-shaped`); omit for default 24-bit WAV TPDF
        #[arg(long)]
        dither: Option<String>,
        /// Bit depth for dithering (8..=24, typically 16 or 24)
        #[arg(long, default_value_t = 24, value_parser = parse_dither_bits)]
        dither_bits: u16,
        /// Treat input as AI-generated audio (telemetry-driven restoration/de-fizz, avoids artificial high widening)
        #[arg(long)]
        ai_track: bool,
        /// Override HPSS erratic shimmer suppression ratio (0.0..=1.0) on AI/harmonic material
        #[arg(long, value_parser = parse_unit_interval)]
        hpss_shimmer: Option<f64>,
    },

    /// Master audio to ITU-R BS.1770-4 LUFS target, -1.0 dBTP True-Peak ceiling, Reference Match EQ & ISO 3901 ISRC
    Master {
        /// Input mix WAV file
        input: PathBuf,
        /// Output mastered WAV path
        #[arg(short, long)]
        output: PathBuf,
        /// Target Integrated LUFS (-14.0 for streaming, -9.0 for competitive club/pop)
        #[arg(long, default_value_t = -14.0, allow_negative_numbers = true, value_parser = parse_target_lufs)]
        target_lufs: f64,
        /// True-Peak brickwall ceiling in dBTP, -24.0 to 0.0 (-1.0 recommended by DistroKid/Spotify)
        #[arg(long, default_value_t = -1.0, allow_negative_numbers = true, value_parser = parse_ceiling_dbtp)]
        ceiling_dbtp: f64,
        /// Optional reference track WAV to match spectral balance against
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Optional built-in Tonal Balance genre target (`pop`, `hiphop`, `neosoul`, `edm`, `rock`, `jazz`)
        #[arg(long)]
        genre: Option<String>,
        /// Optional 12-character ISO 3901 ISRC code (e.g. US-S1Z-26-00001)
        #[arg(long)]
        isrc: Option<String>,
        /// Track title for ID3v2.3 / RIFF metadata
        #[arg(long, default_value = "Untitled Track")]
        title: String,
        /// Artist name for ID3v2.3 / RIFF metadata
        #[arg(long, default_value = "Independent Artist")]
        artist: String,
    },

    /// Flagship 1-Command AI/Live Song Ingest → Stem Split, De-Fizz, MIDI Triggers, Master & REAPER 7.82 .RPP Session
    Ingest {
        /// Input song or mix (Suno Pro, YuE2, ACE-Step, Lyria, or live stereo bounce)
        input: PathBuf,
        /// Output project folder for stems, MIDI, master WAV, and REAPER .RPP session
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Project tempo in BPM (30 to 300)
        #[arg(long, default_value_t = 120.0, value_parser = parse_bpm)]
        bpm: f64,
        /// Song key/scale (e.g. "D minor")
        #[arg(long, default_value = "D minor")]
        scale: String,
        /// Chord progression for aligned MIDI chord & sub-bass reinforcement tracks
        #[arg(long, default_value = "Dm9 Bbmaj7 Gm9 A7b13")]
        chords: String,
        /// Optional 12-character ISO 3901 ISRC code
        #[arg(long)]
        isrc: Option<String>,
        /// Fail instead of falling back to the DSP stem split when no GPU or Colab worker
        /// produces stems (same as FORGE_WORKER_REQUIRE_NEURAL=1)
        #[arg(long)]
        require_neural: bool,
        /// Optional directory of `.wav` drum one-shots for phase-locked Kick/Snare/HiHat reinforcement
        #[arg(long)]
        replace_drums: Option<PathBuf>,
        /// Wet mix ratio for sample-layered drum reinforcement (0.0..=1.0)
        #[arg(long, default_value_t = 0.5, value_parser = parse_unit_interval)]
        drum_replace_mix: f64,
        /// WSOLA time-warp separated stems onto a constant `--bpm` grid instead of emitting variable `<TEMPOENVEX>`
        #[arg(long)]
        quantize_grid: bool,
    },

    /// Install native ForgeAudio JSFX plugins into REAPER 7.82 or send live UDP OSC commands
    Reaper {
        /// Install all 9 ForgeAudio JSFX plugins into REAPER Effects directory
        #[arg(long)]
        install_jsfx: bool,
        /// Optional custom target directory for JSFX installation
        #[arg(long)]
        jsfx_dir: Option<PathBuf>,
        /// Optional live OSC address to send to REAPER (e.g. "/action/40044" for Play/Stop)
        #[arg(long)]
        osc_addr: Option<String>,
        /// UDP host:port for REAPER OSC surface
        #[arg(long, default_value = "127.0.0.1:9000")]
        osc_host: String,
    },

    /// Index a local drum sample folder by spectral centroid & sub-120Hz ratio (XO / Atlas 2 equivalent)
    SampleIndex {
        /// Folder containing .wav drum one-shots
        folder: PathBuf,
    },

    /// Generate music via Google DeepMind Lyria 3.5 (`lyria-3.5`) or Lyria 3 Clip (`lyria-3-clip-preview`)
    Lyria {
        /// Musical style and structure prompt
        #[arg(short, long)]
        prompt: String,
        /// Output WAV/MP3 path
        #[arg(short, long)]
        output: PathBuf,
        /// Model ID (`lyria-3.5` or `lyria-3-clip-preview`)
        #[arg(long, default_value = "lyria-3.5")]
        model: String,
    },

    /// Neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer (local Python GPU worker)
    Denoise {
        /// Input audio file
        input: PathBuf,
        /// Output directory for denoised audio
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Backend (`auto`, `deepfilternet` / `deepfilternet3`, or `mel-band-roformer` / `roformer`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Optional attenuation limit in dB (0..=100, DeepFilterNet3)
        #[arg(long)]
        attenuation_limit_db: Option<f64>,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 300, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Polyphonic audio-to-MIDI transcription via basic-pitch (local Python GPU worker)
    Transcribe {
        /// Input audio file
        input: PathBuf,
        /// Output directory for transcribed MIDI
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 300, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Professional studio neural stem separation (BS-RoFormer 1297 / Mel-Band-RoFormer / BS-RoFormer SW cascade, Karaoke Lead/Backing, DrumSep & bandwidth restore)
    Separate {
        /// Input audio file (WAV, FLAC, MP3, OGG, M4A)
        input: PathBuf,
        /// Output directory for separated WAV stems
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Primary stem count: 2 (Vocals/Inst), 4 (Vocals/Drums/Bass/Other), or 6 (+Guitar/Keys)
        #[arg(long, default_value_t = 6)]
        stems: u8,
        /// Separation quality tier: `fast` (1-pass), `studio` (BS-RoFormer 1297 + SW/HTDemucs cascade), or `ultra` (dual-RoFormer ensemble)
        #[arg(long, default_value = "studio")]
        quality: String,
        /// Disable BS-RoFormer de-reverb pass on isolated vocals
        #[arg(long)]
        no_dereverb: bool,
        /// Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals (MDX23C DrumSep)
        #[arg(long)]
        drum_sep: bool,
        /// Also split isolated vocals into Lead Vocals and Backing Vocals / Harmonies (RoFormer Karaoke)
        #[arg(long)]
        split_lead_backing: bool,
        /// Also split isolated vocals into Male and Female vocal stems (BS-RoFormer Male/Female)
        #[arg(long)]
        split_male_female: bool,
        /// Also split isolated guitar into Electric Guitar and Acoustic Guitar
        #[arg(long)]
        split_guitar: bool,
        /// Also extract Strings, Brass, Woodwinds, and Synth from the residual Other stem
        #[arg(long)]
        split_orchestral: bool,
        /// Also extract live Crowd / Audience / Room Noise from the residual Other stem
        #[arg(long)]
        split_crowd: bool,
        /// Enable full 23-slot hierarchical pro studio cascade (all sub-splitters)
        #[arg(long)]
        all_stems: bool,
        /// Run optional Apollo / spectral bandwidth restoration post-pass on separated stems
        #[arg(long)]
        restore: bool,
        /// Run Mel-Band-RoFormer stem bleed suppression pass on separated stems
        #[arg(long)]
        debleed: bool,
        /// Run Aspiration Mel-Band-RoFormer breath/aspiration cleanup on isolated vocals
        #[arg(long)]
        de_aspiration: bool,
        /// Run Mel-Band-RoFormer de-reverb/de-echo pass (alternative to BS-RoFormer deverb)
        #[arg(long)]
        dereverb_echo: bool,
        /// Backend override: `auto`, `local_gpu`, `colab`, or `dsp`
        #[arg(long)]
        backend: Option<String>,
    },

    /// Inspect or headlessly bounce/export stems from a ForgeStudio DAW `.forgeproj` session
    Daw {
        #[command(subcommand)]
        action: DawAction,
    },

    /// Neural bandwidth restoration & AI artifact cleanup (Apollo / FlowHigh / AudioSR / SonicMaster / spectral exciter)
    NeuralRestore {
        /// Input audio file
        input: PathBuf,
        /// Output directory for restored audio
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Backend (`auto`, `apollo`, `flowhigh`, `audiosr`, `sonicmaster`, `exciter`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Optional restoration instruction prompt (used by SonicMaster)
        #[arg(long)]
        prompt: Option<String>,
        /// Chunk size in seconds (0.5..=120.0)
        #[arg(long, value_parser = parse_chunk_secs)]
        chunk_secs: Option<f64>,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 600, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Rescue robotic / phasey AI vocals (Resemble Enhance, Seed-VC zero-shot timbre, or RoFormer chain)
    VocalRescue {
        /// Input vocal stem
        input: PathBuf,
        /// Output directory for rescued vocal
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Backend (`auto`, `resemble-enhance`, `seed-vc`, `roformer-clean`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Optional clean human reference vocal WAV for Seed-VC timbre rescue
        #[arg(long)]
        reference_voice: Option<PathBuf>,
        /// Disable de-reverb/de-echo cleaning step
        #[arg(long)]
        no_dereverb: bool,
        /// Disable vocal aspiration suppression step
        #[arg(long)]
        no_de_aspiration: bool,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 600, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Extract beats, downbeats, median BPM, and tempo-drift map (`beat-this` / `librosa` / Rust onset DP)
    BeatTrack {
        /// Input audio file
        input: PathBuf,
        /// Optional BPM hint (30 to 300) for Rust fallback tracker
        #[arg(long, value_parser = parse_bpm)]
        hint_bpm: Option<f64>,
        /// Backend (`auto`, `beat-this`, `librosa`, `energy`, `rust`, `pure_rust_onset`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 300, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Evaluate perceptual audio quality (Meta Audiobox Aesthetics PQ/PC/CE/CU, SingMOS, or acoustic proxy)
    Evaluate {
        /// Input audio file
        input: PathBuf,
        /// Optional reference track WAV for paired perceptual comparison
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Treat input as vocal stem (prioritizes SingMOS)
        #[arg(long)]
        vocal: bool,
        /// Backend (`auto`, `audiobox`, `singmos`, `acoustic-proxy`, `rust`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 300, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
    },

    /// Track fundamental frequency (`F0`) contour via `RMVPE` / `FCPE` / `autocorr` worker or pure-Rust YIN fallback
    F0Track {
        /// Input audio file
        input: PathBuf,
        /// Backend (`auto`, `rmvpe`, `fcpe`, `autocorr`, `rust`)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Frame hop size in seconds (0.002..=0.25)
        #[arg(long, default_value_t = 0.01, value_parser = parse_hop_secs)]
        hop_secs: f64,
        /// Worker timeout in seconds (1..=3600)
        #[arg(long, default_value_t = 300, value_parser = parse_timeout_secs)]
        timeout_secs: u64,
        /// Emit raw JSON output
        #[arg(long)]
        json: bool,
    },

    /// Audit a multi-channel WAV/RF64 ADM BWF file (ITU-R BS.1770-5 + Dolby Atmos QC)
    SpatialQc {
        /// Path to WAV/RF64 ADM BWF audio file
        input: PathBuf,
        /// Emit raw JSON output
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum DawAction {
    /// Validate and print a summary (tracks, objects, aux buses, PDC, BPM) of a `.forgeproj` file
    Inspect {
        /// Path to `.forgeproj` project file
        project: PathBuf,
        /// Emit raw JSON output
        #[arg(long)]
        json: bool,
    },
    /// Headless 24-bit WAV mastering bounce (with ITU-R BS.1770-4 audit) and optional stem export
    Bounce {
        /// Path to `.forgeproj` project file
        project: PathBuf,
        /// Output 24-bit WAV master bounce path
        #[arg(short, long)]
        output: PathBuf,
        /// Optional output directory for batch per-track stem WAV export
        #[arg(long)]
        stems_dir: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    // Windows defaults to a 1 MiB main-thread stack, which unoptimized debug builds
    // of a multi-subcommand `clap` CLI can exceed during `Cli::parse()`. Run on an 8 MiB stack.
    std::thread::Builder::new()
        .name("forge-cli".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| run_cli(Cli::parse()))?
        .join()
        .map_err(|_| anyhow::anyhow!("forge-cli worker thread panicked"))?
}

fn run_cli(cli: Cli) -> anyhow::Result<()> {
    let mut rem = Some(cli.command);
    for stage in [
        try_dispatch_restore,
        dispatch_mix_cli,
        dispatch_delivery_cli,
        dispatch_studio_cli,
        dispatch_worker_ai,
    ] {
        let Some(cmd) = rem else {
            return Ok(());
        };
        rem = stage(cmd)?;
    }
    if let Some(cmd) = rem {
        dispatch_worker_analysis(cmd)?;
    }
    Ok(())
}

fn try_dispatch_restore(command: Commands) -> anyhow::Result<Option<Commands>> {
    match command {
        Commands::Restore {
            input,
            output,
            defizz_sensitivity,
            defizz_depth,
            delta_listen,
            tilt,
            stereo_link,
            mid_side,
            hum_hz,
            tape_drive,
            hard_mode,
            hpss_shimmer,
            ultrasonic_lpf_hz,
            ai_mode,
        } => {
            run_restore_cli(RestoreCliArgs {
                input,
                output,
                defizz_sensitivity,
                defizz_depth,
                delta_listen,
                tilt,
                stereo_link,
                mid_side,
                hum_hz,
                tape_drive,
                hard_mode,
                hpss_shimmer,
                ultrasonic_lpf_hz,
                ai_mode,
            })?;
            Ok(None)
        }
        other => Ok(Some(other)),
    }
}

fn dispatch_mix_cli(command: Commands) -> anyhow::Result<Option<Commands>> {
    match command {
        Commands::Audit { input, json } => run_audit_cli(&input, json)?,
        Commands::SpatialQc { input, json } => run_spatial_qc_cli(&input, json)?,
        Commands::Unmask {
            target,
            sidechain,
            output,
            mode,
            delta_listen,
            mid_side,
        } => run_unmask_cli(&target, &sidechain, &output, &mode, delta_listen, mid_side)?,
        Commands::PhaseAlign {
            reference,
            target,
            output,
            vocal_dtw,
            elastic,
        } => run_phase_align_cli(&reference, &target, &output, vocal_dtw, elastic)?,
        Commands::Vocal {
            input,
            output_dir,
            scale,
            retune,
            harmonize,
            align_to,
        } => run_vocal_cli(
            &input,
            &output_dir,
            &scale,
            retune,
            harmonize,
            align_to.as_deref(),
        )?,
        Commands::Theory {
            chords,
            style,
            groove,
            bpm,
            output_midi,
            output_abc,
            abc_input,
        } => run_theory_cli(TheoryCliArgs {
            chords,
            style,
            groove,
            bpm,
            output_midi,
            output_abc,
            abc_input,
        })?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}

fn run_spatial_qc_cli(input: &std::path::Path, json: bool) -> anyhow::Result<()> {
    let report = audit_spatial_adm_bwf(input)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "Atmos Spatial QC ({:?}): {:.2} LKFS | {:.2} dBTP | {} Hz / {}-bit | ADM axml={} chna={} | Ready: {}",
        report.layout,
        report.integrated_lkfs,
        report.true_peak_dbtp,
        report.sample_rate,
        report.bits_per_sample,
        report.adm.has_axml,
        report.adm.has_chna,
        if report.atmos_ready { "YES" } else { "NO" }
    );
    Ok(())
}

fn run_audit_cli(input: &std::path::Path, json: bool) -> anyhow::Result<()> {
    let buf = AudioBuffer::load(input)?;
    let report = audit_audio(&buf);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "=== ForgeAudio Studio Telemetry Audit: {} ===",
        input.display()
    );
    println!(
        "Format        : {} Hz | {} ch | {:.2} s",
        report.sample_rate, report.channels, report.duration_secs
    );
    println!("Integrated    : {:.2} LUFS", report.integrated_lufs);
    println!(
        "Short-Term Max: {:.2} LUFS | Momentary Max: {:.2} LUFS",
        report.short_term_max_lufs, report.momentary_max_lufs
    );
    println!(
        "Loudness Range: {:.2} LU   | Crest Factor : {:.2} dB",
        report.loudness_range_lu, report.crest_factor_db
    );
    println!(
        "True Peak     : {:.2} dBTP @ {:.2}s (Sample Peak: {:.2} dBFS)",
        report.true_peak_dbtp, report.true_peak_timestamp_secs, report.sample_peak_dbfs
    );
    println!(
        "Stereo Phase  : {:.2} (Sub-160Hz: {:.2} | 2-12kHz: {:.2})",
        report.stereo_correlation, report.sub_160hz_correlation, report.high_mid_stereo_correlation
    );
    println!(
        "AI Artifacts  : Fizz Index {:.1}/100 (Mid p95: {:.1} dB | Side p95: {:.1} dB)",
        report.ai_fizz_index, report.mid_fizz_p95_db, report.side_fizz_p95_db
    );
    println!(
        "Bandwidth/Snap: Effective BW {:.0} Hz | Transient Sharpness {:.1} dB | DistroKid Ready: {}",
        report.effective_bandwidth_hz,
        report.transient_sharpness_db,
        if report.distrokid_ready { "YES" } else { "NO" }
    );
    for rec in &report.recommendations {
        println!("  • {rec}");
    }
    Ok(())
}

struct RestoreCliArgs {
    input: PathBuf,
    output: PathBuf,
    defizz_sensitivity: Option<f64>,
    defizz_depth: Option<f64>,
    delta_listen: bool,
    tilt: f64,
    stereo_link: f64,
    mid_side: Option<f64>,
    hum_hz: Option<f64>,
    tape_drive: Option<f64>,
    hard_mode: bool,
    hpss_shimmer: f64,
    ultrasonic_lpf_hz: Option<f64>,
    ai_mode: bool,
}

fn build_restoration_cfg(
    a: &RestoreCliArgs,
    pre_audit: &forge_core::AuditReport,
) -> RestorationConfig {
    let mut cfg = if a.ai_mode {
        RestorationConfig::from_audit(pre_audit, true)
    } else {
        RestorationConfig::default()
    };
    cfg.hum_fundamental_hz = match a.hum_hz {
        Some(hz) if hz > 0.0 => Some(hz),
        Some(_) => None,
        None if !a.ai_mode => Some(60.0),
        None => cfg.hum_fundamental_hz,
    };
    if let Some(drive) = a.tape_drive {
        cfg.analog_tape_drive = drive;
    } else if !a.ai_mode {
        cfg.analog_tape_drive = 0.15;
    }
    cfg
}

fn build_defizz_cfg(a: &RestoreCliArgs, stage1: &AudioBuffer) -> DeFizzConfig {
    let mut cfg = if a.ai_mode {
        DeFizzConfig::ai_mix_from_audit(&audit_audio(stage1))
    } else {
        DeFizzConfig::default()
    };
    cfg.sensitivity_db = a.defizz_sensitivity.unwrap_or(cfg.sensitivity_db);
    cfg.max_depth_db = a.defizz_depth.unwrap_or(cfg.max_depth_db);
    cfg.delta_listen = a.delta_listen;
    cfg.tilt_db_per_oct = a.tilt;
    cfg.stereo_link = a.stereo_link;
    cfg.mid_side_balance = a
        .mid_side
        .unwrap_or(if a.ai_mode { cfg.mid_side_balance } else { 0.0 });
    cfg.hard_mode |= a.hard_mode;
    if a.hpss_shimmer > 0.0 {
        cfg.hpss_shimmer_suppress = a.hpss_shimmer;
    }
    if let Some(lpf_hz) = a.ultrasonic_lpf_hz {
        cfg.ultrasonic_lpf_hz = Some(lpf_hz);
    }
    cfg
}

fn run_restore_cli(a: RestoreCliArgs) -> anyhow::Result<()> {
    let buf = AudioBuffer::load(&a.input)?;
    let rest_cfg = build_restoration_cfg(&a, &audit_audio(&buf));
    let stage1 = run_full_restoration_chain(&buf, &rest_cfg)?;
    let defizz_cfg = build_defizz_cfg(&a, &stage1);
    let cleaned = apply_spectral_defizz(&stage1, &defizz_cfg)?;
    cleaned.save_wav(&a.output, 24)?;
    let post_audit = audit_audio(&cleaned);
    println!(
        "Restored audio saved to {} (Post AI Fizz Index: {:.1}/100, Sub-160Hz Correlation: {:.2})",
        a.output.display(),
        post_audit.ai_fizz_index,
        post_audit.sub_160hz_correlation
    );
    Ok(())
}

fn run_unmask_cli(
    target: &std::path::Path,
    sidechain: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
    delta_listen: bool,
    mid_side: f64,
) -> anyhow::Result<()> {
    let t_buf = AudioBuffer::load(target)?;
    let sc_buf = AudioBuffer::load(sidechain)?;
    let valid_mode = validate_choice(mode, "mode", &["vocal", "kick"])?;
    let mut cfg = if valid_mode == "kick" {
        UnmaskConfig::kick_over_bass()
    } else {
        UnmaskConfig::vocal_over_instruments()
    };
    cfg.delta_listen = delta_listen;
    cfg.mid_side_balance = mid_side;
    let unmasked = apply_spectral_unmask(&t_buf, &sc_buf, &cfg)?;
    unmasked.save_wav(output, 24)?;
    println!("Sidechain-unmasked stem written to {}", output.display());
    Ok(())
}

fn run_phase_align_cli(
    reference: &std::path::Path,
    target: &std::path::Path,
    output: &std::path::Path,
    vocal_dtw: bool,
    elastic: bool,
) -> anyhow::Result<()> {
    let ref_buf = AudioBuffer::load(reference)?;
    let tgt_buf = AudioBuffer::load(target)?;
    if vocal_dtw {
        let (warped, markers) = if elastic {
            align_vocal_double_dtw_elastic(&ref_buf, &tgt_buf, 120.0)?
        } else {
            align_vocal_double_dtw(&ref_buf, &tgt_buf, 120.0)?
        };
        warped.save_wav(output, 24)?;
        println!(
            "DTW-aligned track written to {} (elastic={elastic}, {} stretch markers)",
            output.display(),
            markers.len()
        );
    } else {
        let (aligned, info) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 35.0)?;
        aligned.save_wav(output, 24)?;
        println!(
            "Phase-aligned track written to {} (Delay corrected: {} samples / {:.2} ms, Polarity inverted: {})",
            output.display(),
            info.delay_samples,
            info.delay_ms,
            info.invert_polarity
        );
    }
    Ok(())
}

fn run_vocal_cli(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    scale: &str,
    retune: f64,
    harmonize: bool,
    align_to: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let raw = AudioBuffer::load(input)?;
    let surgically_clean = clean_vocal_surgical(&raw, true, true, -12.0)?;
    let parsed_scale = VocalScale::parse(scale)?;
    let contour = track_vocal_pitch(&surgically_clean, &parsed_scale, retune);
    let corr_curve: Vec<(f64, f64)> = contour
        .iter()
        .map(|f| (f.time_secs, f.correction_semitones))
        .collect();
    let mut tuned = shift_vocal_pitch_formant_preserved(&surgically_clean, &corr_curve, 0.0)?;
    if let Some(guide_path) = align_to {
        let guide_buf = AudioBuffer::load(guide_path)?;
        let (warped, markers) = align_vocal_double_dtw(&guide_buf, &tuned, 120.0)?;
        tuned = warped;
        println!(
            "FastDTW aligned vocal to guide ({} REAPER stretch markers generated)",
            markers.len()
        );
    }
    let lead_out = output_dir.join("Vocal_Lead_Tuned_Clean.wav");
    tuned.save_wav(&lead_out, 24)?;
    println!(
        "Tuned & cleaned lead vocal written to {}",
        lead_out.display()
    );
    if harmonize {
        let (h_up, h_down, h_dbl) = generate_vocal_harmony_stack(&tuned, &parsed_scale)?;
        h_up.save_wav(output_dir.join("Vocal_Harmony_3rd_Above.wav"), 24)?;
        h_down.save_wav(output_dir.join("Vocal_Harmony_3rd_Below.wav"), 24)?;
        h_dbl.save_wav(output_dir.join("Vocal_Formant_Double.wav"), 24)?;
    }
    Ok(())
}

struct TheoryCliArgs {
    chords: String,
    style: String,
    groove: String,
    bpm: f64,
    output_midi: PathBuf,
    output_abc: Option<PathBuf>,
    abc_input: Option<PathBuf>,
}

fn write_theory_outputs(
    a: &TheoryCliArgs,
    reharm: &[String],
    chord_notes: &[NoteEvent],
    bass_notes: &[NoteEvent],
) -> anyhow::Result<()> {
    if let Some(abc_in_path) = &a.abc_input {
        let raw_abc = std::fs::read_to_string(abc_in_path)?;
        let (_abc_chords, melody_notes) = parse_abc_score(&raw_abc, 480)?;
        write_midi_file(
            &a.output_midi,
            a.bpm,
            480,
            &[
                ("ABC_Melody", &melody_notes),
                ("Voiced_Chords", chord_notes),
                ("Sub_Bass", bass_notes),
            ],
        )?;
    } else {
        write_midi_file(
            &a.output_midi,
            a.bpm,
            480,
            &[("Voiced_Chords", chord_notes), ("Sub_Bass", bass_notes)],
        )?;
    }
    println!(
        "Voice-led MIDI written to {} (Progression: {})",
        a.output_midi.display(),
        reharm.join(" -> ")
    );
    if let Some(abc_path) = &a.output_abc {
        let abc_str = export_abc_score("ForgeAudio Progression", "C", a.bpm.round() as u32, reharm);
        std::fs::write(abc_path, abc_str)?;
        println!("ABC lead sheet written to {}", abc_path.display());
    }
    Ok(())
}

fn run_theory_cli(a: TheoryCliArgs) -> anyhow::Result<()> {
    let v_style = match validate_choice(
        &a.style,
        "style",
        &["neo-soul", "drop2", "close", "cinematic"],
    )? {
        "drop2" => VoicingStyle::Drop2,
        "close" => VoicingStyle::Close,
        "cinematic" => VoicingStyle::CinematicSpread,
        _ => VoicingStyle::NeoSoulRootless,
    };
    let g_style = match validate_choice(
        &a.groove,
        "groove",
        &["dilla", "mpc58", "strum", "straight"],
    )? {
        "straight" => GrooveStyle::Straight,
        "mpc58" => GrooveStyle::MpcSwing58,
        "strum" => GrooveStyle::AcousticStrum,
        _ => GrooveStyle::NeoSoulDilla,
    };
    let raw_chords: Vec<String> = a
        .chords
        .split_whitespace()
        .map(ToString::to_string)
        .collect();
    let reharm = reharmonize_progression(&raw_chords, v_style);
    let (chord_notes, bass_notes) = solve_voice_leading(&reharm, v_style, g_style, 480, 4)?;
    write_theory_outputs(&a, &reharm, &chord_notes, &bass_notes)
}

fn dispatch_delivery_cli(command: Commands) -> anyhow::Result<Option<Commands>> {
    match command {
        Commands::Master {
            input,
            output,
            target_lufs,
            ceiling_dbtp,
            reference,
            genre,
            isrc,
            title,
            artist,
        } => run_master_cli(MasterCliArgs {
            input,
            output,
            target_lufs,
            ceiling_dbtp,
            reference,
            genre,
            isrc,
            title,
            artist,
        })?,
        Commands::Deliver {
            input,
            output,
            target,
            lufs,
            ceiling_dbtp,
            no_repair,
            no_defizz,
            vocal,
            reference,
            genre,
            limiter_style,
            channel_link,
            dither,
            dither_bits,
            ai_track,
            hpss_shimmer,
        } => run_deliver_cli(DeliverCliArgs {
            input,
            output,
            target,
            lufs,
            ceiling_dbtp,
            no_repair,
            no_defizz,
            vocal,
            reference,
            genre,
            limiter_style,
            channel_link,
            dither,
            dither_bits,
            ai_track,
            hpss_shimmer,
        })?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}

fn dispatch_studio_cli(command: Commands) -> anyhow::Result<Option<Commands>> {
    match command {
        Commands::Ingest {
            input,
            output_dir,
            bpm,
            scale,
            chords,
            isrc,
            require_neural,
            replace_drums,
            drum_replace_mix,
            quantize_grid,
        } => run_ingest_cli(IngestCliArgs {
            input,
            output_dir,
            bpm,
            scale,
            chords,
            isrc,
            require_neural,
            replace_drums,
            drum_replace_mix,
            quantize_grid,
        })?,
        Commands::Reaper {
            install_jsfx,
            jsfx_dir,
            osc_addr,
            osc_host,
        } => run_reaper_cli(
            install_jsfx,
            jsfx_dir.as_deref(),
            osc_addr.as_deref(),
            &osc_host,
        )?,
        Commands::SampleIndex { folder } => {
            let indexed = index_drum_sample_library(&folder)?;
            println!("{}", serde_json::to_string_pretty(&indexed)?);
        }
        Commands::Lyria {
            prompt,
            output,
            model,
        } => run_lyria_cli(&prompt, &output, &model)?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}

fn run_reaper_cli(
    install_jsfx: bool,
    jsfx_dir: Option<&std::path::Path>,
    osc_addr: Option<&str>,
    osc_host: &str,
) -> anyhow::Result<()> {
    if install_jsfx {
        let installed = install_jsfx_plugins(jsfx_dir)?;
        println!("Installed {} ForgeAudio JSFX plugins:", installed.len());
        for p in installed {
            println!("  • {}", p.display());
        }
    }
    if let Some(addr) = osc_addr {
        send_reaper_osc(osc_host, addr, Some(1.0))?;
        println!("Sent UDP OSC message '{addr}' to {osc_host}");
    }
    Ok(())
}

fn run_lyria_cli(prompt: &str, output: &std::path::Path, model: &str) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (saved, lyrics) = rt.block_on(generate_music_lyria(prompt, model, output))?;
    println!("Lyria audio saved to {}", saved.display());
    if let Some(l) = lyrics {
        println!("\nGenerated Structure / Lyrics:\n{l}");
    }
    Ok(())
}

fn dispatch_worker_ai(command: Commands) -> anyhow::Result<Option<Commands>> {
    match command {
        Commands::Denoise {
            input,
            output_dir,
            backend,
            attenuation_limit_db,
            timeout_secs,
        } => run_denoise_cli(
            &input,
            &output_dir,
            &backend,
            attenuation_limit_db,
            timeout_secs,
        )?,
        Commands::Transcribe {
            input,
            output_dir,
            timeout_secs,
        } => {
            let res = transcribe_midi_hybrid(
                &input,
                &output_dir,
                std::time::Duration::from_secs(timeout_secs),
            )?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Commands::Separate {
            input,
            output_dir,
            stems,
            quality,
            no_dereverb,
            drum_sep,
            split_lead_backing,
            split_male_female,
            split_guitar,
            split_orchestral,
            split_crowd,
            all_stems,
            restore,
            debleed,
            de_aspiration,
            dereverb_echo,
            backend,
        } => run_separate_cmd(SeparateCliArgs {
            input,
            output_dir,
            stems,
            quality,
            no_dereverb,
            drum_sep,
            split_lead_backing,
            split_male_female,
            split_guitar,
            split_orchestral,
            split_crowd,
            all_stems,
            restore,
            debleed,
            de_aspiration,
            dereverb_echo,
            backend,
        })?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}

fn dispatch_worker_analysis(command: Commands) -> anyhow::Result<()> {
    match command {
        Commands::Daw { action } => run_daw_cli(action)?,
        Commands::NeuralRestore {
            input,
            output_dir,
            backend,
            prompt,
            chunk_secs,
            timeout_secs,
        } => run_neural_restore_cli(
            &input,
            &output_dir,
            &backend,
            prompt.as_deref(),
            chunk_secs,
            timeout_secs,
        )?,
        Commands::VocalRescue {
            input,
            output_dir,
            backend,
            reference_voice,
            no_dereverb,
            no_de_aspiration,
            timeout_secs,
        } => run_vocal_rescue_cli(
            &input,
            &output_dir,
            &backend,
            reference_voice.as_deref(),
            no_dereverb,
            no_de_aspiration,
            timeout_secs,
        )?,
        Commands::BeatTrack {
            input,
            hint_bpm,
            backend,
            timeout_secs,
        } => run_beat_track_cli(&input, hint_bpm, &backend, timeout_secs)?,
        Commands::Evaluate {
            input,
            reference,
            vocal,
            backend,
            timeout_secs,
        } => run_evaluate_cli(&input, reference.as_deref(), vocal, &backend, timeout_secs)?,
        Commands::F0Track {
            input,
            backend,
            hop_secs,
            timeout_secs,
            json,
        } => run_f0_track_cli(&input, &backend, hop_secs, timeout_secs, json)?,
        _ => unreachable!("all CLI commands handled across dispatch stages"),
    }
    Ok(())
}

fn run_denoise_cli(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    backend: &str,
    attenuation_limit_db: Option<f64>,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let worker_backend = match validate_choice(
        backend,
        "backend",
        &[
            "auto",
            "clearervoice",
            "deepfilternet",
            "deepfilternet3",
            "mel-band-roformer",
            "roformer",
        ],
    )? {
        "deepfilternet3" => "deepfilternet",
        "roformer" => "mel-band-roformer",
        other => other,
    };
    let res = denoise_hybrid(
        input,
        output_dir,
        worker_backend,
        attenuation_limit_db,
        std::time::Duration::from_secs(timeout_secs),
    )?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

struct SeparateCliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    stems: u8,
    quality: String,
    no_dereverb: bool,
    drum_sep: bool,
    split_lead_backing: bool,
    split_male_female: bool,
    split_guitar: bool,
    split_orchestral: bool,
    split_crowd: bool,
    all_stems: bool,
    restore: bool,
    debleed: bool,
    de_aspiration: bool,
    dereverb_echo: bool,
    backend: Option<String>,
}

fn run_separate_cmd(a: SeparateCliArgs) -> anyhow::Result<()> {
    if !matches!(a.stems, 2 | 4 | 6) {
        anyhow::bail!("--stems must be 2, 4, or 6 (got {})", a.stems);
    }
    let mut cfg = SeparationConfig::from_env();
    cfg.stems = a.stems;
    cfg.quality = parse_quality(&a.quality)?;
    cfg.dereverb = !a.no_dereverb;
    cfg.drum_sep = a.drum_sep;
    cfg.split_lead_backing = a.split_lead_backing;
    cfg.split_male_female = a.split_male_female;
    cfg.split_guitar = a.split_guitar;
    cfg.split_orchestral = a.split_orchestral;
    cfg.split_crowd = a.split_crowd;
    if a.all_stems {
        cfg.enable_all_stems();
        cfg.dereverb &= !a.no_dereverb;
    }
    cfg.restore = a.restore;
    cfg.debleed = a.debleed;
    cfg.de_aspiration = a.de_aspiration;
    cfg.dereverb_echo = a.dereverb_echo;
    if let Some(b) = &a.backend {
        cfg.backend = parse_backend(b, cfg.colab_url.as_deref())?;
    }
    run_separate_cli(&a.input, &a.output_dir, &cfg)
}

fn run_neural_restore_cli(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    backend: &str,
    prompt: Option<&str>,
    chunk_secs: Option<f64>,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let backend = validate_backend_choice(
        backend,
        &[
            "auto",
            "clearervoice",
            "apollo",
            "flowhigh",
            "audiosr",
            "sonicmaster",
            "exciter",
        ],
    )?;
    let res = neural_restore_hybrid(
        input,
        output_dir,
        backend,
        prompt,
        chunk_secs,
        std::time::Duration::from_secs(timeout_secs),
    )?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

fn run_vocal_rescue_cli(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    backend: &str,
    reference_voice: Option<&std::path::Path>,
    no_dereverb: bool,
    no_de_aspiration: bool,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let backend = validate_backend_choice(
        backend,
        &["auto", "resemble-enhance", "seed-vc", "roformer-clean"],
    )?;
    let res = vocal_rescue_hybrid(
        input,
        output_dir,
        backend,
        reference_voice,
        !no_dereverb,
        !no_de_aspiration,
        std::time::Duration::from_secs(timeout_secs),
    )?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

fn run_beat_track_cli(
    input: &std::path::Path,
    hint_bpm: Option<f64>,
    backend: &str,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let backend = validate_backend_choice(
        backend,
        &[
            "auto",
            "beat-this",
            "librosa",
            "energy",
            "rust",
            "pure_rust_onset",
        ],
    )?;
    let res = track_beats_hybrid(
        input,
        hint_bpm,
        backend,
        std::time::Duration::from_secs(timeout_secs),
    )?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

fn run_evaluate_cli(
    input: &std::path::Path,
    reference: Option<&std::path::Path>,
    vocal: bool,
    backend: &str,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let backend = validate_backend_choice(
        backend,
        &["auto", "audiobox", "singmos", "acoustic-proxy", "rust"],
    )?;
    let res = evaluate_quality_hybrid(
        input,
        reference,
        vocal,
        backend,
        std::time::Duration::from_secs(timeout_secs),
    )?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

struct MasterCliArgs {
    input: PathBuf,
    output: PathBuf,
    target_lufs: f64,
    ceiling_dbtp: f64,
    reference: Option<PathBuf>,
    genre: Option<String>,
    isrc: Option<String>,
    title: String,
    artist: String,
}

fn run_master_cli(a: MasterCliArgs) -> anyhow::Result<()> {
    let mut buf = AudioBuffer::load(&a.input)?;
    if let Some(ref_path) = &a.reference {
        let ref_buf = AudioBuffer::load(ref_path)?;
        let match_curve = compute_reference_match_eq(&buf, &ref_buf, 4.5);
        apply_18_band_eq(&mut buf, &match_curve);
    } else if let Some(ref g_str) = a.genre {
        let g = validate_choice(
            g_str,
            "genre",
            &["pop", "hiphop", "neosoul", "edm", "rock", "jazz"],
        )?;
        let target = GenreTonalTarget::parse(g);
        let genre_curve = compute_genre_tonal_balance_eq(&buf, target, 3.5);
        apply_18_band_eq(&mut buf, &genre_curve);
    }

    let mastered = normalize_and_limit(&buf, a.target_lufs, a.ceiling_dbtp)?;
    if let Some(raw_isrc) = &a.isrc {
        let canonical = validate_isrc(raw_isrc)?;
        let tmp = TempFileGuard::new(a.output.with_extension("pre_tag.wav"));
        mastered.save_wav(tmp.path(), 24)?;
        let meta = ReleaseMetadata {
            title: a.title,
            artist: a.artist,
            album: "Single".to_string(),
            isrc: canonical.clone(),
            bpm: None,
            key: None,
            year: Some(2026),
        };
        write_distrokid_metadata(tmp.path(), a.output.as_path(), &meta)?;
        println!(
            "Mastered & tagged with ISRC {canonical} -> {}",
            a.output.display()
        );
    } else {
        mastered.save_wav(&a.output, 24)?;
    }

    let audit = audit_audio(&mastered);
    println!(
        "Final Master: {:.2} LUFS | {:.2} dBTP True Peak | LRA {:.2} LU | DistroKid Ready: {}",
        audit.integrated_lufs, audit.true_peak_dbtp, audit.loudness_range_lu, audit.distrokid_ready
    );
    Ok(())
}

struct DeliverCliArgs {
    input: PathBuf,
    output: PathBuf,
    target: String,
    lufs: Option<f64>,
    ceiling_dbtp: Option<f64>,
    no_repair: bool,
    no_defizz: bool,
    vocal: bool,
    reference: Option<PathBuf>,
    genre: Option<String>,
    limiter_style: String,
    channel_link: f64,
    dither: Option<String>,
    dither_bits: u16,
    ai_track: bool,
    hpss_shimmer: Option<f64>,
}

fn configure_deliver_options(a: &DeliverCliArgs) -> anyhow::Result<(DeliveryOptions, bool)> {
    let target = match (a.lufs, a.ceiling_dbtp) {
        (Some(lufs), Some(ceiling)) => DeliveryTarget::custom(lufs, ceiling)?,
        _ => DeliveryTarget::parse(&a.target)?,
    };
    let mut options = DeliveryOptions::for_target(target);
    options.repair = !a.no_repair;
    options.defizz = !a.no_defizz;
    options.vocal = a.vocal;
    options.ai_track_mode = a.ai_track;
    options.limiter_style = match validate_choice(
        &a.limiter_style,
        "limiter-style",
        &["transparent", "punchy", "dynamic", "aggressive"],
    )? {
        "punchy" => LimiterStyle::Punchy,
        "dynamic" => LimiterStyle::Dynamic,
        "aggressive" => LimiterStyle::Aggressive,
        _ => LimiterStyle::Transparent,
    };
    options.channel_link = a.channel_link;
    let explicit_dither = a.dither.is_some();
    options.dither = match a
        .dither
        .as_deref()
        .unwrap_or("none")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "none" => DitherMode::None,
        "tpdf" => DitherMode::Tpdf,
        "noise-shaped" | "noise_shaped" | "shaped" => DitherMode::NoiseShapedHighPass,
        other => anyhow::bail!("unknown --dither '{other}' (expected none, tpdf, or noise-shaped)"),
    };
    options.dither_bits = a.dither_bits;
    options.tone_match = match (&a.reference, &a.genre) {
        (Some(path), _) => Some(ToneMatch::Reference(AudioBuffer::load(path)?)),
        (None, Some(g_str)) => {
            let g = validate_choice(
                g_str,
                "genre",
                &["pop", "hiphop", "neosoul", "edm", "rock", "jazz"],
            )?;
            Some(ToneMatch::Genre(GenreTonalTarget::parse(g)))
        }
        (None, None) => None,
    };
    Ok((options, explicit_dither))
}

fn print_delivery_summary(report: &forge_dsp::DeliveryReport) {
    println!(
        "{} ({:.1} LUFS, {:.1} dBTP): {:.2} -> {:.2} LUFS, {:.2} -> {:.2} dBTP",
        report.target,
        report.target_lufs,
        report.ceiling_dbtp,
        report.before.integrated_lufs,
        report.after.integrated_lufs,
        report.before.true_peak_dbtp,
        report.after.true_peak_dbtp
    );
    for check in &report.checks {
        let mark = if check.passed { "PASS" } else { "FAIL" };
        println!("  [{mark}] {}: {}", check.name, check.detail);
    }
}

fn run_deliver_cli(a: DeliverCliArgs) -> anyhow::Result<()> {
    let (mut options, explicit_dither) = configure_deliver_options(&a)?;
    let buf = AudioBuffer::load(&a.input)?;
    if let Some(shimmer) = a.hpss_shimmer {
        let mut cfg = DeFizzConfig::ai_mix_from_audit(&audit_audio(&buf));
        cfg.hpss_shimmer_suppress = shimmer;
        options.defizz_config = Some(cfg);
    }
    let (master, report) = deliver(&buf, &options)?;
    if options.dither != DitherMode::None {
        let pcm_bits = if options.dither_bits <= 16 { 16 } else { 24 };
        master.save_wav_no_dither(&a.output, pcm_bits)?;
    } else if explicit_dither {
        master.save_wav_no_dither(&a.output, 24)?;
    } else {
        master.save_wav(&a.output, 24)?;
    }
    let report_path = a.output.with_extension("delivery.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    print_delivery_summary(&report);
    if !report.ready {
        std::process::exit(2);
    }
    Ok(())
}

fn rust_f0_track_cli(
    input: &std::path::Path,
    hop_secs: f64,
    warnings: Vec<String>,
) -> anyhow::Result<F0TrackResult> {
    let buf = AudioBuffer::load(input)?;
    let scale = VocalScale::parse("C chromatic")?;
    let raw_frames = track_vocal_pitch(&buf, &scale, 1.0);
    let frames: Vec<F0TrackFrame> = raw_frames
        .iter()
        .map(|f| F0TrackFrame {
            time_secs: f.time_secs,
            f0_hz: f.f0_hz,
            voiced: f.f0_hz > 0.0,
            confidence: if f.f0_hz > 0.0 { 0.85 } else { 0.0 },
        })
        .collect();
    let mut voiced_hz: Vec<f64> = frames
        .iter()
        .filter(|f| f.voiced && f.f0_hz.is_finite() && f.f0_hz > 0.0)
        .map(|f| f.f0_hz)
        .collect();
    voiced_hz.sort_by(f64::total_cmp);
    let median_f0_hz = voiced_hz.get(voiced_hz.len() / 2).copied();
    let voiced_ratio = if frames.is_empty() {
        0.0
    } else {
        (voiced_hz.len() as f64) / (frames.len() as f64)
    };
    Ok(F0TrackResult {
        backend: "rust-yin".to_string(),
        sample_rate: buf.sample_rate,
        hop_secs,
        voiced_ratio,
        median_f0_hz,
        frames,
        warnings,
    })
}

fn run_f0_track_cli(
    input: &std::path::Path,
    backend: &str,
    hop_secs: f64,
    timeout_secs: u64,
    json: bool,
) -> anyhow::Result<()> {
    let backend = validate_choice(
        backend,
        "backend",
        &["auto", "swiftf0", "rmvpe", "fcpe", "autocorr", "rust"],
    )?;
    let timeout = std::time::Duration::from_secs(timeout_secs);
    let res = if backend == "rust" {
        rust_f0_track_cli(input, hop_secs, Vec::new())?
    } else {
        match resolve_default_local_worker()
            .and_then(|worker| run_local_f0_track(&worker, input, backend, hop_secs, timeout))
        {
            Ok(res) => res,
            Err(err) if backend == "auto" => rust_f0_track_cli(
                input,
                hop_secs,
                vec![format!(
                    "worker skipped ({err}), used pure-Rust YIN fallback"
                )],
            )?,
            Err(err) => return Err(err.into()),
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&res)?);
    } else {
        println!(
            "F0 Track ({}, {} Hz, hop {:.3}s): voiced {:.1}%, median {:.2} Hz ({} frames)",
            res.backend,
            res.sample_rate,
            res.hop_secs,
            res.voiced_ratio * 100.0,
            res.median_f0_hz.unwrap_or(0.0),
            res.frames.len()
        );
    }
    Ok(())
}

struct IngestCliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    bpm: f64,
    scale: String,
    chords: String,
    isrc: Option<String>,
    require_neural: bool,
    replace_drums: Option<PathBuf>,
    drum_replace_mix: f64,
    quantize_grid: bool,
}

#[inline(never)]
fn run_ingest_cli(args: IngestCliArgs) -> anyhow::Result<()> {
    let parsed_scale = VocalScale::parse(&args.scale)?;
    let validated_isrc = args.isrc.as_deref().map(validate_isrc).transpose()?;
    std::fs::create_dir_all(&args.output_dir)?;
    let stems_dir = args.output_dir.join("stems");
    let midi_dir = args.output_dir.join("midi");
    std::fs::create_dir_all(&stems_dir)?;
    std::fs::create_dir_all(&midi_dir)?;

    let full_mix = AudioBuffer::load(&args.input)?;
    let beat_grid = analyze_beat_grid(&full_mix, Some(args.bpm));

    // 1. Separate stems
    let mut config = SeparationConfig::from_env();
    config.require_neural |= args.require_neural;
    let separation = separate_stems(&args.input, &stems_dir, &config)?;
    let mut bundle = separation.bundle;

    let mut primary_stems: Vec<(PathBuf, AudioBuffer)> = Vec::new();
    let mut vocal_pitch_frames = Vec::new();
    let mut cleaned_vocal: Option<AudioBuffer> = None;

    let mut vocal_stem_idx: Option<usize> = None;

    // 2. Clean & Spectral De-Fizz Lead Vocal (ai_vocal preset)
    if let Some(ref voc_path) = bundle.vocals {
        let voc_buf = AudioBuffer::load(voc_path)?;
        let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
        let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::ai_vocal())?;
        vocal_pitch_frames = track_vocal_pitch(&voc_defizz, &parsed_scale, 0.85);
        voc_defizz.save_wav(voc_path, 24)?;
        cleaned_vocal = Some(voc_defizz.clone());
        vocal_stem_idx = Some(primary_stems.len());
        primary_stems.push((voc_path.clone(), voc_defizz));
    }

    // 3. De-Fizz & Vocal-Unmask Instrumental / Other stem
    if let Some(ref other_path) = bundle.other {
        let other_buf = AudioBuffer::load(other_path)?;
        let defizz_cfg = DeFizzConfig::ai_mix_from_audit(&audit_audio(&other_buf));
        let mut other_clean = apply_spectral_defizz(&other_buf, &defizz_cfg)?;
        if let Some(ref voc) = cleaned_vocal {
            other_clean =
                apply_spectral_unmask(&other_clean, voc, &UnmaskConfig::vocal_over_instruments())?;
        }
        other_clean.save_wav(other_path, 24)?;
        primary_stems.push((other_path.clone(), other_clean));
    }

    // 4. Extract Drum MIDI Triggers + optional Phase-Locked Sample Layering
    let mut drum_buf = if let Some(ref drum_path) = bundle.drums {
        AudioBuffer::load(drum_path)?
    } else {
        full_mix.clone()
    };
    let (kick_notes, snare_notes, hat_notes) = extract_drum_midi_triggers(&drum_buf, args.bpm, 480);

    let drum_midi_path = midi_dir.join("01_Drum_Triggers.mid");

    let mut drum_hits_layered = 0_usize;
    if let Some(ref samples_dir) = args.replace_drums {
        let library = index_drum_sample_library(samples_dir)?;
        let tv = |ns: &[NoteEvent]| -> Vec<(u32, u8)> {
            ns.iter().map(|n| (n.start_tick, n.velocity)).collect()
        };
        let hits = DrumHit::from_note_groups(
            [
                ("Kick", tv(&kick_notes)),
                ("Snare", tv(&snare_notes)),
                ("HiHat", tv(&hat_notes)),
            ],
            args.bpm,
            480,
            drum_buf.sample_rate,
        );
        let dry_ref = bundle.drums.as_ref().map(|_| &drum_buf);
        let (layered, count) = render_drum_sample_layer(
            drum_buf.num_frames(),
            drum_buf.sample_rate,
            &hits,
            &library,
            args.drum_replace_mix,
            dry_ref,
        )?;
        drum_hits_layered = count;
        let out_drum = bundle
            .drums
            .clone()
            .unwrap_or_else(|| stems_dir.join("Drums_Layered.wav"));
        layered.save_wav(&out_drum, 24)?;
        bundle.drums = Some(out_drum);
        drum_buf = layered;
    }
    let mut drum_stem_idx: Option<usize> = None;
    if let Some(ref drums_path) = bundle.drums {
        drum_stem_idx = Some(primary_stems.len());
        primary_stems.push((drums_path.clone(), drum_buf));
    }

    // 5. Load remaining primary stems (Bass, optional Guitar/Keys from 6-stem mode)
    for opt_path in [&bundle.bass, &bundle.guitar, &bundle.keys]
        .into_iter()
        .flatten()
    {
        primary_stems.push((opt_path.clone(), AudioBuffer::load(opt_path)?));
    }

    // 6. Optional WSOLA Beat-Grid Quantization across all primary stems
    let mut quantized_applied = false;
    if args.quantize_grid && beat_grid.beats_secs.len() >= 2 {
        for (path, buf) in &mut primary_stems {
            let (warped, _) = quantize_to_constant_bpm(buf, &beat_grid.beats_secs, args.bpm)?;
            warped.save_wav(path, 24)?;
            *buf = warped;
        }
        if let Some(v_idx) = vocal_stem_idx {
            vocal_pitch_frames = track_vocal_pitch(&primary_stems[v_idx].1, &parsed_scale, 0.85);
        }
        quantized_applied = true;
    }

    let drum_for_midi = if let Some(d_idx) = drum_stem_idx {
        primary_stems[d_idx].1.clone()
    } else if quantized_applied {
        quantize_to_constant_bpm(&full_mix, &beat_grid.beats_secs, args.bpm)?.0
    } else {
        full_mix.clone()
    };
    let (kick_notes, snare_notes, hat_notes) =
        extract_drum_midi_triggers(&drum_for_midi, args.bpm, 480);
    write_midi_file(
        &drum_midi_path,
        args.bpm,
        480,
        &[
            ("Kick_Trigger_C1", &kick_notes),
            ("Snare_Trigger_D1", &snare_notes),
            ("HiHat_Trigger_F#1", &hat_notes),
        ],
    )?;

    // 7. Solve Voice-Led Chords & Phase-Locked Sub-Bass MIDI
    let raw_chords: Vec<String> = args
        .chords
        .split_whitespace()
        .map(ToString::to_string)
        .collect();
    let reharm = reharmonize_progression(&raw_chords, VoicingStyle::NeoSoulRootless);
    let (chord_notes, bass_notes) = solve_voice_leading(
        &reharm,
        VoicingStyle::NeoSoulRootless,
        GrooveStyle::NeoSoulDilla,
        480,
        4,
    )?;

    let chords_midi_path = midi_dir.join("02_Voiced_Chords.mid");
    let bass_midi_path = midi_dir.join("03_Sub_Bass_Reinforce.mid");
    write_midi_file(
        &chords_midi_path,
        args.bpm,
        480,
        &[("Voiced_Chords", &chord_notes)],
    )?;
    write_midi_file(&bass_midi_path, args.bpm, 480, &[("Sub_Bass", &bass_notes)])?;

    let abc_path = args.output_dir.join("Lead_Sheet.abc");
    std::fs::write(
        &abc_path,
        export_abc_score(
            "ForgeAudio Studio Session",
            &args.scale,
            args.bpm.round() as u32,
            &reharm,
        ),
    )?;

    // 8. Re-sum cleaned stems and render telemetry-driven AI-aware DistroKid Master
    let refs: Vec<&AudioBuffer> = primary_stems.iter().map(|(_, b)| b).collect();
    let (mastered, delivery) =
        forge_dsp::resum_and_deliver_ai_stems(&refs, &full_mix, DeliveryTarget::Spotify)?;
    let master_path = args.output_dir.join("Master_DistroKid_24bit_48kHz.wav");
    mastered.save_wav(&master_path, 24)?;
    std::fs::write(
        master_path.with_extension("delivery.json"),
        serde_json::to_string_pretty(&delivery)?,
    )?;

    let buf_duration_secs =
        |b: &AudioBuffer| (b.num_frames() as f64) / f64::from(b.sample_rate.max(1));
    let duration_secs = primary_stems
        .iter()
        .map(|(_, b)| buf_duration_secs(b))
        .chain(std::iter::once(buf_duration_secs(&mastered)))
        .fold(0.0_f64, f64::max)
        .max(buf_duration_secs(&full_mix));
    let stem_duration = |path: Option<&PathBuf>| -> f64 {
        path.and_then(|p| {
            primary_stems
                .iter()
                .find(|(sp, _)| sp == p)
                .map(|(_, b)| buf_duration_secs(b))
        })
        .unwrap_or(duration_secs)
    };

    // 9. Build Complete REAPER 7.82 Multitrack .RPP Project (with <TEMPOENVEX> when unquantized & drifting)
    let mut rpp = ReaperProjectBuilder::new(
        "ForgeAudio Studio Production",
        args.bpm,
        full_mix.sample_rate,
    );
    rpp.isrc = validated_isrc;
    if !quantized_applied && beat_grid.max_bpm_drift >= 1.5 {
        rpp.tempo_map = beat_grid.tempo_map.clone();
    }
    rpp.add_default_song_structure_markers();

    rpp.tracks.push(
        ReaperTrackSpec::audio(
            "01 Lead Vocal (Dry De-Fizzed)",
            bundle.vocals.clone(),
            stem_duration(bundle.vocals.as_ref()),
            0.85,
            0x0155_88FF,
        )
        .with_jsfx(["Forge_Vocal_Surgical.jsfx", "Forge_OpenSoothe_DeFizz.jsfx"])
        .with_pitch_envelope(vocal_pitch_frames),
    );
    rpp.tracks.push(ReaperTrackSpec::audio(
        "02 Drums Stem",
        bundle.drums.clone(),
        stem_duration(bundle.drums.as_ref()),
        0.80,
        0x0144_CC44,
    ));
    rpp.tracks.push(
        ReaperTrackSpec::audio(
            "03 Bass Stem (Sidechain Unmasked + Mono Sub)",
            bundle.bass.clone(),
            stem_duration(bundle.bass.as_ref()),
            0.82,
            0x01FF_8844,
        )
        .with_jsfx([
            "Forge_SubBass_Reinforce.jsfx",
            "Forge_OpenSpacer_Unmask.jsfx",
        ])
        .with_sidechain(Some(1)),
    );
    rpp.tracks.push(
        ReaperTrackSpec::audio(
            "04 Instruments Bus (Vocal Unmasked + De-Fizzed)",
            bundle.other.clone(),
            stem_duration(bundle.other.as_ref()),
            0.78,
            0x01CC_66FF,
        )
        .with_jsfx([
            "Forge_OpenSoothe_DeFizz.jsfx",
            "Forge_OpenSpacer_Unmask.jsfx",
        ])
        .with_sidechain(Some(0)),
    );
    rpp.tracks.push(ReaperTrackSpec::midi(
        "05 MIDI Drum Triggers (Kick C1 / Snare D1 / Hat F#1)",
        drum_midi_path,
        duration_secs,
        0.90,
        0x0144_FFAA,
    ));
    rpp.tracks.push(ReaperTrackSpec::midi(
        "06 MIDI Voiced Chords (Neo-Soul Drop-2)",
        chords_midi_path,
        duration_secs,
        0.85,
        0x01FF_CC44,
    ));
    rpp.tracks.push(ReaperTrackSpec::midi(
        "07 MIDI Phase-Pure Sub-Bass / 808",
        bass_midi_path,
        duration_secs,
        0.88,
        0x01FF_6644,
    ));
    if let Some(ref guitar) = bundle.guitar {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "08 Guitar Stem",
            Some(guitar.clone()),
            stem_duration(Some(guitar)),
            0.78,
            0x0144_AAFF,
        ));
    }
    if let Some(ref keys) = bundle.keys {
        rpp.tracks.push(ReaperTrackSpec::audio(
            "09 Keys / Piano Stem",
            Some(keys.clone()),
            stem_duration(Some(keys)),
            0.78,
            0x01AA_FF44,
        ));
    }

    let rpp_path = args.output_dir.join("ForgeAudio_Session.rpp");
    rpp.save_rpp(&rpp_path)?;

    println!("=== ForgeAudio 1-Command Studio Ingest Complete ===");
    println!("Stem separation     : {}", separation.backend.as_str());
    for reason in &separation.fallbacks {
        println!("  fallback          : {reason}");
    }
    println!(
        "Beat Grid           : {:.2} median BPM | max drift {:.2} BPM | quantized={}",
        beat_grid.median_bpm, beat_grid.max_bpm_drift, quantized_applied
    );
    if drum_hits_layered > 0 {
        println!("Drum Sample Layer   : {drum_hits_layered} hits layered");
    }
    println!(
        "Master Verification : {} ({:.2} LUFS, {:.2} dBTP)",
        if delivery.ready { "READY" } else { "NOT READY" },
        delivery.after.integrated_lufs,
        delivery.after.true_peak_dbtp
    );
    println!("REAPER 7.82 Project : {}", rpp_path.display());
    println!("DistroKid Master WAV: {}", master_path.display());
    println!("ABC Lead Sheet      : {}", abc_path.display());
    Ok(())
}

fn validate_choice<'a>(raw: &str, flag: &str, allowed: &[&'a str]) -> anyhow::Result<&'a str> {
    let trimmed = raw.trim();
    allowed
        .iter()
        .copied()
        .find(|c| c.eq_ignore_ascii_case(trimmed))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported --{flag} '{trimmed}' (expected one of {})",
                allowed.join(", ")
            )
        })
}

fn validate_backend_choice<'a>(raw: &str, allowed: &[&'a str]) -> anyhow::Result<&'a str> {
    validate_choice(raw, "backend", allowed)
}

fn parse_hum_hz(raw: &str) -> Result<f64, String> {
    let v: f64 = raw.parse().map_err(|e| format!("{e}"))?;
    if v.is_finite() && (v == 0.0 || (40.0..=70.0).contains(&v)) {
        Ok(v)
    } else {
        Err(format!("hum-hz ({v}) must be 0 (off) or in 40..=70 Hz"))
    }
}

fn parse_hop_secs(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "hop-secs", 0.002, 0.25)
}

fn parse_dither_bits(raw: &str) -> Result<u16, String> {
    let v: u16 = raw.parse().map_err(|e| format!("{e}"))?;
    if (8..=24).contains(&v) {
        Ok(v)
    } else {
        Err(format!("dither-bits ({v}) must be in 8..=24"))
    }
}

fn parse_target_lufs(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "target-lufs", -40.0, -4.0)
}

fn parse_ceiling_dbtp(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "ceiling-dbtp", -24.0, 0.0)
}

fn parse_lpf_hz(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "ultrasonic-lpf-hz", 1000.0, 22_000.0)
}

fn parse_chunk_secs(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "chunk-secs", 0.5, 120.0)
}

fn parse_timeout_secs(raw: &str) -> Result<u64, String> {
    let v: u64 = raw.parse().map_err(|e| format!("{e}"))?;
    if (1..=3600).contains(&v) {
        Ok(v)
    } else {
        Err(format!("timeout-secs ({v}) must be in 1..=3600"))
    }
}

#[inline(never)]
fn run_daw_cli(action: DawAction) -> anyhow::Result<()> {
    match action {
        DawAction::Inspect { project, json } => {
            let proj = DawProject::load_from_file(&project)?;
            let mut cache = ClipCache::default();
            let mut missing_clips = Vec::new();
            for tr in &proj.tracks {
                for obj in &tr.objects {
                    if let Err(e) = cache.load_clip(&obj.source_path) {
                        missing_clips.push(format!("{}: {e}", obj.source_path.display()));
                    }
                }
            }
            let engine = ObjectOrientedEngine::compile(&proj, &cache);
            let summary = proj.inspect_summary(engine.total_pdc_samples(), missing_clips);
            if json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                println!(
                    "ForgeStudio Project '{}' | {} Hz | {:.1} BPM ({}) | {} tracks ({} objects, {} MIDI clips) | {} aux buses | PDC: {} spl",
                    summary.name,
                    summary.sample_rate,
                    summary.bpm,
                    summary.time_signature,
                    summary.tracks,
                    summary.audio_objects,
                    summary.midi_clips,
                    summary.aux_buses,
                    summary.total_pdc_samples
                );
                if !summary.missing_clips.is_empty() {
                    eprintln!(
                        "Warning: {} clip(s) missing on disk",
                        summary.missing_clips.len()
                    );
                }
            }
        }
        DawAction::Bounce {
            project,
            output,
            stems_dir,
        } => {
            let proj = DawProject::load_from_file(&project)?;
            let mut cache = ClipCache::default();
            for tr in &proj.tracks {
                for obj in &tr.objects {
                    cache.load_clip(&obj.source_path)?;
                }
            }
            let report = ObjectOrientedEngine::bounce_to_wav(&proj, &cache, &output)?;
            let mut exported_stems = Vec::new();
            if let Some(dir) = stems_dir {
                exported_stems = ObjectOrientedEngine::bounce_stems(&proj, &cache, &dir)?;
            }
            let result = serde_json::json!({
                "output": output.display().to_string(),
                "integrated_lufs": report.integrated_lufs,
                "true_peak_dbtp": report.true_peak_dbtp,
                "stems_exported": exported_stems.len(),
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

#[inline(never)]
fn run_separate_cli(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    cfg: &SeparationConfig,
) -> anyhow::Result<()> {
    let report = separate_stems(input, output_dir, cfg)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Parses a `--bpm` value and rejects tempos outside `forge_theory::BPM_RANGE`.
fn parse_bpm(raw: &str) -> Result<f64, String> {
    let bpm: f64 = raw.parse().map_err(|e| format!("{e}"))?;
    validate_bpm(bpm).map_err(|e| e.to_string())
}

fn parse_bounded_f64(raw: &str, name: &str, min: f64, max: f64) -> Result<f64, String> {
    let v: f64 = raw.parse().map_err(|e| format!("{e}"))?;
    if v.is_finite() && (min..=max).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{name} ({v}) must be finite and in {min}..={max}"))
    }
}

fn parse_tilt(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "tilt", -6.0, 6.0)
}

fn parse_unit_interval(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "value", 0.0, 1.0)
}

fn parse_mid_side(raw: &str) -> Result<f64, String> {
    parse_bounded_f64(raw, "mid-side", -1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpm_outside_the_supported_range_is_rejected() {
        for bad in ["0", "-5", "29.9", "301", "NaN", "inf"] {
            let theory: &[&str] = &["forge", "theory", "-o", "x.mid", "--bpm", bad];
            let ingest: &[&str] = &["forge", "ingest", "mix.wav", "-o", "out", "--bpm", bad];
            for cmd in [theory, ingest] {
                assert!(Cli::try_parse_from(cmd).is_err(), "{cmd:?} was accepted");
            }
        }
        for good in ["30", "92", "300"] {
            let cmd = ["forge", "theory", "-o", "x.mid", "--bpm", good];
            assert!(Cli::try_parse_from(cmd).is_ok(), "{cmd:?} was rejected");
        }
    }

    #[test]
    fn negative_loudness_and_ceiling_values_parse() {
        let cli = Cli::try_parse_from([
            "forge",
            "deliver",
            "mix.wav",
            "-o",
            "out/master.wav",
            "--lufs",
            "-10",
            "--ceiling-dbtp",
            "-1.5",
        ])
        .expect("deliver with negative custom target parses");
        match cli.command {
            Commands::Deliver {
                lufs, ceiling_dbtp, ..
            } => {
                assert_eq!(lufs, Some(-10.0));
                assert_eq!(ceiling_dbtp, Some(-1.5));
            }
            _ => panic!("expected deliver"),
        }

        let cli = Cli::try_parse_from([
            "forge",
            "master",
            "mix.wav",
            "-o",
            "master.wav",
            "--target-lufs",
            "-14",
            "--ceiling-dbtp",
            "-1",
        ])
        .expect("master with negative targets parses");
        match cli.command {
            Commands::Master {
                target_lufs,
                ceiling_dbtp,
                ..
            } => {
                assert_eq!(target_lufs, -14.0);
                assert_eq!(ceiling_dbtp, -1.0);
            }
            _ => panic!("expected master"),
        }
    }

    #[test]
    fn ai_suite_subcommands_and_new_flags_parse_and_validate() {
        for ok_cmd in [
            vec![
                "forge",
                "restore",
                "in.wav",
                "-o",
                "out.wav",
                "--ai-mode",
                "--hard-mode",
                "--hpss-shimmer",
                "0.45",
                "--ultrasonic-lpf-hz",
                "16000",
            ],
            vec![
                "forge",
                "deliver",
                "in.wav",
                "-o",
                "out.wav",
                "--ai-track",
                "--hpss-shimmer",
                "0.5",
            ],
            vec![
                "forge",
                "phase-align",
                "ref.wav",
                "tgt.wav",
                "-o",
                "out.wav",
                "--vocal-dtw",
                "--elastic",
            ],
            vec![
                "forge",
                "separate",
                "in.wav",
                "-o",
                "stems",
                "--debleed",
                "--de-aspiration",
                "--dereverb-echo",
            ],
            vec![
                "forge",
                "ingest",
                "mix.wav",
                "-o",
                "proj",
                "--replace-drums",
                "kit",
                "--drum-replace-mix",
                "0.6",
                "--quantize-grid",
            ],
            vec![
                "forge",
                "neural-restore",
                "in.wav",
                "-o",
                "out",
                "--backend",
                "apollo",
                "--prompt",
                "remove shimmer",
                "--chunk-secs",
                "8.0",
                "--timeout-secs",
                "120",
            ],
            vec![
                "forge",
                "vocal-rescue",
                "vox.wav",
                "-o",
                "out",
                "--backend",
                "seed-vc",
                "--reference-voice",
                "clean.wav",
                "--no-dereverb",
                "--no-de-aspiration",
            ],
            vec![
                "forge",
                "beat-track",
                "mix.wav",
                "--hint-bpm",
                "120",
                "--backend",
                "rust",
            ],
            vec![
                "forge",
                "evaluate",
                "mix.wav",
                "--reference",
                "ref.wav",
                "--vocal",
                "--backend",
                "rust",
            ],
        ] {
            assert!(
                Cli::try_parse_from(&ok_cmd).is_ok(),
                "{ok_cmd:?} failed to parse"
            );
        }

        for bad_cmd in [
            vec![
                "forge",
                "restore",
                "in.wav",
                "-o",
                "out.wav",
                "--ultrasonic-lpf-hz",
                "100",
            ],
            vec![
                "forge",
                "neural-restore",
                "in.wav",
                "-o",
                "out",
                "--chunk-secs",
                "0.1",
            ],
            vec!["forge", "beat-track", "in.wav", "--timeout-secs", "0"],
        ] {
            assert!(
                Cli::try_parse_from(&bad_cmd).is_err(),
                "{bad_cmd:?} must be rejected"
            );
        }
    }

    #[test]
    fn restore_ai_mode_leaves_mid_side_and_tape_drive_unset_by_default() {
        let ai_cli =
            Cli::try_parse_from(["forge", "restore", "in.wav", "-o", "out.wav", "--ai-mode"])
                .expect("parse ai-mode restore");
        match ai_cli.command {
            Commands::Restore {
                mid_side,
                tape_drive,
                ai_mode,
                ..
            } => {
                assert!(ai_mode);
                assert_eq!(mid_side, None);
                assert_eq!(tape_drive, None);
            }
            other => {
                let _ = other;
                panic!("unexpected command variant");
            }
        }

        let explicit_cli = Cli::try_parse_from([
            "forge",
            "restore",
            "in.wav",
            "-o",
            "out.wav",
            "--ai-mode",
            "--mid-side",
            "0.4",
            "--tape-drive",
            "0.25",
        ])
        .expect("parse explicit restore overrides");
        match explicit_cli.command {
            Commands::Restore {
                mid_side,
                tape_drive,
                ai_mode,
                ..
            } => {
                assert!(ai_mode);
                assert_eq!(mid_side, Some(0.4));
                assert_eq!(tape_drive, Some(0.25));
            }
            other => {
                let _ = other;
                panic!("unexpected command variant");
            }
        }
    }

    #[test]
    fn wave4_cli_bounds_and_enum_choices_are_validated() {
        for bad_cmd in [
            vec![
                "forge", "restore", "in.wav", "-o", "out.wav", "--hum-hz", "500",
            ],
            vec!["forge", "vocal", "in.wav", "-o", "out", "--retune", "1.5"],
            vec![
                "forge",
                "deliver",
                "in.wav",
                "-o",
                "out.wav",
                "--dither-bits",
                "32",
            ],
            vec![
                "forge",
                "deliver",
                "in.wav",
                "-o",
                "out.wav",
                "--channel-link",
                "-0.2",
            ],
            vec![
                "forge",
                "master",
                "in.wav",
                "-o",
                "out.wav",
                "--target-lufs",
                "2.0",
            ],
            vec![
                "forge",
                "master",
                "in.wav",
                "-o",
                "out.wav",
                "--ceiling-dbtp",
                "1.0",
            ],
            vec!["forge", "f0-track", "in.wav", "--hop-secs", "0.0001"],
        ] {
            assert!(
                Cli::try_parse_from(&bad_cmd).is_err(),
                "{bad_cmd:?} must be rejected by value_parser"
            );
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let sr = 48_000_u32;
        let samples: Vec<f64> = (0..(sr / 2))
            .map(|i| 0.3 * (2.0 * std::f64::consts::PI * 220.0 * (i as f64) / f64::from(sr)).sin())
            .collect();
        AudioBuffer::from_channels(vec![samples.clone(), samples], sr)
            .expect("buf")
            .save_wav(&wav, 24)
            .expect("save");
        let out_wav = dir.path().join("out.wav");

        for bad_exec in [
            vec![
                "forge",
                "unmask",
                wav.to_str().unwrap(),
                "-s",
                wav.to_str().unwrap(),
                "-o",
                out_wav.to_str().unwrap(),
                "--mode",
                "guitar",
            ],
            vec![
                "forge",
                "theory",
                "-o",
                dir.path().join("bad.mid").to_str().unwrap(),
                "--style",
                "baroque",
            ],
            vec![
                "forge",
                "theory",
                "-o",
                dir.path().join("bad2.mid").to_str().unwrap(),
                "--groove",
                "waltz",
            ],
            vec![
                "forge",
                "master",
                wav.to_str().unwrap(),
                "-o",
                out_wav.to_str().unwrap(),
                "--genre",
                "polka",
            ],
        ] {
            let parsed = Cli::try_parse_from(&bad_exec).expect("clap parse");
            assert!(
                run_cli(parsed).is_err(),
                "{bad_exec:?} must error cleanly on unknown enum choice"
            );
        }
    }

    #[test]
    fn cli_end_to_end_audit_theory_f0_daw_and_reaper_execute() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("voc.wav");
        let sr = 48_000_u32;
        let samples: Vec<f64> = (0..sr)
            .map(|i| 0.35 * (2.0 * std::f64::consts::PI * 220.0 * (i as f64) / f64::from(sr)).sin())
            .collect();
        AudioBuffer::from_channels(vec![samples.clone(), samples], sr)
            .expect("buf")
            .save_wav(&wav, 24)
            .expect("save");

        let audit_cli = Cli::try_parse_from(["forge", "audit", wav.to_str().unwrap(), "--json"])
            .expect("audit parse");
        run_cli(audit_cli).expect("audit e2e");

        let abc_in = dir.path().join("sheet.abc");
        std::fs::write(&abc_in, "X:1\nM:4/4\nL:1/4\nK:C\n\"C\"C E G c |\n").expect("abc");
        let out_mid = dir.path().join("theory.mid");
        let out_abc = dir.path().join("out.abc");
        let theory_cli = Cli::try_parse_from([
            "forge",
            "theory",
            "--chords",
            "Dm7 G7 Cmaj7 A7",
            "-o",
            out_mid.to_str().unwrap(),
            "--abc-input",
            abc_in.to_str().unwrap(),
            "--output-abc",
            out_abc.to_str().unwrap(),
        ])
        .expect("theory parse");
        run_cli(theory_cli).expect("theory e2e");
        assert!(out_mid.exists());
        assert!(out_abc.exists());

        let f0_cli = Cli::try_parse_from([
            "forge",
            "f0-track",
            wav.to_str().unwrap(),
            "--backend",
            "rust",
            "--json",
        ])
        .expect("f0-track parse");
        run_cli(f0_cli).expect("f0-track e2e");

        let proj_path = dir.path().join("session.forgeproj");
        let proj = DawProject::new("CliE2E", 48_000, 120.0);
        proj.save_to_file(&proj_path).expect("save proj");
        let daw_cli = Cli::try_parse_from([
            "forge",
            "daw",
            "inspect",
            proj_path.to_str().unwrap(),
            "--json",
        ])
        .expect("daw inspect parse");
        run_cli(daw_cli).expect("daw inspect e2e");

        let jsfx_dir = dir.path().join("jsfx");
        let reaper_cli = Cli::try_parse_from([
            "forge",
            "reaper",
            "--install-jsfx",
            "--jsfx-dir",
            jsfx_dir.to_str().unwrap(),
        ])
        .expect("reaper parse");
        run_cli(reaper_cli).expect("reaper install_jsfx e2e");
        assert!(jsfx_dir.join("Forge_Master_Limiter.jsfx").exists());
    }
}
