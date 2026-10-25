//! `forge` — Unified CLI for ForgeAudio (`audio-forge`).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use forge_core::AudioBuffer;
use forge_dsp::{
    align_phase_gcc_phat, align_vocal_double_dtw, apply_18_band_eq, apply_spectral_defizz,
    apply_spectral_unmask, audit_audio, clean_vocal_surgical, compute_genre_tonal_balance_eq,
    compute_reference_match_eq, deliver, normalize_and_limit, run_full_restoration_chain,
    DeFizzConfig, DeliveryOptions, DeliveryTarget, GenreTonalTarget, RestorationConfig, ToneMatch,
    UnmaskConfig,
};
use forge_reaper::{
    install_jsfx_plugins, send_reaper_osc, validate_isrc, write_distrokid_metadata,
    ReaperProjectBuilder, ReaperTrackSpec, ReleaseMetadata,
};
use forge_theory::{
    export_abc_score, extract_drum_midi_triggers, reharmonize_progression, solve_voice_leading,
    write_midi_file, GrooveStyle, VoicingStyle,
};
use forge_vocal::{
    generate_vocal_harmony_stack, shift_vocal_pitch_formant_preserved, track_vocal_pitch,
    VocalScale,
};
use forge_worker::{
    generate_music_lyria, index_drum_sample_library, separate_stems, WorkerBackend,
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
        /// Sensitivity in dB for OpenSoothe spectral resonance suppression
        #[arg(long, default_value_t = 3.5)]
        defizz_sensitivity: f64,
        /// Maximum suppression depth in dB
        #[arg(long, default_value_t = 9.0)]
        defizz_depth: f64,
        /// Output only the removed harshness/fizz (Delta listen mode)
        #[arg(long)]
        delta_listen: bool,
        /// Fundamental hum frequency (50 or 60 Hz) to notch out with 10 harmonics
        #[arg(long, default_value_t = 60.0)]
        hum_hz: f64,
        /// Airwindows Console8 / ChowTape analog warmth drive (0.0 to 1.0)
        #[arg(long, default_value_t = 0.15)]
        tape_drive: f64,
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
    },

    /// Align multi-mic phase & polarity via GCC-PHAT cross-correlation (Auto-Align equivalent)
    PhaseAlign {
        /// Reference microphone track (e.g. Snare Top or Bass DI)
        reference: PathBuf,
        /// Target microphone track to align (e.g. Overhead or Bass Amp)
        target: PathBuf,
        /// Output phase-aligned WAV path
        #[arg(short, long)]
        output: PathBuf,
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
        #[arg(long, default_value_t = 0.85)]
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
        /// Tempo in BPM
        #[arg(long, default_value_t = 92.0)]
        bpm: f64,
        /// Output multi-track MIDI file (.mid)
        #[arg(short, long)]
        output_midi: PathBuf,
        /// Optional output ABC notation score file (.abc)
        #[arg(long)]
        output_abc: Option<PathBuf>,
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
        #[arg(long, requires = "ceiling_dbtp")]
        lufs: Option<f64>,
        /// Custom true-peak ceiling in dBTP (requires --lufs)
        #[arg(long, requires = "lufs")]
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
    },

    /// Master audio to ITU-R BS.1770-4 LUFS target, -1.0 dBTP True-Peak ceiling, Reference Match EQ & ISO 3901 ISRC
    Master {
        /// Input mix WAV file
        input: PathBuf,
        /// Output mastered WAV path
        #[arg(short, long)]
        output: PathBuf,
        /// Target Integrated LUFS (-14.0 for streaming, -9.0 for competitive club/pop)
        #[arg(long, default_value_t = -14.0)]
        target_lufs: f64,
        /// True-Peak brickwall ceiling in dBTP (-1.0 dBTP recommended by DistroKid/Spotify)
        #[arg(long, default_value_t = -1.0)]
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
        /// Project tempo in BPM
        #[arg(long, default_value_t = 120.0)]
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
    },

    /// Install native ForgeAudio JSFX plugins into REAPER 7.82 or send live UDP OSC commands
    Reaper {
        /// Install all 4 ForgeAudio JSFX plugins into REAPER Effects directory
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { input, json } => {
            let buf = AudioBuffer::load(&input)?;
            let report = audit_audio(&buf);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
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
                    "Stereo Phase  : {:.2} (Sub-160Hz Mono Coherence: {:.2})",
                    report.stereo_correlation, report.sub_160hz_correlation
                );
                println!(
                    "AI Fizz Index : {:.1} / 100.0 | DistroKid Ready: {}",
                    report.ai_fizz_index,
                    if report.distrokid_ready { "YES" } else { "NO" }
                );
                if !report.recommendations.is_empty() {
                    println!("\nRecommendations:");
                    for rec in &report.recommendations {
                        println!("  • {rec}");
                    }
                }
            }
        }

        Commands::Restore {
            input,
            output,
            defizz_sensitivity,
            defizz_depth,
            delta_listen,
            hum_hz,
            tape_drive,
        } => {
            let buf = AudioBuffer::load(&input)?;
            let rest_cfg = RestorationConfig {
                hum_fundamental_hz: if hum_hz > 0.0 { Some(hum_hz) } else { None },
                analog_tape_drive: tape_drive,
                ..RestorationConfig::default()
            };
            let stage1 = run_full_restoration_chain(&buf, &rest_cfg)?;
            let defizz_cfg = DeFizzConfig {
                sensitivity_db: defizz_sensitivity,
                max_depth_db: defizz_depth,
                delta_listen,
                ..DeFizzConfig::default()
            };
            let cleaned = apply_spectral_defizz(&stage1, &defizz_cfg)?;
            cleaned.save_wav(&output, 24)?;
            let post_audit = audit_audio(&cleaned);
            println!(
                "Restored audio saved to {} (Post AI Fizz Index: {:.1}/100, Sub-160Hz Correlation: {:.2})",
                output.display(),
                post_audit.ai_fizz_index,
                post_audit.sub_160hz_correlation
            );
        }

        Commands::Unmask {
            target,
            sidechain,
            output,
            mode,
        } => {
            let t_buf = AudioBuffer::load(&target)?;
            let sc_buf = AudioBuffer::load(&sidechain)?;
            let cfg = if mode.eq_ignore_ascii_case("kick") {
                UnmaskConfig::kick_over_bass()
            } else {
                UnmaskConfig::vocal_over_instruments()
            };
            let unmasked = apply_spectral_unmask(&t_buf, &sc_buf, &cfg)?;
            unmasked.save_wav(&output, 24)?;
            println!("Sidechain-unmasked stem written to {}", output.display());
        }

        Commands::PhaseAlign {
            reference,
            target,
            output,
        } => {
            let ref_buf = AudioBuffer::load(&reference)?;
            let tgt_buf = AudioBuffer::load(&target)?;
            let (aligned, info) = align_phase_gcc_phat(&ref_buf, &tgt_buf, 35.0)?;
            aligned.save_wav(&output, 24)?;
            println!(
                "Phase-aligned track written to {} (Delay corrected: {} samples / {:.2} ms, Polarity inverted: {})",
                output.display(),
                info.delay_samples,
                info.delay_ms,
                info.invert_polarity
            );
        }

        Commands::Vocal {
            input,
            output_dir,
            scale,
            retune,
            harmonize,
            align_to,
        } => {
            std::fs::create_dir_all(&output_dir)?;
            let raw = AudioBuffer::load(&input)?;
            let surgically_clean = clean_vocal_surgical(&raw, true, true, -12.0)?;
            let parsed_scale = VocalScale::parse(&scale)?;
            let contour = track_vocal_pitch(&surgically_clean, &parsed_scale, retune);

            let corr_curve: Vec<(f64, f64)> = contour
                .iter()
                .map(|f| (f.time_secs, f.correction_semitones))
                .collect();
            let mut tuned =
                shift_vocal_pitch_formant_preserved(&surgically_clean, &corr_curve, 0.0)?;

            if let Some(guide_path) = align_to {
                let guide_buf = AudioBuffer::load(&guide_path)?;
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
                let p_up = output_dir.join("Vocal_Harmony_3rd_Above.wav");
                let p_down = output_dir.join("Vocal_Harmony_3rd_Below.wav");
                let p_dbl = output_dir.join("Vocal_Formant_Double.wav");
                h_up.save_wav(&p_up, 24)?;
                h_down.save_wav(&p_down, 24)?;
                h_dbl.save_wav(&p_dbl, 24)?;
                println!(
                    "3-part harmony stack written to:\n  • {}\n  • {}\n  • {}",
                    p_up.display(),
                    p_down.display(),
                    p_dbl.display()
                );
            }
        }

        Commands::Theory {
            chords,
            style,
            groove,
            bpm,
            output_midi,
            output_abc,
        } => {
            let v_style = match style.to_ascii_lowercase().as_str() {
                "drop2" => VoicingStyle::Drop2,
                "close" => VoicingStyle::Close,
                "cinematic" => VoicingStyle::CinematicSpread,
                _ => VoicingStyle::NeoSoulRootless,
            };
            let g_style = match groove.to_ascii_lowercase().as_str() {
                "straight" => GrooveStyle::Straight,
                "mpc58" => GrooveStyle::MpcSwing58,
                "strum" => GrooveStyle::AcousticStrum,
                _ => GrooveStyle::NeoSoulDilla,
            };

            let raw_chords: Vec<String> =
                chords.split_whitespace().map(ToString::to_string).collect();
            let reharm = reharmonize_progression(&raw_chords, v_style);
            let (chord_notes, bass_notes) = solve_voice_leading(&reharm, v_style, g_style, 480, 4)?;

            write_midi_file(
                &output_midi,
                bpm,
                480,
                &[("Voiced_Chords", &chord_notes), ("Sub_Bass", &bass_notes)],
            )?;
            println!(
                "Voice-led MIDI written to {} (Progression: {})",
                output_midi.display(),
                reharm.join(" -> ")
            );

            if let Some(abc_path) = output_abc {
                let abc_str =
                    export_abc_score("ForgeAudio Progression", "C", bpm.round() as u32, &reharm);
                std::fs::write(&abc_path, abc_str)?;
                println!("ABC lead sheet written to {}", abc_path.display());
            }
        }

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
        } => {
            let mut buf = AudioBuffer::load(&input)?;
            if let Some(ref_path) = reference {
                let ref_buf = AudioBuffer::load(&ref_path)?;
                let match_curve = compute_reference_match_eq(&buf, &ref_buf, 4.5);
                apply_18_band_eq(&mut buf, &match_curve);
            } else if let Some(ref g_str) = genre {
                let target = GenreTonalTarget::parse(g_str);
                let genre_curve = compute_genre_tonal_balance_eq(&buf, target, 3.5);
                apply_18_band_eq(&mut buf, &genre_curve);
            }

            let mastered = normalize_and_limit(&buf, target_lufs, ceiling_dbtp)?;
            if let Some(raw_isrc) = isrc {
                let canonical = validate_isrc(&raw_isrc)?;
                let tmp = output.with_extension("pre_tag.wav");
                mastered.save_wav(&tmp, 24)?;
                let meta = ReleaseMetadata {
                    title,
                    artist,
                    album: "Single".to_string(),
                    isrc: canonical.clone(),
                    bpm: None,
                    key: None,
                    year: Some(2026),
                };
                write_distrokid_metadata(&tmp, &output, &meta)?;
                let _ = std::fs::remove_file(&tmp);
                println!(
                    "Mastered & tagged with ISRC {canonical} -> {}",
                    output.display()
                );
            } else {
                mastered.save_wav(&output, 24)?;
            }

            let audit = audit_audio(&mastered);
            println!(
                "Final Master: {:.2} LUFS | {:.2} dBTP True Peak | LRA {:.2} LU | DistroKid Ready: {}",
                audit.integrated_lufs, audit.true_peak_dbtp, audit.loudness_range_lu, audit.distrokid_ready
            );
        }

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
        } => {
            let target = match (lufs, ceiling_dbtp) {
                (Some(lufs), Some(ceiling)) => DeliveryTarget::custom(lufs, ceiling)?,
                _ => DeliveryTarget::parse(&target)?,
            };
            let mut options = DeliveryOptions::for_target(target);
            options.repair = !no_repair;
            options.defizz = !no_defizz;
            options.vocal = vocal;
            options.tone_match = match (reference, genre) {
                (Some(path), _) => Some(ToneMatch::Reference(AudioBuffer::load(&path)?)),
                (None, Some(g)) => Some(ToneMatch::Genre(GenreTonalTarget::parse(&g))),
                (None, None) => None,
            };

            let buf = AudioBuffer::load(&input)?;
            let (master, report) = deliver(&buf, &options)?;
            master.save_wav(&output, 24)?;
            let report_path = output.with_extension("delivery.json");
            std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

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
            println!(
                "{} -> {} (report: {})",
                if report.ready { "READY" } else { "NOT READY" },
                output.display(),
                report_path.display()
            );
            if !report.ready {
                std::process::exit(2);
            }
        }

        Commands::Ingest {
            input,
            output_dir,
            bpm,
            scale,
            chords,
            isrc,
        } => {
            std::fs::create_dir_all(&output_dir)?;
            let stems_dir = output_dir.join("stems");
            let midi_dir = output_dir.join("midi");
            std::fs::create_dir_all(&stems_dir)?;
            std::fs::create_dir_all(&midi_dir)?;

            // 1. Separate stems
            let bundle = separate_stems(&input, &stems_dir, &WorkerBackend::Auto)?;

            // 2. Clean & Spectral De-Fizz the Vocal and Instrumental stems
            let mut vocal_pitch_frames = Vec::new();
            let parsed_scale = VocalScale::parse(&scale)?;
            if let Some(ref voc_path) = bundle.vocals {
                let voc_buf = AudioBuffer::load(voc_path)?;
                let voc_clean = clean_vocal_surgical(&voc_buf, true, true, -10.0)?;
                let voc_defizz = apply_spectral_defizz(&voc_clean, &DeFizzConfig::default())?;
                vocal_pitch_frames = track_vocal_pitch(&voc_defizz, &parsed_scale, 0.85);
                voc_defizz.save_wav(voc_path, 24)?;
            }

            // 3. Extract Drum MIDI Triggers (Kick, Snare, Hats)
            let drum_src = bundle.drums.as_ref().unwrap_or(&input);
            let drum_buf = AudioBuffer::load(drum_src)?;
            let duration_secs = drum_buf.duration_secs();
            let (kick_notes, snare_notes, hat_notes) =
                extract_drum_midi_triggers(&drum_buf, bpm, 480);

            let drum_midi_path = midi_dir.join("01_Drum_Triggers.mid");
            write_midi_file(
                &drum_midi_path,
                bpm,
                480,
                &[
                    ("Kick_Trigger_C1", &kick_notes),
                    ("Snare_Trigger_D1", &snare_notes),
                    ("HiHat_Trigger_F#1", &hat_notes),
                ],
            )?;

            // 4. Solve Voice-Led Chords & Phase-Locked Sub-Bass MIDI
            let raw_chords: Vec<String> =
                chords.split_whitespace().map(ToString::to_string).collect();
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
                bpm,
                480,
                &[("Voiced_Chords", &chord_notes)],
            )?;
            write_midi_file(&bass_midi_path, bpm, 480, &[("Sub_Bass", &bass_notes)])?;

            let abc_path = output_dir.join("Lead_Sheet.abc");
            std::fs::write(
                &abc_path,
                export_abc_score(
                    "ForgeAudio Studio Session",
                    &scale,
                    bpm.round() as u32,
                    &reharm,
                ),
            )?;

            // 5. Render DistroKid-Ready Master (-14.0 LUFS, -1.0 dBTP)
            let full_mix = AudioBuffer::load(&input)?;
            let restored = run_full_restoration_chain(&full_mix, &RestorationConfig::default())?;
            let defizzed = apply_spectral_defizz(&restored, &DeFizzConfig::default())?;
            let mastered = normalize_and_limit(&defizzed, -14.0, -1.0)?;
            let master_path = output_dir.join("Master_DistroKid_24bit_48kHz.wav");
            mastered.save_wav(&master_path, 24)?;

            // 6. Build Complete REAPER 7.82 Multitrack .RPP Project
            let mut rpp = ReaperProjectBuilder::new(
                "ForgeAudio Studio Production",
                bpm,
                full_mix.sample_rate,
            );
            if let Some(raw_isrc) = isrc {
                rpp.isrc = Some(validate_isrc(&raw_isrc)?);
            }
            rpp.add_default_song_structure_markers();

            // Track 0: Lead Vocal (with surgical strip + OpenSoothe + editable Take Pitch Envelope)
            rpp.tracks.push(ReaperTrackSpec {
                name: "01 Lead Vocal (Dry De-Fizzed)".to_string(),
                file_path: bundle.vocals.clone(),
                is_midi: false,
                duration_secs,
                volume_lin: 0.85,
                pan: 0.0,
                color: 0x0155_88FF,
                jsfx_inserts: vec![
                    "Forge_Vocal_Surgical.jsfx".to_string(),
                    "Forge_OpenSoothe_DeFizz.jsfx".to_string(),
                ],
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: vocal_pitch_frames,
            });

            // Track 1: Drums Stem
            rpp.tracks.push(ReaperTrackSpec {
                name: "02 Drums Stem".to_string(),
                file_path: bundle.drums.clone(),
                is_midi: false,
                duration_secs,
                volume_lin: 0.80,
                pan: 0.0,
                color: 0x0144_CC44,
                jsfx_inserts: Vec::new(),
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });

            // Track 2: Bass Stem (Sidechain Unmasked by Track 1 Drums + Sub-Bass Mono Reinforce)
            rpp.tracks.push(ReaperTrackSpec {
                name: "03 Bass Stem (Sidechain Unmasked + Mono Sub)".to_string(),
                file_path: bundle.bass.clone(),
                is_midi: false,
                duration_secs,
                volume_lin: 0.82,
                pan: 0.0,
                color: 0x01FF_8844,
                jsfx_inserts: vec![
                    "Forge_SubBass_Reinforce.jsfx".to_string(),
                    "Forge_OpenSpacer_Unmask.jsfx".to_string(),
                ],
                sidechain_from_track: Some(1),
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });

            // Track 3: Instruments / Synths / Guitars (Sidechain Unmasked by Track 0 Lead Vocal)
            rpp.tracks.push(ReaperTrackSpec {
                name: "04 Instruments Bus (Vocal Unmasked + De-Fizzed)".to_string(),
                file_path: bundle.other.clone(),
                is_midi: false,
                duration_secs,
                volume_lin: 0.78,
                pan: 0.0,
                color: 0x01CC_66FF,
                jsfx_inserts: vec![
                    "Forge_OpenSoothe_DeFizz.jsfx".to_string(),
                    "Forge_OpenSpacer_Unmask.jsfx".to_string(),
                ],
                sidechain_from_track: Some(0),
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });

            // Tracks 4-6: Aligned MIDI Tracks (Drum Triggers, Voiced Chords, Sub-Bass)
            rpp.tracks.push(ReaperTrackSpec {
                name: "05 MIDI Drum Triggers (Kick C1 / Snare D1 / Hat F#1)".to_string(),
                file_path: Some(drum_midi_path),
                is_midi: true,
                duration_secs,
                volume_lin: 0.90,
                pan: 0.0,
                color: 0x0144_FFAA,
                jsfx_inserts: Vec::new(),
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });
            rpp.tracks.push(ReaperTrackSpec {
                name: "06 MIDI Voiced Chords (Neo-Soul Drop-2)".to_string(),
                file_path: Some(chords_midi_path),
                is_midi: true,
                duration_secs,
                volume_lin: 0.85,
                pan: 0.0,
                color: 0x01FF_CC44,
                jsfx_inserts: Vec::new(),
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });
            rpp.tracks.push(ReaperTrackSpec {
                name: "07 MIDI Phase-Pure Sub-Bass / 808".to_string(),
                file_path: Some(bass_midi_path),
                is_midi: true,
                duration_secs,
                volume_lin: 0.88,
                pan: 0.0,
                color: 0x01FF_6644,
                jsfx_inserts: Vec::new(),
                sidechain_from_track: None,
                stretch_markers: Vec::new(),
                pitch_envelope: Vec::new(),
            });

            let rpp_path = output_dir.join("ForgeAudio_Session.rpp");
            rpp.save_rpp(&rpp_path)?;

            println!("=== ForgeAudio 1-Command Studio Ingest Complete ===");
            println!("REAPER 7.82 Project : {}", rpp_path.display());
            println!("DistroKid Master WAV: {}", master_path.display());
            println!("ABC Lead Sheet      : {}", abc_path.display());
        }

        Commands::Reaper {
            install_jsfx,
            jsfx_dir,
            osc_addr,
            osc_host,
        } => {
            if install_jsfx {
                let installed = install_jsfx_plugins(jsfx_dir.as_deref())?;
                println!("Installed {} ForgeAudio JSFX plugins:", installed.len());
                for p in installed {
                    println!("  • {}", p.display());
                }
            }
            if let Some(addr) = osc_addr {
                send_reaper_osc(&osc_host, &addr, Some(1.0))?;
                println!("Sent UDP OSC message '{addr}' to {osc_host}");
            }
        }

        Commands::SampleIndex { folder } => {
            let indexed = index_drum_sample_library(&folder)?;
            println!("{}", serde_json::to_string_pretty(&indexed)?);
        }

        Commands::Lyria {
            prompt,
            output,
            model,
        } => {
            let (saved, lyrics) = generate_music_lyria(&prompt, &model, &output).await?;
            println!("Lyria audio saved to {}", saved.display());
            if let Some(l) = lyrics {
                println!("\nGenerated Structure / Lyrics:\n{l}");
            }
        }
    }

    Ok(())
}
