#!/usr/bin/env python3
"""Ephemeral GPU Worker for ForgeAudio (`audio-forge`).

Executes heavy PyTorch/ONNX models on the local NVIDIA RTX 3080 or remote Colab A100
and exits immediately after writing artifacts to disk so 100% of VRAM and system RAM
are returned to REAPER 7.82.

Supported tasks:
- `separate`: Mel-Band-RoFormer (Kim Vocal, MIT) + optional BS-RoFormer de-reverb (`--stems 2`),
  or HTDemucs 6-stem separation (`--stems 6`: vocals, drums, bass, guitar, keys, other).
- `denoise`: Neural audio denoising via DeepFilterNet3 (`df.enhance`) with automatic fallback
  to Mel-Band-RoFormer denoise (`audio-separator`) for heavy noise/room rescue.
- `transcribe`: Polyphonic audio-to-MIDI transcription via Spotify `basic-pitch`.
- `yue2-remote`: Calls mrfakename/yue2-3b or a custom Colab YuE2-3B endpoint.
"""

from __future__ import annotations

import argparse
import array
import json
import sys
import wave
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

# Normalized stem tags (lowercase, alphanumerics only) for each role.
VOCAL_TAGS = frozenset({"vocals", "vocal"})
INSTRUMENTAL_TAGS = frozenset({"instrumental", "instrumentals", "other"})
DRUMS_TAGS = frozenset({"drums", "drum"})
BASS_TAGS = frozenset({"bass"})
GUITAR_TAGS = frozenset({"guitar"})
KEYS_TAGS = frozenset({"piano", "keys", "keyboard"})
KICK_TAGS = frozenset({"kick", "bd", "bassdrum"})
SNARE_TAGS = frozenset({"snare", "sd"})
TOMS_TAGS = frozenset({"toms", "tom"})
HIHAT_TAGS = frozenset({"hh", "hihat", "hihats", "hats"})
CYMBALS_TAGS = frozenset({"cymbals", "cymbal", "crash", "ride", "overheads", "oh"})
LEAD_VOCAL_TAGS = frozenset({"vocals", "vocal", "lead", "leadvocals", "karaoke"})
BACKING_VOCAL_TAGS = frozenset(
    {"instrumental", "backing", "backingvocals", "harmonies", "nolead", "other"}
)
MALE_VOCAL_TAGS = frozenset({"male", "malevocal", "malevocals"})
FEMALE_VOCAL_TAGS = frozenset({"female", "femalevocal", "femalevocals"})
ELECTRIC_GUITAR_TAGS = frozenset({"electric", "electricguitar", "eguitar"})
ACOUSTIC_GUITAR_TAGS = frozenset({"acoustic", "acousticguitar", "aguitar", "other"})
STRINGS_TAGS = frozenset({"strings", "bowedstrings", "violin", "cello"})
BRASS_TAGS = frozenset({"brass", "horns", "trumpet", "trombone"})
WIND_TAGS = frozenset({"wind", "winds", "woodwinds", "reeds", "flute", "sax"})
SYNTH_TAGS = frozenset({"synth", "synthesizer", "pad", "lead"})
CROWD_TAGS = frozenset({"crowd", "applause", "audience", "noise"})
REVERB_TAGS = frozenset({"reverb", "wet", "echo", "ambience"})
DRY_TAGS = frozenset({"noreverb", "dry"})
DENOISED_TAGS = frozenset({"dry", "nonoise", "denoised", "clean"})

BS_ROFORMER_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL_ROFORMER_VOCAL_MODEL = "model_mel_band_roformer_kim_vocals.ckpt"
DEREVERB_MODEL = "deverb_bs_roformer_8_384dim_10depth.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
MALE_FEMALE_MODEL = "mel_band_roformer_male_female_aufr33.ckpt"
GUITAR_SUB_MODEL = "mel_band_roformer_guitar_elec_acoust.ckpt"
STRINGS_MODEL = "mel_band_roformer_strings.ckpt"
BRASS_MODEL = "mel_band_roformer_brass.ckpt"
WIND_MODEL = "mel_band_roformer_woodwinds.ckpt"
SYNTH_MODEL = "mel_band_roformer_synth.ckpt"
CROWD_MODEL = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
DENOISE_ROFORMER_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
SCNET_FOUR_STEM_MODEL = "scnet_xl_ihf_sdr_10.08.ckpt"
BS_ROFORMER_SIX_STEM_MODEL = "BS-Roformer-SW.ckpt"
FOUR_STEM_MODEL = "htdemucs_ft.yaml"
SIX_STEM_MODEL = "htdemucs_6s.yaml"
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
APOLLO_RESTORE_MODEL = "apollo_universal_model.ckpt"
VALID_QUALITIES = frozenset({"fast", "studio", "ultra"})


def stem_tag(filename: str) -> str | None:
    """Return the stem tag audio-separator put in an output name, or None.

    Output names look like ``<input>_(<Stem>)_<model>.wav``. The input name can contain
    anything, including ``_(Vocals)_``, so only the last ``_(...)`` group counts.
    """
    name = Path(filename).name
    start = name.rfind("_(")
    end = name.find(")", start + 2) if start >= 0 else -1
    if end < 0:
        return None
    return name[start + 2 : end]


def _normalize(tag: str) -> str:
    return "".join(ch for ch in tag.lower() if ch.isalnum())


def pick_stem(
    files: list[str], wanted: frozenset[str], output_dir: Path
) -> Path | None:
    """Return the first output whose stem tag is in ``wanted``, verified inside output_dir."""
    resolved_dir = output_dir.resolve()
    for f in files:
        tag = stem_tag(f)
        if tag is not None and _normalize(tag) in wanted:
            path = Path(f)
            candidate = path if path.is_absolute() else output_dir / path
            resolved = candidate.resolve()
            if not resolved.is_relative_to(resolved_dir):
                raise RuntimeError(
                    f"Separator wrote stem outside output_dir: {resolved}"
                )
            return candidate
    return None


def _blend_wav_pair(
    path_a: Path, path_b: Path, out_path: Path, weight_a: float = 0.6
) -> Path:
    """Weighted float64 PCM blend of two matching 16-bit WAV stems (stdlib only).

    Falls back to ``path_a`` if inputs are not readable 16-bit PCM WAVs (e.g. in unit mocks).
    """
    try:
        with wave.open(str(path_a), "rb") as wa, wave.open(str(path_b), "rb") as wb:
            if (
                wa.getnchannels() != wb.getnchannels()
                or wa.getsampwidth() != wb.getsampwidth()
                or wa.getframerate() != wb.getframerate()
                or wa.getsampwidth() != 2
            ):
                return path_a
            nframes = min(wa.getnframes(), wb.getnframes())
            sa = array.array("h")
            sb = array.array("h")
            sa.frombytes(wa.readframes(nframes))
            sb.frombytes(wb.readframes(nframes))
            wb_f = 1.0 - weight_a
            blended = array.array(
                "h",
                (
                    max(-32768, min(32767, round(a * weight_a + b * wb_f)))
                    for a, b in zip(sa, sb, strict=False)
                ),
            )
            with wave.open(str(out_path), "wb") as wo:
                wo.setnchannels(wa.getnchannels())
                wo.setsampwidth(2)
                wo.setframerate(wa.getframerate())
                wo.writeframes(blended.tobytes())
            return out_path
    except (wave.Error, OSError):
        return path_a


def _run_vocal_stage(
    sep: object,
    resolved_input: Path,
    resolved_output: Path,
    quality: str,
) -> tuple[Path | None, Path | None]:
    """Run Stage 1 vocal/instrumental extraction according to quality tier."""
    if quality == "fast":
        sep.load_model(model_filename=MEL_ROFORMER_VOCAL_MODEL)  # type: ignore[attr-defined]
        files = sep.separate(str(resolved_input))  # type: ignore[attr-defined]
        return (
            pick_stem(files, VOCAL_TAGS, resolved_output),
            pick_stem(files, INSTRUMENTAL_TAGS, resolved_output),
        )

    # Studio & Ultra start with MVSEP SOTA BS-RoFormer Viperx 1297 (~12.98 dB SDR)
    sep.load_model(model_filename=BS_ROFORMER_VOCAL_MODEL)  # type: ignore[attr-defined]
    bs_files = sep.separate(str(resolved_input))  # type: ignore[attr-defined]
    bs_vocal = pick_stem(bs_files, VOCAL_TAGS, resolved_output)
    bs_inst = pick_stem(bs_files, INSTRUMENTAL_TAGS, resolved_output)

    if quality != "ultra":
        return bs_vocal, bs_inst

    # Ultra: Dual-RoFormer ensemble (BS-RoFormer 0.6 + Mel-Band-RoFormer 0.4)
    sep.load_model(model_filename=MEL_ROFORMER_VOCAL_MODEL)  # type: ignore[attr-defined]
    mel_files = sep.separate(str(resolved_input))  # type: ignore[attr-defined]
    mel_vocal = pick_stem(mel_files, VOCAL_TAGS, resolved_output)
    mel_inst = pick_stem(mel_files, INSTRUMENTAL_TAGS, resolved_output)

    base = _safe_stem(resolved_input)
    if bs_vocal and mel_vocal and bs_vocal.exists() and mel_vocal.exists():
        vocal_out = (
            resolved_output / f"{base}_(Vocals)_roformer_ensemble.wav"
        ).resolve()
        if vocal_out.is_relative_to(resolved_output):
            bs_vocal = _blend_wav_pair(bs_vocal, mel_vocal, vocal_out, weight_a=0.6)
    if bs_inst and mel_inst and bs_inst.exists() and mel_inst.exists():
        inst_out = (
            resolved_output / f"{base}_(Instrumental)_roformer_ensemble.wav"
        ).resolve()
        if inst_out.is_relative_to(resolved_output):
            bs_inst = _blend_wav_pair(bs_inst, mel_inst, inst_out, weight_a=0.6)
    return bs_vocal, bs_inst


def _separate_multistem_with_fallback(
    sep: object,
    cascade_input: Path,
    stems: int,
) -> list[str]:
    """Run SOTA multi-stem separation (SCNet XL 4-stem / BS-RoFormer SW 6-stem) with HTDemucs fallback."""
    primary_model = SCNET_FOUR_STEM_MODEL if stems == 4 else BS_ROFORMER_SIX_STEM_MODEL
    fallback_model = FOUR_STEM_MODEL if stems == 4 else SIX_STEM_MODEL
    try:
        sep.load_model(model_filename=primary_model)  # type: ignore[attr-defined]
        return sep.separate(str(cascade_input))  # type: ignore[attr-defined]
    except (RuntimeError, FileNotFoundError, ValueError, OSError):
        sep.load_model(model_filename=fallback_model)  # type: ignore[attr-defined]
        return sep.separate(str(cascade_input))  # type: ignore[attr-defined]


def _restore_wav_bandwidth(src_path: Path, out_path: Path, drive: float = 0.12) -> Path:
    """Phase-coherent spectral bandwidth restoration fallback for heavily limited 16-bit PCM WAVs.

    Synthesizes gentle upper-band harmonic air from the first-order high-frequency difference
    signal and blends it at ``drive`` (~ -18 dB). Falls back to ``src_path`` for non-16-bit/mock files.
    """
    try:
        with wave.open(str(src_path), "rb") as wi:
            if wi.getsampwidth() != 2:
                return src_path
            nch = wi.getnchannels()
            sr = wi.getframerate()
            samples = array.array("h")
            samples.frombytes(wi.readframes(wi.getnframes()))
        if len(samples) <= nch:
            return src_path
        restored = array.array("h", samples)
        inv_norm = 1.0 / 32768.0
        for ch in range(nch):
            prev = samples[ch] * inv_norm
            for idx in range(ch + nch, len(samples), nch):
                cur = samples[idx] * inv_norm
                hp = cur - prev
                prev = cur
                # Soft cubic harmonic air excitation strictly on high-frequency transients/air
                air = hp * (1.0 + 0.5 * hp * hp)
                y = cur + drive * air
                restored[idx] = max(-32768, min(32767, round(y * 32767.0)))
        with wave.open(str(out_path), "wb") as wo:
            wo.setnchannels(nch)
            wo.setsampwidth(2)
            wo.setframerate(sr)
            wo.writeframes(restored.tobytes())
        return out_path
    except (wave.Error, OSError):
        return src_path


def _restore_stem(
    sep: object,
    stem_path: Path | None,
    resolved_output: Path,
) -> Path | None:
    """Restore high-frequency bandwidth on a separated stem via Apollo or spectral exciter fallback."""
    if stem_path is None or not stem_path.exists():
        return stem_path
    try:
        sep.load_model(model_filename=APOLLO_RESTORE_MODEL)  # type: ignore[attr-defined]
        out_files = sep.separate(str(stem_path))  # type: ignore[attr-defined]
        if out_files:
            cand = Path(out_files[0])
            resolved = (
                cand if cand.is_absolute() else resolved_output / cand
            ).resolve()
            if resolved.is_relative_to(resolved_output) and resolved.is_file():
                return resolved
    except (RuntimeError, FileNotFoundError, ValueError, OSError):
        pass
    out_path = (resolved_output / f"{_safe_stem(stem_path)}_restored.wav").resolve()
    if not out_path.is_relative_to(resolved_output):
        return stem_path
    return _restore_wav_bandwidth(stem_path, out_path)


def _try_optional_split(
    sep: object,
    src_path: Path | None,
    model_filename: str,
    primary_tags: frozenset[str],
    residual_tags: frozenset[str],
    resolved_output: Path,
) -> tuple[Path | None, Path | None]:
    """Run an optional binary RoFormer sub-splitter; returns ``(None, src_path)`` if unavailable."""
    if src_path is None or not src_path.exists():
        return None, src_path
    try:
        sep.load_model(model_filename=model_filename)  # type: ignore[attr-defined]
        out_files = sep.separate(str(src_path))  # type: ignore[attr-defined]
    except (RuntimeError, FileNotFoundError, ValueError, OSError):
        return None, src_path
    primary = pick_stem(out_files, primary_tags, resolved_output)
    residual = pick_stem(out_files, residual_tags, resolved_output) or src_path
    return primary, residual


def _run_orchestral_sub_splitters(
    sep: object,
    residual: Path | None,
    resolved_output: Path,
    base_stem: str,
) -> tuple[dict[str, Path | None], Path | None]:
    """Cascade Strings, Brass, Woodwinds, and Synth RoFormer sub-splitters on residual ``other``."""
    extracted: dict[str, Path | None] = {
        "strings": None,
        "brass": None,
        "wind": None,
        "synth": None,
    }
    specs = (
        ("strings", STRINGS_MODEL, STRINGS_TAGS),
        ("brass", BRASS_MODEL, BRASS_TAGS),
        ("wind", WIND_MODEL, WIND_TAGS),
        ("synth", SYNTH_MODEL, SYNTH_TAGS),
    )
    for key, model_name, tags in specs:
        stem, next_residual = _try_optional_split(
            sep, residual, model_name, tags, INSTRUMENTAL_TAGS, resolved_output
        )
        extracted[key] = stem
        # Keep cascaded residual filenames short so multi-stage runs never exceed OS NAME_MAX (255 B).
        if (
            next_residual is not None
            and next_residual != residual
            and next_residual.exists()
        ):
            short_res = (
                resolved_output / f"{base_stem}_(Residual_{key}).wav"
            ).resolve()
            if short_res.is_relative_to(resolved_output):
                next_residual.replace(short_res)
                next_residual = short_res
        residual = next_residual
    return extracted, residual


def run_separate(
    input_path: Path,
    output_dir: Path,
    dereverb: bool = True,
    stems: int = 2,
    quality: str = "studio",
    drum_sep: bool = False,
    split_lead_backing: bool = False,
    split_male_female: bool = False,
    split_guitar: bool = False,
    split_orchestral: bool = False,
    split_crowd: bool = False,
    restore: bool = False,
) -> dict[str, str | None]:
    if stems not in {2, 4, 6}:
        raise ValueError(f"stems must be 2, 4, or 6, got {stems!r}")
    if quality not in VALID_QUALITIES:
        raise ValueError(
            f"quality must be one of {sorted(VALID_QUALITIES)}, got {quality!r}"
        )
    resolved_input = input_path.resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input audio file not found: {resolved_input}")
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator is not installed in the worker venv. "
            "Run: uv sync --project python-worker --extra gpu (or --extra cpu for CPU)"
        ) from exc

    sep = Separator(output_dir=str(resolved_output), output_format="WAV")
    drums: Path | None = None
    bass: Path | None = None
    guitar: Path | None = None
    keys: Path | None = None
    other: Path | None = None
    kick: Path | None = None
    snare: Path | None = None
    toms: Path | None = None
    hihat: Path | None = None
    cymbals: Path | None = None
    lead_vocals: Path | None = None
    backing_vocals: Path | None = None
    male_vocals: Path | None = None
    female_vocals: Path | None = None
    vocal_reverb: Path | None = None
    electric_guitar: Path | None = None
    acoustic_guitar: Path | None = None
    crowd: Path | None = None
    orch_stems: dict[str, Path | None] = {
        "strings": None,
        "brass": None,
        "wind": None,
        "synth": None,
    }

    if stems == 2:
        vocal_file, other = _run_vocal_stage(
            sep, resolved_input, resolved_output, quality
        )
    elif quality == "fast":
        # Fast single-pass 4-stem or 6-stem Demucs
        sep.load_model(model_filename=FOUR_STEM_MODEL if stems == 4 else SIX_STEM_MODEL)
        files = sep.separate(str(resolved_input))
        vocal_file = pick_stem(files, VOCAL_TAGS, resolved_output)
        drums = pick_stem(files, DRUMS_TAGS, resolved_output)
        bass = pick_stem(files, BASS_TAGS, resolved_output)
        guitar = pick_stem(files, GUITAR_TAGS, resolved_output) if stems == 6 else None
        keys = pick_stem(files, KEYS_TAGS, resolved_output) if stems == 6 else None
        other = pick_stem(files, INSTRUMENTAL_TAGS, resolved_output)
    else:
        # Studio / Ultra Pro Cascade:
        # Stage 1: Bleedless BS-RoFormer (or Dual-RoFormer Ensemble) extracts Vocals + Instrumental
        vocal_file, inst_file = _run_vocal_stage(
            sep, resolved_input, resolved_output, quality
        )
        # Stage 3: SOTA SCNet XL (4-stem) / BS-RoFormer SW (6-stem) on vocal-free Instrumental
        #          with automatic HTDemucs fallback if checkpoint is unavailable.
        cascade_input = (
            inst_file
            if (inst_file is not None and inst_file.exists())
            else resolved_input
        )
        files = _separate_multistem_with_fallback(sep, cascade_input, stems)
        drums = pick_stem(files, DRUMS_TAGS, resolved_output)
        bass = pick_stem(files, BASS_TAGS, resolved_output)
        guitar = pick_stem(files, GUITAR_TAGS, resolved_output) if stems == 6 else None
        keys = pick_stem(files, KEYS_TAGS, resolved_output) if stems == 6 else None
        other = pick_stem(files, INSTRUMENTAL_TAGS, resolved_output) or inst_file

    # Stage 2a: Optional BS-RoFormer De-Reverb on isolated vocal stem (exposes both dry & wet tail)
    if dereverb and vocal_file and vocal_file.exists():
        sep.load_model(model_filename=DEREVERB_MODEL)
        deverb_files = sep.separate(str(vocal_file))
        vocal_reverb = pick_stem(deverb_files, REVERB_TAGS, resolved_output)
        vocal_file = pick_stem(deverb_files, DRY_TAGS, resolved_output) or vocal_file

    # Stage 2b: Optional RoFormer Karaoke pass (Lead Vocals vs. Backing Vocals / Harmonies)
    if split_lead_backing and vocal_file and vocal_file.exists():
        sep.load_model(model_filename=KARAOKE_MODEL)
        karaoke_files = sep.separate(str(vocal_file))
        lead_vocals = pick_stem(karaoke_files, LEAD_VOCAL_TAGS, resolved_output)
        backing_vocals = pick_stem(karaoke_files, BACKING_VOCAL_TAGS, resolved_output)

    # Stage 2c: Optional Male vs. Female Vocal split
    if split_male_female:
        male_vocals, female_cand = _try_optional_split(
            sep,
            vocal_file,
            MALE_FEMALE_MODEL,
            MALE_VOCAL_TAGS,
            FEMALE_VOCAL_TAGS,
            resolved_output,
        )
        female_vocals = female_cand if male_vocals is not None else None

    # Stage 4: Optional 5-piece DrumSep on isolated drums stem
    if drum_sep and drums and drums.exists():
        sep.load_model(model_filename=DRUMSEP_MODEL)
        drum_files = sep.separate(str(drums))
        kick = pick_stem(drum_files, KICK_TAGS, resolved_output)
        snare = pick_stem(drum_files, SNARE_TAGS, resolved_output)
        toms = pick_stem(drum_files, TOMS_TAGS, resolved_output)
        hihat = pick_stem(drum_files, HIHAT_TAGS, resolved_output)
        cymbals = pick_stem(drum_files, CYMBALS_TAGS, resolved_output)

    # Stage 5: Optional Electric vs. Acoustic Guitar sub-split
    if split_guitar and guitar and guitar.exists():
        electric_guitar, acoustic_cand = _try_optional_split(
            sep,
            guitar,
            GUITAR_SUB_MODEL,
            ELECTRIC_GUITAR_TAGS,
            ACOUSTIC_GUITAR_TAGS,
            resolved_output,
        )
        acoustic_guitar = acoustic_cand if electric_guitar is not None else None

    # Stage 6: Optional Orchestral & Synth sub-splitters on residual `other`
    if split_orchestral and other and other.exists():
        orch_stems, other = _run_orchestral_sub_splitters(
            sep, other, resolved_output, _safe_stem(resolved_input)
        )

    # Stage 7: Optional Crowd / Audience / Live Room extraction
    if split_crowd:
        crowd, other = _try_optional_split(
            sep, other, CROWD_MODEL, CROWD_TAGS, INSTRUMENTAL_TAGS, resolved_output
        )

    stems_map: dict[str, Path | None] = {
        "vocals": vocal_file,
        "lead_vocals": lead_vocals,
        "backing_vocals": backing_vocals,
        "male_vocals": male_vocals,
        "female_vocals": female_vocals,
        "vocal_reverb": vocal_reverb,
        "drums": drums,
        "bass": bass,
        "guitar": guitar,
        "electric_guitar": electric_guitar,
        "acoustic_guitar": acoustic_guitar,
        "keys": keys,
        "strings": orch_stems["strings"],
        "brass": orch_stems["brass"],
        "wind": orch_stems["wind"],
        "synth": orch_stems["synth"],
        "crowd": crowd,
        "other": other,
        "kick": kick,
        "snare": snare,
        "toms": toms,
        "hihat": hihat,
        "cymbals": cymbals,
    }

    # Stage 8: Optional Apollo / spectral bandwidth restoration post-pass
    if restore:
        stems_map = {
            k: _restore_stem(sep, v, resolved_output) for k, v in stems_map.items()
        }

    return {k: (str(v) if v else None) for k, v in stems_map.items()}


def run_transcribe(input_path: Path, output_dir: Path) -> dict[str, str]:
    """Run polyphonic audio-to-MIDI transcription via Spotify basic-pitch."""
    resolved_input = input_path.resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input audio file not found: {resolved_input}")
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise RuntimeError(
            "basic-pitch is not installed in the worker venv. "
            "Run: uv sync --project python-worker"
        ) from exc

    _model_output, midi_data, _note_events = predict(str(resolved_input))
    out_midi = (
        resolved_output / f"{_safe_stem(resolved_input)}_basic_pitch.mid"
    ).resolve()
    if not out_midi.is_relative_to(resolved_output):
        raise ValueError(f"Refusing to write outside output directory: {out_midi}")
    midi_data.write(str(out_midi))
    return {"midi": str(out_midi)}


def _safe_stem(path: Path) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in path.stem
    )
    return cleaned.strip("._") or "audio"


def _denoise_with_deepfilternet(
    input_path: Path, output_dir: Path, attenuation_limit_db: float | None
) -> Path:
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df()
    audio, _ = load_audio(str(input_path), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio, atten_lim_db=attenuation_limit_db)
    out_path = (
        output_dir / f"{_safe_stem(input_path)}_denoised_deepfilternet.wav"
    ).resolve()
    if not out_path.is_relative_to(output_dir):
        raise ValueError(f"Refusing to write outside output directory: {out_path}")
    save_audio(str(out_path), enhanced, df_state.sr())
    return out_path


def _denoise_with_roformer(input_path: Path, output_dir: Path) -> Path:
    from audio_separator.separator import Separator

    sep = Separator(output_dir=str(output_dir), output_format="WAV")
    sep.load_model(model_filename=DENOISE_ROFORMER_MODEL)
    outputs = sep.separate(str(input_path))
    cleaned = pick_stem(outputs, DENOISED_TAGS, output_dir)
    if cleaned is None:
        raise RuntimeError(
            f"RoFormer denoise produced no clean stem in outputs: {outputs!r}"
        )
    resolved_cleaned = cleaned.resolve()
    if not resolved_cleaned.is_relative_to(output_dir):
        raise ValueError(
            f"Refusing to return path outside output directory: {resolved_cleaned}"
        )
    return resolved_cleaned


def run_denoise(
    input_path: Path,
    output_dir: Path,
    attenuation_limit_db: float | None = None,
    backend: str = "auto",
) -> dict[str, str]:
    """Run neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer denoise."""
    if backend not in {"auto", "deepfilternet", "mel-band-roformer"}:
        raise ValueError(
            f"Unsupported denoise backend {backend!r}; "
            "expected auto, deepfilternet, or mel-band-roformer"
        )
    if attenuation_limit_db is not None and not (0.0 <= attenuation_limit_db <= 100.0):
        raise ValueError(
            f"attenuation_limit_db must be finite and in [0, 100], got {attenuation_limit_db!r}"
        )
    resolved_input = input_path.resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input audio file not found: {resolved_input}")
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    if backend in {"auto", "deepfilternet"}:
        try:
            out_path = _denoise_with_deepfilternet(
                resolved_input, resolved_output, attenuation_limit_db
            )
            return {"denoised": str(out_path), "backend": "deepfilternet"}
        except (ImportError, RuntimeError, OSError) as exc:
            if backend == "deepfilternet":
                raise RuntimeError(
                    "deepfilternet failed or is not installed in the worker venv. "
                    "DeepFilterNet3 pins numpy<2; run it isolated via "
                    "`uv run --with deepfilternet` or use --backend auto/mel-band-roformer."
                ) from exc

    try:
        out_path = _denoise_with_roformer(resolved_input, resolved_output)
        return {"denoised": str(out_path), "backend": "mel-band-roformer"}
    except ImportError as exc:
        raise RuntimeError(
            "Neither deepfilternet nor audio-separator is installed in the worker venv."
        ) from exc


def run_yue2_remote(
    style: str,
    lyrics: str,
    planning_mode: str,
    ode_steps: int,
    seed: int,
    space_id: str,
    timeout_s: float = 600.0,
) -> dict[str, str]:
    if not (timeout_s > 0.0 and timeout_s < float("inf")):
        raise ValueError(f"timeout_s must be positive and finite, got {timeout_s!r}")
    from gradio_client import Client

    client = Client(space_id)
    job = client.submit(
        style=style,
        lyrics=lyrics,
        planning_mode=planning_mode,
        render_quality=ode_steps,
        seed=seed,
        api_name="/generate_song",
    )
    try:
        mp3_path, flac_path, abc_score, elapsed = job.result(timeout=timeout_s)
    except FutureTimeoutError as exc:
        job.cancel()
        raise RuntimeError(
            f"YuE2 Space '{space_id}' did not answer within {timeout_s:g} s"
        ) from exc
    return {
        "mp3_path": str(mp3_path),
        "flac_path": str(flac_path),
        "abc_score": str(abc_score),
        "elapsed": str(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAudio Ephemeral GPU Worker")
    sub = parser.add_subparsers(dest="command", required=True)

    sep_p = sub.add_parser(
        "separate",
        help="Run studio-grade BS-RoFormer/Mel-RoFormer cascade (2/4/6-stem) + optional DrumSep",
    )
    sep_p.add_argument("--input", type=Path, required=True)
    sep_p.add_argument("--output-dir", type=Path, required=True)
    sep_p.add_argument("--no-dereverb", action="store_true")
    sep_p.add_argument("--stems", type=int, default=2, choices=[2, 4, 6])
    sep_p.add_argument(
        "--quality",
        default="studio",
        choices=["fast", "studio", "ultra"],
        help="Separation quality tier: fast (1-pass), studio (BS-RoFormer 1297 cascade), ultra (dual-RoFormer ensemble)",
    )
    sep_p.add_argument(
        "--drum-sep",
        action="store_true",
        help="Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals (MDX23C DrumSep)",
    )
    sep_p.add_argument(
        "--split-lead-backing",
        action="store_true",
        help="Also split isolated vocals into Lead Vocals and Backing Vocals / Harmonies (RoFormer Karaoke)",
    )
    sep_p.add_argument(
        "--split-male-female",
        action="store_true",
        help="Also split isolated vocals into Male and Female vocal stems (Mel-Band-RoFormer)",
    )
    sep_p.add_argument(
        "--split-guitar",
        action="store_true",
        help="Also split isolated guitar into Electric Guitar and Acoustic Guitar",
    )
    sep_p.add_argument(
        "--split-orchestral",
        action="store_true",
        help="Also extract Strings, Brass, Woodwinds, and Synth from residual Other stem",
    )
    sep_p.add_argument(
        "--split-crowd",
        action="store_true",
        help="Also extract live Crowd / Audience / Room Noise from residual Other stem",
    )
    sep_p.add_argument(
        "--all-stems",
        action="store_true",
        help="Enable full 23-slot hierarchical pro cascade (6-stem + DrumSep + Karaoke + Male/Female + Guitar + Orchestral/Synth + Crowd)",
    )
    sep_p.add_argument(
        "--restore",
        action="store_true",
        help="Run optional Apollo / spectral bandwidth restoration post-pass on separated stems",
    )

    tr_p = sub.add_parser(
        "transcribe", help="Run polyphonic audio-to-MIDI transcription via basic-pitch"
    )
    tr_p.add_argument("--input", type=Path, required=True)
    tr_p.add_argument("--output-dir", type=Path, required=True)

    den_p = sub.add_parser(
        "denoise", help="Run neural denoising (DeepFilterNet3 / Mel-Band-RoFormer)"
    )
    den_p.add_argument("--input", type=Path, required=True)
    den_p.add_argument("--output-dir", type=Path, required=True)
    den_p.add_argument(
        "--attenuation-limit-db",
        type=float,
        default=None,
        help="Optional max attenuation limit in dB (0..100, DeepFilterNet3 only)",
    )
    den_p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "deepfilternet", "mel-band-roformer"],
    )

    yue_p = sub.add_parser(
        "yue2-remote", help="Generate song + ABC score via YuE2-3B endpoint"
    )
    yue_p.add_argument("--style", required=True)
    yue_p.add_argument("--lyrics", required=True)
    yue_p.add_argument(
        "--planning-mode", default="full", choices=["full", "melody", "off"]
    )
    yue_p.add_argument("--ode-steps", type=int, default=32, choices=[16, 32])
    yue_p.add_argument("--seed", type=int, default=42)
    yue_p.add_argument("--space", default="mrfakename/yue2-3b")
    yue_p.add_argument(
        "--timeout", type=float, default=600.0, help="Seconds to wait for the Space"
    )

    args = parser.parse_args()
    try:
        res: Mapping[str, str | None]
        if args.command == "separate":
            all_on = bool(args.all_stems)
            res = run_separate(
                args.input,
                args.output_dir,
                dereverb=not args.no_dereverb,
                stems=6 if all_on else args.stems,
                quality=args.quality,
                drum_sep=all_on or args.drum_sep,
                split_lead_backing=all_on or args.split_lead_backing,
                split_male_female=all_on or args.split_male_female,
                split_guitar=all_on or args.split_guitar,
                split_orchestral=all_on or args.split_orchestral,
                split_crowd=all_on or args.split_crowd,
                restore=args.restore,
            )
            print(json.dumps(res))
        elif args.command == "transcribe":
            res = run_transcribe(args.input, args.output_dir)
            print(json.dumps(res))
        elif args.command == "denoise":
            res = run_denoise(
                args.input,
                args.output_dir,
                attenuation_limit_db=args.attenuation_limit_db,
                backend=args.backend,
            )
            print(json.dumps(res))
        elif args.command == "yue2-remote":
            res = run_yue2_remote(
                args.style,
                args.lyrics,
                args.planning_mode,
                args.ode_steps,
                args.seed,
                args.space,
                args.timeout,
            )
            print(json.dumps(res))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
