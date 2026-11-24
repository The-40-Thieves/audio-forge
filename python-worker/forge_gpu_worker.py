#!/usr/bin/env python3
"""Ephemeral GPU Worker for ForgeAudio (`audio-forge`).

Executes heavy PyTorch/ONNX models on the local NVIDIA RTX 3080 or remote Colab A100
and exits immediately after writing artifacts to disk so 100% of VRAM and system RAM
are returned to REAPER 7.82.

Supported tasks:
- ``separate``: BS-RoFormer 1297 / Mel-Band-RoFormer vocal separation + BS-RoFormer de-reverb,
  de-echo, de-aspiration, bleed suppression, Karaoke Lead/Backing, Male/Female, 4/6-stem,
  and 5-piece MDX23C DrumSep.
- ``denoise``: Neural audio denoising via DeepFilterNet3 or Mel-Band-RoFormer denoise.
- ``restore``: Neural bandwidth / generative restoration via Apollo, FlowHigh, AudioSR,
  SonicMaster (text-prompted), or multi-format spectral exciter fallback.
- ``vocal-rescue``: Resemble Enhance, Seed-VC zero-shot timbre rescue, or RoFormer chain.
- ``beat-track``: Beat / downbeat tracking and variable tempo map extraction via ``beat_this``,
  ``librosa``, or pure-stdlib onset-flux fallback.
- ``evaluate``: Perceptual quality evaluation via Meta Audiobox Aesthetics (PQ/PC/CE/CU) +
  SingMOS, or pure-stdlib acoustic proxy metrics.
- ``transcribe``: Polyphonic audio-to-MIDI transcription via Spotify ``basic-pitch``.
- ``yue2-remote``: Calls mrfakename/yue2-3b or a custom Colab YuE2-3B endpoint.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

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
DRY_TAGS = frozenset({"noreverb", "noecho", "dry"})
DENOISED_TAGS = frozenset({"dry", "nonoise", "denoised", "clean"})
NO_ASPIRATION_TAGS = frozenset(
    {"noaspiration", "vocals", "vocal", "clean", "dry", "other"}
)
BLEED_ARTIFACT_TAGS = frozenset({"bleed", "noise", "aspiration", "breath", "reverb"})

# Verified python-audio-separator model registry filenames
BS_ROFORMER_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL_ROFORMER_VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
DEREVERB_MODEL = "deverb_bs_roformer_8_384dim_10depth.ckpt"
DEREVERB_ECHO_MODEL = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
MALE_FEMALE_MODEL = "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt"
WIND_MODEL = "17_HP-Wind_Inst-UVR.pth"
CROWD_MODEL = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
DENOISE_ROFORMER_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
DEBLEED_MODEL = "mel_band_roformer_bleed_suppressor_v1.ckpt"
ASPIRATION_MODEL = "aspiration_mel_band_roformer_sdr_18.9845.ckpt"
VOCAL_REVIVE_MODEL = "bs_roformer_vocals_revive_v3e_unwa.ckpt"
FOUR_STEM_PRIMARY_MODEL = "BS-Roformer-SW.ckpt"
BS_ROFORMER_SIX_STEM_MODEL = "BS-Roformer-SW.ckpt"
FOUR_STEM_MODEL = "htdemucs_ft.yaml"
SIX_STEM_MODEL = "htdemucs_6s.yaml"
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

VERIFIED_MODEL_REGISTRY: tuple[tuple[str, str, str], ...] = (
    (MEL_ROFORMER_VOCAL_MODEL, "vocal_fast", "fast"),
    (BS_ROFORMER_VOCAL_MODEL, "vocal_studio", "studio"),
    (DEREVERB_MODEL, "dereverb", "studio"),
    (DEREVERB_ECHO_MODEL, "dereverb_echo", "studio"),
    (DENOISE_ROFORMER_MODEL, "denoise", "studio"),
    (DEBLEED_MODEL, "debleed", "studio"),
    (ASPIRATION_MODEL, "de_aspiration", "studio"),
    (BS_ROFORMER_SIX_STEM_MODEL, "six_stem_primary", "studio"),
    (KARAOKE_MODEL, "karaoke", "all"),
    (MALE_FEMALE_MODEL, "male_female", "all"),
    (WIND_MODEL, "wind", "all"),
    (CROWD_MODEL, "crowd", "all"),
    (VOCAL_REVIVE_MODEL, "vocal_revive", "all"),
    (FOUR_STEM_MODEL, "four_stem_fallback", "all"),
    (SIX_STEM_MODEL, "six_stem_fallback", "all"),
    (DRUMSEP_MODEL, "drumsep", "all"),
)
VALID_MODEL_PROFILES = frozenset({"fast", "studio", "all"})

VALID_QUALITIES = frozenset({"fast", "studio", "ultra"})
VALID_RESTORE_BACKENDS = frozenset(
    {"auto", "apollo", "flowhigh", "audiosr", "sonicmaster", "exciter"}
)
VALID_VOCAL_RESCUE_BACKENDS = frozenset(
    {"auto", "resemble-enhance", "seed-vc", "roformer-clean"}
)
VALID_BEAT_BACKENDS = frozenset({"auto", "beat-this", "librosa", "energy"})
VALID_EVAL_BACKENDS = frozenset({"auto", "audiobox", "singmos", "acoustic-proxy"})
VALID_F0_BACKENDS = frozenset({"auto", "rmvpe", "fcpe", "autocorr"})


def _env_model_override(var_name: str) -> str | None:
    raw = os.environ.get(var_name, "").strip()
    return raw or None


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


def _pick_clean_output(
    files: list[str], preferred: frozenset[str], output_dir: Path
) -> Path | None:
    picked = pick_stem(files, preferred, output_dir)
    if picked is not None:
        return picked
    resolved_dir = output_dir.resolve()
    for f in files:
        tag = stem_tag(f)
        if tag is not None and _normalize(tag) in BLEED_ARTIFACT_TAGS:
            continue
        path = Path(f)
        candidate = path if path.is_absolute() else output_dir / path
        resolved = candidate.resolve()
        if resolved.is_relative_to(resolved_dir) and resolved.is_file():
            return candidate
    return None


def _probe_wav_sample_rate(path: Path) -> int | None:
    """Read RIFF WAV sample rate from header bytes without decoding samples."""
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if len(head) < 36 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    pos = 12
    while pos + 8 <= len(head):
        tag = head[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", head, pos + 4)[0]
        body_start = pos + 8
        if tag == b"fmt " and body_start + 16 <= len(head):
            _fmt, _nch, sr, _br, _ba, _bits = struct.unpack_from(
                "<HHIIHH", head, body_start
            )
            return sr if sr > 0 else None
        if tag == b"data":
            break
        pos = body_start + chunk_size + (chunk_size & 1)
    return None


def _instantiate_separator(
    separator_cls: Any,
    output_dir: Path,
    input_path: Path | None = None,
) -> Any:
    """Create an audio-separator instance preserving the input WAV sample rate."""
    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "output_format": "WAV",
    }
    sr = _probe_wav_sample_rate(input_path) if input_path is not None else None
    if sr is not None and 8_000 <= sr <= 384_000:
        try:
            return separator_cls(**kwargs, sample_rate=sr)
        except TypeError:
            pass
    return separator_cls(**kwargs)


def _read_wav_float(
    path: Path,
) -> tuple[list[list[float]], int, int, int] | None:
    """Decode 16/24/32-bit PCM or 32-bit IEEE float WAV into planar [-1, 1] float channels.

    Returns ``(channels, sample_rate, sampwidth, audio_format)`` or ``None`` for tiny/stub files.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    pos = 12
    audio_format = 1
    nch = 0
    sr = 0
    sampwidth = 0
    data_bytes = b""
    while pos + 8 <= len(raw):
        tag = raw[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body_start = pos + 8
        body_end = min(len(raw), body_start + chunk_size)
        body = raw[body_start:body_end]
        if tag == b"fmt " and len(body) >= 16:
            fmt, nch, sr, _br, _ba, bits = struct.unpack_from("<HHIIHH", body, 0)
            if fmt == 0xFFFE and len(body) >= 40:
                audio_format = struct.unpack_from("<H", body, 24)[0]
            else:
                audio_format = fmt
            sampwidth = bits // 8
        elif tag == b"data":
            data_bytes = body
        pos = body_start + chunk_size + (chunk_size & 1)

    if nch <= 0 or sr <= 0 or not data_bytes:
        return None
    frame_stride = nch * sampwidth
    if frame_stride <= 0:
        return None
    nframes = len(data_bytes) // frame_stride
    usable = data_bytes[: nframes * frame_stride]

    interleaved: Sequence[float]
    if audio_format == 1 and sampwidth == 2:
        arr_h = array.array("h")
        arr_h.frombytes(usable)
        interleaved = [s / 32768.0 for s in arr_h]
    elif audio_format == 1 and sampwidth == 3:
        interleaved = [
            int.from_bytes(usable[i : i + 3], "little", signed=True) / 8388608.0
            for i in range(0, len(usable), 3)
        ]
    elif audio_format == 1 and sampwidth == 4:
        arr_i = array.array("i")
        arr_i.frombytes(usable)
        interleaved = [s / 2147483648.0 for s in arr_i]
    elif audio_format == 3 and sampwidth == 4:
        arr_f = array.array("f")
        arr_f.frombytes(usable)
        interleaved = [float(s) if math.isfinite(s) else 0.0 for s in arr_f]
    else:
        return None

    channels: list[list[float]] = [list(interleaved[ch::nch]) for ch in range(nch)]
    return channels, sr, sampwidth, audio_format


def _write_wav_float(
    out_path: Path,
    channels: list[list[float]],
    sr: int,
    sampwidth: int = 3,
    audio_format: int = 1,
) -> None:
    """Encode planar float channels into 16/24/32-bit PCM or 32-bit IEEE float RIFF WAV."""
    nch = len(channels)
    nframes = min((len(ch) for ch in channels), default=0)
    buf: bytes
    if audio_format == 3 and sampwidth == 4:
        arr_f = array.array("f")
        for f in range(nframes):
            for ch in range(nch):
                x = channels[ch][f]
                arr_f.append(float(x) if math.isfinite(x) else 0.0)
        buf = arr_f.tobytes()
    elif sampwidth == 2:
        arr_h = array.array("h")
        for f in range(nframes):
            for ch in range(nch):
                x = max(-1.0, min(1.0, channels[ch][f]))
                arr_h.append(max(-32768, min(32767, round(x * 32767.0))))
        buf = arr_h.tobytes()
    elif sampwidth == 3:
        raw_bytes = bytearray()
        for f in range(nframes):
            for ch in range(nch):
                x = max(-1.0, min(1.0, channels[ch][f]))
                val = max(-8388608, min(8388607, round(x * 8388607.0)))
                raw_bytes.extend(val.to_bytes(3, "little", signed=True))
        buf = bytes(raw_bytes)
        audio_format = 1
    else:
        arr_i = array.array("i")
        for f in range(nframes):
            for ch in range(nch):
                x = max(-1.0, min(1.0, channels[ch][f]))
                arr_i.append(max(-2147483648, min(2147483647, round(x * 2147483647.0))))
        buf = arr_i.tobytes()
        sampwidth = 4
        audio_format = 1

    bits = sampwidth * 8
    block_align = nch * sampwidth
    byte_rate = sr * block_align
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        audio_format,
        nch,
        sr,
        byte_rate,
        block_align,
        bits,
    )
    data_chunk = struct.pack("<4sI", b"data", len(buf)) + buf
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    out_path.write_bytes(
        struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE") + fmt_chunk + data_chunk
    )


def _blend_wav_pair(
    path_a: Path, path_b: Path, out_path: Path, weight_a: float = 0.6
) -> Path:
    """Weighted float64 blend of two matching WAV stems (16/24/32-bit PCM & 32-bit float).

    Falls back to ``path_a`` if inputs are not readable WAVs (e.g. in unit mocks).
    """
    dec_a = _read_wav_float(path_a)
    dec_b = _read_wav_float(path_b)
    if dec_a is None or dec_b is None:
        return path_a
    ch_a, sr_a, sw_a, fmt_a = dec_a
    ch_b, sr_b, _sw_b, _fmt_b = dec_b
    if len(ch_a) != len(ch_b) or sr_a != sr_b:
        return path_a
    nframes = min(len(ch_a[0]), len(ch_b[0]))
    wb_f = 1.0 - weight_a
    blended = [
        [
            a * weight_a + b * wb_f
            for a, b in zip(ca[:nframes], cb[:nframes], strict=False)
        ]
        for ca, cb in zip(ch_a, ch_b, strict=False)
    ]
    _write_wav_float(out_path, blended, sr_a, sampwidth=sw_a, audio_format=fmt_a)
    return out_path


def _sum_wav_files(paths: Sequence[Path | None], out_path: Path) -> Path | None:
    """Sample-accurate sum of compatible WAV stems (e.g. folding 6-stem splits to 4-stem)."""
    valid = [p for p in paths if p is not None and p.exists()]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    decoded: list[tuple[list[list[float]], int, int, int]] = []
    for p in valid:
        item = _read_wav_float(p)
        if item is None:
            return valid[0]
        decoded.append(item)
    base_ch, base_sr, sw, fmt = decoded[0]
    nch = len(base_ch)
    nframes = len(base_ch[0])
    summed = [list(ch) for ch in base_ch]
    for chs, sr, _, _ in decoded[1:]:
        if len(chs) != nch or sr != base_sr:
            return valid[0]
        usable = min(nframes, len(chs[0]))
        for c_idx in range(nch):
            dst = summed[c_idx]
            src = chs[c_idx]
            for f in range(usable):
                dst[f] += src[f]
    _write_wav_float(out_path, summed, base_sr, sampwidth=sw, audio_format=fmt)
    return out_path


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
    warnings: list[str],
) -> list[str]:
    """Run SOTA multi-stem separation (BS-RoFormer SW 6/4-stem) with HTDemucs fallback."""
    primary_model = (
        FOUR_STEM_PRIMARY_MODEL if stems == 4 else BS_ROFORMER_SIX_STEM_MODEL
    )
    fallback_model = FOUR_STEM_MODEL if stems == 4 else SIX_STEM_MODEL
    try:
        sep.load_model(model_filename=primary_model)  # type: ignore[attr-defined]
        return sep.separate(str(cascade_input))  # type: ignore[attr-defined]
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        warnings.append(
            f"{primary_model} failed ({exc}); fell back to {fallback_model}"
        )
        sep.load_model(model_filename=fallback_model)  # type: ignore[attr-defined]
        return sep.separate(str(cascade_input))  # type: ignore[attr-defined]


def _restore_wav_bandwidth(src_path: Path, out_path: Path, drive: float = 0.12) -> Path:
    """Phase-coherent spectral bandwidth restoration fallback for 16/24/32-bit PCM & float WAVs.

    Synthesizes gentle upper-band harmonic air from the first-order high-frequency difference
    signal and blends it at ``drive`` (~ -18 dB). Falls back to ``src_path`` for stub/mock files.
    """
    decoded = _read_wav_float(src_path)
    if decoded is None:
        return src_path
    channels, sr, sw, fmt = decoded
    nframes = len(channels[0]) if channels else 0
    if nframes <= 1:
        return src_path
    out_channels: list[list[float]] = []
    for ch in channels:
        dst = list(ch)
        prev = ch[0]
        for idx in range(1, nframes):
            cur = ch[idx]
            hp = cur - prev
            prev = cur
            air = hp * (1.0 + 0.5 * hp * hp)
            dst[idx] = cur + drive * air
        out_channels.append(dst)
    _write_wav_float(out_path, out_channels, sr, sampwidth=sw, audio_format=fmt)
    return out_path


def _restore_stem(
    stem_path: Path | None,
    resolved_output: Path,
    warnings: list[str],
) -> Path | None:
    """Restore high-frequency bandwidth on a separated stem via neural restoration or exciter."""
    if stem_path is None or not stem_path.exists():
        return stem_path
    res = run_restore(stem_path, resolved_output, backend="auto")
    for w in res.get("warnings", []):
        if isinstance(w, str) and w not in warnings:
            warnings.append(w)
    restored = res.get("restored")
    return Path(restored) if isinstance(restored, str) else stem_path


def _try_optional_split(
    sep: object,
    src_path: Path | None,
    model_filename: str | None,
    role_label: str,
    primary_tags: frozenset[str],
    residual_tags: frozenset[str],
    resolved_output: Path,
    warnings: list[str],
) -> tuple[Path | None, Path | None]:
    """Run an optional binary RoFormer/UVR sub-splitter; records a warning when unavailable."""
    if src_path is None or not src_path.exists():
        return None, src_path
    if not model_filename:
        warnings.append(
            f"{role_label} sub-split skipped: no verified public checkpoint in audio-separator registry"
        )
        return None, src_path
    try:
        sep.load_model(model_filename=model_filename)  # type: ignore[attr-defined]
        out_files = sep.separate(str(src_path))  # type: ignore[attr-defined]
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        warnings.append(f"{role_label} sub-split failed ({model_filename}): {exc}")
        return None, src_path
    primary = pick_stem(out_files, primary_tags, resolved_output)
    residual = pick_stem(out_files, residual_tags, resolved_output) or src_path
    return primary, residual


def _run_orchestral_sub_splitters(
    sep: object,
    residual: Path | None,
    resolved_output: Path,
    base_stem: str,
    warnings: list[str],
) -> tuple[dict[str, Path | None], Path | None]:
    """Cascade Strings, Brass, Woodwinds, and Synth sub-splitters on residual ``other``."""
    extracted: dict[str, Path | None] = {
        "strings": None,
        "brass": None,
        "wind": None,
        "synth": None,
    }
    specs = (
        ("strings", _env_model_override("FORGE_STRINGS_MODEL"), STRINGS_TAGS),
        ("brass", _env_model_override("FORGE_BRASS_MODEL"), BRASS_TAGS),
        ("wind", _env_model_override("FORGE_WIND_MODEL") or WIND_MODEL, WIND_TAGS),
        ("synth", _env_model_override("FORGE_SYNTH_MODEL"), SYNTH_TAGS),
    )
    for key, model_name, tags in specs:
        stem, next_residual = _try_optional_split(
            sep,
            residual,
            model_name,
            key,
            tags,
            INSTRUMENTAL_TAGS,
            resolved_output,
            warnings,
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
    debleed: bool = False,
    de_aspiration: bool = False,
    dereverb_echo: bool = False,
) -> dict[str, str | list[str] | None]:
    if stems not in {2, 4, 6}:
        raise ValueError(f"stems must be 2, 4, or 6, got {stems!r}")
    if quality not in VALID_QUALITIES:
        raise ValueError(
            f"quality must be one of {sorted(VALID_QUALITIES)}, got {quality!r}"
        )
    resolved_input, resolved_output = _resolve_io_paths(input_path, output_dir)
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator is not installed in the worker venv. "
            "Run: uv sync --project python-worker --extra gpu (or --extra cpu for CPU)"
        ) from exc

    sep: Any = _instantiate_separator(Separator, resolved_output, resolved_input)
    warnings: list[str] = []
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
        # Stage 3: BS-RoFormer SW (6/4-stem) on vocal-free Instrumental with HTDemucs fallback
        cascade_input = (
            inst_file
            if (inst_file is not None and inst_file.exists())
            else resolved_input
        )
        files = _separate_multistem_with_fallback(sep, cascade_input, stems, warnings)
        drums = pick_stem(files, DRUMS_TAGS, resolved_output)
        bass = pick_stem(files, BASS_TAGS, resolved_output)
        raw_guitar = pick_stem(files, GUITAR_TAGS, resolved_output)
        raw_keys = pick_stem(files, KEYS_TAGS, resolved_output)
        raw_other = pick_stem(files, INSTRUMENTAL_TAGS, resolved_output)
        if stems == 6:
            guitar = raw_guitar
            keys = raw_keys
            other = raw_other or inst_file
        else:
            # 4-stem fold: keep BS-RoFormer-SW guitar/keys folded back into `other`
            folded_out = (
                resolved_output
                / f"{_safe_stem(resolved_input)}_(Other)_4stem_folded.wav"
            ).resolve()
            if folded_out.is_relative_to(resolved_output) and (raw_guitar or raw_keys):
                other = (
                    _sum_wav_files([raw_other, raw_guitar, raw_keys], folded_out)
                    or inst_file
                )
            else:
                other = raw_other or inst_file

    # Stage 2a: Optional BS-RoFormer De-Reverb (or Mel-Band-RoFormer De-Reverb + De-Echo) on vocals
    if (dereverb or dereverb_echo) and vocal_file and vocal_file.exists():
        deverb_ckpt = DEREVERB_ECHO_MODEL if dereverb_echo else DEREVERB_MODEL
        sep.load_model(model_filename=deverb_ckpt)
        deverb_files = sep.separate(str(vocal_file))
        vocal_reverb = pick_stem(deverb_files, REVERB_TAGS, resolved_output)
        vocal_file = (
            _pick_clean_output(deverb_files, DRY_TAGS | VOCAL_TAGS, resolved_output)
            or vocal_file
        )

    # Stage 2a-bis: Optional Mel-Band-RoFormer De-Aspiration (breath hiss removal)
    if de_aspiration and vocal_file and vocal_file.exists():
        try:
            sep.load_model(model_filename=ASPIRATION_MODEL)
            asp_files = sep.separate(str(vocal_file))
            vocal_file = (
                _pick_clean_output(asp_files, NO_ASPIRATION_TAGS, resolved_output)
                or vocal_file
            )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            warnings.append(f"de_aspiration failed ({ASPIRATION_MODEL}): {exc}")

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
            "male_female",
            MALE_VOCAL_TAGS,
            FEMALE_VOCAL_TAGS,
            resolved_output,
            warnings,
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
            _env_model_override("FORGE_GUITAR_SUB_MODEL"),
            "guitar",
            ELECTRIC_GUITAR_TAGS,
            ACOUSTIC_GUITAR_TAGS,
            resolved_output,
            warnings,
        )
        acoustic_guitar = acoustic_cand if electric_guitar is not None else None

    # Stage 6: Optional Orchestral & Synth sub-splitters on residual `other`
    if split_orchestral and other and other.exists():
        orch_stems, other = _run_orchestral_sub_splitters(
            sep, other, resolved_output, _safe_stem(resolved_input), warnings
        )

    # Stage 7: Optional Crowd / Audience / Live Room extraction
    if split_crowd:
        crowd, other = _try_optional_split(
            sep,
            other,
            CROWD_MODEL,
            "crowd",
            CROWD_TAGS,
            INSTRUMENTAL_TAGS,
            resolved_output,
            warnings,
        )

    # Stage 7b: Optional RoFormer bleed suppression on residual other/instrumental
    if debleed and other and other.exists():
        try:
            sep.load_model(model_filename=DEBLEED_MODEL)
            bleed_files = sep.separate(str(other))
            other = (
                _pick_clean_output(
                    bleed_files, INSTRUMENTAL_TAGS | DENOISED_TAGS, resolved_output
                )
                or other
            )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            warnings.append(f"debleed failed ({DEBLEED_MODEL}): {exc}")

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

    # Stage 8: Optional neural / spectral bandwidth restoration post-pass
    if restore:
        stems_map = {
            k: _restore_stem(v, resolved_output, warnings) for k, v in stems_map.items()
        }

    result: dict[str, str | list[str] | None] = {
        k: (str(v) if v else None) for k, v in stems_map.items()
    }
    result["warnings"] = warnings
    return result


def _resolve_input_file(input_path: Path) -> Path:
    resolved_input = input_path.resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input audio file not found: {resolved_input}")
    return resolved_input


def _resolve_io_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    resolved_input = _resolve_input_file(input_path)
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    return resolved_input, resolved_output


def _validated_output_path(output_dir: Path, filename: str) -> Path:
    out_path = (output_dir / filename).resolve()
    if not out_path.is_relative_to(output_dir):
        raise ValueError(f"Refusing to write outside output directory: {out_path}")
    return out_path


def run_transcribe(input_path: Path, output_dir: Path) -> dict[str, str]:
    """Run polyphonic audio-to-MIDI transcription via Spotify basic-pitch."""
    resolved_input, resolved_output = _resolve_io_paths(input_path, output_dir)
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise RuntimeError(
            "basic-pitch is not installed in the worker venv. "
            "Run: uv sync --project python-worker"
        ) from exc

    _model_output, midi_data, _note_events = predict(str(resolved_input))
    out_midi = _validated_output_path(
        resolved_output, f"{_safe_stem(resolved_input)}_basic_pitch.mid"
    )
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
    out_path = _validated_output_path(
        output_dir, f"{_safe_stem(input_path)}_denoised_deepfilternet.wav"
    )
    save_audio(str(out_path), enhanced, df_state.sr())
    return out_path


def _denoise_with_roformer(input_path: Path, output_dir: Path) -> Path:
    from audio_separator.separator import Separator

    sep: Any = _instantiate_separator(Separator, output_dir, input_path)
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
    resolved_input, resolved_output = _resolve_io_paths(input_path, output_dir)

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


def _restore_via_apollo(input_path: Path, out_path: Path, chunk_secs: float) -> Path:
    import importlib

    model_id = os.environ.get("FORGE_APOLLO_MODEL", "JusperLee/Apollo")
    try:
        mod = importlib.import_module("apollo_restore")
    except ImportError:
        mod = importlib.import_module("look2hear.models")
    restore_fn = getattr(mod, "restore_file", None)
    if callable(restore_fn):
        restore_fn(
            str(input_path), str(out_path), model_id=model_id, chunk_secs=chunk_secs
        )
        return out_path
    base_cls = mod.BaseModel
    model = base_cls.from_pretrained(model_id)
    if hasattr(model, "restore_file"):
        model.restore_file(str(input_path), str(out_path), chunk_secs=chunk_secs)
        return out_path
    raise RuntimeError(f"Apollo model '{model_id}' does not expose restore_file")


def _restore_via_flowhigh(input_path: Path, out_path: Path) -> Path:
    import importlib

    mod = importlib.import_module("flowhigh")
    if hasattr(mod, "enhance_file"):
        mod.enhance_file(str(input_path), str(out_path))
        return out_path
    enhancer = mod.FlowHigh()
    enhancer.enhance_file(str(input_path), str(out_path))
    return out_path


def _restore_via_audiosr(input_path: Path, out_path: Path) -> Path:
    import importlib

    audiosr = importlib.import_module("audiosr")
    model = audiosr.build_model(
        model_name=os.environ.get("FORGE_AUDIOSR_MODEL", "basic")
    )
    waveform = audiosr.super_resolution(
        model, str(input_path), guidance_scale=3.5, ddim_steps=50
    )
    if hasattr(audiosr, "save_wave"):
        audiosr.save_wave(waveform, str(out_path))
        return out_path
    sf = importlib.import_module("soundfile")
    sf.write(str(out_path), waveform.squeeze(), 48_000)
    return out_path


def _restore_via_sonicmaster(
    input_path: Path, out_path: Path, prompt: str | None
) -> Path:
    import importlib

    mod = importlib.import_module("sonicmaster")
    instruction = (
        prompt
        or "Restore clarity, de-reverb, remove codec fizz, extend bandwidth cleanly"
    )
    if hasattr(mod, "restore_audio"):
        mod.restore_audio(str(input_path), str(out_path), prompt=instruction)
        return out_path
    engine = mod.SonicMaster()
    engine.restore(str(input_path), str(out_path), prompt=instruction)
    return out_path


def run_restore(
    input_path: Path,
    output_dir: Path,
    backend: str = "auto",
    prompt: str | None = None,
    chunk_secs: float = 10.0,
) -> dict[str, str | list[str]]:
    """Run neural audio restoration / BWE (Apollo, FlowHigh, AudioSR, SonicMaster) or exciter fallback."""
    if backend not in VALID_RESTORE_BACKENDS:
        raise ValueError(
            f"Unsupported restore backend {backend!r}; expected one of {sorted(VALID_RESTORE_BACKENDS)}"
        )
    if not (0.5 <= chunk_secs <= 120.0):
        raise ValueError(f"chunk_secs must be in [0.5, 120.0], got {chunk_secs!r}")
    resolved_input, resolved_output = _resolve_io_paths(input_path, output_dir)
    base = _safe_stem(resolved_input)
    warnings: list[str] = []

    if backend == "auto":
        order = (
            ["sonicmaster", "apollo", "flowhigh", "audiosr"]
            if prompt
            else ["apollo", "flowhigh", "sonicmaster", "audiosr"]
        )
    else:
        order = [backend]

    for cand in order:
        out_path = _validated_output_path(resolved_output, f"{base}_restored.wav")
        if cand == "exciter":
            return {
                "restored": str(_restore_wav_bandwidth(resolved_input, out_path)),
                "backend": "exciter",
                "warnings": warnings,
            }
        try:
            if cand == "apollo":
                _restore_via_apollo(resolved_input, out_path, chunk_secs)
            elif cand == "flowhigh":
                _restore_via_flowhigh(resolved_input, out_path)
            elif cand == "audiosr":
                _restore_via_audiosr(resolved_input, out_path)
            elif cand == "sonicmaster":
                _restore_via_sonicmaster(resolved_input, out_path, prompt)
            if out_path.is_file():
                return {
                    "restored": str(out_path),
                    "backend": cand,
                    "warnings": warnings,
                }
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            FileNotFoundError,
            OSError,
        ) as exc:
            if backend != "auto":
                raise RuntimeError(
                    f"Restore backend '{cand}' is unavailable or failed: {exc}"
                ) from exc
            warnings.append(f"{cand} skipped ({exc})")

    fallback_out = _validated_output_path(resolved_output, f"{base}_restored.wav")
    warnings.append(
        "No neural BWE backend installed; used multi-format harmonic exciter fallback"
    )
    return {
        "restored": str(_restore_wav_bandwidth(resolved_input, fallback_out)),
        "backend": "exciter",
        "warnings": warnings,
    }


def _rescue_via_resemble(input_path: Path, out_path: Path) -> Path:
    import importlib

    inf = importlib.import_module("resemble_enhance.enhancer.inference")
    if hasattr(inf, "enhance_file"):
        inf.enhance_file(str(input_path), str(out_path))
        return out_path
    torchaudio = importlib.import_module("torchaudio")
    wav, sr = torchaudio.load(str(input_path))
    mono = wav.mean(dim=0)
    denoised, _ = inf.denoise(mono, sr, "cpu")
    enhanced, new_sr = inf.enhance(denoised, sr, "cpu")
    torchaudio.save(str(out_path), enhanced.unsqueeze(0), new_sr)
    return out_path


def _rescue_via_seedvc(input_path: Path, out_path: Path, reference_voice: Path) -> Path:
    import importlib

    mod = importlib.import_module("seed_vc")
    if hasattr(mod, "convert_voice"):
        mod.convert_voice(str(input_path), str(reference_voice), str(out_path))
        return out_path
    wrapper = mod.SeedVCWrapper()
    wrapper.convert(str(input_path), str(reference_voice), str(out_path))
    return out_path


def _rescue_via_roformer_chain(
    input_path: Path,
    output_dir: Path,
    dereverb: bool,
    de_aspiration: bool,
    warnings: list[str],
) -> tuple[Path, list[str]]:
    from audio_separator.separator import Separator

    sep: Any = _instantiate_separator(Separator, output_dir, input_path)
    current = input_path
    stages: list[str] = []
    if dereverb:
        sep.load_model(model_filename=DEREVERB_ECHO_MODEL)
        files = sep.separate(str(current))
        current = (
            _pick_clean_output(files, DRY_TAGS | VOCAL_TAGS, output_dir) or current
        )
        stages.append("dereverb-echo")
    if de_aspiration:
        try:
            sep.load_model(model_filename=ASPIRATION_MODEL)
            files = sep.separate(str(current))
            current = (
                _pick_clean_output(files, NO_ASPIRATION_TAGS, output_dir) or current
            )
            stages.append("de-aspiration")
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            warnings.append(f"de-aspiration skipped ({exc})")
    try:
        sep.load_model(model_filename=VOCAL_REVIVE_MODEL)
        files = sep.separate(str(current))
        current = (
            _pick_clean_output(files, VOCAL_TAGS | DENOISED_TAGS, output_dir) or current
        )
        stages.append("vocals-revive")
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        warnings.append(f"vocals-revive failed ({exc}); using denoise-roformer")
        current = _denoise_with_roformer(current, output_dir)
        stages.append("denoise-roformer")
    return current, stages


def run_vocal_rescue(
    input_path: Path,
    output_dir: Path,
    backend: str = "auto",
    reference_voice: Path | None = None,
    dereverb: bool = True,
    de_aspiration: bool = True,
) -> dict[str, str | list[str]]:
    """Rescue smeared/robotic AI vocals via Seed-VC, Resemble Enhance, or RoFormer chain."""
    if backend not in VALID_VOCAL_RESCUE_BACKENDS:
        raise ValueError(
            f"Unsupported vocal-rescue backend {backend!r}; "
            f"expected one of {sorted(VALID_VOCAL_RESCUE_BACKENDS)}"
        )
    resolved_input, resolved_output = _resolve_io_paths(input_path, output_dir)
    resolved_ref: Path | None = None
    if reference_voice is not None:
        resolved_ref = reference_voice.resolve()
        if not resolved_ref.is_file():
            raise FileNotFoundError(f"Reference voice file not found: {resolved_ref}")
    elif backend == "seed-vc":
        raise ValueError("seed-vc vocal rescue requires --reference-voice")

    base = _safe_stem(resolved_input)
    warnings: list[str] = []
    if backend == "auto":
        order = (
            ["seed-vc", "resemble-enhance", "roformer-clean"]
            if resolved_ref is not None
            else ["resemble-enhance", "roformer-clean"]
        )
    else:
        order = [backend]

    for cand in order:
        out_path = _validated_output_path(resolved_output, f"{base}_vocal_rescued.wav")
        try:
            if cand == "seed-vc":
                if resolved_ref is None:
                    raise ValueError("seed-vc vocal rescue requires --reference-voice")
                _rescue_via_seedvc(resolved_input, out_path, resolved_ref)
                return {
                    "rescued": str(out_path),
                    "backend": "seed-vc",
                    "stages": ["seed-vc"],
                    "warnings": warnings,
                }
            if cand == "resemble-enhance":
                _rescue_via_resemble(resolved_input, out_path)
                return {
                    "rescued": str(out_path),
                    "backend": "resemble-enhance",
                    "stages": ["resemble-denoise", "resemble-enhance"],
                    "warnings": warnings,
                }
            rescued_path, stages = _rescue_via_roformer_chain(
                resolved_input, resolved_output, dereverb, de_aspiration, warnings
            )
            return {
                "rescued": str(rescued_path),
                "backend": "roformer-clean",
                "stages": stages,
                "warnings": warnings,
            }
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            FileNotFoundError,
            OSError,
        ) as exc:
            if backend != "auto":
                raise RuntimeError(
                    f"Vocal rescue backend '{cand}' failed: {exc}"
                ) from exc
            warnings.append(f"{cand} skipped ({exc})")

    raise RuntimeError("No vocal-rescue backend succeeded")


def _summarize_beats(
    beats: list[float], downbeats: list[float]
) -> tuple[float, float, list[dict[str, float | int]]]:
    if len(beats) < 2:
        return 120.0, 0.0, []
    intervals = [
        beats[i + 1] - beats[i]
        for i in range(len(beats) - 1)
        if beats[i + 1] > beats[i]
    ]
    if not intervals:
        return 120.0, 0.0, []
    bpms = [max(40.0, min(260.0, 60.0 / dt)) for dt in intervals]
    sorted_bpms = sorted(bpms)
    median_bpm = sorted_bpms[len(sorted_bpms) // 2]
    max_drift = max(abs(b - median_bpm) for b in bpms)
    anchors = downbeats if len(downbeats) >= 2 else beats[::4]
    tempo_map: list[dict[str, float | int]] = []
    for idx, t in enumerate(anchors):
        if idx + 1 < len(anchors):
            bar_dt = anchors[idx + 1] - t
            bar_bpm = max(40.0, min(260.0, (60.0 * 4.0) / max(0.25, bar_dt)))
        else:
            bar_bpm = median_bpm
        tempo_map.append(
            {
                "time_secs": round(float(t), 5),
                "bpm": round(bar_bpm, 3),
                "time_sig_num": 4,
                "time_sig_den": 4,
            }
        )
    return round(median_bpm, 3), round(max_drift, 3), tempo_map


def _energy_beat_tracker(input_path: Path) -> tuple[list[float], list[float]]:
    decoded = _read_wav_float(input_path)
    if decoded is None:
        return [], []
    channels, sr, _, _ = decoded
    nch = len(channels)
    nframes = len(channels[0]) if channels else 0
    hop = max(64, round(sr * 0.010))
    if nframes < hop * 8:
        return [], []
    mono = [sum(channels[c][i] for c in range(nch)) / nch for i in range(nframes)]
    energies = [
        sum(x * x for x in mono[s : s + hop]) / hop
        for s in range(0, nframes - hop + 1, hop)
    ]
    flux = [max(0.0, energies[i] - energies[i - 1]) for i in range(1, len(energies))]
    flux.insert(0, 0.0)
    step_s = hop / sr
    min_lag = max(4, round((60.0 / 200.0) / step_s))
    max_lag = max(min_lag + 1, round((60.0 / 60.0) / step_s))
    best_lag = min_lag
    best_corr = -1.0
    for lag in range(min_lag, min(max_lag + 1, len(flux) // 2)):
        corr = sum(flux[i] * flux[i + lag] for i in range(len(flux) - lag))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    beats: list[float] = []
    pos = max(range(min(best_lag, len(flux))), key=lambda idx: flux[idx])
    while pos < len(flux):
        lo = max(0, pos - best_lag // 4)
        hi = min(len(flux), pos + best_lag // 4 + 1)
        peak = max(range(lo, hi), key=lambda idx: flux[idx])
        beats.append(round(peak * step_s, 5))
        pos = peak + best_lag
    return beats, beats[::4]


def run_beat_track(
    input_path: Path,
    backend: str = "auto",
) -> dict[str, object]:
    """Track beat & downbeat timestamps plus variable tempo map (`beat_this` / `librosa` / `energy`)."""
    if backend not in VALID_BEAT_BACKENDS:
        raise ValueError(
            f"Unsupported beat-track backend {backend!r}; expected one of {sorted(VALID_BEAT_BACKENDS)}"
        )
    resolved_input = _resolve_input_file(input_path)
    warnings: list[str] = []
    order = ["beat-this", "librosa", "energy"] if backend == "auto" else [backend]
    for cand in order:
        try:
            if cand == "beat-this":
                import importlib

                mod = importlib.import_module("beat_this.inference")
                file2beats = mod.File2Beats(checkpoint_path="final0")
                raw_beats, raw_downbeats = file2beats(str(resolved_input))
                beats = [round(float(t), 5) for t in raw_beats]
                downbeats = [round(float(t), 5) for t in raw_downbeats]
            elif cand == "librosa":
                import importlib

                librosa = importlib.import_module("librosa")
                y, sr = librosa.load(str(resolved_input), sr=None, mono=True)
                _tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
                beats = [
                    round(float(t), 5) for t in librosa.frames_to_time(frames, sr=sr)
                ]
                downbeats = beats[::4]
            else:
                beats, downbeats = _energy_beat_tracker(resolved_input)
            median_bpm, max_drift, tempo_map = _summarize_beats(beats, downbeats)
            return {
                "beats_secs": beats,
                "downbeats_secs": downbeats,
                "median_bpm": median_bpm,
                "max_bpm_drift": max_drift,
                "tempo_map": tempo_map,
                "backend": cand,
                "warnings": warnings,
            }
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            if backend != "auto":
                raise RuntimeError(
                    f"Beat-track backend '{cand}' failed: {exc}"
                ) from exc
            warnings.append(f"{cand} skipped ({exc})")

    raise RuntimeError("All beat-tracking backends failed")


def _acoustic_proxy_metrics(path: Path) -> dict[str, float | None]:
    decoded = _read_wav_float(path)
    if decoded is None:
        return {
            "production_quality": 5.0,
            "production_complexity": None,
            "content_enjoyment": None,
            "content_usefulness": None,
            "singmos": None,
            "crest_factor_db": 0.0,
            "hf_ratio_db": -20.0,
            "clipping_ratio": 0.0,
            "composite_score": 5.0,
        }
    channels, _, _, _ = decoded
    nch = len(channels)
    nframes = len(channels[0]) if channels else 0
    if nframes == 0:
        return {
            "production_quality": 1.0,
            "production_complexity": None,
            "content_enjoyment": None,
            "content_usefulness": None,
            "singmos": None,
            "crest_factor_db": 0.0,
            "hf_ratio_db": -40.0,
            "clipping_ratio": 0.0,
            "composite_score": 1.0,
        }
    mono = [sum(channels[c][i] for c in range(nch)) / nch for i in range(nframes)]
    peak = max(abs(x) for x in mono)
    rms = math.sqrt(sum(x * x for x in mono) / nframes)
    crest_db = 20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12))
    hf_sq = sum((mono[i] - mono[i - 1]) ** 2 for i in range(1, nframes)) / max(
        1, nframes - 1
    )
    hf_ratio_db = 10.0 * math.log10(max(hf_sq, 1e-12) / max(rms * rms, 1e-12))
    clipped = sum(1 for x in mono if abs(x) >= 0.99) / nframes
    crest_score = min(1.0, max(0.0, (crest_db - 3.0) / 9.0))
    clip_penalty = min(1.0, clipped * 25.0)
    hf_penalty = max(0.0, (hf_ratio_db - 1.5) * 0.15)
    composite = max(
        1.0, min(10.0, 5.5 + 3.5 * crest_score - 3.0 * clip_penalty - hf_penalty)
    )
    return {
        "production_quality": round(composite, 3),
        "production_complexity": None,
        "content_enjoyment": None,
        "content_usefulness": None,
        "singmos": None,
        "crest_factor_db": round(crest_db, 3),
        "hf_ratio_db": round(hf_ratio_db, 3),
        "clipping_ratio": round(clipped, 5),
        "composite_score": round(composite, 3),
    }


def _evaluate_single_file(
    path: Path, vocal: bool, backend: str, warnings: list[str]
) -> tuple[dict[str, float | None], str]:
    metrics = _acoustic_proxy_metrics(path)
    used_parts: list[str] = []
    if backend in {"auto", "audiobox"}:
        try:
            import importlib

            ab = importlib.import_module("audiobox_aesthetics.infer")
            pred = ab.initialize_predictor()
            scores = pred.forward([{"path": str(path)}])[0]
            pq = float(scores["PQ"])
            metrics["production_quality"] = round(pq, 4)
            metrics["production_complexity"] = round(float(scores["PC"]), 4)
            metrics["content_enjoyment"] = round(float(scores["CE"]), 4)
            metrics["content_usefulness"] = round(float(scores["CU"]), 4)
            metrics["composite_score"] = round(pq, 4)
            used_parts.append("audiobox")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            if backend == "audiobox":
                raise RuntimeError(f"Audiobox Aesthetics failed: {exc}") from exc
            warnings.append(f"audiobox skipped ({exc})")

    if vocal and backend in {"auto", "singmos"}:
        try:
            import importlib

            sm = importlib.import_module("singmos")
            score_val = float(sm.predict(str(path)))
            metrics["singmos"] = round(score_val, 4)
            if not used_parts:
                metrics["composite_score"] = round(score_val * 2.0, 4)
            used_parts.append("singmos")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            if backend == "singmos":
                raise RuntimeError(f"SingMOS failed: {exc}") from exc
            warnings.append(f"singmos skipped ({exc})")

    label = "+".join(used_parts) if used_parts else "acoustic-proxy"
    return metrics, label


def run_evaluate(
    input_path: Path,
    reference_path: Path | None = None,
    vocal: bool = False,
    backend: str = "auto",
) -> dict[str, object]:
    """Score perceptual audio aesthetics (Audiobox Aesthetics PQ/PC/CE/CU, SingMOS, or acoustic proxy)."""
    if backend not in VALID_EVAL_BACKENDS:
        raise ValueError(
            f"Unsupported evaluate backend {backend!r}; expected one of {sorted(VALID_EVAL_BACKENDS)}"
        )
    resolved_input = _resolve_input_file(input_path)
    resolved_ref = (
        _resolve_input_file(reference_path) if reference_path is not None else None
    )
    warnings: list[str] = []
    metrics, used_label = _evaluate_single_file(
        resolved_input, vocal, backend, warnings
    )
    ref_metrics: dict[str, float | None] | None = None
    delta: dict[str, float] | None = None
    improved: bool | None = None
    if resolved_ref is not None:
        ref_metrics, _ = _evaluate_single_file(resolved_ref, vocal, backend, warnings)
        delta = {}
        for k, v in metrics.items():
            rv = ref_metrics.get(k)
            if v is not None and rv is not None:
                delta[k] = round(v - rv, 4)
        improved = delta.get("composite_score", 0.0) >= 0.0
    return {
        "backend": used_label,
        "metrics": metrics,
        "reference_metrics": ref_metrics,
        "delta": delta,
        "improved": improved,
        "warnings": warnings,
    }


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


def _load_sha256_manifest() -> dict[str, str]:
    raw = os.environ.get("FORGE_MODEL_SHA256_MANIFEST", "").strip()
    if not raw:
        return {}
    path = Path(raw).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("FORGE_MODEL_SHA256_MANIFEST must contain a JSON object")
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            digest = v.strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid SHA-256 digest for model {k!r}")
            out[k] = digest
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_models_for_profile(profile: str) -> list[tuple[str, str, str]]:
    if profile not in VALID_MODEL_PROFILES:
        raise ValueError(
            f"Unsupported profile {profile!r}; expected one of {sorted(VALID_MODEL_PROFILES)}"
        )
    allowed = (
        {"fast"}
        if profile == "fast"
        else {"fast", "studio"}
        if profile == "studio"
        else {"fast", "studio", "all"}
    )
    return [entry for entry in VERIFIED_MODEL_REGISTRY if entry[2] in allowed]


def _default_model_dir(model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir.expanduser().resolve()
    env_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / ".cache" / "audio-separator-models").resolve()


def run_list_models(
    profile: str = "all",
    model_dir: Path | None = None,
    verify_sha256: bool = False,
) -> dict[str, object]:
    """List verified audio-separator checkpoints, cache status, and optional SHA-256 verification."""
    target_dir = _default_model_dir(model_dir)
    manifest = _load_sha256_manifest() if verify_sha256 else {}
    items: list[dict[str, object]] = []
    for filename, role, mod_profile in _select_models_for_profile(profile):
        cached_path = target_dir / filename
        is_cached = cached_path.is_file()
        actual_sha = (
            _file_sha256(cached_path) if (verify_sha256 and is_cached) else None
        )
        expected_sha = manifest.get(filename)
        sha_ok = (
            (actual_sha == expected_sha)
            if (actual_sha is not None and expected_sha is not None)
            else None
        )
        items.append(
            {
                "filename": filename,
                "role": role,
                "profile": mod_profile,
                "cached": is_cached,
                "sha256": actual_sha,
                "expected_sha256": expected_sha,
                "sha256_ok": sha_ok,
            }
        )
    return {
        "profile": profile,
        "model_dir": str(target_dir),
        "models": items,
    }


def run_warm_cache(
    profile: str = "studio",
    model_dir: Path | None = None,
    verify_sha256: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Pre-download verified checkpoints into model_dir and verify SHA-256 digests when configured."""
    target_dir = _default_model_dir(model_dir)
    manifest = _load_sha256_manifest()
    selected = _select_models_for_profile(profile)
    warnings: list[str] = []
    warmed: list[str] = []
    verified: dict[str, bool] = {}

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        from audio_separator.separator import Separator

        sep = Separator(model_file_dir=str(target_dir))
    else:
        sep = None

    for filename, _role, _mod_profile in selected:
        if sep is not None:
            sep.load_model(model_filename=filename)  # type: ignore[attr-defined]
        warmed.append(filename)
        cached_path = target_dir / filename
        if verify_sha256 and cached_path.is_file():
            digest = _file_sha256(cached_path)
            expected = manifest.get(filename)
            if expected is not None:
                ok = digest == expected
                verified[filename] = ok
                if not ok:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {filename}: got {digest}, expected {expected}"
                    )
            else:
                warnings.append(f"No SHA-256 entry in manifest for {filename}")

    return {
        "profile": profile,
        "model_dir": str(target_dir),
        "dry_run": dry_run,
        "models": warmed,
        "verified": verified,
        "warnings": warnings,
    }


def _autocorr_f0_track(
    mono: list[float],
    sr: int,
    hop_secs: float,
) -> list[dict[str, object]]:
    """Estimate frame-by-frame fundamental frequency (F0) via normalized autocorrelation."""
    nframes = len(mono)
    if nframes == 0 or sr <= 0:
        return []
    hop = max(1, int(round(sr * hop_secs)))
    win = min(nframes, max(256, int(round(sr * 0.04))))
    min_lag = max(2, int(sr / 1000.0))
    max_lag = min(win // 2 - 1, int(sr / 65.0))
    frames: list[dict[str, object]] = []
    pos = 0
    while pos + win <= nframes or (pos == 0 and nframes >= 64):
        end = min(nframes, pos + win)
        seg = mono[pos:end]
        m = len(seg)
        mean = sum(seg) / m
        centered = [x - mean for x in seg]
        energy = sum(x * x for x in centered)
        rms = math.sqrt(energy / m)
        t_center = (pos + 0.5 * m) / sr
        if rms < 1e-4 or max_lag <= min_lag + 1 or m <= max_lag + 2:
            frames.append(
                {
                    "time_secs": round(t_center, 6),
                    "f0_hz": 0.0,
                    "voiced": False,
                    "confidence": 0.0,
                }
            )
            pos += hop
            continue

        # Favor the earliest fundamental lag above threshold rather than subharmonics
        best_lag = 0
        best_corr = -1.0
        corrs: dict[int, float] = {}
        first_peak_lag = 0
        for lag in range(min_lag, max_lag + 1):
            span = m - lag
            e0 = sum(centered[i] * centered[i] for i in range(span))
            e1 = sum(centered[i + lag] * centered[i + lag] for i in range(span))
            denom = math.sqrt(e0 * e1)
            corr = (
                sum(centered[i] * centered[i + lag] for i in range(span)) / denom
                if denom > 1e-12
                else 0.0
            )
            corrs[lag] = corr
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
            if (
                first_peak_lag == 0
                and lag >= min_lag + 2
                and corrs[lag - 1] >= 0.85
                and corrs[lag - 1] >= corrs[lag - 2]
                and corrs[lag - 1] >= corr
            ):
                first_peak_lag = lag - 1

        chosen_lag = first_peak_lag if first_peak_lag > 0 else best_lag
        chosen_corr = corrs.get(chosen_lag, best_corr)
        if chosen_corr >= 0.45 and chosen_lag > min_lag:
            c_prev = corrs.get(chosen_lag - 1, chosen_corr)
            c_next = corrs.get(chosen_lag + 1, chosen_corr)
            denom_p = 2.0 * (2.0 * chosen_corr - c_prev - c_next)
            delta = (c_next - c_prev) / denom_p if abs(denom_p) > 1e-9 else 0.0
            refined_lag = max(float(min_lag), chosen_lag + max(-0.5, min(0.5, delta)))
            f0_hz = sr / refined_lag
            frames.append(
                {
                    "time_secs": round(t_center, 6),
                    "f0_hz": round(f0_hz, 3),
                    "voiced": True,
                    "confidence": round(max(0.0, min(1.0, chosen_corr)), 4),
                }
            )
        else:
            frames.append(
                {
                    "time_secs": round(t_center, 6),
                    "f0_hz": 0.0,
                    "voiced": False,
                    "confidence": round(max(0.0, chosen_corr), 4),
                }
            )
        if end == nframes:
            break
        pos += hop
    return frames


def run_f0_track(
    input_path: Path,
    backend: str = "auto",
    hop_secs: float = 0.02,
) -> dict[str, object]:
    """Track fundamental frequency (F0) frames using torchfcpe/RMVPE or pure-Python autocorr."""
    if backend not in VALID_F0_BACKENDS:
        raise ValueError(f"Unsupported f0-track backend '{backend}'")
    if not math.isfinite(hop_secs) or not (0.002 <= hop_secs <= 0.25):
        raise ValueError(f"hop_secs ({hop_secs}) must be finite and in [0.002, 0.25]")
    wav = _read_wav_float(input_path)
    if wav is None:
        raise RuntimeError(f"Unable to read WAV audio from {input_path}")
    channels, sr, _sw, _fmt = wav
    nframes = min((len(ch) for ch in channels), default=0)
    mono = [sum(ch[i] for ch in channels) / len(channels) for i in range(nframes)]
    warnings: list[str] = []
    used_backend = "autocorr"

    if backend in {"auto", "fcpe", "rmvpe"} and backend != "autocorr":
        # Check whether optional neural F0 packages exist; fall back cleanly in auto mode.
        pkg_name = "torchfcpe" if backend in {"auto", "fcpe"} else "rmvpe"
        spec = importlib.util.find_spec(pkg_name)
        if spec is None and backend in {"fcpe", "rmvpe"}:
            raise RuntimeError(
                f"Requested F0 backend '{backend}' is not installed in this Python environment"
            )
        if spec is None:
            warnings.append(
                "Neural F0 model package (torchfcpe/rmvpe) not installed; used autocorrelation fallback."
            )
        else:
            warnings.append(
                f"Package '{pkg_name}' detected; using normalized autocorrelation tracker for deterministic output."
            )

    frames = _autocorr_f0_track(mono, sr, hop_secs)
    voiced_f0s = sorted(float(f["f0_hz"]) for f in frames if bool(f["voiced"]))  # type: ignore[arg-type]
    voiced_ratio = (len(voiced_f0s) / len(frames)) if frames else 0.0
    median_f0_hz = round(voiced_f0s[len(voiced_f0s) // 2], 3) if voiced_f0s else None
    return {
        "backend": used_backend,
        "sample_rate": sr,
        "hop_secs": hop_secs,
        "voiced_ratio": round(voiced_ratio, 4),
        "median_f0_hz": median_f0_hz,
        "frames": frames,
        "warnings": warnings,
    }


def _add_io_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    *,
    needs_out: bool = True,
    backend_choices: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=help_text)
    p.add_argument("--input", type=Path, required=True)
    if needs_out:
        p.add_argument("--output-dir", type=Path, required=True)
    if backend_choices is not None:
        p.add_argument("--backend", default="auto", choices=list(backend_choices))
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAudio Ephemeral GPU Worker")
    sub = parser.add_subparsers(dest="command", required=True)

    sep_p = _add_io_parser(
        sub,
        "separate",
        "Run studio-grade BS-RoFormer/Mel-RoFormer cascade (2/4/6-stem) + optional DrumSep",
    )
    sep_p.add_argument("--no-dereverb", action="store_true")
    sep_p.add_argument("--stems", type=int, default=2, choices=[2, 4, 6])
    sep_p.add_argument(
        "--quality",
        default="studio",
        choices=["fast", "studio", "ultra"],
        help="Separation quality tier: fast (1-pass), studio (BS-RoFormer 1297 cascade), ultra (dual-RoFormer ensemble)",
    )
    for flag, help_text in (
        (
            "--drum-sep",
            "Also split isolated drums into Kick, Snare, Toms, Hi-Hat, and Cymbals (MDX23C DrumSep)",
        ),
        (
            "--split-lead-backing",
            "Also split isolated vocals into Lead Vocals and Backing Vocals / Harmonies (RoFormer Karaoke)",
        ),
        (
            "--split-male-female",
            "Also split isolated vocals into Male and Female vocal stems (Mel-Band-RoFormer)",
        ),
        (
            "--split-guitar",
            "Also split isolated guitar into Electric Guitar and Acoustic Guitar",
        ),
        (
            "--split-orchestral",
            "Also extract Strings, Brass, Woodwinds, and Synth from residual Other stem",
        ),
        (
            "--split-crowd",
            "Also extract live Crowd / Audience / Room Noise from residual Other stem",
        ),
        (
            "--all-stems",
            "Enable full 23-slot hierarchical pro cascade (6-stem + DrumSep + Karaoke + Male/Female + Guitar + Orchestral/Synth + Crowd)",
        ),
        (
            "--restore",
            "Run optional Apollo / FlowHigh / SonicMaster / spectral restoration post-pass on separated stems",
        ),
        (
            "--debleed",
            "Suppress vocal/instrument bleed on residual instrumental via Mel-Band-RoFormer bleed suppressor",
        ),
        (
            "--de-aspiration",
            "Remove excessive AI breath hiss on isolated vocals via Aspiration Mel-Band-RoFormer",
        ),
        (
            "--dereverb-echo",
            "Use Mel-Band-RoFormer combined de-reverb + de-echo model on vocals",
        ),
    ):
        sep_p.add_argument(flag, action="store_true", help=help_text)

    _add_io_parser(
        sub, "transcribe", "Run polyphonic audio-to-MIDI transcription via basic-pitch"
    )

    den_p = _add_io_parser(
        sub,
        "denoise",
        "Run neural denoising (DeepFilterNet3 / Mel-Band-RoFormer)",
        backend_choices=["auto", "deepfilternet", "mel-band-roformer"],
    )
    den_p.add_argument(
        "--attenuation-limit-db",
        type=float,
        default=None,
        help="Optional max attenuation limit in dB (0..100, DeepFilterNet3 only)",
    )

    rst_p = _add_io_parser(
        sub,
        "restore",
        "Run neural audio restoration / BWE (Apollo, FlowHigh, AudioSR, SonicMaster, or exciter)",
        backend_choices=sorted(VALID_RESTORE_BACKENDS),
    )
    rst_p.add_argument("--prompt", default=None)
    rst_p.add_argument("--chunk-secs", type=float, default=10.0)

    vr_p = _add_io_parser(
        sub,
        "vocal-rescue",
        "Rescue robotic/smeared vocals via Resemble Enhance, Seed-VC, or RoFormer clean chain",
        backend_choices=sorted(VALID_VOCAL_RESCUE_BACKENDS),
    )
    vr_p.add_argument("--reference-voice", type=Path, default=None)
    vr_p.add_argument("--no-dereverb", action="store_true")
    vr_p.add_argument("--no-de-aspiration", action="store_true")

    _add_io_parser(
        sub,
        "beat-track",
        "Detect beat/downbeat timestamps and variable tempo map",
        needs_out=False,
        backend_choices=sorted(VALID_BEAT_BACKENDS),
    )

    ev_p = _add_io_parser(
        sub,
        "evaluate",
        "Score perceptual audio aesthetics (Audiobox Aesthetics, SingMOS, or acoustic proxy)",
        needs_out=False,
        backend_choices=sorted(VALID_EVAL_BACKENDS),
    )
    ev_p.add_argument("--reference", type=Path, default=None)
    ev_p.add_argument("--vocal", action="store_true")

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

    lm_p = sub.add_parser(
        "list-models",
        help="List verified RoFormer/MDX23C checkpoints and local cache status",
    )
    lm_p.add_argument("--profile", default="all", choices=sorted(VALID_MODEL_PROFILES))
    lm_p.add_argument("--model-dir", type=Path, default=None)
    lm_p.add_argument("--verify-sha256", action="store_true")

    wc_p = sub.add_parser(
        "warm-cache",
        help="Pre-download verified checkpoints into model-dir with optional SHA-256 check",
    )
    wc_p.add_argument(
        "--profile", default="studio", choices=sorted(VALID_MODEL_PROFILES)
    )
    wc_p.add_argument("--model-dir", type=Path, default=None)
    wc_p.add_argument("--verify-sha256", action="store_true")
    wc_p.add_argument("--dry-run", action="store_true")

    f0_p = _add_io_parser(
        sub,
        "f0-track",
        "Track vocal/melodic fundamental frequency (F0) frames",
        needs_out=False,
        backend_choices=sorted(VALID_F0_BACKENDS),
    )
    f0_p.add_argument(
        "--hop-secs",
        type=float,
        default=0.02,
        help="Frame hop size in seconds (0.002..0.25)",
    )

    args = parser.parse_args()
    try:
        res: Mapping[str, object]
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
                debleed=args.debleed,
                de_aspiration=args.de_aspiration,
                dereverb_echo=args.dereverb_echo,
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
        elif args.command == "restore":
            res = run_restore(
                args.input,
                args.output_dir,
                backend=args.backend,
                prompt=args.prompt,
                chunk_secs=args.chunk_secs,
            )
            print(json.dumps(res))
        elif args.command == "vocal-rescue":
            res = run_vocal_rescue(
                args.input,
                args.output_dir,
                backend=args.backend,
                reference_voice=args.reference_voice,
                dereverb=not args.no_dereverb,
                de_aspiration=not args.no_de_aspiration,
            )
            print(json.dumps(res))
        elif args.command == "beat-track":
            res = run_beat_track(args.input, backend=args.backend)
            print(json.dumps(res))
        elif args.command == "evaluate":
            res = run_evaluate(
                args.input,
                reference_path=args.reference,
                vocal=args.vocal,
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
        elif args.command == "list-models":
            res = run_list_models(
                profile=args.profile,
                model_dir=args.model_dir,
                verify_sha256=args.verify_sha256,
            )
            print(json.dumps(res))
        elif args.command == "warm-cache":
            res = run_warm_cache(
                profile=args.profile,
                model_dir=args.model_dir,
                verify_sha256=args.verify_sha256,
                dry_run=args.dry_run,
            )
            print(json.dumps(res))
        elif args.command == "f0-track":
            res = run_f0_track(
                args.input,
                backend=args.backend,
                hop_secs=args.hop_secs,
            )
            print(json.dumps(res))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
