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
import json
import sys
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

# Normalized stem tags (lowercase, alphanumerics only) for each role.
VOCAL_TAGS = frozenset({"vocals", "vocal"})
INSTRUMENTAL_TAGS = frozenset({"instrumental", "instrumentals", "other"})
DRUMS_TAGS = frozenset({"drums", "drum"})
BASS_TAGS = frozenset({"bass"})
GUITAR_TAGS = frozenset({"guitar"})
KEYS_TAGS = frozenset({"piano", "keys", "keyboard"})
DRY_TAGS = frozenset({"noreverb", "dry"})
DENOISED_TAGS = frozenset({"dry", "nonoise", "denoised", "clean"})
DENOISE_ROFORMER_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
SIX_STEM_MODEL = "htdemucs_6s.yaml"


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
            candidate = (path if path.is_absolute() else resolved_dir / path).resolve()
            if not candidate.is_relative_to(resolved_dir):
                raise RuntimeError(
                    f"Separator wrote stem outside output_dir: {candidate}"
                )
            return candidate
    return None


def run_separate(
    input_path: Path, output_dir: Path, dereverb: bool = True, stems: int = 2
) -> dict[str, str | None]:
    if stems not in {2, 6}:
        raise ValueError(f"stems must be 2 or 6, got {stems!r}")
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
            "Run: uv pip install 'audio-separator[gpu]'"
        ) from exc

    sep = Separator(output_dir=str(resolved_output), output_format="WAV")
    if stems == 6:
        sep.load_model(model_filename=SIX_STEM_MODEL)
        files = sep.separate(str(resolved_input))
        vocal_file = pick_stem(files, VOCAL_TAGS, resolved_output)
        if dereverb and vocal_file and vocal_file.exists():
            sep.load_model(model_filename="deverb_bs_roformer_8_384dim_10depth.ckpt")
            deverb_files = sep.separate(str(vocal_file))
            vocal_file = (
                pick_stem(deverb_files, DRY_TAGS, resolved_output) or vocal_file
            )
        drums = pick_stem(files, DRUMS_TAGS, resolved_output)
        bass = pick_stem(files, BASS_TAGS, resolved_output)
        guitar = pick_stem(files, GUITAR_TAGS, resolved_output)
        keys = pick_stem(files, KEYS_TAGS, resolved_output)
        other = pick_stem(files, INSTRUMENTAL_TAGS, resolved_output)
        return {
            "vocals": str(vocal_file) if vocal_file else None,
            "drums": str(drums) if drums else None,
            "bass": str(bass) if bass else None,
            "guitar": str(guitar) if guitar else None,
            "keys": str(keys) if keys else None,
            "other": str(other) if other else None,
        }

    # 1. Mel-Band-RoFormer Kim Vocal (MIT license, ~11 dB SDR)
    sep.load_model(model_filename="model_mel_band_roformer_kim_vocals.ckpt")
    primary_files = sep.separate(str(resolved_input))

    vocal_file = pick_stem(primary_files, VOCAL_TAGS, resolved_output)
    inst_file = pick_stem(primary_files, INSTRUMENTAL_TAGS, resolved_output)

    # 2. Optional BS-RoFormer De-Reverb on isolated vocal stem
    if dereverb and vocal_file and vocal_file.exists():
        sep.load_model(model_filename="deverb_bs_roformer_8_384dim_10depth.ckpt")
        deverb_files = sep.separate(str(vocal_file))
        vocal_file = pick_stem(deverb_files, DRY_TAGS, resolved_output) or vocal_file

    return {
        "vocals": str(vocal_file) if vocal_file else None,
        "drums": None,
        "bass": None,
        "guitar": None,
        "keys": None,
        "other": str(inst_file) if inst_file else None,
    }


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
            "Run: uv pip install basic-pitch"
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
                    "Run: uv pip install deepfilternet"
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
        help="Run Mel-Band-RoFormer (2-stem) or HTDemucs (6-stem) separation",
    )
    sep_p.add_argument("--input", type=Path, required=True)
    sep_p.add_argument("--output-dir", type=Path, required=True)
    sep_p.add_argument("--no-dereverb", action="store_true")
    sep_p.add_argument("--stems", type=int, default=2, choices=[2, 6])

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
        if args.command == "separate":
            res = run_separate(
                args.input,
                args.output_dir,
                dereverb=not args.no_dereverb,
                stems=args.stems,
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
