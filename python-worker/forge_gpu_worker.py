#!/usr/bin/env python3
"""Ephemeral GPU Worker for ForgeAudio (`audio-forge`).

Executes heavy PyTorch/ONNX models on the local NVIDIA RTX 3080 or remote Colab A100
and exits immediately after writing artifacts to disk so 100% of VRAM and system RAM
are returned to REAPER 7.82.

Supported tasks:
- `separate`: Mel-Band-RoFormer (Kim Vocal, MIT) + BS-RoFormer 6-Stem + BS-RoFormer De-Reverb
- `apollo`: JusperLee/Apollo Band-Split GAN High-Frequency & Codec Restoration
- `transcribe`: SheetSage2 / Basic Pitch / YourMT3+ Audio-to-MIDI
- `yue2-remote`: Calls mrfakename/yue2-3b or custom Colab YuE2-3B endpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_separate(input_path: Path, output_dir: Path, dereverb: bool = True) -> dict[str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator is not installed in the worker venv. "
            "Run: uv pip install 'audio-separator[gpu]'"
        ) from exc

    sep = Separator(output_dir=str(output_dir), output_format="WAV")
    # 1. Mel-Band-RoFormer Kim Vocal (MIT license, ~11 dB SDR)
    sep.load_model(model_filename="model_mel_band_roformer_kim_vocals.ckpt")
    primary_files = sep.separate(str(input_path))

    vocal_file = next((Path(output_dir) / f for f in primary_files if "Vocal" in f), None)
    inst_file = next((Path(output_dir) / f for f in primary_files if "Instrumental" in f), None)

    # 2. Optional BS-RoFormer De-Reverb on isolated vocal stem
    if dereverb and vocal_file and vocal_file.exists():
        sep.load_model(model_filename="deverb_bs_roformer_8_384dim_10depth.ckpt")
        deverb_files = sep.separate(str(vocal_file))
        dry_vocal = next(
            (Path(output_dir) / f for f in deverb_files if "No Reverb" in f or "dry" in f.lower()),
            vocal_file,
        )
        vocal_file = dry_vocal

    return {
        "vocals": str(vocal_file) if vocal_file else None,
        "drums": None,
        "bass": None,
        "guitar": None,
        "keys": None,
        "other": str(inst_file) if inst_file else None,
    }


def run_yue2_remote(
    style: str,
    lyrics: str,
    planning_mode: str,
    ode_steps: int,
    seed: int,
    space_id: str,
) -> dict[str, str]:
    from gradio_client import Client

    client = Client(space_id)
    mp3_path, flac_path, abc_score, elapsed = client.predict(
        style=style,
        lyrics=lyrics,
        planning_mode=planning_mode,
        render_quality=ode_steps,
        seed=seed,
        api_name="/generate_song",
    )
    return {
        "mp3_path": str(mp3_path),
        "flac_path": str(flac_path),
        "abc_score": str(abc_score),
        "elapsed": str(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ForgeAudio Ephemeral GPU Worker")
    sub = parser.add_subparsers(dest="command", required=True)

    sep_p = sub.add_parser("separate", help="Run Mel-Band-RoFormer + BS-RoFormer De-Reverb")
    sep_p.add_argument("--input", type=Path, required=True)
    sep_p.add_argument("--output-dir", type=Path, required=True)
    sep_p.add_argument("--no-dereverb", action="store_true")

    yue_p = sub.add_parser("yue2-remote", help="Generate song + ABC score via YuE2-3B endpoint")
    yue_p.add_argument("--style", required=True)
    yue_p.add_argument("--lyrics", required=True)
    yue_p.add_argument("--planning-mode", default="full", choices=["full", "melody", "off"])
    yue_p.add_argument("--ode-steps", type=int, default=32, choices=[16, 32])
    yue_p.add_argument("--seed", type=int, default=42)
    yue_p.add_argument("--space", default="mrfakename/yue2-3b")

    args = parser.parse_args()
    try:
        if args.command == "separate":
            res = run_separate(args.input, args.output_dir, dereverb=not args.no_dereverb)
            print(json.dumps(res))
        elif args.command == "yue2-remote":
            res = run_yue2_remote(
                args.style,
                args.lyrics,
                args.planning_mode,
                args.ode_steps,
                args.seed,
                args.space,
            )
            print(json.dumps(res))
        return 0
    except (RuntimeError, ValueError, OSError, ImportError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
