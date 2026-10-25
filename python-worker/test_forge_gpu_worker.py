"""Unit tests for the GPU worker's stem picking and timeouts.

No GPU, models or network: ``audio_separator`` and ``gradio_client`` are replaced with fakes.
Run with ``python -m unittest discover -s python-worker``.
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
from concurrent.futures import Future
from pathlib import Path
from unittest import TestCase, main, mock

import forge_gpu_worker as worker

KIM = "model_mel_band_roformer_kim_vocals"
DEVERB = "deverb_bs_roformer_8_384dim_10depth"


class FakeSeparator:
    """Mimics audio-separator: ``<input stem>_(<Stem>)_<model>.wav`` names, instrumental first."""

    def __init__(self, output_dir: str, output_format: str) -> None:
        self.output_dir = Path(output_dir)
        self.model = ""

    def load_model(self, model_filename: str) -> None:
        self.model = Path(model_filename).stem

    def separate(self, audio_file: str) -> list[str]:
        stems = (
            ("Reverb", "No Reverb")
            if self.model == DEVERB
            else ("Instrumental", "Vocals")
        )
        base = Path(audio_file).stem
        names = [f"{base}_({stem})_{self.model}.wav" for stem in stems]
        for name in names:
            (self.output_dir / name).write_bytes(b"RIFF")
        return names


def fake_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class StemTagTests(TestCase):
    def test_tag_is_the_last_group_even_when_the_input_name_has_one(self) -> None:
        name = f"Lead_Vocal_(Vocals)_mix_(Instrumental)_{KIM}.wav"
        self.assertEqual(worker.stem_tag(name), "Instrumental")

    def test_names_without_a_tag_have_none(self) -> None:
        self.assertIsNone(worker.stem_tag("Lead_Vocal_mix.wav"))
        self.assertIsNone(worker.stem_tag("broken_(Vocals.wav"))

    def test_pick_stem_ignores_substrings_of_the_input_name(self) -> None:
        files = [
            f"Lead_Vocal_mix_(Instrumental)_{KIM}.wav",
            f"Lead_Vocal_mix_(Vocals)_{KIM}.wav",
        ]
        out = Path("out")
        self.assertEqual(
            worker.pick_stem(files, worker.VOCAL_TAGS, out), out / files[1]
        )
        self.assertEqual(
            worker.pick_stem(files, worker.INSTRUMENTAL_TAGS, out), out / files[0]
        )


class RunSeparateTests(TestCase):
    def run_separate(self, input_name: str, dereverb: bool) -> dict[str, str | None]:
        separator = fake_module("audio_separator.separator", Separator=FakeSeparator)
        modules = {
            "audio_separator": fake_module("audio_separator"),
            "audio_separator.separator": separator,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, modules),
        ):
            out = Path(tmp)
            result = worker.run_separate(out / input_name, out, dereverb=dereverb)
            return {k: (Path(v).name if v else None) for k, v in result.items()}

    def test_vocal_named_input_returns_the_vocal_stem(self) -> None:
        result = self.run_separate("Lead_Vocal_mix.wav", dereverb=False)
        self.assertEqual(result["vocals"], f"Lead_Vocal_mix_(Vocals)_{KIM}.wav")
        self.assertEqual(result["other"], f"Lead_Vocal_mix_(Instrumental)_{KIM}.wav")

    def test_dry_named_input_returns_the_dereverbed_vocal(self) -> None:
        result = self.run_separate("dry_take.wav", dereverb=True)
        self.assertEqual(
            result["vocals"], f"dry_take_(Vocals)_{KIM}_(No Reverb)_{DEVERB}.wav"
        )


class YueTimeoutTests(TestCase):
    def test_a_space_that_never_answers_times_out(self) -> None:
        never_done: Future[tuple[str, str, str, str]] = Future()

        class FakeClient:
            def __init__(self, space_id: str) -> None:
                self.space_id = space_id

            def submit(self, **_kwargs: object) -> Future[tuple[str, str, str, str]]:
                return never_done

            def predict(self, **_kwargs: object) -> tuple[str, str, str, str]:
                # The blocking call the worker must not use: it ignores any timeout.
                time.sleep(3.0)
                return ("a.mp3", "a.flac", "X:1", "3")

        modules = {"gradio_client": fake_module("gradio_client", Client=FakeClient)}
        started = time.monotonic()
        with (
            mock.patch.dict(sys.modules, modules),
            self.assertRaises(RuntimeError) as ctx,
        ):
            worker.run_yue2_remote("soul", "la", "full", 32, 1, "x/y", timeout_s=0.2)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("did not answer within 0.2 s", str(ctx.exception))
        self.assertTrue(never_done.cancelled())


if __name__ == "__main__":
    main()
