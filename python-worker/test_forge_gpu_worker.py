"""Unit tests for the GPU worker's stem picking and timeouts.

No GPU, models or network: ``audio_separator`` and ``gradio_client`` are replaced with fakes.
Run with ``python -m unittest discover -s python-worker``.
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
import types
import wave
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from pathlib import Path
from unittest import TestCase, main, mock

import forge_gpu_worker as worker

MEL_VOCAL = "vocals_mel_band_roformer"
BS_1297 = "model_bs_roformer_ep_317_sdr_12.9755"
DEVERB = "deverb_bs_roformer_8_384dim_10depth"
DEVERB_ECHO = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2"
MALE_FEMALE = "bs_roformer_male_female_by_aufr33_sdr_7.2889"
WIND_STEM = "17_HP-Wind_Inst-UVR"
DEBLEED = "mel_band_roformer_bleed_suppressor_v1"
ASPIRATION = "aspiration_mel_band_roformer_sdr_18.9845"


class FakeSeparator:
    """Mimics audio-separator: ``<input stem>_(<Stem>)_<model>.wav`` names, instrumental first."""

    def __init__(self, output_dir: str, output_format: str) -> None:
        self.output_dir = Path(output_dir)
        self.model = ""

    def load_model(self, model_filename: str) -> None:
        self.model = Path(model_filename).stem

    def write_stems(self, audio_file: str, stems: tuple[str, ...]) -> list[str]:
        base = Path(audio_file).stem
        names = [f"{base}_({stem})_{self.model}.wav" for stem in stems]
        for name in names:
            (self.output_dir / name).write_bytes(b"RIFF")
        return names

    def separate(self, audio_file: str) -> list[str]:
        stems = (
            ("Reverb", "No Reverb")
            if self.model == DEVERB
            else ("Instrumental", "Vocals")
        )
        return self.write_stems(audio_file, stems)


def fake_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def separator_modules(
    separator_cls: type[FakeSeparator],
    **extra: types.ModuleType | None,
) -> dict[str, types.ModuleType | None]:
    return {
        "audio_separator": fake_module("audio_separator"),
        "audio_separator.separator": fake_module(
            "audio_separator.separator", Separator=separator_cls
        ),
        **extra,
    }


@contextlib.contextmanager
def patched_temp_audio(
    modules: Mapping[str, types.ModuleType | None],
    input_name: str | None = None,
) -> Iterator[tuple[Path, Path]]:
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.dict(sys.modules, modules),
    ):
        out = Path(tmp)
        inp = out / (input_name or "input.wav")
        if input_name is not None:
            inp.write_bytes(b"RIFF")
        yield out, inp


class StemTagTests(TestCase):
    def test_tag_is_the_last_group_even_when_the_input_name_has_one(self) -> None:
        name = f"Lead_Vocal_(Vocals)_mix_(Instrumental)_{MEL_VOCAL}.wav"
        self.assertEqual(worker.stem_tag(name), "Instrumental")

    def test_names_without_a_tag_have_none(self) -> None:
        self.assertIsNone(worker.stem_tag("Lead_Vocal_mix.wav"))
        self.assertIsNone(worker.stem_tag("broken_(Vocals.wav"))

    def test_pick_stem_ignores_substrings_of_the_input_name(self) -> None:
        files = [
            f"Lead_Vocal_mix_(Instrumental)_{MEL_VOCAL}.wav",
            f"Lead_Vocal_mix_(Vocals)_{MEL_VOCAL}.wav",
        ]
        out = Path("out")
        self.assertEqual(
            worker.pick_stem(files, worker.VOCAL_TAGS, out), out / files[1]
        )
        self.assertEqual(
            worker.pick_stem(files, worker.INSTRUMENTAL_TAGS, out), out / files[0]
        )

    def test_pick_stem_rejects_path_traversal(self) -> None:
        out = Path("out")
        with self.assertRaises(RuntimeError):
            worker.pick_stem(
                [f"../escape_(Vocals)_{MEL_VOCAL}.wav"], worker.VOCAL_TAGS, out
            )


class RunSeparateTests(TestCase):
    def run_separate(self, input_name: str, dereverb: bool) -> dict[str, str | None]:
        with patched_temp_audio(separator_modules(FakeSeparator), input_name) as (
            out,
            inp,
        ):
            result = worker.run_separate(inp, out, dereverb=dereverb)
            return {
                k: (Path(v).name if isinstance(v, str) else None)
                for k, v in result.items()
            }

    def test_missing_input_fails_fast_before_loading_models(self) -> None:
        constructed = False

        class GuardSeparator(FakeSeparator):
            def __init__(self, output_dir: str, output_format: str) -> None:
                nonlocal constructed
                constructed = True
                super().__init__(output_dir, output_format)

        with patched_temp_audio(separator_modules(GuardSeparator)) as (out, _inp):
            with self.assertRaises(FileNotFoundError):
                worker.run_separate(out / "does_not_exist.wav", out, dereverb=False)
        self.assertFalse(constructed)

    def test_vocal_named_input_returns_the_vocal_stem(self) -> None:
        result = self.run_separate("Lead_Vocal_mix.wav", dereverb=False)
        self.assertEqual(result["vocals"], f"Lead_Vocal_mix_(Vocals)_{BS_1297}.wav")
        self.assertEqual(
            result["other"], f"Lead_Vocal_mix_(Instrumental)_{BS_1297}.wav"
        )

    def test_dry_named_input_returns_the_dereverbed_vocal(self) -> None:
        result = self.run_separate("dry_take.wav", dereverb=True)
        self.assertEqual(
            result["vocals"], f"dry_take_(Vocals)_{BS_1297}_(No Reverb)_{DEVERB}.wav"
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

    def test_non_positive_timeout_is_rejected(self) -> None:
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                worker.run_yue2_remote(
                    "soul", "la", "full", 32, 1, "x/y", timeout_s=bad
                )


class RunDenoiseTests(TestCase):
    def test_missing_input_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                worker.run_denoise(out / "missing.wav", out, backend="deepfilternet")

    def test_invalid_attenuation_limit_and_backend_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            inp = out / "take.wav"
            inp.write_bytes(b"RIFF")
            for bad in (-1.0, 120.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    worker.run_denoise(inp, out, attenuation_limit_db=bad)
            with self.assertRaises(ValueError):
                worker.run_denoise(inp, out, backend="unknown")

    def test_deepfilternet_backend_writes_denoised_wav(self) -> None:
        seen_atten: list[float | None] = []

        class FakeState:
            def sr(self) -> int:
                return 48_000

        def init_df() -> tuple[object, FakeState, str]:
            return (object(), FakeState(), "")

        def load_audio(_path: str, sr: int) -> tuple[bytes, int]:
            return (b"AUDIO", sr)

        def enhance(
            _model: object,
            _state: FakeState,
            audio: bytes,
            atten_lim_db: float | None = None,
        ) -> bytes:
            seen_atten.append(atten_lim_db)
            return audio

        def save_audio(path: str, _audio: bytes, _sr: int) -> None:
            Path(path).write_bytes(b"RIFF")

        modules = {
            "df": fake_module("df"),
            "df.enhance": fake_module(
                "df.enhance",
                init_df=init_df,
                load_audio=load_audio,
                enhance=enhance,
                save_audio=save_audio,
            ),
        }
        with patched_temp_audio(modules, "noisy_vox.wav") as (out, inp):
            res = worker.run_denoise(
                inp, out, attenuation_limit_db=18.0, backend="auto"
            )
            self.assertEqual(res["backend"], "deepfilternet")
            self.assertEqual(
                Path(res["denoised"]).name, "noisy_vox_denoised_deepfilternet.wav"
            )
            self.assertTrue(Path(res["denoised"]).is_file())
            self.assertEqual(seen_atten, [18.0])

    def test_auto_falls_back_to_roformer_when_deepfilternet_absent(self) -> None:
        class DenoiseSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                return self.write_stems(audio_file, ("Noise", "Dry"))

        modules = separator_modules(DenoiseSeparator, df=None, **{"df.enhance": None})
        with patched_temp_audio(modules, "room_vox.wav") as (out, inp):
            res = worker.run_denoise(inp, out, backend="auto")
            self.assertEqual(res["backend"], "mel-band-roformer")
            self.assertIn("_(Dry)_", Path(res["denoised"]).name)
            self.assertTrue(Path(res["denoised"]).is_file())


class SixStemAndTranscribeTests(TestCase):
    def test_six_stem_separation_populates_all_six_slots(self) -> None:
        class SixStemSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                return self.write_stems(
                    audio_file, ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other")
                )

        with patched_temp_audio(
            separator_modules(SixStemSeparator), "full_mix.wav"
        ) as (out, inp):
            res = worker.run_separate(inp, out, dereverb=False, stems=6, quality="fast")
            expected = {
                "vocals": "full_mix_(Vocals)_htdemucs_6s.wav",
                "drums": "full_mix_(Drums)_htdemucs_6s.wav",
                "bass": "full_mix_(Bass)_htdemucs_6s.wav",
                "guitar": "full_mix_(Guitar)_htdemucs_6s.wav",
                "keys": "full_mix_(Piano)_htdemucs_6s.wav",
                "other": "full_mix_(Other)_htdemucs_6s.wav",
            }
            for key, filename in expected.items():
                val = res[key]
                self.assertIsInstance(val, str, f"missing 6-stem slot: {key}")
                assert isinstance(val, str)
                self.assertEqual(Path(val).name, filename)
                self.assertTrue(Path(val).is_file())

    def test_studio_cascade_and_drumsep_populate_all_eleven_slots(self) -> None:
        calls: list[tuple[str, str]] = []

        class CascadeSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                calls.append((self.model, Path(audio_file).name))
                stems: tuple[str, ...]
                if self.model in {"BS-Roformer-SW", "htdemucs_6s"}:
                    stems = ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other")
                elif self.model == "MDX23C-DrumSep-aufr33-jarredou":
                    stems = ("Kick", "Snare", "Toms", "HH", "Cymbals")
                elif (
                    self.model == "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956"
                ):
                    stems = ("Vocals", "Instrumental")
                else:
                    stems = ("Instrumental", "Vocals")
                return self.write_stems(audio_file, stems)

        with patched_temp_audio(separator_modules(CascadeSeparator), "song.wav") as (
            out,
            inp,
        ):
            res = worker.run_separate(
                inp,
                out,
                dereverb=False,
                stems=6,
                quality="studio",
                drum_sep=True,
                split_lead_backing=True,
            )
            self.assertEqual(calls[0], (BS_1297, "song.wav"))
            self.assertEqual(
                calls[1], ("BS-Roformer-SW", f"song_(Instrumental)_{BS_1297}.wav")
            )
            self.assertEqual(
                calls[2][0], "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956"
            )
            self.assertEqual(calls[3][0], "MDX23C-DrumSep-aufr33-jarredou")
            for key in (
                "vocals",
                "lead_vocals",
                "backing_vocals",
                "drums",
                "bass",
                "guitar",
                "keys",
                "other",
                "kick",
                "snare",
                "toms",
                "hihat",
                "cymbals",
            ):
                self.assertIsNotNone(res[key], f"missing cascade slot: {key}")

    def test_four_stem_sw_falls_back_to_htdemucs_and_spectral_restore_runs(
        self,
    ) -> None:
        loaded_models: list[str] = []

        def write_pcm(path: Path) -> None:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48_000)
                pcm = [((i % 8) - 4) * 2000 for i in range(64)]
                w.writeframes(array.array("h", pcm).tobytes())

        class FallbackSeparator(FakeSeparator):
            def load_model(self, model_filename: str) -> None:
                loaded_models.append(model_filename)
                if model_filename == worker.FOUR_STEM_PRIMARY_MODEL:
                    raise FileNotFoundError(f"missing checkpoint: {model_filename}")
                super().load_model(model_filename)

            def separate(self, audio_file: str) -> list[str]:
                base = Path(audio_file).stem
                stems: tuple[str, ...]
                if self.model == "htdemucs_ft":
                    stems = ("Vocals", "Drums", "Bass", "Other")
                else:
                    stems = ("Instrumental", "Vocals")
                names = [f"{base}_({stem})_{self.model}.wav" for stem in stems]
                for name in names:
                    write_pcm(self.output_dir / name)
                return names

        with patched_temp_audio(separator_modules(FallbackSeparator)) as (out, _inp):
            inp = out / "master.wav"
            write_pcm(inp)
            res = worker.run_separate(
                inp, out, dereverb=False, stems=4, quality="studio", restore=True
            )
            self.assertIn(worker.FOUR_STEM_PRIMARY_MODEL, loaded_models)
            self.assertIn(worker.FOUR_STEM_MODEL, loaded_models)
            vocals_path = res["vocals"]
            self.assertIsInstance(vocals_path, str)
            assert isinstance(vocals_path, str)
            self.assertTrue(
                Path(vocals_path).name.endswith("_restored.wav"), vocals_path
            )
            drums_path = res["drums"]
            self.assertIsInstance(drums_path, str)
            assert isinstance(drums_path, str)
            self.assertTrue(Path(drums_path).is_file())
            warnings_val = res.get("warnings")
            self.assertIsInstance(warnings_val, list)
            assert isinstance(warnings_val, list)
            self.assertTrue(
                any(worker.FOUR_STEM_PRIMARY_MODEL in str(w) for w in warnings_val)
            )

    def test_four_stem_sw_folds_guitar_and_keys_into_other(self) -> None:
        def write_const(path: Path, sample_val: int) -> None:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48_000)
                w.writeframes(array.array("h", [sample_val] * 32).tobytes())

        class FoldSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                base = Path(audio_file).stem
                if self.model == "BS-Roformer-SW":
                    vals = {
                        "Vocals": 100,
                        "Drums": 200,
                        "Bass": 300,
                        "Guitar": 1000,
                        "Piano": 2000,
                        "Other": 500,
                    }
                else:
                    vals = {"Instrumental": 500, "Vocals": 100}
                names = [f"{base}_({s})_{self.model}.wav" for s in vals]
                for name, (_, val) in zip(names, vals.items(), strict=True):
                    write_const(self.output_dir / name, val)
                return names

        with patched_temp_audio(separator_modules(FoldSeparator)) as (out, _inp):
            inp = out / "mix.wav"
            write_const(inp, 1000)
            res = worker.run_separate(
                inp, out, dereverb=False, stems=4, quality="studio"
            )
            self.assertIsNone(res["guitar"])
            self.assertIsNone(res["keys"])
            other_str = res["other"]
            self.assertIsInstance(other_str, str)
            assert isinstance(other_str, str)
            samples, sr, sampwidth, afmt = worker._read_wav_float(Path(other_str)) or (
                [],
                0,
                0,
                0,
            )
            self.assertEqual((len(samples), sr, sampwidth, afmt), (2, 48_000, 2, 1))
            # Guitar (1000) + Piano (2000) + Other (500) = 3500
            self.assertAlmostEqual(samples[0][0] * 32768.0, 3500.0, delta=2.0)

    def test_debleed_de_aspiration_dereverb_echo_and_unconfigured_warnings(
        self,
    ) -> None:
        loaded: list[str] = []

        class FlagsSeparator(FakeSeparator):
            def load_model(self, model_filename: str) -> None:
                loaded.append(model_filename)
                super().load_model(model_filename)

            def separate(self, audio_file: str) -> list[str]:
                if self.model == DEVERB_ECHO:
                    return self.write_stems(audio_file, ("Reverb", "No Reverb"))
                if self.model == ASPIRATION:
                    return self.write_stems(audio_file, ("Other", "No Aspiration"))
                if self.model == DEBLEED:
                    return self.write_stems(audio_file, ("Clean", "Bleed"))
                if self.model in {"BS-Roformer-SW", "htdemucs_6s"}:
                    return self.write_stems(
                        audio_file,
                        ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"),
                    )
                if self.model == WIND_STEM:
                    return self.write_stems(audio_file, ("Wind", "Instrumental"))
                return self.write_stems(audio_file, ("Instrumental", "Vocals"))

        clean_env = {
            k: ""
            for k in (
                "FORGE_GUITAR_SUB_MODEL",
                "FORGE_STRINGS_MODEL",
                "FORGE_BRASS_MODEL",
                "FORGE_SYNTH_MODEL",
            )
        }
        with (
            mock.patch.dict(os.environ, clean_env),
            patched_temp_audio(separator_modules(FlagsSeparator), "song.wav") as (
                out,
                inp,
            ),
        ):
            res = worker.run_separate(
                inp,
                out,
                dereverb=True,
                dereverb_echo=True,
                de_aspiration=True,
                debleed=True,
                stems=6,
                quality="studio",
                split_guitar=True,
                split_orchestral=True,
            )
            self.assertIn(worker.DEREVERB_ECHO_MODEL, loaded)
            self.assertIn(worker.ASPIRATION_MODEL, loaded)
            self.assertIn(worker.DEBLEED_MODEL, loaded)
            self.assertIsNotNone(res["wind"])
            warnings = res["warnings"]
            self.assertIsInstance(warnings, list)
            assert isinstance(warnings, list)
            joined = "\n".join(warnings)
            for missing in ("guitar", "strings", "brass", "synth"):
                self.assertIn(missing, joined.lower())

    def test_full_hierarchical_23_slot_cascade_populates_all_sub_stems(self) -> None:
        model_outputs: dict[str, tuple[str, ...]] = {
            BS_1297: ("Instrumental", "Vocals"),
            "BS-Roformer-SW": ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"),
            DEVERB: ("Reverb", "No Reverb"),
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956": (
                "Vocals",
                "Instrumental",
            ),
            MALE_FEMALE: ("Male", "Female"),
            "MDX23C-DrumSep-aufr33-jarredou": (
                "Kick",
                "Snare",
                "Toms",
                "HH",
                "Cymbals",
            ),
            "custom_guitar_split": ("Electric Guitar", "Acoustic Guitar"),
            "custom_strings": ("Strings", "Other"),
            "custom_brass": ("Brass", "Other"),
            WIND_STEM: ("Wind", "Instrumental"),
            "custom_synth": ("Synth", "Other"),
            "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144": ("Crowd", "Other"),
        }

        class FullCascadeSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                return self.write_stems(audio_file, model_outputs[self.model])

        env_overrides = {
            "FORGE_GUITAR_SUB_MODEL": "custom_guitar_split.ckpt",
            "FORGE_STRINGS_MODEL": "custom_strings.ckpt",
            "FORGE_BRASS_MODEL": "custom_brass.ckpt",
            "FORGE_SYNTH_MODEL": "custom_synth.ckpt",
        }
        with (
            mock.patch.dict(os.environ, env_overrides),
            patched_temp_audio(
                separator_modules(FullCascadeSeparator), "symphony.wav"
            ) as (out, inp),
        ):
            res = worker.run_separate(
                inp,
                out,
                dereverb=True,
                stems=6,
                quality="studio",
                drum_sep=True,
                split_lead_backing=True,
                split_male_female=True,
                split_guitar=True,
                split_orchestral=True,
                split_crowd=True,
            )
            expected_keys = (
                "vocals",
                "lead_vocals",
                "backing_vocals",
                "male_vocals",
                "female_vocals",
                "vocal_reverb",
                "drums",
                "bass",
                "guitar",
                "electric_guitar",
                "acoustic_guitar",
                "keys",
                "strings",
                "brass",
                "wind",
                "synth",
                "crowd",
                "other",
                "kick",
                "snare",
                "toms",
                "hihat",
                "cymbals",
            )
            self.assertEqual(len(expected_keys), 23)
            for key in expected_keys:
                val = res[key]
                self.assertIsNotNone(val, f"missing hierarchical slot: {key}")
                assert isinstance(val, str)
                self.assertTrue(Path(val).is_file())
            self.assertEqual(res["warnings"], [])

    def test_ultra_ensemble_blends_bs_and_mel_roformer_wavs(self) -> None:
        def write_pcm(path: Path, sample_val: int) -> None:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48_000)
                w.writeframes(array.array("h", [sample_val] * 64).tobytes())

        class EnsembleSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                base = Path(audio_file).stem
                val = 10_000 if self.model == BS_1297 else 20_000
                names = [
                    f"{base}_(Instrumental)_{self.model}.wav",
                    f"{base}_(Vocals)_{self.model}.wav",
                ]
                for name in names:
                    write_pcm(self.output_dir / name, val)
                return names

        with patched_temp_audio(separator_modules(EnsembleSeparator)) as (out, _inp):
            inp = out / "mix.wav"
            write_pcm(inp, 15_000)
            res = worker.run_separate(
                inp, out, dereverb=False, stems=2, quality="ultra"
            )
            vocal_str = res["vocals"]
            self.assertIsNotNone(vocal_str)
            assert isinstance(vocal_str, str)
            vocal_path = Path(vocal_str)
            self.assertEqual(vocal_path.name, "mix_(Vocals)_roformer_ensemble.wav")
            with wave.open(str(vocal_path), "rb") as w:
                samples = array.array("h")
                samples.frombytes(w.readframes(4))
            # 0.6 * 10_000 + 0.4 * 20_000 = 14_000
            self.assertEqual(samples[0], 14_000)

    def test_transcribe_writes_midi_file(self) -> None:
        class FakeMidi:
            def write(self, path: str) -> None:
                Path(path).write_bytes(b"MThd")

        def fake_predict(_audio: str) -> tuple[object, FakeMidi, list[object]]:
            return (None, FakeMidi(), [])

        modules: dict[str, types.ModuleType | None] = {
            "basic_pitch": fake_module("basic_pitch"),
            "basic_pitch.inference": fake_module(
                "basic_pitch.inference", predict=fake_predict
            ),
        }
        with patched_temp_audio(modules, "piano_solo.wav") as (out, inp):
            res = worker.run_transcribe(inp, out)
            self.assertEqual(Path(res["midi"]).name, "piano_solo_basic_pitch.mid")
            self.assertTrue(Path(res["midi"]).is_file())


class MultiFormatWavTests(TestCase):
    @staticmethod
    def _write_float32_wav(path: Path, samples: list[float], sr: int = 48_000) -> None:
        payload = struct.pack(f"<{len(samples)}f", *samples)
        fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, 2, sr, sr * 8, 8, 32)
        data_chunk = struct.pack("<4sI", b"data", len(payload)) + payload
        header = struct.pack(
            "<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE"
        )
        path.write_bytes(header + fmt_chunk + data_chunk)

    def test_blend_and_restore_work_across_pcm16_24_32_and_float32(self) -> None:
        vals = [
            0.25 * math.sin(2.0 * math.pi * 220.0 * i / 48_000.0) for i in range(256)
        ]
        interleaved = [v for v in vals for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 16-bit, 24-bit, 32-bit PCM via pure stdlib writer, and IEEE float32
            for sw in (2, 3, 4):
                p_a = root / f"a_{sw}.wav"
                p_b = root / f"b_{sw}.wav"
                worker._write_wav_float(
                    p_a, [vals[:], vals[:]], 48_000, sw, audio_format=1
                )
                worker._write_wav_float(
                    p_b,
                    [[x * 0.5 for x in vals], [x * 0.5 for x in vals]],
                    48_000,
                    sw,
                    audio_format=1,
                )
                blended = worker._blend_wav_pair(p_a, p_b, root / f"mix_{sw}.wav", 0.5)
                restored = worker._restore_wav_bandwidth(p_a, root / f"rst_{sw}.wav")
                decoded = worker._read_wav_float(blended)
                self.assertIsNotNone(decoded)
                assert decoded is not None
                self.assertEqual(decoded[2], sw)
                self.assertEqual(decoded[3], 1)
                self.assertTrue(restored.is_file())

            f32_path = root / "f32.wav"
            self._write_float32_wav(f32_path, interleaved)
            decoded_f = worker._read_wav_float(f32_path)
            self.assertIsNotNone(decoded_f)
            assert decoded_f is not None
            restored_f = worker._restore_wav_bandwidth(f32_path, root / "rst_f32.wav")
            decoded_rst = worker._read_wav_float(restored_f)
            self.assertIsNotNone(decoded_rst)
            assert decoded_rst is not None
            self.assertEqual(decoded_rst[2], 4)
            self.assertEqual(decoded_rst[3], 3)


class NoDeadCheckpointsTests(TestCase):
    def test_no_unverified_or_dead_checkpoints_in_worker_or_notebook(self) -> None:
        dead_models = (
            "model_mel_band_roformer_kim_vocals",
            "scnet_xl_ihf",
            "model_bs_roformer_6s",
            "model_mel_band_roformer_male_female",
            "model_bs_roformer_guitar_elec_acoust",
            "model_bs_roformer_strings",
            "model_bs_roformer_brass",
            "model_bs_roformer_wind",
            "model_bs_roformer_synth",
            "apollo_universal_restore",
        )
        repo_root = Path(__file__).resolve().parent.parent
        worker_src = (repo_root / "python-worker" / "forge_gpu_worker.py").read_text(
            encoding="utf-8"
        )
        nb_src = (
            repo_root / "notebooks" / "ForgeAudio_Colab_A100_Worker.ipynb"
        ).read_text(encoding="utf-8")
        # Ensure notebook parses as valid JSON too
        json.loads(nb_src)
        for bad in dead_models:
            self.assertNotIn(bad, worker_src)
            self.assertNotIn(bad, nb_src)

        # Ensure MCP forge://models resource stays in sync with forge_gpu_worker.py constants.
        mcp_res = (
            repo_root / "crates" / "forge-mcp" / "src" / "resources.rs"
        ).read_text(encoding="utf-8")
        for expected in (
            worker.MEL_ROFORMER_VOCAL_MODEL,
            worker.BS_ROFORMER_VOCAL_MODEL,
            worker.BS_ROFORMER_SIX_STEM_MODEL,
            worker.DEREVERB_MODEL,
            worker.DEREVERB_ECHO_MODEL,
            worker.DENOISE_ROFORMER_MODEL,
            worker.DEBLEED_MODEL,
            worker.ASPIRATION_MODEL,
            worker.VOCAL_REVIVE_MODEL,
            worker.DRUMSEP_MODEL,
            worker.KARAOKE_MODEL,
            '"rmvpe", "fcpe", "autocorr"',
        ):
            self.assertIn(expected, mcp_res)
        for bad_f0 in ("torchcrepe", "librosa-yin"):
            self.assertNotIn(bad_f0, mcp_res)


class NewTasksTests(TestCase):
    @staticmethod
    def _write_click_wav(path: Path, sr: int = 48_000, secs: float = 3.0) -> None:
        frames = int(sr * secs)
        step = int(sr * 0.5)  # 120 BPM
        ch = [0.0] * frames
        for start in range(sr // 4, frames - 400, step):
            for k in range(300):
                ch[start + k] = 0.8 * math.exp(-k / 50.0)
        worker._write_wav_float(path, [ch, ch], sr, 2, audio_format=1)

    def test_restore_exciter_fallback_and_invalid_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "ai.wav"
            self._write_click_wav(inp, secs=1.0)
            res = worker.run_restore(inp, root / "out", backend="exciter")
            self.assertEqual(res["backend"], "exciter")
            self.assertTrue(Path(str(res["restored"])).is_file())
            with self.assertRaises(ValueError):
                worker.run_restore(inp, root / "out", backend="bogus")

    def test_vocal_rescue_roformer_chain_and_seed_vc_requires_ref(self) -> None:
        class RescueSep(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                if self.model in {DEVERB_ECHO, DEVERB}:
                    return self.write_stems(audio_file, ("Reverb", "No Reverb"))
                if self.model == ASPIRATION:
                    return self.write_stems(audio_file, ("Other", "No Aspiration"))
                return self.write_stems(audio_file, ("Vocals", "Other"))

        with patched_temp_audio(separator_modules(RescueSep), "ai_vox.wav") as (
            out,
            inp,
        ):
            res = worker.run_vocal_rescue(inp, out, backend="roformer-clean")
            self.assertEqual(res["backend"], "roformer-clean")
            self.assertTrue(Path(str(res["rescued"])).is_file())
            self.assertIn("dereverb-echo", res["stages"])
            with self.assertRaises(ValueError):
                worker.run_vocal_rescue(
                    inp, out, backend="seed-vc", reference_voice=None
                )

    def test_beat_track_and_evaluate_pure_stdlib_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "beat.wav"
            ref = root / "ref.wav"
            self._write_click_wav(inp, secs=3.0)
            self._write_click_wav(ref, secs=3.0)

            beats = worker.run_beat_track(inp, backend="energy")
            self.assertEqual(beats["backend"], "energy")
            beats_secs = beats["beats_secs"]
            self.assertIsInstance(beats_secs, list)
            assert isinstance(beats_secs, list)
            self.assertGreater(len(beats_secs), 2)
            median_bpm = beats["median_bpm"]
            self.assertIsInstance(median_bpm, (int, float))
            assert isinstance(median_bpm, (int, float))
            self.assertGreater(float(median_bpm), 0.0)

            ev = worker.run_evaluate(
                inp, reference_path=ref, vocal=True, backend="acoustic-proxy"
            )
            self.assertEqual(ev["backend"], "acoustic-proxy")
            self.assertIn("metrics", ev)
            self.assertIsNotNone(ev["reference_metrics"])

    def test_list_models_and_warm_cache_with_sha256_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            fast_file = model_dir / worker.MEL_ROFORMER_VOCAL_MODEL
            fast_file.write_bytes(b"weights-v1")
            digest = hashlib.sha256(b"weights-v1").hexdigest()
            manifest = root / "sha256.json"
            manifest.write_text(
                json.dumps({worker.MEL_ROFORMER_VOCAL_MODEL: digest}),
                encoding="utf-8",
            )
            prev_manifest = os.environ.get("FORGE_MODEL_SHA256_MANIFEST")
            os.environ["FORGE_MODEL_SHA256_MANIFEST"] = str(manifest)
            try:
                listed = worker.run_list_models(
                    profile="fast", model_dir=model_dir, verify_sha256=True
                )
                models = listed["models"]
                self.assertIsInstance(models, list)
                assert isinstance(models, list)
                self.assertEqual(len(models), 1)
                self.assertTrue(models[0]["cached"])
                self.assertTrue(models[0]["sha256_ok"])

                warmed = worker.run_warm_cache(
                    profile="fast",
                    model_dir=model_dir,
                    verify_sha256=True,
                    dry_run=True,
                )
                self.assertEqual(
                    warmed["verified"], {worker.MEL_ROFORMER_VOCAL_MODEL: True}
                )

                manifest.write_text(
                    json.dumps({worker.MEL_ROFORMER_VOCAL_MODEL: "0" * 64}),
                    encoding="utf-8",
                )
                with self.assertRaises(RuntimeError):
                    worker.run_warm_cache(
                        profile="fast",
                        model_dir=model_dir,
                        verify_sha256=True,
                        dry_run=True,
                    )
            finally:
                if prev_manifest is None:
                    os.environ.pop("FORGE_MODEL_SHA256_MANIFEST", None)
                else:
                    os.environ["FORGE_MODEL_SHA256_MANIFEST"] = prev_manifest

    def test_f0_track_autocorr_detects_220hz_and_rejects_bad_args(self) -> None:
        sr = 48_000
        dur_frames = int(sr * 0.4)
        tone = [
            0.4 * math.sin(2.0 * math.pi * 220.0 * i / sr) for i in range(dur_frames)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "a3_tone.wav"
            worker._write_wav_float(wav_path, [tone], sr)
            res = worker.run_f0_track(wav_path, backend="autocorr", hop_secs=0.02)
            self.assertEqual(res["backend"], "autocorr")
            self.assertEqual(res["sample_rate"], sr)
            self.assertIsNotNone(res["median_f0_hz"])
            med = float(res["median_f0_hz"])  # type: ignore[arg-type]
            self.assertLess(abs(med - 220.0), 1.5, f"expected ~220 Hz, got {med}")
            with self.assertRaises(ValueError):
                worker.run_f0_track(wav_path, backend="bogus-f0", hop_secs=0.02)
            with self.assertRaises(ValueError):
                worker.run_f0_track(wav_path, backend="autocorr", hop_secs=0.0)

    def test_worker_path_traversal_and_root_containment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "workspace"
            allowed.mkdir()
            wav_ok = allowed / "ok.wav"
            self._write_click_wav(wav_ok, secs=0.2)
            outside = root / "outside.wav"
            self._write_click_wav(outside, secs=0.2)

            with self.assertRaises(ValueError):
                worker._validate_worker_path(allowed / ".." / "outside.wav")

            prev_root = os.environ.get("FORGE_WORKER_ROOT")
            os.environ["FORGE_WORKER_ROOT"] = str(allowed)
            try:
                self.assertEqual(
                    worker._validate_worker_path(wav_ok, must_exist_file=True),
                    wav_ok.resolve(),
                )
                with self.assertRaises(PermissionError):
                    worker._validate_worker_path(outside, must_exist_file=True)
            finally:
                if prev_root is None:
                    os.environ.pop("FORGE_WORKER_ROOT", None)
                else:
                    os.environ["FORGE_WORKER_ROOT"] = prev_root

    def test_evaluate_undecodable_non_wav_raises_instead_of_fabricating_score(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_mp3 = Path(tmp) / "corrupt.mp3"
            bad_mp3.write_bytes(b"not-valid-audio-stream")
            with (
                mock.patch.dict(
                    sys.modules,
                    {"soundfile": None, "librosa": None, "audioread": None},
                ),
                self.assertRaises(RuntimeError),
            ):
                worker.run_evaluate(bad_mp3, backend="acoustic-proxy")

    def test_f0_explicit_neural_backends_invoke_model_or_fail_cleanly(self) -> None:
        sr = 48_000
        dur = int(sr * 0.25)
        tone = [0.4 * math.sin(2.0 * math.pi * 220.0 * i / sr) for i in range(dur)]
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "a3.wav"
            worker._write_wav_float(wav_path, [tone], sr)

            with (
                mock.patch.dict(sys.modules, {"torchfcpe": None, "rmvpe": None}),
            ):
                with self.assertRaises(RuntimeError):
                    worker.run_f0_track(wav_path, backend="fcpe", hop_secs=0.02)
                with self.assertRaises(RuntimeError):
                    worker.run_f0_track(wav_path, backend="rmvpe", hop_secs=0.02)
                auto_res = worker.run_f0_track(wav_path, backend="auto", hop_secs=0.02)
                self.assertEqual(auto_res["backend"], "autocorr")
                self.assertTrue(auto_res["warnings"])

            class FakeTensor:
                def __init__(self, vals: list[float]) -> None:
                    self._vals = vals
                    self.ndim = 1

                def unsqueeze(self, _dim: int) -> FakeTensor:
                    return self

                def tolist(self) -> list[float]:
                    return list(self._vals)

            class FakeTorch:
                float32 = "float32"

                class cuda:
                    @staticmethod
                    def is_available() -> bool:
                        return False

                @staticmethod
                def tensor(vals: list[float], **_kw: object) -> FakeTensor:
                    return FakeTensor(vals)

            class FakeFcpeModel:
                def __init__(self) -> None:
                    self.seen_kwargs: dict[str, object] = {}

                def infer(self, _t: object, **kw: object) -> list[float]:
                    self.seen_kwargs = dict(kw)
                    return [220.0, 220.0, 0.0, 221.5]

            fake_model = FakeFcpeModel()

            class FakeFcpeMod:
                @staticmethod
                def spawn_bundled_infer_model(device: str = "cpu") -> FakeFcpeModel:
                    _ = device
                    return fake_model

            with mock.patch.dict(
                sys.modules, {"torchfcpe": FakeFcpeMod(), "torch": FakeTorch()}
            ):
                fcpe_res = worker.run_f0_track(wav_path, backend="fcpe", hop_secs=0.02)
                self.assertEqual(fcpe_res["backend"], "fcpe")
                self.assertEqual(fcpe_res["median_f0_hz"], 220.0)
                self.assertEqual(
                    fake_model.seen_kwargs.get("output_interp_target_length"), 13
                )
                fcpe_frames = fcpe_res["frames"]
                assert isinstance(fcpe_frames, list)
                self.assertEqual(len(fcpe_frames), 13)
                self.assertLessEqual(float(fcpe_frames[-1]["time_secs"]), 0.25)  # type: ignore[arg-type]

            class LegacyFcpeModel:
                def infer(
                    self,
                    _t: object,
                    *,
                    sr: int,
                    decoder_mode: str,
                    threshold: float,
                    f0_min: float,
                    f0_max: float,
                    interp_uv: bool,
                ) -> list[float]:
                    _ = (sr, decoder_mode, threshold, f0_min, f0_max, interp_uv)
                    return [220.0] * 25

            class LegacyFcpeMod:
                @staticmethod
                def spawn_bundled_infer_model(device: str = "cpu") -> LegacyFcpeModel:
                    _ = device
                    return LegacyFcpeModel()

            with mock.patch.dict(
                sys.modules, {"torchfcpe": LegacyFcpeMod(), "torch": FakeTorch()}
            ):
                legacy_res = worker.run_f0_track(
                    wav_path, backend="fcpe", hop_secs=0.02
                )
                legacy_frames = legacy_res["frames"]
                assert isinstance(legacy_frames, list)
                self.assertEqual(len(legacy_frames), 13)
                self.assertLessEqual(float(legacy_frames[-1]["time_secs"]), 0.25)  # type: ignore[arg-type]

            class FakeRmvpeModel:
                def __init__(self, *_a: object, **_kw: object) -> None:
                    pass

                def infer_from_audio(
                    self, _mono: object, thred: float = 0.03
                ) -> list[float]:
                    _ = thred
                    # Simulate native 10 ms hop (25 frames for 0.25 s audio)
                    return [220.0] * 12 + [440.0] * 13

            fake_rmvpe_mod = types.ModuleType("rmvpe")
            fake_rmvpe_mod.RMVPE = FakeRmvpeModel  # type: ignore[attr-defined]
            with mock.patch.dict(
                sys.modules, {"rmvpe": fake_rmvpe_mod, "torch": FakeTorch()}
            ):
                rmvpe_res = worker.run_f0_track(
                    wav_path, backend="rmvpe", hop_secs=0.02
                )
                self.assertEqual(rmvpe_res["backend"], "rmvpe")
                rmvpe_frames = rmvpe_res["frames"]
                assert isinstance(rmvpe_frames, list)
                self.assertEqual(len(rmvpe_frames), 13)
                self.assertAlmostEqual(float(rmvpe_frames[0]["f0_hz"]), 220.0)  # type: ignore[arg-type]
                self.assertAlmostEqual(float(rmvpe_frames[-1]["f0_hz"]), 440.0)  # type: ignore[arg-type]
                self.assertAlmostEqual(
                    float(rmvpe_frames[-1]["time_secs"]),  # type: ignore[arg-type]
                    0.24,
                    delta=0.001,
                )

    def test_restore_and_roformer_chain_never_escape_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp_dir = root / "in"
            out_dir = root / "out"
            inp_dir.mkdir()
            out_dir.mkdir()
            fake_mp3 = inp_dir / "song.mp3"
            fake_mp3.write_bytes(b"ID3not-a-real-mp3")
            out_wav = out_dir / "restored.wav"

            restored = worker._restore_wav_bandwidth(fake_mp3, out_wav)
            self.assertTrue(restored.resolve().is_relative_to(out_dir.resolve()))
            self.assertTrue(restored.is_file())

            class NoOpSep:
                def __init__(self, **_kw: object) -> None:
                    pass

                def load_model(self, model_filename: str = "") -> None:
                    _ = model_filename

                def separate(self, _inp: str) -> list[str]:
                    return []

            fake_sep_mod = types.ModuleType("audio_separator.separator")
            fake_sep_mod.Separator = NoOpSep  # type: ignore[attr-defined]
            fake_pkg = types.ModuleType("audio_separator")
            warnings: list[str] = []
            with mock.patch.dict(
                sys.modules,
                {
                    "audio_separator": fake_pkg,
                    "audio_separator.separator": fake_sep_mod,
                },
            ):
                rescued, _stages = worker._rescue_via_roformer_chain(
                    fake_mp3, out_dir, False, False, warnings
                )
            self.assertTrue(rescued.resolve().is_relative_to(out_dir.resolve()))
            self.assertTrue(rescued.is_file())

    def test_rescue_via_resemble_selects_cuda_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "in.wav"
            out = root / "out.wav"
            NewTasksTests._write_click_wav(inp, secs=0.2)
            seen_devices: list[str] = []

            class FakeTensor:
                def mean(self, dim: int = 0) -> FakeTensor:
                    _ = dim
                    return self

                def unsqueeze(self, dim: int = 0) -> FakeTensor:
                    _ = dim
                    return self

            fake_inf = types.ModuleType("resemble_enhance.enhancer.inference")

            def _denoise(mono: object, sr: int, device: str) -> tuple[object, int]:
                seen_devices.append(device)
                return mono, sr

            def _enhance(denoised: object, sr: int, device: str) -> tuple[object, int]:
                seen_devices.append(device)
                return denoised, sr

            fake_inf.denoise = _denoise  # type: ignore[attr-defined]
            fake_inf.enhance = _enhance  # type: ignore[attr-defined]
            fake_resemble = types.ModuleType("resemble_enhance")
            fake_resemble_enh = types.ModuleType("resemble_enhance.enhancer")
            fake_ta = types.ModuleType("torchaudio")
            fake_ta.load = lambda _p: (FakeTensor(), 48_000)  # type: ignore[attr-defined]
            fake_ta.save = lambda p, _w, _sr: Path(p).write_bytes(b"RIFF")  # type: ignore[attr-defined]

            for cuda_ok, expected_dev in ((True, "cuda"), (False, "cpu")):
                seen_devices.clear()
                fake_torch = types.ModuleType("torch")
                fake_torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
                    is_available=lambda ok=cuda_ok: ok
                )
                with mock.patch.dict(
                    sys.modules,
                    {
                        "resemble_enhance": fake_resemble,
                        "resemble_enhance.enhancer": fake_resemble_enh,
                        "resemble_enhance.enhancer.inference": fake_inf,
                        "torchaudio": fake_ta,
                        "torch": fake_torch,
                    },
                ):
                    worker._rescue_via_resemble(inp, out)
                self.assertEqual(seen_devices, [expected_dev, expected_dev])


if __name__ == "__main__":
    main()
