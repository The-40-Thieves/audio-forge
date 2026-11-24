"""Unit tests for the GPU worker's stem picking and timeouts.

No GPU, models or network: ``audio_separator`` and ``gradio_client`` are replaced with fakes.
Run with ``python -m unittest discover -s python-worker``.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
import types
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from pathlib import Path
from unittest import TestCase, main, mock

import forge_gpu_worker as worker

KIM = "model_mel_band_roformer_kim_vocals"
BS_1297 = "model_bs_roformer_ep_317_sdr_12.9755"
DEVERB = "deverb_bs_roformer_8_384dim_10depth"


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

    def test_pick_stem_rejects_path_traversal(self) -> None:
        out = Path("out")
        with self.assertRaises(RuntimeError):
            worker.pick_stem([f"../escape_(Vocals)_{KIM}.wav"], worker.VOCAL_TAGS, out)


class RunSeparateTests(TestCase):
    def run_separate(self, input_name: str, dereverb: bool) -> dict[str, str | None]:
        with patched_temp_audio(separator_modules(FakeSeparator), input_name) as (
            out,
            inp,
        ):
            result = worker.run_separate(inp, out, dereverb=dereverb)
            return {k: (Path(v).name if v else None) for k, v in result.items()}

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
                self.assertIsNotNone(val, f"missing 6-stem slot: {key}")
                assert val is not None
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

    def test_scnet_four_stem_falls_back_to_htdemucs_and_spectral_restore_runs(
        self,
    ) -> None:
        import array
        import wave

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
                if model_filename in {
                    worker.SCNET_FOUR_STEM_MODEL,
                    worker.APOLLO_RESTORE_MODEL,
                }:
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
            self.assertIn(worker.SCNET_FOUR_STEM_MODEL, loaded_models)
            self.assertIn(worker.FOUR_STEM_MODEL, loaded_models)
            self.assertIn(worker.APOLLO_RESTORE_MODEL, loaded_models)
            vocals_path = res["vocals"]
            self.assertIsNotNone(vocals_path)
            assert vocals_path is not None
            self.assertTrue(
                Path(vocals_path).name.endswith("_restored.wav"), vocals_path
            )
            drums_path = res["drums"]
            self.assertIsNotNone(drums_path)
            assert drums_path is not None
            self.assertTrue(Path(drums_path).is_file())

    def test_full_hierarchical_23_slot_cascade_populates_all_sub_stems(self) -> None:
        model_outputs: dict[str, tuple[str, ...]] = {
            BS_1297: ("Instrumental", "Vocals"),
            "BS-Roformer-SW": ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"),
            DEVERB: ("Reverb", "No Reverb"),
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956": (
                "Vocals",
                "Instrumental",
            ),
            "mel_band_roformer_male_female_aufr33": ("Male", "Female"),
            "MDX23C-DrumSep-aufr33-jarredou": (
                "Kick",
                "Snare",
                "Toms",
                "HH",
                "Cymbals",
            ),
            "mel_band_roformer_guitar_elec_acoust": (
                "Electric Guitar",
                "Acoustic Guitar",
            ),
            "mel_band_roformer_strings": ("Strings", "Other"),
            "mel_band_roformer_brass": ("Brass", "Other"),
            "mel_band_roformer_woodwinds": ("Woodwinds", "Other"),
            "mel_band_roformer_synth": ("Synth", "Other"),
            "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144": ("Crowd", "Other"),
        }

        class FullCascadeSeparator(FakeSeparator):
            def separate(self, audio_file: str) -> list[str]:
                return self.write_stems(audio_file, model_outputs[self.model])

        with patched_temp_audio(
            separator_modules(FullCascadeSeparator), "symphony.wav"
        ) as (out, inp):
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
                assert val is not None
                self.assertTrue(Path(val).is_file())

    def test_ultra_ensemble_blends_bs_and_mel_roformer_wavs(self) -> None:
        import array
        import wave

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
            assert vocal_str is not None
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


if __name__ == "__main__":
    main()
