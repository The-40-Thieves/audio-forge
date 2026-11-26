"""Independent FFmpeg ebur128 reference meter validation for ForgeAudio spatial QC.

Generates bed-only (permuted 5.1), object-only (panning + gain trajectory),
and mixed bed+object ADM BWF fixtures, independently renders the 5.1 speaker
signals in Python, meters them via WSL ``ffmpeg -filter_complex ebur128=peak=true``,
and compares against ``forge spatial-qc --layout 5.1 --json``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

SR = 48_000
SECS = 2.0
FRAMES = int(SR * SECS)
REPO_ROOT = Path(__file__).resolve().parent.parent

SPEAKERS_5_1: list[tuple[str, float]] = [
    ("M+030", 30.0),
    ("M-030", -30.0),
    ("M+000", 0.0),
    ("LFE", 0.0),
    ("M+110", 110.0),
    ("M-110", -110.0),
]
RING_5_1 = [2, 0, 4, 5, 1]


def _win_to_wsl(win_path: Path) -> str:
    resolved = str(win_path.resolve())
    if len(resolved) >= 2 and resolved[1] == ":":
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return resolved.replace("\\", "/")


def _write_24bit_wav(
    path: Path, channels: list[list[float]], extra_chunks: list[tuple[bytes, bytes]]
) -> None:
    n_ch = len(channels)
    n_frames = len(channels[0])
    block_align = n_ch * 3
    pcm = bytearray(n_frames * block_align)
    off = 0
    for i in range(n_frames):
        for ch in range(n_ch):
            v = max(-1.0, min(1.0, channels[ch][i]))
            q = round(v * 8_388_607.0)
            q = max(-8_388_608, min(8_388_607, q))
            pcm[off : off + 3] = q.to_bytes(3, "little", signed=True)
            off += 3

    def pack_chunk(cid: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) % 2 == 1 else b""
        return cid + struct.pack("<I", len(payload)) + payload + pad

    fmt = struct.pack("<HHIIHH", 1, n_ch, SR, SR * block_align, block_align, 24)
    chunks = [pack_chunk(b"fmt ", fmt)]
    for cid, payload in extra_chunks:
        chunks.append(pack_chunk(cid, payload))
    chunks.append(pack_chunk(b"data", bytes(pcm)))
    body = b"".join(chunks)
    riff = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    path.write_bytes(riff)


def _chna_bytes(records: list[tuple[int, str, str, str]]) -> bytes:
    buf = bytearray(struct.pack("<HH", len(records), len(records)))
    for trk, uid, trk_ref, pack_ref in records:
        rec = bytearray(40)
        rec[0:2] = struct.pack("<H", trk)
        rec[2:14] = uid.encode("ascii")[:12].ljust(12, b"\x00")
        rec[14:28] = trk_ref.encode("ascii")[:14].ljust(14, b"\x00")
        rec[28:39] = pack_ref.encode("ascii")[:11].ljust(11, b"\x00")
        buf.extend(rec)
    return bytes(buf)


def _vbap_2d_5_1(az_deg: float) -> list[float]:
    gains = [0.0] * 6
    ux = -math.sin(math.radians(az_deg))
    uy = math.cos(math.radians(az_deg))
    for k in range(len(RING_5_1)):
        ia = RING_5_1[k]
        ib = RING_5_1[(k + 1) % len(RING_5_1)]
        ax = -math.sin(math.radians(SPEAKERS_5_1[ia][1]))
        ay = math.cos(math.radians(SPEAKERS_5_1[ia][1]))
        bx = -math.sin(math.radians(SPEAKERS_5_1[ib][1]))
        by = math.cos(math.radians(SPEAKERS_5_1[ib][1]))
        det = ax * by - ay * bx
        if abs(det) < 1e-9:
            continue
        ga = (ux * by - uy * bx) / det
        gb = (ax * uy - ay * ux) / det
        if ga >= -1e-6 and gb >= -1e-6:
            ga, gb = max(0.0, ga), max(0.0, gb)
            norm = math.hypot(ga, gb)
            gains[ia] = ga / norm
            gains[ib] = gb / norm
            return gains
    return gains


def _sine(freq: float, amp: float) -> list[float]:
    return [amp * math.sin(2.0 * math.pi * freq * i / SR) for i in range(FRAMES)]


def _ffmpeg_ebur128(wav_path: Path) -> tuple[float, float]:
    wsl_bin = shutil.which("wsl")
    if wsl_bin:
        cmd = [
            wsl_bin,
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            _win_to_wsl(wav_path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("Neither wsl nor ffmpeg binary found on PATH")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-i",
            str(wav_path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    stderr = res.stderr
    m_i = re.findall(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", stderr)
    m_p = re.findall(r"\bPeak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", stderr)
    if not m_i or not m_p:
        raise RuntimeError(f"Could not parse ffmpeg ebur128 output:\n{stderr}")
    return float(m_i[-1]), float(m_p[-1])


def _forge_cli_bin() -> Path:
    exe = REPO_ROOT / "target" / "debug" / ("forge.exe" if os.name == "nt" else "forge")
    if not exe.exists():
        cargo_bin = shutil.which("cargo")
        if not cargo_bin:
            raise RuntimeError("cargo not found on PATH")
        subprocess.run(
            [cargo_bin, "build", "-p", "forge-cli", "--locked"],
            cwd=REPO_ROOT,
            check=True,
            timeout=300,
        )
    return exe


def _run_forge_qc(wav_path: Path) -> dict[str, object]:
    exe = _forge_cli_bin()
    res = subprocess.run(
        [str(exe), "spatial-qc", str(wav_path), "--layout", "5.1", "--json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return dict(json.loads(res.stdout))


def _make_fixtures(root: Path) -> list[tuple[str, Path, Path]]:
    half = FRAMES // 2
    s_l = _sine(1000.0, 0.03)
    s_r = _sine(1000.0, 0.03)
    s_c = _sine(1000.0, 0.025)
    s_lfe = _sine(80.0, 0.4)
    s_ls = _sine(1000.0, 0.03)
    s_rs = _sine(1000.0, 0.03)

    # 1. Permuted 5.1 bed-only: [C, LFE, Ls, Rs, L, R]
    bed_adm = root / "bed_permuted.wav"
    bed_ref = root / "bed_ref_5_1.wav"
    bed_chna = _chna_bytes(
        [
            (1, "ATU_00000001", "AT_00010003_01", "AP_00010002"),
            (2, "ATU_00000002", "AT_00010004_01", "AP_00010002"),
            (3, "ATU_00000003", "AT_00010005_01", "AP_00010002"),
            (4, "ATU_00000004", "AT_00010006_01", "AP_00010002"),
            (5, "ATU_00000005", "AT_00010001_01", "AP_00010002"),
            (6, "ATU_00000006", "AT_00010002_01", "AP_00010002"),
        ]
    )
    bed_axml = (
        b"<ebuCoreMain><audioFormatExtended>"
        b'<audioProgramme audioProgrammeID="APR_1"><audioContentIDRef>ACO_1</audioContentIDRef></audioProgramme>'
        b'<audioContent audioContentID="ACO_1"><audioObjectIDRef>AO_1</audioObjectIDRef></audioContent>'
        b'<audioObject audioObjectID="AO_1">'
        b"<audioTrackUIDRef>ATU_00000001</audioTrackUIDRef><audioTrackUIDRef>ATU_00000002</audioTrackUIDRef>"
        b"<audioTrackUIDRef>ATU_00000003</audioTrackUIDRef><audioTrackUIDRef>ATU_00000004</audioTrackUIDRef>"
        b"<audioTrackUIDRef>ATU_00000005</audioTrackUIDRef><audioTrackUIDRef>ATU_00000006</audioTrackUIDRef>"
        b"</audioObject>"
        b'<audioTrackUID UID="ATU_00000001"><audioTrackFormatIDRef>AT_00010003_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000002"><audioTrackFormatIDRef>AT_00010004_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000003"><audioTrackFormatIDRef>AT_00010005_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000004"><audioTrackFormatIDRef>AT_00010006_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000005"><audioTrackFormatIDRef>AT_00010001_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000006"><audioTrackFormatIDRef>AT_00010002_01</audioTrackFormatIDRef></audioTrackUID>'
        b"</audioFormatExtended></ebuCoreMain>"
    )
    _write_24bit_wav(
        bed_adm,
        [s_c, s_lfe, s_ls, s_rs, s_l, s_r],
        [(b"bext", bytes(602)), (b"chna", bed_chna), (b"axml", bed_axml)],
    )
    _write_24bit_wav(bed_ref, [s_l, s_r, s_c, s_lfe, s_ls, s_rs], [])

    # 2. Object-only: center (0 deg, gain 1.0) first 1s -> left-surround (+110 deg, +3 dB) second 1s
    obj_src = _sine(1000.0, 0.04)
    obj_gain_2 = 10.0 ** (3.0 / 20.0)
    g0 = _vbap_2d_5_1(0.0)
    g1 = _vbap_2d_5_1(110.0)
    obj_ref_chs = [[0.0] * FRAMES for _ in range(6)]
    for i in range(FRAMES):
        gains = g0 if i < half else [v * obj_gain_2 for v in g1]
        for ch in range(6):
            obj_ref_chs[ch][i] = obj_src[i] * gains[ch]
    obj_adm = root / "obj_motion.wav"
    obj_ref = root / "obj_ref_5_1.wav"
    obj_chna = _chna_bytes([(1, "ATU_00000010", "AT_00031001_01", "AP_00031001")])
    obj_axml = (
        b"<ebuCoreMain><audioFormatExtended>"
        b'<audioProgramme audioProgrammeID="APR_1"><audioContentIDRef>ACO_1</audioContentIDRef></audioProgramme>'
        b'<audioContent audioContentID="ACO_1"><audioObjectIDRef>AO_1</audioObjectIDRef></audioContent>'
        b'<audioObject audioObjectID="AO_1"><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef><audioTrackUIDRef>ATU_00000010</audioTrackUIDRef></audioObject>'
        b'<audioPackFormat audioPackFormatID="AP_00031001" typeDefinition="Objects"><audioChannelFormatIDRef>AC_00031001</audioChannelFormatIDRef></audioPackFormat>'
        b'<audioChannelFormat audioChannelFormatID="AC_00031001" typeDefinition="Objects">'
        b'<audioBlockFormat rtime="00:00:00.00000" duration="00:00:01.00000"><position coordinate="azimuth">0.0</position><position coordinate="elevation">0.0</position><gain gainUnit="linear">1.0</gain></audioBlockFormat>'
        b'<audioBlockFormat rtime="00:00:01.00000" duration="00:00:01.00000"><position coordinate="azimuth">110.0</position><position coordinate="elevation">0.0</position><gain gainUnit="dB">3.0</gain></audioBlockFormat>'
        b"</audioChannelFormat>"
        b'<audioTrackFormat audioTrackFormatID="AT_00031001_01"><audioStreamFormatIDRef>AS_00031001</audioStreamFormatIDRef></audioTrackFormat>'
        b'<audioStreamFormat audioStreamFormatID="AS_00031001"><audioChannelFormatIDRef>AC_00031001</audioChannelFormatIDRef><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef></audioStreamFormat>'
        b'<audioTrackUID UID="ATU_00000010"><audioTrackFormatIDRef>AT_00031001_01</audioTrackFormatIDRef><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef></audioTrackUID>'
        b"</audioFormatExtended></ebuCoreMain>"
    )
    _write_24bit_wav(
        obj_adm,
        [obj_src],
        [(b"bext", bytes(602)), (b"chna", obj_chna), (b"axml", obj_axml)],
    )
    _write_24bit_wav(obj_ref, obj_ref_chs, [])

    # 3. Mixed 5.1 bed + moving object (channel 7)
    mix_adm = root / "mixed_5_1_plus_obj.wav"
    mix_ref = root / "mixed_ref_5_1.wav"
    mix_ref_chs = [
        [s + o for s, o in zip(bed_c, obj_c)]
        for bed_c, obj_c in zip([s_l, s_r, s_c, s_lfe, s_ls, s_rs], obj_ref_chs)
    ]
    mix_chna = _chna_bytes(
        [
            (1, "ATU_00000001", "AT_00010001_01", "AP_00010002"),
            (2, "ATU_00000002", "AT_00010002_01", "AP_00010002"),
            (3, "ATU_00000003", "AT_00010003_01", "AP_00010002"),
            (4, "ATU_00000004", "AT_00010004_01", "AP_00010002"),
            (5, "ATU_00000005", "AT_00010005_01", "AP_00010002"),
            (6, "ATU_00000006", "AT_00010006_01", "AP_00010002"),
            (7, "ATU_00000010", "AT_00031001_01", "AP_00031001"),
        ]
    )
    mix_axml = (
        b"<ebuCoreMain><audioFormatExtended>"
        b'<audioProgramme audioProgrammeID="APR_1"><audioContentIDRef>ACO_1</audioContentIDRef></audioProgramme>'
        b'<audioContent audioContentID="ACO_1"><audioObjectIDRef>AO_BED</audioObjectIDRef><audioObjectIDRef>AO_OBJ</audioObjectIDRef></audioContent>'
        b'<audioObject audioObjectID="AO_BED">'
        b"<audioTrackUIDRef>ATU_00000001</audioTrackUIDRef><audioTrackUIDRef>ATU_00000002</audioTrackUIDRef>"
        b"<audioTrackUIDRef>ATU_00000003</audioTrackUIDRef><audioTrackUIDRef>ATU_00000004</audioTrackUIDRef>"
        b"<audioTrackUIDRef>ATU_00000005</audioTrackUIDRef><audioTrackUIDRef>ATU_00000006</audioTrackUIDRef>"
        b"</audioObject>"
        b'<audioObject audioObjectID="AO_OBJ"><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef><audioTrackUIDRef>ATU_00000010</audioTrackUIDRef></audioObject>'
        b'<audioPackFormat audioPackFormatID="AP_00031001" typeDefinition="Objects"><audioChannelFormatIDRef>AC_00031001</audioChannelFormatIDRef></audioPackFormat>'
        b'<audioChannelFormat audioChannelFormatID="AC_00031001" typeDefinition="Objects">'
        b'<audioBlockFormat rtime="00:00:00.00000" duration="00:00:01.00000"><position coordinate="azimuth">0.0</position><position coordinate="elevation">0.0</position><gain gainUnit="linear">1.0</gain></audioBlockFormat>'
        b'<audioBlockFormat rtime="00:00:01.00000" duration="00:00:01.00000"><position coordinate="azimuth">110.0</position><position coordinate="elevation">0.0</position><gain gainUnit="dB">3.0</gain></audioBlockFormat>'
        b"</audioChannelFormat>"
        b'<audioTrackUID UID="ATU_00000001"><audioTrackFormatIDRef>AT_00010001_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000002"><audioTrackFormatIDRef>AT_00010002_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000003"><audioTrackFormatIDRef>AT_00010003_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000004"><audioTrackFormatIDRef>AT_00010004_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000005"><audioTrackFormatIDRef>AT_00010005_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000006"><audioTrackFormatIDRef>AT_00010006_01</audioTrackFormatIDRef></audioTrackUID>'
        b'<audioTrackUID UID="ATU_00000010"><audioTrackFormatIDRef>AT_00031001_01</audioTrackFormatIDRef><audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef></audioTrackUID>'
        b"</audioFormatExtended></ebuCoreMain>"
    )
    _write_24bit_wav(
        mix_adm,
        [s_l, s_r, s_c, s_lfe, s_ls, s_rs, obj_src],
        [(b"bext", bytes(602)), (b"chna", mix_chna), (b"axml", mix_axml)],
    )
    _write_24bit_wav(mix_ref, mix_ref_chs, [])
    with wave.open(str(bed_ref), "rb") as w:
        if w.getnchannels() != 6:
            raise RuntimeError(
                f"Expected 6 channels in reference WAV, got {w.getnchannels()}"
            )

    return [
        ("bed_permuted", bed_adm, bed_ref),
        ("object_motion", obj_adm, obj_ref),
        ("mixed_bed_obj", mix_adm, mix_ref),
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = _make_fixtures(Path(tmp))
        for name, adm_wav, ref_wav in fixtures:
            ff_lkfs, ff_tp = _ffmpeg_ebur128(ref_wav)
            rep = _run_forge_qc(adm_wav)
            fg_lkfs = float(rep["integrated_lkfs"])  # type: ignore[arg-type]
            fg_tp = float(rep["true_peak_dbtp"])  # type: ignore[arg-type]
            dl = abs(fg_lkfs - ff_lkfs)
            dtp = abs(fg_tp - ff_tp)
            print(
                f"[{name}] kind={rep['programme_kind']} forge={fg_lkfs:.3f} LKFS / {fg_tp:.3f} dBTP | ffmpeg={ff_lkfs:.3f} LUFS / {ff_tp:.3f} dBTP | delta=({dl:.3f} LU, {dtp:.3f} dB)"
            )
            if dl > 0.15 or dtp > 0.15:
                raise RuntimeError(
                    f"Tolerance exceeded on {name}: dLU={dl:.3f}, dTP={dtp:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
