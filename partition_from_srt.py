#!/usr/bin/env python3
"""
Particiona audios según todos los .srt de una carpeta de entrada.

Salida fija: partitioned_audios/<nombre_del_srt_sin_ext>/
  por cada partición: {id}.wav y {id}.txt

El WAV de cada SRT se busca en wav_audios/<mismo_stem_que_el_srt>.wav
"""

# from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_DIR = SCRIPT_DIR / "wav_audios"
# Salida siempre esta carpeta
PARTITIONED_AUDIOS_DIR = SCRIPT_DIR / "partitioned_audios"

TIME_LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
    re.MULTILINE,
)


def srt_timestamp_to_seconds(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    h, m, rest = ts.split(":")
    sec = float(rest)
    return int(h) * 3600 + int(m) * 60 + sec


def parse_srt_partitions(srt_path: Path) -> list[dict]:
    text = srt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    raw_blocks = re.split(r"\n\s*\n", text)
    partitions: list[dict] = []

    for block in raw_blocks:
        lines = [ln.rstrip() for ln in block.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        if len(lines) < 3:
            continue

        partition_id = lines[0].strip()
        time_line = lines[1].strip()
        m = TIME_LINE_RE.match(time_line)
        if not m:
            continue

        partitions.append(
            {
                "id": partition_id,
                "start_sec": srt_timestamp_to_seconds(m.group(1)),
                "end_sec": srt_timestamp_to_seconds(m.group(2)),
                "text": "\n".join(lines[2:]).strip(),
            }
        )

    return partitions


def extract_segment_to_wav(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    out_wav: Path,
) -> None:
    data, samplerate = sf.read(str(audio_path))
    n_frames = data.shape[0] if data.ndim > 1 else len(data)
    duration_sec = n_frames / samplerate

    start = max(0.0, float(start_sec))
    end = min(float(end_sec), duration_sec)
    if start >= end:
        raise ValueError(
            f"Intervalo vacío o fuera de duración ({duration_sec:.3f}s): "
            f"start={start}, end={end}"
        )

    i0 = int(start * samplerate)
    i1 = int(end * samplerate)
    segment = data[i0:i1]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), segment, samplerate)


def partition_audio_from_srt(
    srt_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    audio_dir: str | Path | None = None,
) -> tuple[Path, int]:
    """
    Un solo .srt → particiones en PARTITIONED_AUDIOS_DIR/<stem_srt>/.
    """
    srt_path = Path(srt_path).resolve()
    if not srt_path.is_file():
        raise FileNotFoundError(f"No existe el SRT: {srt_path}")

    stem = srt_path.stem
    if audio_path is not None:
        wav_path = Path(audio_path).resolve()
    else:
        base = Path(audio_dir) if audio_dir is not None else DEFAULT_AUDIO_DIR
        wav_path = (base / f"{stem}.wav").resolve()

    if not wav_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el audio: {wav_path} (esperado mismo nombre que el .srt)"
        )

    session_dir = PARTITIONED_AUDIOS_DIR / stem
    session_dir.mkdir(parents=True, exist_ok=True)

    partitions = parse_srt_partitions(srt_path)
    if not partitions:
        raise ValueError(f"No se pudo parsear ninguna partición en: {srt_path}")

    for p in partitions:
        safe_id = re.sub(r"[^\w\-]+", "_", str(p["id"])).strip("_") or "segment"
        extract_segment_to_wav(
            wav_path, p["start_sec"], p["end_sec"], session_dir / f"{safe_id}.wav"
        )
        (session_dir / f"{safe_id}.txt").write_text(p["text"], encoding="utf-8")

    return session_dir, len(partitions)


@dataclass
class PartitionFolderResult:
    """Resultado de procesar todos los .srt de una carpeta."""

    srt_input_folder: Path
    ok: list[tuple[Path, Path, int]] = field(default_factory=list)
    """(ruta .srt, carpeta partitioned_audios/..., num_particiones)."""
    failed: list[tuple[Path, str]] = field(default_factory=list)
    """(.srt, mensaje de error)."""


def partition_from_srt_folder(
    srt_input_folder: str | Path,
    *,
    audio_dir: str | Path | None = None,
) -> PartitionFolderResult:
    """
    Procesa **todos** los archivos `.srt` en `srt_input_folder`.

    Para cada SRT se crea una carpeta en `partitioned_audios/` con el mismo nombre
    base que el SRT (sin `.srt`), con los `.wav` y `.txt` por partición.

    Args:
        srt_input_folder: Carpeta que contiene los .srt (p. ej. `transcription`).
        audio_dir: Carpeta de los .wav (por defecto `wav_audios` junto al script).

    Returns:
        PartitionFolderResult con listas de éxitos y fallos (un .srt sin WAV no
        detiene el resto).
    """
    folder = Path(srt_input_folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {folder}")

    result = PartitionFolderResult(srt_input_folder=folder)
    srts = sorted(folder.glob("*.srt"), key=lambda p: p.name.lower())

    for srt in srts:
        try:
            out_dir, n = partition_audio_from_srt(srt, audio_dir=audio_dir)
            result.ok.append((srt, out_dir, n))
        except (FileNotFoundError, ValueError, OSError) as e:
            result.failed.append((srt, str(e)))

    return result


if __name__ == "__main__":
    # Ejemplo: carpeta de transcripciones por defecto del proyecto
    r = partition_from_srt_folder(SCRIPT_DIR / "audios_srt")
    for srt, out, n in r.ok:
        print(f"OK  {srt.name} → {out} ({n} particiones)")
    for srt, err in r.failed:
        print(f"ERR {srt.name}: {err}")
