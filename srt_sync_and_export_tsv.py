#!/usr/bin/env python3
"""
1) Copia las marcas de tiempo desde audios_srt hacia los .srt en transcription_srt
   (mismo nombre de archivo), conservando el texto de la transcripción.
2) Exporta cada .srt bajo transcription_srt a un .tsv en transcription_tsv/
   (misma jerarquía de subcarpetas; columnas: start, end en segundos, text).

Ejecución: python srt_sync_and_export_tsv.py (sin argumentos; rutas en variables globales).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


# --- Configuración (editar aquí) ---
BASE_DIR = Path(__file__).resolve().parent
AUDIOS_SRT_DIR = BASE_DIR / "audios_srt"
TRANSCRIPTION_SRT_DIR = BASE_DIR / "transcription_srt"
# Carpeta donde se escriben todos los .tsv (se replica la ruta relativa bajo transcription_srt).
TSV_OUTPUT_ROOT: Path | None = BASE_DIR / "transcription_tsv"
# Para volcar los .tsv junto a cada .srt otra vez, usar: TSV_OUTPUT_ROOT = None


def parse_srt(content: str) -> list[dict]:
    """Parsea un SRT en lista de segmentos: index (str), time_line, text."""
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n", content)
    segments: list[dict] = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        if "-->" not in time_line:
            continue
        text = "\n".join(lines[2:]) if len(lines) > 2 else ""
        segments.append({"index": idx_line, "time_line": time_line, "text": text})
    return segments


def strip_trailing_empty_text_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """
    Quita solo del final los segmentos cuyo texto está vacío o es solo espacios.
    En audios_srt suele haber un último bloque con tiempos pero sin línea de texto.
    """
    segs = list(segments)
    dropped = 0
    while segs and not segs[-1]["text"].strip():
        segs.pop()
        dropped += 1
    return segs, dropped


def format_srt(segments: list[dict]) -> str:
    parts: list[str] = []
    for seg in segments:
        parts.append(f"{seg['index']}\n{seg['time_line']}\n{seg['text']}")
    return "\n\n".join(parts) + ("\n" if parts else "")


def merge_reference_times(
    ref_segments: list[dict],
    trans_segments: list[dict],
    *,
    file_label: str = "",
) -> list[dict]:
    """
    Usa time_line de referencia por índice; conserva text de transcripción.
    Los segmentos finales de la referencia sin texto se ignoran (no cuentan como error).
    """
    ref_raw = len(ref_segments)
    ref_used, dropped_empty_tail = strip_trailing_empty_text_segments(ref_segments)
    n_ref = len(ref_used)
    n_trans = len(trans_segments)
    merged: list[dict] = []
    for i in range(n_trans):
        text = trans_segments[i]["text"]
        idx = str(i + 1)
        if i < n_ref:
            time_line = ref_used[i]["time_line"]
        else:
            time_line = trans_segments[i]["time_line"]
        merged.append({"index": idx, "time_line": time_line, "text": text})

    prefix = f"{file_label}: " if file_label else ""

    # Bloque final vacío en audios_srt: no es error; no avisar si ya alinea con la transcripción.
    if dropped_empty_tail > 0 and n_ref != n_trans:
        print(
            f"  Contexto ({prefix}en referencia hay {ref_raw} bloques; "
            f"se ignoraron {dropped_empty_tail} segmento(s) final(es) sin texto → "
            f"quedan {n_ref} intervalos útiles; transcripción tiene {n_trans}).",
            file=sys.stderr,
        )

    if n_ref != n_trans:
        if n_ref < n_trans:
            print(
                f"  Aviso ({prefix}desajuste real): hay más segmentos en la transcripción "
                f"que intervalos útiles en la referencia ({n_ref} vs {n_trans}). "
                f"Los últimos {n_trans - n_ref} segmento(s) mantienen sus tiempos del ASR.",
                file=sys.stderr,
            )
        else:
            print(
                f"  Aviso ({prefix}desajuste real): la referencia tiene más segmentos con texto "
                f"que la transcripción ({n_ref} vs {n_trans}). "
                f"Sobran {n_ref - n_trans} intervalo(s) en audios_srt sin pareja.",
                file=sys.stderr,
            )

    return merged


def srt_timestamp_to_seconds(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    h_str, m_str, s_str = ts.split(":")
    return int(h_str) * 3600 + int(m_str) * 60 + float(s_str)


def parse_time_line(line: str) -> tuple[float, float]:
    left, _, right = line.partition("-->")
    return srt_timestamp_to_seconds(left), srt_timestamp_to_seconds(right.strip())


def srt_to_tsv_rows(segments: list[dict]) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for seg in segments:
        start, end = parse_time_line(seg["time_line"])
        text_one_line = " ".join(seg["text"].split())
        rows.append((start, end, text_one_line))
    return rows


def write_tsv(path: Path, rows: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["start", "end", "text"])
        for start, end, text in rows:
            w.writerow([f"{start:.3f}", f"{end:.3f}", text])


def sync_times_from_reference() -> None:
    if not AUDIOS_SRT_DIR.is_dir():
        print(f"No existe la carpeta de referencia: {AUDIOS_SRT_DIR}", file=sys.stderr)
        return
    if not TRANSCRIPTION_SRT_DIR.is_dir():
        print(f"No existe: {TRANSCRIPTION_SRT_DIR}", file=sys.stderr)
        return

    for srt_path in sorted(TRANSCRIPTION_SRT_DIR.rglob("*.srt")):
        ref_path = AUDIOS_SRT_DIR / srt_path.name
        if not ref_path.is_file():
            print(f"Sin referencia, se omite sync: {srt_path.name}", file=sys.stderr)
            continue
        ref_text = ref_path.read_text(encoding="utf-8")
        trans_text = srt_path.read_text(encoding="utf-8")
        ref_segs = parse_srt(ref_text)
        trans_segs = parse_srt(trans_text)
        if not trans_segs:
            print(f"SRT vacío o inválido: {srt_path}", file=sys.stderr)
            continue
        rel = srt_path.relative_to(BASE_DIR)
        print(f"Sincronizando tiempos: {rel}")
        merged = merge_reference_times(ref_segs, trans_segs, file_label=str(rel))
        srt_path.write_text(format_srt(merged), encoding="utf-8")


def export_all_transcription_srt_to_tsv() -> None:
    if not TRANSCRIPTION_SRT_DIR.is_dir():
        return
    for srt_path in sorted(TRANSCRIPTION_SRT_DIR.rglob("*.srt")):
        content = srt_path.read_text(encoding="utf-8")
        segs = parse_srt(content)
        if not segs:
            print(f"Omitido TSV (sin segmentos): {srt_path}", file=sys.stderr)
            continue
        if TSV_OUTPUT_ROOT is None:
            tsv_path = srt_path.with_suffix(".tsv")
        else:
            rel = srt_path.relative_to(TRANSCRIPTION_SRT_DIR)
            tsv_path = TSV_OUTPUT_ROOT / rel.with_suffix(".tsv")
        rows = srt_to_tsv_rows(segs)
        write_tsv(tsv_path, rows)
        print(f"TSV: {tsv_path.relative_to(BASE_DIR)}")


def main() -> None:
    print("Paso 1: copiar tiempos desde audios_srt → transcription_srt", flush=True)
    sync_times_from_reference()
    print("Paso 2: exportar .tsv desde transcription_srt", flush=True)
    export_all_transcription_srt_to_tsv()
    print("Listo.")


if __name__ == "__main__":
    main()
