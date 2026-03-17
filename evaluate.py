"""
Evalúa CER sobre todos los audios listados en un CSV de referencias.

Uso:
  python evaluate.py
  python evaluate.py --audios-dir audios --references data/references.csv

El CSV debe tener columnas: filename, reference
  - filename: nombre del .wav (ej. 10006.wav)
  - reference: transcripción correcta (ground truth) para ese audio
"""

import argparse
import csv
from pathlib import Path

from metrics import compute_cer, compute_cer_batch
from transcribe import transcribe


def load_references(csv_path: Path) -> list[tuple[str, str]]:
    """Carga pares (filename, reference) desde un CSV con columnas filename, reference."""
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "filename" in reader.fieldnames and "reference" in reader.fieldnames:
            for row in reader:
                fn = (row.get("filename") or "").strip()
                ref = (row.get("reference") or "").strip()
                if fn and not fn.startswith("#"):
                    pairs.append((fn, ref))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe todos los audios del CSV y calcula CER por archivo y promedio"
    )
    parser.add_argument(
        "--audios-dir",
        type=Path,
        default=Path("audios"),
        help="Carpeta donde están los archivos .wav (default: audios)",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/references.csv"),
        help="Ruta al CSV con columnas filename, reference (default: data/references.csv)",
    )
    parser.add_argument("--lang", default="que_Latn", help="Código de idioma (default: que_Latn)")
    parser.add_argument(
        "--model",
        default="omniASR_CTC_1B_v2",
        help="Model card para inferencia (default: omniASR_CTC_1B_v2)",
    )
    args = parser.parse_args()

    if not args.references.exists():
        raise SystemExit(f"No se encontró el archivo de referencias: {args.references}")

    pairs = load_references(args.references)
    if not pairs:
        raise SystemExit(
            f"No hay filas válidas en {args.references}. "
            "Asegúrate de tener columnas 'filename' y 'reference' y filas sin '#' al inicio."
        )

    audios_dir = args.audios_dir.resolve()
    if not audios_dir.is_dir():
        raise SystemExit(f"No existe la carpeta de audios: {audios_dir}")

    references = []
    hypotheses = []
    results = []  # (filename, reference, hypothesis, cer)

    for i, (filename, reference) in enumerate(pairs, 1):
        audio_path = audios_dir / filename
        if not audio_path.exists():
            print(f"[{i}/{len(pairs)}] OMITIDO (no existe): {filename}")
            continue

        print(f"[{i}/{len(pairs)}] Transcribiendo: {filename} ...")
        try:
            hypothesis = transcribe(str(audio_path), lang=args.lang, model_card=args.model)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        cer = compute_cer(reference=reference, hypothesis=hypothesis)
        references.append(reference)
        hypotheses.append(hypothesis)
        results.append((filename, reference, hypothesis, cer))
        print(f"  reference:  {reference}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  CER:        {cer:.4f}")
        print()

    if not results:
        raise SystemExit("No se procesó ningún audio. Revisa rutas y que los archivos existan.")

    # CER promedio (mismo que jiwer sobre todas las parejas)
    cer_avg = compute_cer_batch(references=references, hypotheses=hypotheses)
    print()
    print(f"Archivos evaluados: {len(results)}")
    print(f"CER promedio:      {cer_avg:.4f}")


if __name__ == "__main__":
    main()
