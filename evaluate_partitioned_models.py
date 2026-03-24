#!/usr/bin/env python3
"""
Evalúa CER y WER por modelo sobre todas las particiones en ``partitioned_audios/``.

Cada subcarpeta es un audio; dentro, pares ``{id}.wav`` + ``{id}.txt`` (referencia).
Para cada modelo en ``models.MODELS`` se transcribe cada .wav y se compara con el .txt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from metrics import compute_cer, compute_wer, compute_cer_batch, compute_wer_batch
from models import MODELS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARTITIONED_DIR = SCRIPT_DIR / "partitioned_audios"

logger = logging.getLogger(__name__)


def _segment_sort_key(wav: Path) -> tuple:
    stem = wav.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def iter_wav_txt_pairs(partitioned_root: Path) -> list[tuple[str, Path, Path, str]]:
    """
    Recorre subcarpetas de partitioned_audios y devuelve
    (nombre_sesión, wav, txt, referencia).
    """
    jobs: list[tuple[str, Path, Path, str]] = []
    if not partitioned_root.is_dir():
        raise NotADirectoryError(f"No existe: {partitioned_root}")

    for subdir in sorted(partitioned_root.iterdir(), key=lambda p: p.name.lower()):
        if not subdir.is_dir():
            continue
        wavs = sorted(subdir.glob("*.wav"), key=_segment_sort_key)
        for wav in wavs:
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                logger.warning(
                    "Sin referencia .txt para %s (sesión %s) — omitido",
                    wav.name,
                    subdir.name,
                )
                continue
            ref = txt.read_text(encoding="utf-8", errors="replace").strip()
            jobs.append((subdir.name, wav, txt, ref))
    return jobs


@dataclass
class ModelEvalSummary:
    model: str
    n_segments: int = 0
    n_skipped_empty_ref: int = 0
    cer_mean: float = 0.0
    wer_mean: float = 0.0
    cer_micro: float = 0.0
    wer_micro: float = 0.0
    errors: list[str] = field(default_factory=list)


def evaluate_partitioned_models(
    partitioned_audios_dir: str | Path | None = None,
    models: list[str] | None = None,
    *,
    lang: str = "que_Latn",
) -> list[ModelEvalSummary]:
    """
    Transcribe cada partición con cada modelo y calcula CER/WER frente al .txt.

    Args:
        partitioned_audios_dir: Raíz con una carpeta por audio (default: partitioned_audios).
        models: Lista de model cards; por defecto ``models.MODELS``.
        lang: Código de idioma para el ASR (p. ej. que_Latn).

    Returns:
        Lista de resúmenes por modelo.
    """
    root = Path(partitioned_audios_dir or DEFAULT_PARTITIONED_DIR).resolve()
    model_list = list(models) if models is not None else list(MODELS)

    logger.info("Raíz de particiones: %s", root)
    jobs = iter_wav_txt_pairs(root)
    if not jobs:
        logger.error("No hay pares .wav/.txt en %s", root)
        return []

    logger.info(
        "Encontrados %d segmentos en %d sesiones",
        len(jobs),
        len({j[0] for j in jobs}),
    )

    summaries: list[ModelEvalSummary] = []

    for model_card in model_list:
        logger.info("=" * 72)
        logger.info("MODELO: %s", model_card)
        logger.info("=" * 72)

        try:
            logger.info("Cargando pipeline ASR…")
            pipeline = ASRInferencePipeline(model_card=model_card)
            logger.info("Pipeline listo.")
        except Exception as e:
            logger.exception("No se pudo cargar el modelo %s: %s", model_card, e)
            summaries.append(
                ModelEvalSummary(
                    model=model_card,
                    errors=[f"load_failed: {e}"],
                )
            )
            continue

        refs_list: list[str] = []
        hyps_list: list[str] = []
        cers: list[float] = []
        wers: list[float] = []
        skipped = 0
        summary = ModelEvalSummary(model=model_card)

        current_session: str | None = None
        for session, wav, _txt, reference in jobs:
            if session != current_session:
                current_session = session
                logger.info("— Sesión / audio: %s", session)

            seg_id = wav.stem
            if not reference:
                logger.warning(
                    "  [%s] Referencia vacía en %s — omitido",
                    seg_id,
                    wav.name,
                )
                skipped += 1
                continue

            logger.info("  Segmento id=%s | archivo=%s", seg_id, wav.name)
            logger.info("    Transcribiendo…")

            try:
                hypotheses = pipeline.transcribe(
                    [str(wav)],
                    lang=[lang],
                    batch_size=1,
                )
                hypothesis = (hypotheses[0] or "").strip()
            except Exception as e:
                logger.exception("    Error al transcribir %s: %s", wav, e)
                summary.errors.append(f"{wav}: {e}")
                continue

            cer = compute_cer(reference, hypothesis)
            wer = compute_wer(reference, hypothesis)

            logger.info("    REFERENCIA: %s", reference)
            logger.info("    MODELO:     %s", hypothesis or "(vacío)")
            logger.info("    CER: %.6f | WER: %.6f", cer, wer)

            refs_list.append(reference)
            hyps_list.append(hypothesis)
            cers.append(cer)
            wers.append(wer)

        summary.n_segments = len(cers)
        summary.n_skipped_empty_ref = skipped

        if cers:
            summary.cer_mean = sum(cers) / len(cers)
            summary.wer_mean = sum(wers) / len(wers)
            summary.cer_micro = compute_cer_batch(refs_list, hyps_list)
            summary.wer_micro = compute_wer_batch(refs_list, hyps_list)

            logger.info("-" * 72)
            logger.info("RESUMEN MODELO %s", model_card)
            logger.info(
                "  Segmentos evaluados: %d | referencias vacías omitidas: %d",
                summary.n_segments,
                summary.n_skipped_empty_ref,
            )
            logger.info(
                "  CER medio (macro, por segmento): %.6f",
                summary.cer_mean,
            )
            logger.info(
                "  WER medio (macro, por segmento): %.6f",
                summary.wer_mean,
            )
            logger.info(
                "  CER micro (jiwer sobre todos los textos concatenados): %.6f",
                summary.cer_micro,
            )
            logger.info(
                "  WER micro (jiwer sobre todos los textos concatenados): %.6f",
                summary.wer_micro,
            )
        else:
            logger.warning(
                "No se evaluó ningún segmento para %s (errores o sin datos).",
                model_card,
            )

        if summary.errors:
            logger.warning("Errores en segmentos: %d", len(summary.errors))

        summaries.append(summary)

    logger.info("=" * 72)
    logger.info("TABLA FINAL (macro por segmento)")
    logger.info("=" * 72)
    for s in summaries:
        if s.n_segments:
            logger.info(
                "%-28s  CER_macro=%.4f  WER_macro=%.4f  (n=%d)",
                s.model,
                s.cer_mean,
                s.wer_mean,
                s.n_segments,
            )
        else:
            logger.info("%-28s  (sin evaluación)", s.model)

    return summaries


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    evaluate_partitioned_models()
