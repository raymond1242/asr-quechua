#!/usr/bin/env python3
"""
Evalúa CER y WER por modelo sobre todas las particiones en ``partitioned_audios/``.

Cada subcarpeta es un audio; dentro, pares ``{id}.wav`` + ``{id}.txt`` (referencia).
Para cada modelo en ``models.MODELS`` se transcribe cada .wav y se compara con el .txt.

Al finalizar cada modelo (si está activado) escribe SRT por sesión en
``transcription_srt/<modelo_sanitizado>/``, un cue por partición con tiempos según
la duración de cada ``.wav`` en orden.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from metrics import compute_cer, compute_wer, compute_cer_batch, compute_wer_batch
from models import MODELS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARTITIONED_DIR = SCRIPT_DIR / "partitioned_audios"
DEFAULT_SRT_OUTPUT_ROOT = SCRIPT_DIR / "transcription_srt"
# Duración mínima por cue (s) para evitar 00:00:00,000 --> 00:00:00,000
_MIN_SRT_CUE_SECONDS = 0.04

logger = logging.getLogger(__name__)


def _segment_sort_key(wav: Path) -> tuple:
    stem = wav.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def _safe_model_dirname(model_card: str) -> str:
    """Nombre de carpeta seguro para rutas (evita / y : de model cards)."""
    return re.sub(r'[<>:"/\\|?*]+', "_", model_card).strip("._") or "model"


def _wav_duration_seconds(wav: Path) -> float:
    try:
        return float(sf.info(str(wav)).duration)
    except Exception as e:
        logger.warning("No se pudo leer duración de %s (%s); se usa %.3f s", wav, e, _MIN_SRT_CUE_SECONDS)
        return _MIN_SRT_CUE_SECONDS


def format_srt_timestamp(total_seconds: float) -> str:
    """Convierte segundos a ``HH:MM:SS,mmm`` (formato SRT)."""
    if total_seconds < 0:
        total_seconds = 0.0
    total_ms = int(round(total_seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _hypothesis_to_srt_text(hyp: str) -> str:
    """Una línea lógica por cue; saltos de línea en la hipótesis se aplastan."""
    t = (hyp or "").strip()
    if not t:
        return "(vacío)"
    return " ".join(t.splitlines())


def write_hypothesis_srts_for_model(
    model_card: str,
    session_wav_hyp: dict[str, list[tuple[Path, str]]],
    output_root: Path,
) -> list[Path]:
    """
    Escribe un ``.srt`` por sesión (mismo nombre de carpeta que en ``partitioned_audios/``),
    con un subtítulo por partición: tiempos según duración real de cada ``.wav`` en secuencia.
    """
    model_dir = output_root / _safe_model_dirname(model_card)
    model_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for session in sorted(session_wav_hyp.keys(), key=str.lower):
        items = sorted(session_wav_hyp[session], key=lambda pair: _segment_sort_key(pair[0]))
        blocks: list[str] = []
        offset = 0.0
        for cue_idx, (wav, hyp) in enumerate(items, start=1):
            dur = _wav_duration_seconds(wav)
            if dur < _MIN_SRT_CUE_SECONDS:
                dur = _MIN_SRT_CUE_SECONDS
            start = offset
            end = offset + dur
            offset = end
            text = _hypothesis_to_srt_text(hyp)
            blocks.append(
                f"{cue_idx}\n"
                f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
                f"{text}\n"
            )
        if not blocks:
            continue
        out_path = model_dir / f"{session}.srt"
        out_path.write_text("\n".join(blocks), encoding="utf-8")
        written.append(out_path)

    return written


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
    srt_output_dir: str | Path | None = None,
    write_transcription_srt: bool = True,
) -> list[ModelEvalSummary]:
    """
    Transcribe cada partición con cada modelo y calcula CER/WER frente al .txt.

    Args:
        partitioned_audios_dir: Raíz con una carpeta por audio (default: partitioned_audios).
        models: Lista de model cards; por defecto ``models.MODELS``.
        lang: Código de idioma para el ASR (p. ej. que_Latn).
        srt_output_dir: Carpeta raíz donde escribir ``transcription_srt/<modelo>/*.srt``
            (por defecto ``transcription_srt`` junto al script). Ignorado si
            ``write_transcription_srt`` es False.
        write_transcription_srt: Si True, al terminar cada modelo genera SRT por sesión
            con la hipótesis y tiempos alineados a la duración de cada ``.wav``.

    Returns:
        Lista de resúmenes por modelo.
    """
    root = Path(partitioned_audios_dir or DEFAULT_PARTITIONED_DIR).resolve()
    srt_root = (
        Path(srt_output_dir).resolve()
        if srt_output_dir is not None
        else DEFAULT_SRT_OUTPUT_ROOT.resolve()
    )
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
        session_wav_hyp: dict[str, list[tuple[Path, str]]] = defaultdict(list)

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
            session_wav_hyp[session].append((wav, hypothesis))

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

        if write_transcription_srt and session_wav_hyp:
            paths = write_hypothesis_srts_for_model(
                model_card, session_wav_hyp, srt_root
            )
            logger.info(
                "  SRT de transcripción: %d archivo(s) en %s",
                len(paths),
                srt_root / _safe_model_dirname(model_card),
            )
        elif write_transcription_srt and not session_wav_hyp:
            logger.info("  SRT: sin segmentos transcritos para escribir.")

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
