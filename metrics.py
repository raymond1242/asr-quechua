"""
Métricas para evaluar transcripción ASR (CER, WER, etc.).
Usa la librería jiwer: https://github.com/jitsi/jiwer
"""

import jiwer


def compute_cer(reference: str, hypothesis: str) -> float:
    """Calcula el Character Error Rate (CER) entre referencia e hipótesis.

    CER = (sustituciones + inserciones + eliminaciones) / nº caracteres en referencia.
    Útil para evaluar transcripción automática (ASR), especialmente en idiomas
    como el quechua donde la tasa de error por carácter suele ser más estable
    que el WER.

    Args:
        reference: Texto de referencia (ground truth).
        hypothesis: Texto transcrito por el modelo (hipótesis).

    Returns:
        CER en [0, +inf). 0 = transcripción perfecta.
    """
    return jiwer.cer(reference=reference, hypothesis=hypothesis)


def compute_cer_batch(references: list[str], hypotheses: list[str]) -> float:
    """Calcula el CER promedio sobre múltiples pares referencia/hipótesis.

    Args:
        references: Lista de textos de referencia.
        hypotheses: Lista de transcripciones (misma longitud que references).

    Returns:
        CER promedio.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references y hypotheses deben tener la misma longitud, "
            f"got {len(references)} vs {len(hypotheses)}"
        )
    return jiwer.cer(reference=references, hypothesis=hypotheses)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calcula el Word Error Rate (WER) entre referencia e hipótesis.

    WER = (sustituciones + inserciones + eliminaciones) / nº palabras en referencia.
    Métrica estándar en ASR a nivel de palabra.

    Args:
        reference: Texto de referencia (ground truth).
        hypothesis: Texto transcrito por el modelo (hipótesis).

    Returns:
        WER en [0, +inf). 0 = transcripción perfecta.
    """
    return jiwer.wer(reference=reference, hypothesis=hypothesis)


def compute_wer_batch(references: list[str], hypotheses: list[str]) -> float:
    """Calcula el WER promedio sobre múltiples pares referencia/hipótesis.

    Args:
        references: Lista de textos de referencia.
        hypotheses: Lista de transcripciones (misma longitud que references).

    Returns:
        WER promedio.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references y hypotheses deben tener la misma longitud, "
            f"got {len(references)} vs {len(hypotheses)}"
        )
    return jiwer.wer(reference=references, hypothesis=hypotheses)
