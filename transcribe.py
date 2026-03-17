"""Transcripción de audio con Omnilingual ASR."""

from pathlib import Path

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def transcribe(
    audio_path: str,
    lang: str = "que_Latn",
    model_card: str = "omniASR_CTC_300M",
) -> str:
    """Transcribe un archivo de audio con Omnilingual ASR.

    Args:
        audio_path: Ruta al archivo de audio (.wav, .flac, etc.).
        lang: Código de idioma en formato {lang}_{script} (default: que_Latn).
        model_card: Nombre del modelo para inferencia.

    Returns:
        Texto transcrito.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = ASRInferencePipeline(model_card=model_card)
    transcriptions = pipeline.transcribe(
        [str(path)],
        lang=[lang],
        batch_size=1,
    )
    return transcriptions[0]
