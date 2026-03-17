"""Transcripción de audio con Omnilingual ASR."""

from pathlib import Path

import torch
import torchaudio

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def transcribe(
    audio_path: str,
    lang: str = "que_Latn",
    model_card: str = "omniASR_LLM_1B_v2",
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> str:
    """Transcribe a WAV audio file using Omnilingual ASR.

    Args:
        audio_path: Path to the .wav file.
        lang: Language code in {lang}_{script} format (default: que_Latn for Quechua Latin).
        model_card: Model card name to use for inference.
        start_sec: Optional start time in seconds. If set, only this segment is transcribed.
        end_sec: Optional end time in seconds. Used together with start_sec for [start_sec, end_sec].

    Returns:
        The transcription text.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = ASRInferencePipeline(model_card=model_card)

    # Si se pide un segmento de tiempo, cargar audio, recortar y pasar como dict
    if start_sec is not None or end_sec is not None:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        num_frames = waveform.shape[1]

        start_sample = 0 if start_sec is None else int(start_sec * sample_rate)
        end_sample = num_frames if end_sec is None else int(end_sec * sample_rate)
        start_sample = max(0, min(start_sample, num_frames))
        end_sample = max(start_sample, min(end_sample, num_frames))

        segment = waveform[:, start_sample:end_sample]
        # El pipeline acepta dict con "waveform" (channels, time) y "sample_rate"
        audio_input = [
            {"waveform": segment.numpy(), "sample_rate": sample_rate}
        ]
        transcriptions = pipeline.transcribe(audio_input, lang=[lang], batch_size=1)
    else:
        transcriptions = pipeline.transcribe([str(path)], lang=[lang], batch_size=1)

    return transcriptions[0]
