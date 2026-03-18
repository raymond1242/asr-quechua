"""Extrae un intervalo de tiempo de un archivo de audio y lo guarda en OUTPUT_FOLDER."""

from pathlib import Path

import soundfile as sf

# Carpeta global donde se guardan los recortes (puedes cambiarla aquí)
OUTPUT_FOLDER = Path(__file__).resolve().parent / "output_segments"


def extract_audio_interval(
    audio_path: str | Path,
    start_sec: float,
    end_sec: float,
    output_filename: str | None = None,
) -> Path:
    """Crea un nuevo audio con el tramo [start_sec, end_sec) y lo guarda en OUTPUT_FOLDER.

    Args:
        audio_path: Ruta al archivo de audio (p. ej. .wav, .flac).
        start_sec: Inicio del intervalo en segundos (>= 0).
        end_sec: Fin del intervalo en segundos (debe ser > start_sec).
        output_filename: Nombre del archivo de salida. Si es None, se genera a partir
            del nombre original y los tiempos.

    Returns:
        Path al archivo guardado.

    Raises:
        FileNotFoundError: Si no existe el audio de entrada.
        ValueError: Si start_sec >= end_sec o intervalo inválido.
    """
    if start_sec >= end_sec:
        raise ValueError(f"start_sec ({start_sec}) debe ser menor que end_sec ({end_sec})")

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el audio: {audio_path}")

    data, samplerate = sf.read(str(path))
    n_frames = data.shape[0] if data.ndim > 1 else len(data)
    duration_sec = n_frames / samplerate

    start = max(0.0, float(start_sec))
    end = min(float(end_sec), duration_sec)
    if start >= end:
        raise ValueError(
            f"Tras ajustar a la duración del archivo ({duration_sec:.3f}s), "
            f"el intervalo queda vacío: start={start}, end={end}"
        )

    i0 = int(start * samplerate)
    i1 = int(end * samplerate)
    segment = data[i0:i1]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        safe_stem = path.stem.replace(" ", "_")
        output_filename = f"{safe_stem}_{start:.2f}s_{end:.2f}s.wav"

    out_path = OUTPUT_FOLDER / output_filename
    sf.write(str(out_path), segment, samplerate)
    return out_path


if __name__ == "__main__":
    # Ejemplo de uso al ejecutar el módulo (opcional)
    audio_path = "wav_audios/006_recuerdo feliz.wav"
    # verify if the audio file exists
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"No se encontró el audio: {audio_path}")

    out_path = extract_audio_interval(audio_path, 0.0, 2.56)
    print(f"Guardado: {out_path}")
