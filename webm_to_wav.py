"""
Convierte todos los archivos de audio .webm de una carpeta a .wav en otra carpeta.
Usa ffmpeg directamente (compatible con Python 3.13+ donde audioop fue eliminado).
"""

import subprocess
from pathlib import Path

# --- Configuración: carpetas de entrada y salida ---
INPUT_FOLDER = "webm_audios"   # Carpeta con archivos .webm
OUTPUT_FOLDER = "wav_audios"   # Carpeta donde se guardarán los .wav


def convert_webm_to_wav():
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    if not input_path.exists():
        print(f"Error: la carpeta '{INPUT_FOLDER}' no existe.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    webm_files = list(input_path.glob("*.webm"))
    if not webm_files:
        print(f"No se encontraron archivos .webm en '{INPUT_FOLDER}'.")
        return

    for webm_file in webm_files:
        out_file = output_path / (webm_file.stem + ".wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # sobrescribir sin preguntar
                    "-i", str(webm_file),
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",   # 44 kHz
                    "-ac", "1",      # mono
                    str(out_file),
                ],
                check=True,
                capture_output=True,
            )
            print(f"Convertido: {webm_file.name} -> {out_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error al convertir {webm_file.name}: {e.stderr.decode(errors='ignore')}")
        except FileNotFoundError:
            print("Error: ffmpeg no está instalado o no está en el PATH.")
            return


if __name__ == "__main__":
    convert_webm_to_wav()
