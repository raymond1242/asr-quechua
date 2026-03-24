# Quechua Audio Transcription

Transcribe Quechua audio files to text using [omnilingual-asr](https://pypi.org/project/omnilingual-asr/) running in Docker.

## Prerequisites

- **macOS** (tested on Apple Silicon)
- **Docker Desktop** — [Download here](https://www.docker.com/products/docker-desktop/)
  - Recommended: allocate at least **16 GB of RAM** in Docker Desktop → Settings → Resources

## Project Structure

```
quechua-transcription/
├── audios/                 # Tus archivos .wav para transcribir
│   └── *.wav
├── data/
│   └── references.csv      # Listado: filename, reference (transcripción correcta)
├── Dockerfile
├── main.py                 # Transcribe un audio y opcionalmente calcula CER
├── evaluate.py             # Evalúa CER sobre todos los audios del CSV
├── metrics.py              # CER y otras métricas (jiwer)
└── README.md
```

## Setup

### 1. Build the Docker image

```bash
cd /path/to/quechua-transcription
docker build --platform linux/amd64 -t quechua-asr .
```

> **Note:** The `--platform linux/amd64` flag is required because `fairseq2n` (a dependency) only provides x86_64 Linux wheels. On Apple Silicon Macs, Docker will emulate x86 architecture.

### 2. Create a local cache directory

This prevents re-downloading the model (~1.2 GB) every time you start a new container:

```bash
mkdir -p ~/quechua-model-cache
```

## Usage

### Start the container

```bash
docker run -it --platform linux/amd64 \
  -v "/path/to/quechua-transcription:/workspace" \
  -v ~/quechua-model-cache:/root/.cache \
  quechua-asr bash
```

Replace `/path/to/quechua-transcription` with the full path to your project folder. If the path contains spaces, wrap it in quotes.

### Transcribe a single file

```bash
python main.py audios/10004.wav
```

Output:

```
llaqta ayllukuna sapa kuti rimapakunku ruway ninkumanta
```

### Options

```bash
python main.py <audio_file> [--lang LANG] [--model MODEL] [--reference REF]
```

| Argument       | Default              | Description                              |
|----------------|----------------------|------------------------------------------|
| `audio`        | *(required)*         | Path to the `.wav` audio file            |
| `--lang`       | `que_Latn`           | Language code in `{lang}_{script}` format |
| `--model`      | `omniASR_LLM_1B_v2`  | Model card name for inference            |
| `--reference`  | —                    | Texto de referencia o ruta a `.txt` para calcular CER |

### Evaluación con CER (Character Error Rate)

Si tienes la transcripción de referencia (ground truth), puedes evaluar la salida del modelo con la métrica **CER** usando la opción `--reference`:

```bash
# Referencia como texto directo
python main.py audios/10004.wav --reference "llaqta ayllukuna sapa kuti rimapakunku ruway ninkumanta"

# Referencia desde un archivo de texto
python main.py audios/10004.wav --reference referencias/10004.txt
```

La salida incluirá la transcripción y el CER (0 = perfecto, mayor = más errores):

```
llaqta ayllukuna sapa kuti rimapakunku ruway ninkumanta
CER: 0.0234
```

La métrica CER se calcula con la librería [jiwer](https://github.com/jitsi/jiwer), que usa edit distance a nivel de carácter. También puedes usar el módulo `metrics` en tu código:

```python
from metrics import compute_cer, compute_cer_batch

cer = compute_cer(reference="texto referencia", hypothesis="texto transcrito")
# Varios pares a la vez:
cer_avg = compute_cer_batch(references=["ref1", "ref2"], hypotheses=["hyp1", "hyp2"])
```

### Evaluación por lote (todos los audios)

Si tienes muchos audios y las transcripciones de referencia en un Excel, la forma recomendada es:

1. **Exportar (o crear) un CSV** con dos columnas: `filename` (nombre del .wav) y `reference` (transcripción correcta). Guárdalo como `data/references.csv` (UTF-8).
2. Poner todos los `.wav` en la carpeta `audios/`.
3. Ejecutar el script de evaluación (sin pasar rutas de audio; procesa todos los del CSV):

```bash
python evaluate.py
```

Opciones útiles:

```bash
python evaluate.py --audios-dir audios --references data/references.csv
python evaluate.py --output-csv results/cer_results.csv   # guarda resultados en CSV
```

El script transcribe cada audio, calcula el CER frente a la referencia y al final muestra el **CER promedio** sobre todos los archivos. Así evitas tener que mantener un diccionario en código: la fuente de verdad es el CSV (exportado desde tu Excel).

## Dockerfile

La imagen fija **PyTorch 2.8.0 + torchaudio (CPU)**, instala **fairseq2** usando el índice de Meta `…/pt2.8.0/cpu` como **índice principal** (si solo se usa `--extra-index-url`, pip suele quedarse con el wheel CUDA de PyPI), y luego **omnilingual-asr** con `--no-deps` para que no vuelva a subir PyTorch. El detalle está en el `Dockerfile`.

## Docker Quick Reference

| Action                     | Command                                           |
|----------------------------|---------------------------------------------------|
| Build the image            | `docker build --platform linux/amd64 -t quechua-asr .` |
| Start a container          | `docker run -it --platform linux/amd64 -v ...  quechua-asr bash` |
| Exit the container         | `exit`                                            |
| List stopped containers    | `docker ps -a`                                    |
| Restart a stopped container| `docker start -i <container_id>`                  |

## Known Warnings

You may see the following during transcription:

```
[W NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
```

These are harmless — caused by x86 emulation on Apple Silicon. They do not affect the transcription output.

## Troubleshooting

- **`Killed` during model loading:** Increase Docker memory in Docker Desktop → Settings → Resources (16 GB minimum recommended).
- **Empty `/workspace` in container:** Make sure you're using the full, quoted path in the `-v` flag.
- **`fairseq2n` install errors:** Ensure you're building with `--platform linux/amd64`.
- **`OSError: libcudart.so.*`:** PyTorch/torchaudio con CUDA dentro de un contenedor sin bibliotecas NVIDIA. Reconstruye la imagen con el `Dockerfile` actual (fija PyTorch CPU y fairseq2 variante CPU).
- **`fairseq2 requires a CUDA … build of PyTorch … but the installed version is CPU-only`:** Ocurre si solo se fuerza PyTorch CPU con una versión distinta (p. ej. 2.11) pero `fairseq2n` de PyPI sigue siendo la variante CUDA para otra versión. El `Dockerfile` reinstala **torch 2.8.0 + torchaudio 2.8.0 (CPU)** y **fairseq2** desde `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu`. Tras cambios, haz `docker build --platform linux/amd64 --no-cache -t quechua-asr .`.
- **GPU:** Si tienes NVIDIA y quieres CUDA, usa imagen con runtime CUDA compatible, `docker run --gpus all ...`, e instala PyTorch + variante fairseq2 `cu128`/`cu126` según [fairseq2#variants](https://github.com/facebookresearch/fairseq2#variants) (no uses el truco CPU del Dockerfile).