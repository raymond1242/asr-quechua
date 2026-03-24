FROM python:3.11

RUN apt-get update && apt-get install -y libsndfile1

# Sin GPU: no usar `pip install -r requirements.txt` primero — PyPI trae fairseq2 (variante CUDA)
# y torch con CUDA; además `pip install fairseq2 --extra-index-url meta` suele seguir eligiendo PyPI.
# Orden: torch 2.8.0 CPU → fairseq2 con índice *principal* Meta (variante cpu) → deps → omnilingual sin deps.
ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG FAIRSEQ2_CPU_INDEX=https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 \
      --index-url ${PYTORCH_CPU_INDEX} \
 && pip install --no-cache-dir "fairseq2[arrow]>=0.5.2,<=0.6.0" \
      --index-url ${FAIRSEQ2_CPU_INDEX} \
      --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir \
      "pyarrow>=20.0.0" numba pandas numpy kenlm "polars>=1.29.0" \
 && pip install --no-cache-dir omnilingual-asr==0.2.0 --no-deps \
 && pip install --no-cache-dir jiwer soundfile \
 && python -c "import torch; assert str(torch.__version__).startswith('2.8'), torch.__version__" \
 && python -c "import fairseq2n, fairseq2"

WORKDIR /workspace
