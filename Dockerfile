FROM python:3.11
RUN apt-get update && apt-get install -y libsndfile1
RUN pip install omnilingual-asr jiwer
WORKDIR /workspace
