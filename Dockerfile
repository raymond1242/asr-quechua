FROM python:3.11

RUN apt-get update && apt-get install -y libsndfile1

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
