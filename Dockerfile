FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .