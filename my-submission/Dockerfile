FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
COPY predict.py .
COPY ALBERT_HardMoE_Task1.pth .
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python3", "predict.py", "--model-path", "/app/ALBERT_HardMoE_Task1.pth", "--input-file", "$inputDataset/dataset.jsonl", "--output-dir", "$outputDir"]
