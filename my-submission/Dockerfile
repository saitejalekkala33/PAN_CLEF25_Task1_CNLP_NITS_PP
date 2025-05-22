FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY predict.py .
COPY ALBERT_HardMoE_Task1.pth .
ENTRYPOINT ["python3", "predict.py", "--model-path", "/app/ALBERT_HardMoE_Task1.pth", "--input-file", "$inputDataset/dataset.jsonl", "--output-dir", "$outputDir"]
