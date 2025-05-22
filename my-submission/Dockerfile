FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY predict.py .
COPY ALBERT_HardMoE_Task1.pth .
CMD ["sh", "-c", "python3 predict.py --model-path /app/ALBERT_HardMoE_Task1.pth --input-dir $inputDataset --output-dir $outputDir"]
