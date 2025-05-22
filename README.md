# PAN_CLEF25_Task1_CNLP_NITS_PP

This repository contains the code for our submission to **PAN CLEF 2025 Task 1**.

## Usage

To run predictions using a trained model, use the following command:

```bash
python3 predict.py --model-path /path/to/model.pth --input-file /path/to/test.csv --output-dir /path/to/output/dir
```

## Example
If all files (model, input file, and output directory) are in the same directory, the command would look like:
```bash
python3 predict.py --model-path ALBERT_HardMoE_Task1.pth --input-file test.csv --output-dir output
```

## Arguments

--model-path: Path to the .pth model file.

--input-file: Path to the input CSV file containing test data.

--output-dir: Directory where the output predictions will be saved.
