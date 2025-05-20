
# Llama-Prompt-Guard-2-onnx

This repository provides a pipeline to convert Llama-Prompt-Guard-2 models into ONNX format, 
perform optional optimizations and quantization, and evaluate performance on classification datasets.

## 📦 Features

- ✅ Convert `AutoModelForSequenceClassification` models to ONNX.
- 🔄 Optimize the exported ONNX models.
- 📉 Quantize ONNX models to reduce size and improve inference speed.
- 📊 Evaluate accuracy, precision, recall, F1-score, and AUC-ROC.
- 🧪 Compare performance between full and quantized models.

## 🧠 Built With

- Meta LLaMA – Foundation model powering the classifier 
  - [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
  - [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
- 🤗 Hugging Face Transformers – Model and tokenizer loading
- ONNX – Model export and runtime format
- ONNX Runtime – Efficient inference backend

## 🚀 Usage

### 1. Export a Hugging Face Model

```bash
pip install -r requirements.txt
```

```bash
python convert_to_onnx.py \
  --model-name meta-llama/Llama-Prompt-Guard-2-22M \
  --export-onnx \
  --quantize-onnx \
  --update-config
```

This will:
- Download the model and tokenizer
- Save them to `gravitee-io/Llama-Prompt-Guard-2-22M-onnx`
- Export to ONNX (`model.onnx`)
- Optimize and quantize (`model.optim.onnx` and `model.quant.onnx`)
- Update `config.json` with `id2label` and `label2id`

---

### 2. Evaluate the Model

```bash
python evaluate.py \
  --dataset jackhhao/jailbreak-classification \
  --model-path gravitee-io/Llama-Prompt-Guard-2-22M-onnx \
  --quantized \
  --plot-roc
```

You can toggle `--quantized` to evaluate either the full or quantized model.

## 🧪 Evaluation Results

| Model                      | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Inference Time |
|----------------------------|----------|-----------|--------|----------|---------|----------------|
| Llama-Prompt-Guard-2-22M   | 0.9569   | 0.9879    | 0.9260 | 0.9559   | 0.9259  | 33s            |
| Llama-Prompt-Guard-2-22M-q | 0.9473   | 1.0000    | 0.8956 | 0.9449   | 0.9032  | 29s            |
| Llama-Prompt-Guard-2-86M   | 0.9770   | 0.9980    | 0.9564 | 0.9767   | 0.9523  | 1m29s          |
| Llama-Prompt-Guard-2-86M-q | 0.8937   | 1.0000    | 0.7894 | 0.8823   | 0.7263  | 1m15s          |

## 📥 Dataset

We use [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification)
for the evaluation

## 🤗 Usage

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np

# Load model and tokenizer using optimum
model = ORTModelForSequenceClassification.from_pretrained("gravitee-io/Llama-Prompt-Guard-2-22M-onnx", file_name="model.quant.onnx")
tokenizer = AutoTokenizer.from_pretrained("gravitee-io/Llama-Prompt-Guard-2-22M-onnx")

# Tokenize input
text = "Your comment here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Run inference
outputs = model(**inputs)
logits = outputs.logits

# Optional: convert to probabilities
probs = 1 / (1 + np.exp(-logits))
print(probs)
```

You can find and use the quantized ONNX model from the Hugging:xFace repository: 
- [gravitee-io/Llama-Prompt-Guard-2-22M-onnx](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx)
- [gravitee-io/Llama-Prompt-Guard-2-86M-onnx](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-86M-onnx)
