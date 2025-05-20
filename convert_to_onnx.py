import argparse
import json
import os

import onnx
import onnxoptimizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


def save_hf_model(model_name):
    print(f"[*] Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model_path = "gravitee-io/" + model_name.split("/")[1] + '-onnx'
    print(f"[*] Saving model and tokenizer to '{model_path}'...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def export_model_to_onnx(model_dir):
    print("[*] Exporting model to ONNX...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer("Hello, World!", return_tensors="pt")
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        os.path.join(model_dir, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17
    )
    print("[✓] ONNX export complete.")


def optimize_onnx_model(model_dir):
    print("[*] Optimizing ONNX model...")
    model_path = os.path.join(model_dir, "model.onnx")
    optimized_path = os.path.join(model_dir, "model.optim.onnx")

    model = onnx.load(model_path)
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    optimized = onnxoptimizer.optimize(model, passes)
    onnx.save(optimized, optimized_path)
    print("[✓] Optimization complete.")


def quantize_onnx_model(model_dir):
    print("[*] Preparing to quantize ONNX model...")
    optimized_path = os.path.join(model_dir, "model.optim.onnx")
    quantized_path = os.path.join(model_dir, "model.quant.onnx")

    if not os.path.exists(optimized_path):
        print("[!] Optimized model not found. Optimizing now...")
        optimize_onnx_model(model_dir)

    quantize_dynamic(
        optimized_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    print("[✓] Quantization complete.")

def update_config_labels(model_dir):
    print("[*] Updating config.json with id2label and label2id...")

    classes = ["BENIGN", "MALICIOUS"]

    id2label = {str(i): label for i, label in enumerate(classes)}
    label2id = {label: i for i, label in enumerate(classes)}

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["id2label"] = id2label
    config["label2id"] = label2id

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("[✓] config.json updated.")


def main():
    parser = argparse.ArgumentParser(description="ONNX export and quantization pipeline")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-Prompt-Guard-2-22M", help="Name of the model")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX format")
    parser.add_argument("--quantize-onnx", action="store_true", help="Quantize ONNX model (auto-optimizes if needed)")
    parser.add_argument("--update-config", action="store_true", help="Update config.json with labels")
    args = parser.parse_args()

    if args.model_name:
        save_hf_model(args.model_name)

    onnx_path = "gravitee-io/" + args.model_name.split("/")[1] + '-onnx'
    if args.export_onnx:
        export_model_to_onnx(onnx_path)

    if args.quantize_onnx:
        quantize_onnx_model(onnx_path)

    if args.update_config:
        update_config_labels(onnx_path)


if __name__ == "__main__":
    main()
