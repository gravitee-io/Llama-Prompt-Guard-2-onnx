import argparse

import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)
from tqdm import tqdm
from transformers import AutoTokenizer
import polars as pl

def load_model_and_tokenizer(model_path, quantized):
    print("[*] Loading ONNX model and tokenizer...")
    path = model_path + "/" + ("model.quant.onnx" if quantized else "model.onnx")
    ort_session = ort.InferenceSession(path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return ort_session, tokenizer


def predict_logits(text, tokenizer, ort_session):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    outputs = ort_session.run(None, ort_inputs)
    logits = outputs[0]
    return 1 / (1 + np.exp(-logits))


def evaluate(model_path, validation_df, tokenizer, ort_session, plot_roc=False):
    scores = pl.DataFrame({
        "model_path": pl.Series([], dtype=pl.Utf8),
        "accuracy": pl.Series([], dtype=pl.Float64),
        "precision": pl.Series([], dtype=pl.Float64),
        "recall": pl.Series([], dtype=pl.Float64),
        "f1": pl.Series([], dtype=pl.Float64),
        "auc_roc": pl.Series([], dtype=pl.Float64)
    })

    predictions, probs, actual_labels = [], [], []

    for prompt, _type in tqdm(validation_df.iter_rows(), total=validation_df.height):
        probabilities = predict_logits(prompt, tokenizer, ort_session)
        predicted_label = np.argmax(probabilities)

        predictions.append(predicted_label)
        probs.append(probabilities[0][predicted_label])
        actual_labels.append(int(0 if _type == "benign" else 1))

    # Compute metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions)
    recall = recall_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)
    fpr, tpr, _ = roc_curve(actual_labels, probs)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    scores = scores.vstack(pl.DataFrame(
        {"model_path": model_path, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
         "auc_roc": roc_auc}))

    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Threshold)')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    return scores

def build_dataset(args, sample_size=8000):
    dataset = load_dataset(args.dataset)
    df = pl.concat([dataset["train"].to_polars(), pl.DataFrame(dataset["test"].to_polars())])

    if sample_size and sample_size < df.height:
        df = df.sample(n=sample_size, seed=4569)

    return df.drop_nans().drop_nulls()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX Detoxify model")
    parser.add_argument("--dataset", type=str, default="jackhhao/jailbreak-classification",
                        help="Name of the dataset")
    parser.add_argument("--model-paths", type=str, default="gravitee-io/Llama-Prompt-Guard-2-22M-onnx,gravitee-io/Llama-Prompt-Guard-2-86M-onnx",
                        help="Path to quantized ONNX model")
    parser.add_argument("--quantized", action="store_true", help="Plot ROC curves")
    parser.add_argument("--plot-roc", action="store_true", help="Plot ROC curves")
    args = parser.parse_args()

    validation_df = build_dataset(args)

    dfs = []
    for model_path in args.model_paths.split(","):
        session, tokenizer = load_model_and_tokenizer(model_path, args.quantized)
        scores = evaluate(model_path, validation_df, tokenizer, session, args.plot_roc)
        dfs.append(scores)

    print("\nAll scores:")

    with pl.Config(fmt_str_lengths=500):
        results = pl.concat(dfs)
        print(results)

if __name__ == "__main__":
    main()
