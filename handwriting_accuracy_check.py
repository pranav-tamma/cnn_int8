"""
handwriting_accuracy_check.py

Evaluates float vs int8-quantized model accuracy on your OWN photographed/
scanned handwriting samples -- unlike mnist_accuracy_check.py, this runs
images through the FULL camera pipeline (preprocess_like_rtl_pipeline:
threshold, contour-crop, resize, recenter), since that's the part MNIST's
clean, pre-centered images never exercise at all. This is the more
realistic test of what the deployed system will actually see.

LABELING: true labels are parsed straight from each filename -- the first
digit found after "handwritten_" is taken as the label, so
"handwritten_4.png", "handwritten_4b.png", and "handwritten_4_v2.png" all
correctly parse as label 4. Images that don't match this pattern are
skipped with a warning (fix the filename, or extend the regex below).

Requires: train.py, quantize_int8.py, and generated/scales.txt (from a
completed quantize_int8.py calibration run) in the same directory.

Usage:
    python handwriting_accuracy_check.py cnn_weights_mnist_2.pth handwritten_0.png handwritten_1.png ...
    python handwriting_accuracy_check.py cnn_weights_mnist_2.pth --dir handwriting_samples/
"""
import sys
import os
import re
import glob
import io
import contextlib
import numpy as np
import torch

from train import CNN
from quantize_int8 import (
    quantize_tensor_int8, reshape_conv1, reshape_conv2,
    golden_forward_int, preprocess_like_rtl_pipeline, SCALE_INPUT
)


def load_scales(path="generated/scales.txt"):
    scales = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=")
                scales[k.strip()] = float(v.strip())
    return scales


def parse_label(path):
    m = re.search(r'handwritten_(\d)', os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python handwriting_accuracy_check.py <weights.pth> <img1.png> [img2.png ...]")
        print("   or: python handwriting_accuracy_check.py <weights.pth> --dir <folder>")
        sys.exit(1)

    pth_path = args[0]
    if "--dir" in args:
        folder = args[args.index("--dir") + 1]
        image_paths = sorted(glob.glob(os.path.join(folder, "handwritten_*.png")))
        if not image_paths:
            print(f"No handwritten_*.png files found in {folder}")
            sys.exit(1)
    else:
        image_paths = args[1:]

    w = torch.load(pth_path, map_location="cpu")

    model = CNN()
    model.load_state_dict(w)
    model.eval()

    conv1_flat = reshape_conv1(w['conv1.weight'].numpy())
    conv1_q, S_w1 = quantize_tensor_int8(conv1_flat)
    conv2_flat = reshape_conv2(w['conv2.weight'].numpy())
    conv2_q, S_w2 = quantize_tensor_int8(conv2_flat)
    fc_flat = w['fc.weight'].numpy().T
    fc_q, S_wfc = quantize_tensor_int8(fc_flat)

    scales = load_scales()
    S_act1, S_act2 = scales["S_act1"], scales["S_act2"]
    shift1, shift2 = int(scales["relu1_shift"]), int(scales["relu2_shift"])

    conv1_bias_acc = np.round(w['conv1.bias'].numpy() * S_w1 * SCALE_INPUT).astype(np.int64)
    conv2_bias_acc = np.round(w['conv2.bias'].numpy() * S_w2 * S_act1).astype(np.int64)
    fc_bias_acc = np.round(w['fc.bias'].numpy() * S_wfc * S_act2).astype(np.int64)

    results = []
    for path in image_paths:
        label = parse_label(path)
        if label is None:
            print(f"WARNING: couldn't parse a label from '{path}' (expected 'handwritten_<digit>...'), skipping.")
            continue

        image01 = preprocess_like_rtl_pipeline(path)  # full camera pipeline: threshold/crop/resize/recenter

        tensor = torch.tensor(image01, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        float_pred = int(torch.argmax(out, 1).item())

        with contextlib.redirect_stdout(io.StringIO()):
            logits, _, _ = golden_forward_int(
                image01, w, SCALE_INPUT, S_w1, S_w2, S_wfc,
                conv1_bias_acc, conv2_bias_acc, fc_bias_acc, shift1, shift2)
        quant_pred = int(np.argmax(logits))

        float_sorted = torch.softmax(out, dim=1).numpy().flatten()
        top2_float = np.argsort(float_sorted)[-2:][::-1]
        print(f"{os.path.basename(path)}  float margin: "
              f"{float_sorted[top2_float[0]]:.3f} (class {top2_float[0]}) vs "
              f"{float_sorted[top2_float[1]]:.3f} (class {top2_float[1]})")

        quant_sorted_idx = np.argsort(logits)[-2:][::-1]
        print(f"{os.path.basename(path)}  quant margin: "
              f"{logits[quant_sorted_idx[0]]} (class {quant_sorted_idx[0]}) vs "
              f"{logits[quant_sorted_idx[1]]} (class {quant_sorted_idx[1]})")

        results.append({
            "path": path, "label": label,
            "float_pred": float_pred, "quant_pred": quant_pred,
            "float_ok": float_pred == label, "quant_ok": quant_pred == label,
        })

    n = len(results)
    if n == 0:
        print("No valid images to evaluate.")
        return

    float_correct = sum(r["float_ok"] for r in results)
    quant_correct = sum(r["quant_ok"] for r in results)
    agree = sum(r["float_pred"] == r["quant_pred"] for r in results)

    print(f"\n{'File':<28} {'True':>5} {'Float':>6} {'Quant':>6}  Notes")
    print("-" * 65)
    for r in results:
        notes = []
        if not r["float_ok"]:
            notes.append("FLOAT WRONG")
        if not r["quant_ok"]:
            notes.append("QUANT WRONG")
        if r["float_pred"] != r["quant_pred"]:
            notes.append("float/quant disagree")
        print(f"{os.path.basename(r['path']):<28} {r['label']:>5} {r['float_pred']:>6} "
              f"{r['quant_pred']:>6}  {', '.join(notes)}")

    print("\n" + "=" * 60)
    print(f"Float model accuracy:      {float_correct}/{n} = {100*float_correct/n:.2f}%")
    print(f"Quantized model accuracy:  {quant_correct}/{n} = {100*quant_correct/n:.2f}%")
    print(f"Float/quantized agreement: {agree}/{n} = {100*agree/n:.2f}%")
    print("=" * 60)

    only_quant_wrong = [r for r in results if r["float_ok"] and not r["quant_ok"]]
    if only_quant_wrong:
        print(f"\n{len(only_quant_wrong)} image(s) the FLOAT model got right but quantized got wrong "
              f"-- these are real quantization losses, worth a closer look:")
        for r in only_quant_wrong:
            print(f"  {os.path.basename(r['path'])}: true={r['label']}, quant predicted {r['quant_pred']}")
    else:
        print("\nNo cases where float was right and quantized was wrong -- "
              "quantization isn't losing anything on this sample set.")


if __name__ == "__main__":
    main()
