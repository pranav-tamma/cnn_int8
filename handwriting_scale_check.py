"""
handwriting_scale_check.py

Runs a large sample of MNIST test images through the FULL camera pipeline
(preprocess_like_rtl_pipeline: threshold, contour-crop, resize, recenter)
-- not just simple normalization like mnist_accuracy_check.py -- to test
the pipeline+quantization interaction at real scale, using hundreds of
different digit shapes instead of 10 hand-picked photos.

CAVEAT: MNIST images are clean and already centered -- they won't have the
lighting/shadow/paper-texture variation a real camera photo has. This
complements a small set of real photographed samples; it doesn't replace
them. What it DOES give you: a much larger, less cherry-picked sample of
digit *shapes* run through the exact same threshold/crop/recenter/resize
pipeline your real photos go through, which 10 images can't reliably do.

Each MNIST image is temporarily written to disk as a PNG (since
preprocess_like_rtl_pipeline reads from a file path via cv2.imread), run
through the full pipeline, then evaluated with both the float and
quantized models.

Requires: torchvision, train.py, quantize_int8.py, and generated/scales.txt
(from a completed quantize_int8.py calibration run).

Usage:
    python handwriting_scale_check.py cnn_weights_mnist_2.pth --limit 500
"""
import sys
import os
import io
import contextlib
import tempfile
import numpy as np
import torch
from torchvision import datasets

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


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python handwriting_scale_check.py <weights.pth> [--limit N] [--seed S]")
        sys.exit(1)
    pth_path = args[0]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 500
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42

    print("Loading MNIST test set (downloads once, ~10MB, cached afterward)...")
    test_set = datasets.MNIST(root="./mnist_data", train=False, download=True)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_set), size=min(limit, len(test_set)), replace=False)

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

    n_total = len(indices)
    float_correct = 0
    quant_correct = 0
    agree_count = 0
    n_skipped = 0
    per_class_total = np.zeros(10, dtype=int)
    per_class_float_correct = np.zeros(10, dtype=int)
    per_class_quant_correct = np.zeros(10, dtype=int)
    quant_confusion = np.zeros((10, 10), dtype=int)

    print(f"Running {n_total} MNIST images through the full camera pipeline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, idx in enumerate(indices):
            img_pil, label = test_set[int(idx)]
            tmp_path = os.path.join(tmpdir, f"mnist_{idx}.png")
            img_pil.save(tmp_path)

            try:
                image01 = preprocess_like_rtl_pipeline(tmp_path)
            except Exception as e:
                n_skipped += 1
                continue  # pipeline occasionally can't find a contour on a very sparse digit -- rare, skip and count it

            tensor = torch.tensor(image01, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
            float_pred = int(torch.argmax(out, 1).item())

            with contextlib.redirect_stdout(io.StringIO()):
                logits, _, _ = golden_forward_int(
                    image01, w, SCALE_INPUT, S_w1, S_w2, S_wfc,
                    conv1_bias_acc, conv2_bias_acc, fc_bias_acc, shift1, shift2)
            quant_pred = int(np.argmax(logits))

            per_class_total[label] += 1
            if float_pred == label:
                float_correct += 1
                per_class_float_correct[label] += 1
            if quant_pred == label:
                quant_correct += 1
                per_class_quant_correct[label] += 1
            if float_pred == quant_pred:
                agree_count += 1
            quant_confusion[label][quant_pred] += 1

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_total} done...")

    n = n_total - n_skipped
    if n_skipped:
        print(f"\n({n_skipped} image(s) skipped -- pipeline couldn't find a usable contour, rare edge case)")

    print("\n" + "=" * 60)
    print(f"Float model accuracy:      {float_correct}/{n} = {100*float_correct/n:.2f}%")
    print(f"Quantized model accuracy:  {quant_correct}/{n} = {100*quant_correct/n:.2f}%")
    print(f"Float/quantized agreement: {agree_count}/{n} = {100*agree_count/n:.2f}%")
    print("=" * 60)

    print("\nPer-class accuracy:")
    print(f"{'Class':>6} {'Count':>7} {'Float%':>8} {'Quant%':>8}")
    for c in range(10):
        fpct = 100 * per_class_float_correct[c] / per_class_total[c] if per_class_total[c] else 0
        qpct = 100 * per_class_quant_correct[c] / per_class_total[c] if per_class_total[c] else 0
        print(f"{c:>6} {per_class_total[c]:>7} {fpct:>7.2f}% {qpct:>7.2f}%")

    print("\nQuantized model confusion matrix (rows=true label, cols=predicted):")
    print("     " + " ".join(f"{i:>5}" for i in range(10)))
    for i in range(10):
        row = " ".join(f"{quant_confusion[i][j]:>5}" for j in range(10))
        print(f"{i:>3}: {row}")

    print("\nWhere quantized-model misclassifications land (per true class):")
    for true_c in range(10):
        errors = quant_confusion[true_c].copy()
        errors[true_c] = 0
        if errors.sum() > 0:
            top_wrong = int(np.argmax(errors))
            print(f"  True {true_c}: {errors.sum()} wrong, most often predicted as "
                  f"{top_wrong} ({errors[top_wrong]} times)")

    print("\nHow often each class gets OVER-predicted (column sums, excluding correct):")
    for pred_c in range(10):
        col = quant_confusion[:, pred_c].copy()
        col[pred_c] = 0
        if col.sum() > 0:
            print(f"  Class {pred_c} wrongly predicted {col.sum()} times total "
                  f"(true labels were: {[i for i in range(10) if col[i] > 0]})")


if __name__ == "__main__":
    main()
