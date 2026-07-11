"""
mnist_accuracy_check.py

Evaluates the int8-quantized model's accuracy against the official MNIST
test set (10,000 images), alongside the float model as a baseline, and
breaks down where the quantized model's errors land (useful for chasing
the "everything wrong lands on class 8" pattern).

IMPORTANT: this uses raw MNIST test images directly (just normalized to
[0,1]), NOT quantize_int8.py's preprocess_like_rtl_pipeline(). That
pipeline (Otsu threshold, contour-crop, resize, recenter) exists to rescue
handwriting from a camera photo -- MNIST images are already clean,
centered, antialiased 28x28 digits. Running them through Otsu thresholding
would binarize away the antialiasing gradients the float model was
actually trained on, which would test a mismatched pipeline rather than
the model + quantization scheme itself. This script tests the real thing.

Requires:
    - torchvision (for the MNIST dataset: pip install torchvision --break-system-packages)
    - generated/scales.txt already produced by a real quantize_int8.py run
      (S_act1, S_act2, relu1_shift, relu2_shift are loaded from there --
      run quantize_int8.py with your full calibration set first)
    - train.py (defines CNN) and quantize_int8.py in the same directory

Usage:
    python mnist_accuracy_check.py cnn_weights_mnist_2.pth
    python mnist_accuracy_check.py cnn_weights_mnist_2.pth --limit 1000   # quick subset run first

Runtime note: golden_forward_int does its conv1/conv2 convolutions as
explicit Python loops (676 + 121 iterations per image), so the full 10,000
images will take a while (rough ballpark: several minutes to tens of
minutes depending on your machine). Use --limit to sanity-check on a
smaller subset first.
"""
import sys
import io
import contextlib
import numpy as np
import torch
from torchvision import datasets

from train import CNN
from quantize_int8 import (
    quantize_tensor_int8, reshape_conv1, reshape_conv2,
    golden_forward_int, SCALE_INPUT
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
        print("Usage: python mnist_accuracy_check.py <weights.pth> [--limit N]")
        sys.exit(1)
    pth_path = args[0]
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    print("Loading MNIST test set (downloads once, ~10MB, cached afterward)...")
    test_set = datasets.MNIST(root="./mnist_data", train=False, download=True)

    w = torch.load(pth_path, map_location="cpu")

    # --- float model ---
    model = CNN()
    model.load_state_dict(w)
    model.eval()

    # --- quantized model: rebuild weights + load YOUR calibrated scales ---
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

    n = len(test_set) if limit is None else min(limit, len(test_set))

    float_correct = 0
    quant_correct = 0
    agree_count = 0
    per_class_total = np.zeros(10, dtype=int)
    per_class_float_correct = np.zeros(10, dtype=int)
    per_class_quant_correct = np.zeros(10, dtype=int)
    quant_confusion = np.zeros((10, 10), dtype=int)  # [true label][predicted]

    print(f"Evaluating {n} test images{' (subset)' if limit else ''}...")
    for idx in range(n):
        img_pil, label = test_set[idx]
        img = np.array(img_pil, dtype=np.float32) / 255.0  # (28,28)

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        float_pred = int(torch.argmax(out, 1).item())

        # golden_forward_int has several debug $display-style prints inside
        # it (pool1/pool2/raw_fc) -- silence them here, they'd otherwise
        # spam 10,000x.
        with contextlib.redirect_stdout(io.StringIO()):
            logits, _, _ = golden_forward_int(
                img, w, SCALE_INPUT, S_w1, S_w2, S_wfc,
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

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{n} done...")

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
