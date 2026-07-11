"""
int8_camera_model.py

Wraps quantize_int8.py's golden_forward_int with your CURRENT calibrated
weights/scales, so camera_demo.py can predict using the exact int8
quantized math already confirmed to match the real RTL bit-for-bit.

This REPLACES the older fixed_point_model.py, which modeled the original
32-bit Q8.8 design (r1=16, c2=8) from before the int8 rewrite -- that
module no longer reflects the current architecture (int8 weights/
activations, r1=8, c2=16, per-layer calibrated shifts) at all.

Requires, in the same directory:
    generated/scales.txt   (from a completed quantize_int8.py calibration run)
    the same weights.pth file used everywhere else in this project
    quantize_int8.py, train.py
"""
import io
import contextlib
import numpy as np
import torch

from quantize_int8 import (
    quantize_tensor_int8, reshape_conv1, reshape_conv2,
    golden_forward_int, SCALE_INPUT,
)

WEIGHTS_PATH = "cnn_weights_mnist_2.pth"
SCALES_PATH = "generated/scales.txt"


def _load_scales(path=SCALES_PATH):
    scales = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=")
                scales[k.strip()] = float(v.strip())
    return scales


# --- loaded once, at import time -------------------------------------
_w = torch.load(WEIGHTS_PATH, map_location="cpu")

_conv1_flat = reshape_conv1(_w['conv1.weight'].numpy())
_, _S_w1 = quantize_tensor_int8(_conv1_flat)
_conv2_flat = reshape_conv2(_w['conv2.weight'].numpy())
_, _S_w2 = quantize_tensor_int8(_conv2_flat)
_fc_flat = _w['fc.weight'].numpy().T
_, _S_wfc = quantize_tensor_int8(_fc_flat)

_scales = _load_scales()
_S_act1, _S_act2 = _scales["S_act1"], _scales["S_act2"]
_shift1, _shift2 = int(_scales["relu1_shift"]), int(_scales["relu2_shift"])

_conv1_bias_acc = np.round(_w['conv1.bias'].numpy() * _S_w1 * SCALE_INPUT).astype(np.int64)
_conv2_bias_acc = np.round(_w['conv2.bias'].numpy() * _S_w2 * _S_act1).astype(np.int64)
_fc_bias_acc = np.round(_w['fc.bias'].numpy() * _S_wfc * _S_act2).astype(np.int64)


def predict_canvas(canvas):
    """
    canvas: 28x28 uint8 preprocessed digit image (same format
    camera_demo.py's preprocess_frame() produces)

    Returns: (logits, prediction) -- same shape as the old
    fixed_point_model.predict_canvas(), so camera_demo.py's call sites
    don't need to change beyond swapping the import.
    """
    image01 = canvas.astype(np.float32) / 255.0

    # golden_forward_int has several debug $display-style prints inside it
    # (pool1/pool2/raw_fc) -- silence them for a live per-frame call.
    with contextlib.redirect_stdout(io.StringIO()):
        logits, _, _ = golden_forward_int(
            image01, _w, SCALE_INPUT, _S_w1, _S_w2, _S_wfc,
            _conv1_bias_acc, _conv2_bias_acc, _fc_bias_acc, _shift1, _shift2)

    return logits, int(np.argmax(logits))
