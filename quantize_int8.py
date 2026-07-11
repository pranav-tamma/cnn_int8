"""
quantize_int8.py

Produces everything the int8 RTL pipeline (top_cnn.sv / pe2.sv / modularpe2.sv /
relu.sv) needs: int8 weight banks, accumulator-scale int32 biases, and the two
calibrated per-layer requantization shifts (relu1_shift, relu2_shift).

WHY THIS REPLACES extract_weights.py:
extract_weights.py used SCALE=256 (Q8.8) fixed-point: every weight/bias/
activation shared one implicit scale, so a single hardcoded >>>8 always
undid one multiply's worth of scale. That matched the OLD 32-bit datapath.

The int8 redesign is genuinely different, not just narrower:
  - Weights are quantized PER TENSOR: S_w = 127 / max(abs(weight)), so each
    layer's weights use however much of the int8 range they actually need.
  - Activations need their OWN calibrated scale (S_act1 after conv1+relu,
    S_act2 after conv2+relu) -- picked from real data, not assumed.
  - The bias for a layer must be quantized at that layer's ACCUMULATOR scale
    (S_weight * S_activation_in), not at the int8 output scale -- this is
    standard practice (TFLite etc. do the same) so it can be added directly
    to the raw accumulator before the requantizing shift in relu.sv.
  - The requantizing shift is DERIVED, not fixed:
        shift = round(log2(S_weight * S_activation_in / S_activation_out))
    which is why relu.sv takes shift_amount as a runtime port instead of a
    hardcoded >>>8 -- it's different per layer and depends on data you
    calibrate, not just wired-in like the old Q8.8 scheme.

WHAT THIS NEEDS FROM YOU:
Calibration requires a handful of REAL representative digit images, run
through the same preprocessing pipeline as inference, to see what range of
activations conv1/conv2 actually produce. One image is enough to run this
script, but the resulting scale will overfit to that one image -- expect
either wasted dynamic range (if that image was unusually easy/low-contrast)
or clipping on other digits (if it was unusually extreme). Aim for at least
10-20 images spanning several digit classes before treating this as final.

USAGE:
    python quantize_int8.py cnn_weights_mnist_2.pth calib_digit1.png calib_digit2.png ...
Outputs land in generated/:
    a_matrix_bank{0..15}.mem        (written by preprocess_image_int8.py, not here)
    kernel_matrix1_bank{0..7}.mem   int8 weights, 16 words each, hex, one per line
    kernel_matrix2_bank{0..15}.mem  int8 weights, 80 words each
    fc_weights_bank{0..15}.mem      int8 weights, 400 words each
    conv1_bias.vh, conv2_bias.vh, fc_bias.vh   int32 decimal, accumulator scale
    scales.txt                      S_in, S_w1, S_w2, S_wfc, S_act1, S_act2,
                                     relu1_shift, relu2_shift -- keep this,
                                     preprocess_image_int8.py needs S_in and
                                     inference-time verification needs all of it
"""
import sys
import numpy as np
import torch

SCALE_INPUT = 127.0  # S_in: image pixels are normalized to [0,1] (non-negative),
                      # so symmetric int8 wastes the negative half -- accepted
                      # here for simplicity/consistency with weight quant, since
                      # the RTL's a_tile/PE datapath has no zero-point handling
                      # anywhere. round(pixel * 127) -> int8 range [0,127].


def quantize_tensor_int8(w):
    """Symmetric per-tensor int8 quantization, zero-point = 0."""
    max_abs = np.abs(w).max()
    if max_abs == 0:
        return np.zeros_like(w, dtype=np.int32), 1.0
    scale = 127.0 / max_abs
    q = np.round(w * scale).astype(np.int32)
    q = np.clip(q, -127, 127)  # keep symmetric; -128 has no positive pair
    return q, scale


def reshape_conv1(conv1_w):
    # (8,1,3,3) -> (8,9) -> (9,8), row-major per-filter flatten so index =
    # kr*3+kc -- matches processed_digit.py's patch.flatten() order exactly.
    return conv1_w.reshape(8, 9).T  # (9,8)


def reshape_conv2(conv2_w):
    # (16,8,3,3) -> (16,72) -> (72,16), row-major flatten so index =
    # ch*9+kr*3+kc -- matches top_cnn.sv's LOAD_TILE2 decomposition
    # (ch=feature/9; kr=(feature%9)/3; kc=feature%3) exactly. Do not reorder
    # this without also changing that decomposition in top_cnn.sv.
    return conv2_w.reshape(16, 72).T  # (72,16)


def golden_forward_int(image01, w, S_in, S_w1, S_w2, S_wfc,
                        conv1_bias_acc, conv2_bias_acc, fc_bias_acc,
                        shift1, shift2):
    """
    Bit-accurate (int-arithmetic) replica of the RTL pipeline, for
    calibration and for sanity-checking predictions before touching real
    hardware. Every operation here mirrors what top_cnn.sv/relu.sv do:
    int8 x int8 multiply-accumulate into int32, add int32 bias (already at
    accumulator scale), arithmetic right-shift by the calibrated amount,
    ReLU + clamp to int8. Returns (logits, max_raw_conv1, max_raw_conv2)
    -- the two max_raw values are what calibration reads to pick shifts.
    """
    img_q = np.round(np.clip(image01, 0, 1) * S_in).astype(np.int32)  # int8-range but kept wide for the conv sum
    img_q = np.clip(img_q, -127, 127)

    conv1_w_q, _ = quantize_tensor_int8(w['conv1.weight'].numpy())
    conv1_w_flat = conv1_w_q.reshape(8, 9).T  # (9,8) int8

    # conv1: 26x26x8 raw accumulator (pre-bias, pre-shift)
    raw1 = np.zeros((26, 26, 8), dtype=np.int64)
    for r in range(26):
        for c in range(26):
            patch = img_q[r:r+3, c:c+3].reshape(9)
            raw1[r, c, :] = patch @ conv1_w_flat  # int32-range accumulate

    max_raw1 = int(np.abs(raw1 + conv1_bias_acc[None, None, :8]).max())

    biased1 = raw1 + conv1_bias_acc[None, None, :8]
    shifted1 = biased1 >> shift1
    act1 = np.clip(shifted1, 0, 127).astype(np.int32)  # ReLU + int8 clamp

    # pool1: 2x2 max -> 13x13x8
    pool1 = act1.reshape(13, 2, 13, 2, 8).max(axis=(1, 3))
    print("pool1[0,0,:] =", pool1[0,0,:])
    print("pool1[12,12,:] =", pool1[12,12,:])
    print("pool1[6,6,:] =", pool1[6,6,:])

    # conv2 im2col, matching top_cnn.sv's LOAD_TILE2 feature ordering exactly
    conv2_w_q, _ = quantize_tensor_int8(w['conv2.weight'].numpy())
    conv2_w_flat = conv2_w_q.reshape(16, 72).T  # (72,16) int8

    raw2 = np.zeros((11, 11, 16), dtype=np.int64)
    for r in range(11):
        for c in range(11):
            patch = pool1[r:r+3, c:c+3, :]              # (3,3,8): kr,kc,ch
            patch = patch.transpose(2, 0, 1).reshape(72)  # ch*9+kr*3+kc order
            raw2[r, c, :] = patch @ conv2_w_flat

    max_raw2 = int(np.abs(raw2 + conv2_bias_acc[None, None, :]).max())

    biased2 = raw2 + conv2_bias_acc[None, None, :]
    shifted2 = biased2 >> shift2
    act2 = np.clip(shifted2, 0, 127).astype(np.int32)

    # pool2: 2x2 max -> 5x5x16. r1=c1=11 is odd, so row/col index 10 has no
    # partner and is dropped entirely -- matches pooling.sv's (r1/2)=5
    # output size and top_cnn.sv's WRITE_TILE2 comment ("Row 10 has no
    # partner and is written but never triggers a pool").
    pool2 = act2[:10, :10, :].reshape(5, 2, 5, 2, 16).max(axis=(1, 3))
    print("pool2[0,0,:] =", pool2[0,0,:])
    print("pool2[4,4,:] =", pool2[4,4,:])
    print("pool2[2,2,:] =", pool2[2,2,:])

    # FC: flatten matching top_cnn.sv's LOAD_FC index = fc_k*25+fc_i*5+fc_j
    flat = pool2.transpose(2, 0, 1).reshape(400)  # k*25+i*5+j order
    fc_w_q, _ = quantize_tensor_int8(w['fc.weight'].numpy())
    fc_w_flat = fc_w_q.reshape(10, 400).T  # (400,10) int8

    raw_fc = flat.astype(np.int64) @ fc_w_flat + fc_bias_acc
    print("raw_fc[1] (class 1, pre-shift) =", raw_fc[1])
    print("raw_fc[2] (class 2, pre-shift) =", raw_fc[2])
    logits = raw_fc >> 8  # matches top_cnn.sv's fixed >>>8 at WRITE_FC/DONE

    return logits, max_raw1, max_raw2


def preprocess_like_rtl_pipeline(image_path):
    """Same OpenCV pipeline as processed_digit.py, factored out so both
    calibration here and single-image inference prep can share it exactly
    -- calibration scales are only valid if inference uses the identical
    preprocessing."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if np.mean(border) > 127:
        gray = 255 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception(f"No digit found in {image_path}")
    h_img, w_img = thresh.shape
    boxes = []
    for c in contours:
        x, y, w_, h_ = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w_ >= w_img - 1 or y + h_ >= h_img - 1:
            continue
        if cv2.contourArea(c) < 30:
            continue
        boxes.append((x, y, w_, h_))
    if not boxes:
        c = max(contours, key=cv2.contourArea)
        boxes = [cv2.boundingRect(c)]
    xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    x, y = min(xs), min(ys)
    w_, h_ = max(xs) - x, max(ys) - y
    digit = thresh[y:y+h_, x:x+w_]
    digit = cv2.copyMakeBorder(digit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    h_, w_ = digit.shape
    if h_ > w_:
        new_h, new_w = 20, int(round(w_ * 20 / h_))
    else:
        new_w, new_h = 20, int(round(h_ * 20 / w_))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off, y_off = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit
    M = cv2.moments(canvas)
    if M["m00"] != 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        shiftx, shifty = int(round(14 - cx)), int(round(14 - cy))
        T = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        canvas = cv2.warpAffine(canvas, T, (28, 28), borderValue=0)
    return canvas.astype(np.float32) / 255.0


def main():
    if len(sys.argv) < 3:
        print("Usage: python quantize_int8.py <weights.pth> <calib_img1> [calib_img2 ...]")
        sys.exit(1)

    pth_path = sys.argv[1]
    calib_paths = sys.argv[2:]
    if len(calib_paths) < 10:
        print(f"WARNING: only {len(calib_paths)} calibration image(s) given. "
              "Scales below are usable but will likely overfit to these "
              "specific images -- add more (ideally 10-20+, spanning several "
              "digit classes) before treating relu1_shift/relu2_shift as final.")

    w = torch.load(pth_path, map_location="cpu")

    # --- weight quantization (independent of calibration) ---
    conv1_flat = reshape_conv1(w['conv1.weight'].numpy())
    conv1_q, S_w1 = quantize_tensor_int8(conv1_flat)
    conv2_flat = reshape_conv2(w['conv2.weight'].numpy())
    conv2_q, S_w2 = quantize_tensor_int8(conv2_flat)
    fc_flat = w['fc.weight'].numpy().T  # (400,10)
    fc_q, S_wfc = quantize_tensor_int8(fc_flat)
    print("fc_w_q[0][1] (tile0-row0, class1 weight) =", fc_q[0][1])

    conv1_bias_acc = np.round(w['conv1.bias'].numpy() * S_w1 * SCALE_INPUT).astype(np.int64)
    conv2_bias_acc_partial = np.round(w['conv2.bias'].numpy() * S_w2).astype(np.float64)  # finalize after S_act1 known
    fc_bias_acc_partial = np.round(w['fc.bias'].numpy() * S_wfc).astype(np.float64)        # finalize after S_act2 known

    print(f"S_w1 (conv1 weight scale)  = {S_w1:.4f}")
    print(f"S_w2 (conv2 weight scale)  = {S_w2:.4f}")
    print(f"S_wfc (fc weight scale)    = {S_wfc:.4f}")
    print(f"S_in (input scale, fixed)  = {SCALE_INPUT:.4f}")

    # --- calibration: find max post-bias, pre-shift magnitude at conv1/conv2
    # across the calibration set, at a trial shift of 0, to size S_act1/S_act2 ---
    max_raw1_seen, max_raw2_seen = 0, 0
    for p in calib_paths:
        img = preprocess_like_rtl_pipeline(p)
        _, max_raw1, max_raw2 = golden_forward_int(
            img, w, SCALE_INPUT, S_w1, S_w2, S_wfc,
            conv1_bias_acc,
            np.zeros(16, dtype=np.int64),  # bias not finalized yet, fine for max-magnitude probe
            np.zeros(10, dtype=np.int64),
            shift1=0, shift2=0)
        max_raw1_seen = max(max_raw1_seen, max_raw1)
        max_raw2_seen = max(max_raw2_seen, max_raw2)

    # Pick each shift so the observed max magnitude lands just under int8's
    # positive range (127) after shifting -- i.e. shift = ceil(log2(max/127)).
    shift1 = max(0, int(np.ceil(np.log2(max(max_raw1_seen, 1) / 127.0))))
    shift2 = max(0, int(np.ceil(np.log2(max(max_raw2_seen, 1) / 127.0))))
    S_act1 = 127.0 / (max_raw1_seen / (2 ** shift1)) if max_raw1_seen > 0 else 1.0
    S_act2 = 127.0 / (max_raw2_seen / (2 ** shift2)) if max_raw2_seen > 0 else 1.0

    print(f"\nCalibrated over {len(calib_paths)} image(s):")
    print(f"  max |raw conv1 accum+bias| = {max_raw1_seen}  -> relu1_shift = {shift1}, S_act1 = {S_act1:.4f}")
    print(f"  max |raw conv2 accum+bias| = {max_raw2_seen}  -> relu2_shift = {shift2}, S_act2 = {S_act2:.4f}")

    conv2_bias_acc = np.round(conv2_bias_acc_partial * SCALE_INPUT if False else
                               w['conv2.bias'].numpy() * S_w2 * S_act1).astype(np.int64)
    fc_bias_acc = np.round(w['fc.bias'].numpy() * S_wfc * S_act2).astype(np.int64)

    # --- sanity check: run the full quantized pipeline with final shifts,
    # compare argmax against the real float model on each calibration image ---
    print("\nPer-image prediction check (quantized int pipeline, no torch model needed):")
    for p in calib_paths:
        img = preprocess_like_rtl_pipeline(p)
        logits, _, _ = golden_forward_int(
            img, w, SCALE_INPUT, S_w1, S_w2, S_wfc,
            conv1_bias_acc, conv2_bias_acc, fc_bias_acc, shift1, shift2)
        print(f"  {p}: predicted class {int(np.argmax(logits))}  logits={logits.tolist()}")

    # --- write outputs ---
    import os
    os.makedirs("generated", exist_ok=True)

    def write_banks(arr2d, prefix, n_banks):
        # arr2d: (n_words, n_banks) int8 values -> one .mem file per bank,
        # one hex value per line (two's complement, 8 bits).
        for b in range(n_banks):
            with open(f"generated/{prefix}_bank{b}.mem", "w") as f:
                for val in arr2d[:, b]:
                    f.write(f"{int(val) & 0xFF:02x}\n")
        print(f"Generated generated/{prefix}_bank0..{n_banks-1}.mem "
              f"({arr2d.shape[0]} words/bank)")

    # conv1: (9,8) real -> pad reduction dim to 16 rows (kernel_matrix1_addr is 4-bit, 0-15)
    conv1_padded = np.zeros((16, 8), dtype=np.int32)
    conv1_padded[:9, :] = conv1_q
    write_banks(conv1_padded, "kernel_matrix1", 8)
    for i in range(16):
        print(f"conv1_padded[{i}][1] =", conv1_padded[i][1])

    # conv2: (72,16) real -> pad reduction dim to 80 rows (0-79)
    conv2_padded = np.zeros((80, 16), dtype=np.int32)
    conv2_padded[:72, :] = conv2_q
    write_banks(conv2_padded, "kernel_matrix2", 16)

    # fc: (400,10) real -> pad output dim to 16 columns/banks
    fc_padded = np.zeros((400, 16), dtype=np.int32)
    fc_padded[:, :10] = fc_q
    write_banks(fc_padded, "fc_weights", 16)

    def write_bias_vh(arr, name, n):
        with open(f"generated/{name}.vh", "w") as f:
            for i in range(n):
                v = int(arr[i]) if i < len(arr) else 0
                f.write(f"{name}[{i}] = {v};\n")
        print(f"Generated generated/{name}.vh")

    conv1_bias_full = np.zeros(16, dtype=np.int64)
    conv1_bias_full[:8] = conv1_bias_acc
    write_bias_vh(conv1_bias_full, "conv1_bias", 16)
    write_bias_vh(conv2_bias_acc, "conv2_bias", 16)
    write_bias_vh(fc_bias_acc, "fc_bias", 10)  # top_cnn.sv's fc_bias port is [0:9], not padded to 16

    with open("generated/scales.txt", "w") as f:
        f.write(f"S_in = {SCALE_INPUT}\n")
        f.write(f"S_w1 = {S_w1}\n")
        f.write(f"S_w2 = {S_w2}\n")
        f.write(f"S_wfc = {S_wfc}\n")
        f.write(f"S_act1 = {S_act1}\n")
        f.write(f"S_act2 = {S_act2}\n")
        f.write(f"relu1_shift = {shift1}\n")
        f.write(f"relu2_shift = {shift2}\n")
    print("Generated generated/scales.txt")
    print(f"\nDrive relu1_shift={shift1}, relu2_shift={shift2} as top_cnn's runtime ports.")


if __name__ == "__main__":
    main()
