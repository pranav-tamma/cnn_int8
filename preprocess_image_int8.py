"""
preprocess_image_int8.py

Prepares ONE digit image for inference on the int8 RTL pipeline: runs the
same OpenCV preprocessing as calibration, quantizes to int8 using S_in from
scales.txt (produced by quantize_int8.py -- run that first), and writes
a_matrix as 16 BRAM bank files matching top_cnn.sv's banked interface:
    output reg [9:0] a_matrix_addr,           // 0..675
    input signed [7:0] a_matrix_rdata [0:15]  // all 16 banks return together

WHY THIS REPLACES processed_digit.py's a_matrix generation:
processed_digit.py wrote one flat a_matrix.vh with 676*16 individual
`a_matrix[i][j] = value;` lines, matching the OLD design where a_matrix was
a directly-declared top-level array. top_cnn.sv no longer has that port --
a_matrix now lives in 16 external BRAMs (see top_cnn.sv's port comments),
so the deliverable has to be 16 separate per-bank memory-init files, one
word per address, not one big flat array.

USAGE:
    python preprocess_image_int8.py handwritten_2.png
Requires generated/scales.txt to already exist (run quantize_int8.py first).
Outputs generated/a_matrix_bank{0..15}.mem (676 words each, hex, one per line).
"""
import sys
import os
import numpy as np

from quantize_int8 import preprocess_like_rtl_pipeline  # reuse the exact same pipeline as calibration


def load_S_in(scales_path="generated/scales.txt"):
    with open(scales_path) as f:
        for line in f:
            if line.startswith("S_in"):
                return float(line.split("=")[1].strip())
    raise ValueError(f"S_in not found in {scales_path} -- run quantize_int8.py first")


def build_a_matrix(image01, S_in):
    """Same im2col construction as processed_digit.py: 676 rows (26x26
    spatial positions), 16 columns (9 real conv1 patch values + 7 zero
    padding to reach the systolic array's 16-wide operand)."""
    img_q = np.round(np.clip(image01, 0, 1) * S_in).astype(np.int32)
    img_q = np.clip(img_q, -127, 127)

    a_matrix = np.zeros((676, 16), dtype=np.int32)
    idx = 0
    for r in range(26):
        for c in range(26):
            patch = img_q[r:r+3, c:c+3].flatten()  # kr*3+kc order, matches conv1 weight reshape
            a_matrix[idx, 0:9] = patch
            idx += 1
    return a_matrix


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_image_int8.py <image.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    S_in = load_S_in()

    image01 = preprocess_like_rtl_pipeline(image_path)
    a_matrix = build_a_matrix(image01, S_in)

    os.makedirs("generated", exist_ok=True)
    for b in range(16):
        with open(f"generated/a_matrix_bank{b}.mem", "w") as f:
            for val in a_matrix[:, b]:
                f.write(f"{int(val) & 0xFF:02x}\n")
    print(f"Generated generated/a_matrix_bank0..15.mem (676 words/bank) from {image_path}, S_in={S_in}")


if __name__ == "__main__":
    main()
