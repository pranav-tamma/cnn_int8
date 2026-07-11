module relu #(parameter r1 = 16, c1 = 16)(
    // Raw accumulator straight from the systolic array (or, for conv2,
    // the full-width conv2_accumulator) -- still wide/unquantized. Its
    // implicit scale is S_weight * S_activation from whatever quantized
    // this layer's inputs; this module's job is bias-add, ReLU, and
    // requantizing back down to a real int8 activation for the next
    // layer to consume.
    input signed [31:0] results [0:r1-1][0:c1-1],
    // Bias is quantized at the SAME scale as the raw accumulator
    // (S_weight * S_activation), not at the int8 output scale -- this
    // is standard for quantized inference (TFLite etc. do the same) so
    // it can be added directly to the accumulator before the final
    // requantizing shift. Stays 32-bit; biases need the extra headroom,
    // there are only 16 of them so the width costs nothing.
    input signed [31:0] conv_bias [0:15],
    // Per-layer, calibrated from your Python-side quantization scales:
    // shift_amount = round(log2(S_weight * S_activation_in / S_activation_out)).
    // Runtime input (not a localparam) so you can retune per layer
    // without resynthesizing as you calibrate scales.
    input [4:0] shift_amount,
    // Requantized int8 activation, ready to be the NEXT layer's input.
    output reg signed [7:0] relu_results [0:r1-1][0:c1-1]
);

// Single flattened loop over idx = i*c1+j instead of a nested i/j loop
// -- decoded via / and % (c1 is a compile-time constant here, so this
// is cheap constant-divide/mod logic, not a real divider). Same
// hardware either way; this is purely a "no nested for loops" style
// requirement.
integer idx;
integer i, j;
reg signed [31:0] biased;
reg signed [31:0] shifted;

always@(*) begin
    for(idx=0; idx<r1*c1; idx=idx+1) begin
        i = idx / c1;
        j = idx % c1;
        biased = results[i][j] + conv_bias[j];
        shifted = biased >>> shift_amount;
        // ReLU (clamp negative to 0) + saturate to signed int8 range.
        // Since ReLU already floors at 0, only the upper bound (127)
        // can actually be hit -- but it's a real risk: if shift_amount
        // is calibrated even slightly too small for a given layer,
        // large activations will overflow int8 and MUST be clamped,
        // not silently truncated/wrapped (wrapping would turn a large
        // positive activation into small or negative garbage -- silently
        // wrong, not just imprecise).
        if (shifted <= 0)
            relu_results[i][j] = 8'sd0;
        else if (shifted > 127)
            relu_results[i][j] = 8'sd127;
        else
            relu_results[i][j] = shifted[7:0];
    end
end
endmodule