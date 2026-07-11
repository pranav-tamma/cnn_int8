`timescale 1ns/1ps

// Simulation-only testbench for top_cnn.sv. NOT synthesizable, and not the
// AXI wrapper you'll need for the real board -- this exists purely to
// confirm the design behaves correctly with real weight/image data before
// you invest time in the AXI integration. Each *_bank array below stands
// in for one of the external BRAMs top_cnn.sv's port comments describe;
// they're loaded with $readmemh from quantize_int8.py's/preprocess_image_
// int8.py's generated .mem files and read with the same 1-cycle latency
// top_cnn.sv's LOAD_TILE1/LOAD_TILE2/LOAD_FC pipeline already assumes
// (address presented this cycle, data returned next cycle).
module top_cnn_tb;

    // GEN_DIR: absolute path to the folder containing the .mem/.vh files
    // from quantize_int8.py/preprocess_image_int8.py. Vivado's simulator
    // (xsim) runs from its own internal working directory, NOT your
    // project folder -- a relative "generated/..." path silently fails
    // to resolve there (you'll see "cannot be opened for reading"
    // warnings, and every array will just be left at X/0, so the
    // simulation still "finishes" but on garbage data). Update this if
    // your project path is different, or if you regenerate into a
    // different folder.
    parameter GEN_DIR = "C:/Users/prana/cnn_project/generated";

    reg clk;
    reg rst;
    wire done_cnn;

    // --- a_matrix: 16 banks x 676 words, flattened to one 1-D array -----
    // FLATTENED, not [0:15][0:675]: $readmemh's target argument sliced
    // down from a multi-dimensional array (e.g. mem[i] where mem is
    // [0:15][0:675]) isn't reliably portable across simulators -- Vivado's
    // xsim can misinterpret the sliced sub-array's size, producing "too
    // many words specified" for a file that's actually the right length.
    // Flat 1-D array + $readmemh's explicit start/end address arguments
    // is the standard, simulator-agnostic way to load multiple banks.
    reg [7:0] a_matrix_flat [0:16*676-1];
    wire [9:0] a_matrix_addr;
    reg signed [7:0] a_matrix_rdata [0:15];

    // --- kernel_matrix1: 8 banks x 16 words, flattened ------------------
    reg [7:0] kernel_matrix1_flat [0:8*16-1];
    wire [3:0] kernel_matrix1_addr;
    reg signed [7:0] kernel_matrix1_rdata [0:7];

    // --- kernel_matrix2: 16 banks x 80 words, flattened -----------------
    reg [7:0] kernel_matrix2_flat [0:16*80-1];
    wire [6:0] kernel_matrix2_addr;
    reg signed [7:0] kernel_matrix2_rdata [0:15];

    // --- fc_weights: 16 banks x 400 words, flattened --------------------
    reg [7:0] fc_weights_flat [0:16*400-1];
    wire [8:0] fc_weights_addr;
    reg signed [7:0] fc_weights_rdata [0:15];

    // --- biases: loaded via `include, same as top_cnn.sv's ports expect
    reg signed [31:0] conv1_bias [0:15];
    reg signed [31:0] conv2_bias [0:15];
    reg signed [31:0] fc_bias [0:9];

    // --- requantization shifts: from generated/scales.txt's printout,
    // NOT hardcoded here permanently -- update these two lines to match
    // whatever quantize_int8.py prints after you calibrate with real,
    // varied digit images (values below are placeholders).
    reg [4:0] relu1_shift = 8;
    reg [4:0] relu2_shift = 10;

    wire signed [31:0] final_scores [0:9];

    integer i, j;

    // --- clock ---------------------------------------------------------
    always #5 clk = ~clk;  // 100 MHz

    // --- memory loading --------------------------------------------------
    initial begin
        // Each bank i occupies addresses [i*depth : i*depth+depth-1] of
        // the corresponding flat array -- explicit start/end addresses
        // given directly to $readmemh, so there's no multi-dimensional
        // slicing anywhere in this loading path.
        for (i = 0; i < 16; i = i + 1) begin
            $readmemh($sformatf("%s/a_matrix_bank%0d.mem", GEN_DIR, i), a_matrix_flat, i*676, i*676+675);
            $readmemh($sformatf("%s/kernel_matrix2_bank%0d.mem", GEN_DIR, i), kernel_matrix2_flat, i*80, i*80+79);
            $readmemh($sformatf("%s/fc_weights_bank%0d.mem", GEN_DIR, i), fc_weights_flat, i*400, i*400+399);
        end
        for (i = 0; i < 8; i = i + 1)
            $readmemh($sformatf("%s/kernel_matrix1_bank%0d.mem", GEN_DIR, i), kernel_matrix1_flat, i*16, i*16+15);

        // conv1_bias.vh/conv2_bias.vh/fc_bias.vh are lines like
        // `conv1_bias[0] = 327;` -- valid directly inside an initial
        // block as long as the array names here match exactly.
        // `include is a compile-time preprocessor directive -- it can't
        // use the GEN_DIR parameter above (that's a runtime value,
        // `include needs a literal path). Update this path directly if
        // your project folder differs from GEN_DIR's default.
        `include "C:/Users/prana/cnn_project/generated/conv1_bias.vh"
        `include "C:/Users/prana/cnn_project/generated/conv2_bias.vh"
        `include "C:/Users/prana/cnn_project/generated/fc_bias.vh"
    end

    // --- behavioral 1-cycle-latency BRAM read, one always block per bank
    // group. This mirrors real synchronous BRAM: address sampled on the
    // clock edge, data available the FOLLOWING cycle -- matching what
    // top_cnn.sv's LOAD_TILE1/LOAD_TILE2/LOAD_FC pipeline (capture-
    // previous/issue-current) was written to expect. Indexing into the
    // flat arrays as [bank*depth + addr] instead of [bank][addr], since
    // the arrays themselves are now flat 1-D (see declarations above).
    always @(posedge clk) begin
        for (i = 0; i < 16; i = i + 1)
            a_matrix_rdata[i] <= a_matrix_flat[i*676 + a_matrix_addr];
    end

    always @(posedge clk) begin
        for (i = 0; i < 8; i = i + 1)
            kernel_matrix1_rdata[i] <= kernel_matrix1_flat[i*16 + kernel_matrix1_addr];
    end

    always @(posedge clk) begin
        for (i = 0; i < 16; i = i + 1)
            kernel_matrix2_rdata[i] <= kernel_matrix2_flat[i*80 + kernel_matrix2_addr];
    end

    always @(posedge clk) begin
        for (i = 0; i < 16; i = i + 1)
            fc_weights_rdata[i] <= fc_weights_flat[i*400 + fc_weights_addr];
    end

    // --- DUT -------------------------------------------------------------
    top_cnn dut (
        .clk(clk),
        .rst(rst),
        .done_cnn(done_cnn),
        .a_matrix_addr(a_matrix_addr),
        .a_matrix_rdata(a_matrix_rdata),
        .kernel_matrix1_addr(kernel_matrix1_addr),
        .kernel_matrix1_rdata(kernel_matrix1_rdata),
        .kernel_matrix2_addr(kernel_matrix2_addr),
        .kernel_matrix2_rdata(kernel_matrix2_rdata),
        .conv1_bias(conv1_bias),
        .conv2_bias(conv2_bias),
        .relu1_shift(relu1_shift),
        .relu2_shift(relu2_shift),
        .fc_weights_addr(fc_weights_addr),
        .fc_weights_rdata(fc_weights_rdata),
        .fc_bias(fc_bias),
        .final_scores(final_scores)
    );

    // --- stimulus / checking ---------------------------------------------
    initial begin
        clk = 0;
        rst = 1;
        #20;
        rst = 0;

        wait (done_cnn == 1);
        #10;

        $display("--------------------------------");
        $display("CNN FINISHED");
        $display("--------------------------------");
        for (j = 0; j < 10; j = j + 1)
            $display("Class %0d : %0d", j, final_scores[j]);

        // simple argmax, printed for convenience
        begin : argmax_block
            integer best_idx;
            reg signed [31:0] best_val;
            best_idx = 0;
            best_val = final_scores[0];
            for (j = 1; j < 10; j = j + 1) begin
                if (final_scores[j] > best_val) begin
                    best_val = final_scores[j];
                    best_idx = j;
                end
            end
            $display("--------------------------------");
            $display("Prediction : %0d", best_idx);
            $display("Score      : %0d", best_val);
        end

        $finish;
    end

    // Safety timeout in case done_cnn never fires (bad memory load,
    // stuck FSM, etc.) -- without this a failed run just hangs forever
    // in simulation with no explanation.
    initial begin
        #2_000_000;  // 2ms sim time, generous for this design's cycle count
        if (!done_cnn) begin
            $display("TIMEOUT: done_cnn never asserted -- check memory loading and FSM state.");
            $finish;
        end
    end

endmodule