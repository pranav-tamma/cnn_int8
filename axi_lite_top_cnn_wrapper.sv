`timescale 1ns/1ps
//=============================================================================
// axi_lite_top_cnn_wrapper.sv
//
// AXI4-Lite slave wrapping top_cnn.sv's small ports (control/status, shifts,
// bias/rescale arrays, final_scores). The LARGE array ports (a_matrix,
// kernel_matrix1, kernel_matrix2, fc_weights) are NOT handled here -- they
// pass straight through to top-level ports of THIS wrapper, meant to be
// connected in Vivado's Block Design (IP Integrator) to AXI BRAM Controller
// IP + Block Memory Generator, not hand-written RTL. top_cnn's existing
// a_matrix_addr/a_matrix_rdata (etc.) are already shaped exactly like a
// native BRAM port (address in, data out one cycle later) -- an AXI BRAM
// Controller's native-side port connects directly, no glue logic needed.
//
// WHY THIS SPLIT: top_cnn.sv was originally synthesized standalone as the
// top-level module, which meant EVERY port (all the bias/rescale arrays,
// weight/activation BRAM interfaces, final_scores) became a physical FPGA
// pin -- confirmed by direct synthesis report to cost 2832 Bonded IOB
// against a 200-pin budget. Making THIS wrapper the real top-level instead
// means only the ports listed below (mostly small control/status/config
// registers) go through AXI-Lite, and the large arrays go through
// dedicated BRAM controller IP -- neither ends up bound to physical pins
// the way top_cnn's raw ports were.
//
// REGISTER MAP (word-addressed, 4-byte aligned, C_S_AXI_ADDR_WIDTH=9 bits
// covers up to byte offset 0x1FF):
//   0x000  CONTROL      bit0 = rst (drives top_cnn.rst directly; hold high,
//                        write 0 to release and start inference -- same
//                        rst/start convention tb_cnn.sv already uses)
//   0x004  STATUS       bit0 = done_cnn (read-only; writes ignored)
//   0x008  SHIFTS       [4:0]=fc_shift (relu1_shift/relu2_shift REMOVED --
//                        top_cnn.sv's relu instances now use compile-time
//                        RELU1_SHIFT/RELU2_SHIFT localparams, not runtime
//                        ports; this was a deliberate LUT-savings fix,
//                        see relu.sv's own comment for why. Recalibrating
//                        those two now means editing top_cnn.sv directly
//                        and resynthesizing, not writing this register.)
//   0x00C  (reserved)
//   0x010-0x04C  CONV1_BIAS[0:15]      (16 words)
//   0x050-0x08C  CONV2_BIAS[0:15]      (16 words)
//   0x090-0x0B4  FC_BIAS[0:9]          (10 words)
//   0x0B8-0x0F4  CONV1_RESCALE[0:15]   (16 words)
//   0x0F8-0x134  CONV2_RESCALE[0:15]   (16 words)
//   0x138-0x15C  FC_RESCALE[0:9]       (10 words)
//   0x160-0x184  FINAL_SCORES[0:9]     (10 words, read-only; writes ignored)
//
// Total register file: 98 x 32-bit words (index 0..97), addr[8:2] selects
// the word index (addr[1:0] always 0 for 4-byte-aligned accesses, per AXI
// convention -- this slave does not support unaligned or narrow (non-32-
// bit-strobe) accesses, which is standard for a control/config-style
// AXI-Lite slave like this one).
//=============================================================================
module axi_lite_top_cnn_wrapper #(
    parameter C_S_AXI_ADDR_WIDTH = 9,
    parameter C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                              s_axi_aclk,
    input  wire                              s_axi_aresetn,

    // -- AXI4-Lite write address channel --
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire                              s_axi_awvalid,
    output reg                               s_axi_awready,

    // -- AXI4-Lite write data channel --
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output reg                               s_axi_wready,

    // -- AXI4-Lite write response channel --
    output reg  [1:0]                        s_axi_bresp,
    output reg                               s_axi_bvalid,
    input  wire                              s_axi_bready,

    // -- AXI4-Lite read address channel --
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire                              s_axi_arvalid,
    output reg                               s_axi_arready,

    // -- AXI4-Lite read data channel --
    output reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output reg  [1:0]                        s_axi_rresp,
    output reg                               s_axi_rvalid,
    input  wire                              s_axi_rready,

    // -- Pass-through BRAM-style ports: connect to AXI BRAM Controller IP's
    // native port in the Block Design, NOT driven by this module's own
    // logic. Identical shape to top_cnn.sv's own ports -- see that file's
    // port comments for the banking rationale (16 banks x 676/16/80/400
    // words respectively).
    output wire [9:0]        a_matrix_addr,
    input  wire signed [7:0] a_matrix_rdata [0:15],
    output wire [3:0]        kernel_matrix1_addr,
    input  wire signed [7:0] kernel_matrix1_rdata [0:7],
    output wire [6:0]        kernel_matrix2_addr,
    input  wire signed [7:0] kernel_matrix2_rdata [0:15],
    output wire [8:0]        fc_weights_addr,
    input  wire signed [7:0] fc_weights_rdata [0:15]
);

    localparam NREGS = 98;
    localparam IDX_CONTROL       = 0;
    localparam IDX_STATUS        = 1;
    localparam IDX_SHIFTS        = 2;
    localparam IDX_CONV1_BIAS0   = 4;
    localparam IDX_CONV2_BIAS0   = 20;
    localparam IDX_FC_BIAS0      = 36;
    localparam IDX_CONV1_RESC0   = 46;
    localparam IDX_CONV2_RESC0   = 62;
    localparam IDX_FC_RESC0      = 78;
    localparam IDX_FINAL_SCORES0 = 88;

    // ---- register file (drives top_cnn's config ports) -----------------
    reg [31:0] regs [0:NREGS-1];
    integer ri;

    wire cnn_rst    = regs[IDX_CONTROL][0];
    wire done_cnn_w;

    wire signed [31:0] conv1_bias      [0:15];
    wire signed [31:0] conv2_bias      [0:15];
    wire signed [31:0] fc_bias         [0:9];
    wire signed [15:0] conv1_rescale   [0:15];
    wire signed [15:0] conv2_rescale   [0:15];
    wire signed [15:0] fc_rescale      [0:9];
    wire signed [31:0] final_scores_w  [0:9];

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV1_BIAS
            assign conv1_bias[gi] = regs[IDX_CONV1_BIAS0 + gi];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV2_BIAS
            assign conv2_bias[gi] = regs[IDX_CONV2_BIAS0 + gi];
        end
        for (gi = 0; gi < 10; gi = gi + 1) begin : GEN_FC_BIAS
            assign fc_bias[gi] = regs[IDX_FC_BIAS0 + gi];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV1_RESC
            assign conv1_rescale[gi] = regs[IDX_CONV1_RESC0 + gi][15:0];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV2_RESC
            assign conv2_rescale[gi] = regs[IDX_CONV2_RESC0 + gi][15:0];
        end
        for (gi = 0; gi < 10; gi = gi + 1) begin : GEN_FC_RESC
            assign fc_rescale[gi] = regs[IDX_FC_RESC0 + gi][15:0];
        end
    endgenerate

    // ---- top_cnn instantiation ------------------------------------------
    top_cnn dut (
        .clk(s_axi_aclk),
        .rst(cnn_rst),
        .done_cnn(done_cnn_w),
        .a_matrix_addr(a_matrix_addr),
        .a_matrix_rdata(a_matrix_rdata),
        .kernel_matrix1_addr(kernel_matrix1_addr),
        .kernel_matrix1_rdata(kernel_matrix1_rdata),
        .kernel_matrix2_addr(kernel_matrix2_addr),
        .kernel_matrix2_rdata(kernel_matrix2_rdata),
        .conv1_bias(conv1_bias),
        .conv2_bias(conv2_bias),
        .conv1_rescale(conv1_rescale),
        .conv2_rescale(conv2_rescale),
        .fc_rescale(fc_rescale),
        .fc_shift(regs[IDX_SHIFTS][4:0]),
        .fc_weights_addr(fc_weights_addr),
        .fc_weights_rdata(fc_weights_rdata),
        .fc_bias(fc_bias),
        .final_scores(final_scores_w)
    );

    // ---- AXI4-Lite write channel (standard 2-phase handshake) -----------
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_reg;
    reg aw_en;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 0;
            awaddr_reg <= 0;
            aw_en <= 1;
        end else begin
            if (~s_axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                s_axi_awready <= 1;
                awaddr_reg <= s_axi_awaddr;
                aw_en <= 0;
            end else if (s_axi_bready && s_axi_bvalid) begin
                aw_en <= 1;
                s_axi_awready <= 0;
            end else begin
                s_axi_awready <= 0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_wready <= 0;
        end else begin
            if (~s_axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en)
                s_axi_wready <= 1;
            else
                s_axi_wready <= 0;
        end
    end

    wire [6:0] waddr_idx = awaddr_reg[8:2];  // word index, 0..127 (98 used)
    wire write_en = s_axi_wready && s_axi_wvalid && s_axi_awready && s_axi_awvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            // NOTE: this must NOT touch regs[IDX_STATUS] or
            // regs[IDX_FINAL_SCORES0 .. +9] -- those each have their OWN
            // dedicated always block below (mirroring done_cnn/
            // final_scores_w), which ALSO drives them on reset. An
            // earlier version of this loop ran ri across the FULL
            // 0..NREGS-1 range with an `if` guard skipping those specific
            // indices -- that guard is always false for ri==IDX_STATUS or
            // ri in the FINAL_SCORES range, so the assignment inside it
            // can never actually execute there, but synthesis still
            // inferred a driver at those indices anyway (confirmed
            // directly via DRC: "Multiple Driver Nets ... regs_reg[88][0]
            // ... GEN_FINAL_SCORES_MIRROR[0].regs_reg[88][0]" -- a reset
            // tree merging multiple registers under the same `if
            // (!aresetn)` branch doesn't always respect per-index dead-
            // code elimination the way plain simulation does, even when
            // the guard condition is provably always-false for that
            // index). Splitting the LOOP BOUNDS themselves instead -- so
            // ri can never structurally reach those indices at all --
            // removes the driver at the source rather than relying on
            // the tool to prove a branch dead.
            regs[IDX_CONTROL] <= 32'd0;
            // regs[IDX_STATUS] (index 1) intentionally skipped here --
            // owned entirely by its own dedicated block below.
            for (ri = IDX_STATUS + 1; ri < IDX_FINAL_SCORES0; ri = ri + 1)
                regs[ri] <= 32'd0;
            // regs[IDX_FINAL_SCORES0 .. +9] (88..97) intentionally
            // skipped here -- owned entirely by their own dedicated
            // blocks below.
        end else if (write_en) begin
            // STATUS (index 1) and FINAL_SCORES (88..97) are read-only --
            // silently ignore writes there rather than erroring, standard
            // AXI-Lite convention for read-only registers.
            //
            // CHANGED: collapsed from 4 separate byte-strobed writes
            // (`if (wstrb[n]) regs[idx][byte n] <= ...`) down to one
            // unconditional 32-bit write. Every one of these ~94
            // registers (bias/rescale arrays) is only ever written ONCE
            // at startup by software loading weights -- never touched
            // again during actual inference -- so byte-level write
            // granularity was never doing anything useful here. It WAS,
            // however, costing real control-set variety: 4 differently-
            // gated write paths per register instead of 1, contributing
            // to the [Place 30-487] slice-packing failure (too many
            // distinct control sets for the device to pack efficiently,
            // even though raw LUT/FF counts fit). Software loading a
            // register now must write all 4 bytes with WSTRB fully
            // asserted (0xF) -- true of every AXI-Lite master that
            // writes full 32-bit words anyway, so no real workflow cost.
            if (waddr_idx != IDX_STATUS &&
                !(waddr_idx >= IDX_FINAL_SCORES0 && waddr_idx < IDX_FINAL_SCORES0 + 10) &&
                waddr_idx < NREGS) begin
                regs[waddr_idx] <= s_axi_wdata;
            end
        end
    end

    // STATUS register: bit0 mirrors done_cnn continuously (hardware-driven,
    // independent of the AXI write channel above).
    //
    // CHANGED: this used to write regs[IDX_STATUS] (a slot inside the
    // shared regs[] array), even though no other block ever touched that
    // same index -- but synthesis still produced a multi-driven-net
    // error there (confirmed via DRC AND the synthesis log, both
    // pointing at this exact register, even after restructuring the
    // reset loop's BOUNDS to make it structurally impossible to reach
    // index 1). That strongly suggests Vivado's array inference doesn't
    // cleanly handle one shared array being written by multiple
    // different always blocks, even when each block owns a disjoint set
    // of indices with zero logical overlap. Rather than patch this a
    // third time inside the shared array, STATUS now gets its own
    // completely standalone register -- never touched by ANY other
    // always block, so there is no shared array for the tool to get
    // confused about. See the read-mux below for how index 1 now
    // resolves to THIS register instead of regs[1].
    reg [31:0] status_reg;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn)
            status_reg <= 32'd0;
        else
            status_reg[0] <= done_cnn_w;
    end

    // FINAL_SCORES registers: mirror top_cnn's final_scores continuously,
    // same hardware-driven pattern as STATUS above -- and the SAME fix:
    // a standalone array, never touched by any other always block,
    // instead of slots inside the shared regs[] array.
    reg [31:0] final_scores_reg [0:9];
    generate
        for (gi = 0; gi < 10; gi = gi + 1) begin : GEN_FINAL_SCORES_MIRROR
            always @(posedge s_axi_aclk) begin
                if (!s_axi_aresetn)
                    final_scores_reg[gi] <= 32'd0;
                else
                    final_scores_reg[gi] <= final_scores_w[gi];
            end
        end
    endgenerate

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_bvalid <= 0;
            s_axi_bresp  <= 2'b00;
        end else if (write_en && ~s_axi_bvalid) begin
            s_axi_bvalid <= 1;
            s_axi_bresp  <= 2'b00;  // OKAY
        end else if (s_axi_bready && s_axi_bvalid) begin
            s_axi_bvalid <= 0;
        end
    end

    // ---- AXI4-Lite read channel (standard 2-phase handshake) ------------
    reg [C_S_AXI_ADDR_WIDTH-1:0] araddr_reg;
    reg [6:0] raddr_idx_reg;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 0;
            araddr_reg <= 0;
        end else begin
            if (~s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1;
                araddr_reg <= s_axi_araddr;
            end else begin
                s_axi_arready <= 0;
            end
        end
    end

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 0;
            s_axi_rresp  <= 2'b00;
            // NOTE: s_axi_rdata is NOT reset here -- it's driven purely
            // combinationally below (always @(*)), which already settles
            // correctly on reset since raddr_idx_reg (its input) resets
            // to 0. Having BOTH this clocked block and the combinational
            // block assign s_axi_rdata was a real bug: two drivers on
            // the same net, with synthesis picking the constant-0 driver
            // and ignoring the real one -- confirmed directly in the
            // synthesis log ("multi-driven net on pin Q ... other driver
            // is ignored"), which meant EVERY read transaction returned
            // 0 regardless of the register's actual value, on real
            // hardware, silently. This wasn't a resource-count problem,
            // it was a correctness bug -- caught by chasing an unrelated
            // 163-critical-warnings investigation, not by design.
            raddr_idx_reg <= 0;
        end else if (s_axi_arready && s_axi_arvalid && ~s_axi_rvalid) begin
            raddr_idx_reg <= s_axi_araddr[8:2];
            s_axi_rvalid <= 1;
            s_axi_rresp  <= 2'b00;
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 0;
        end
    end

    always @(*) begin
        // STATUS and FINAL_SCORES now live in their own standalone
        // registers (status_reg/final_scores_reg), not in regs[] --
        // see those declarations above for why. regs[IDX_STATUS] and
        // regs[IDX_FINAL_SCORES0..+9] are simply unused, dead slots in
        // the array now (harmless -- nothing ever reads or writes them).
        if (raddr_idx_reg == IDX_STATUS)
            s_axi_rdata = status_reg;
        else if (raddr_idx_reg >= IDX_FINAL_SCORES0 && raddr_idx_reg < IDX_FINAL_SCORES0 + 10)
            s_axi_rdata = final_scores_reg[raddr_idx_reg - IDX_FINAL_SCORES0];
        else if (raddr_idx_reg < NREGS)
            s_axi_rdata = regs[raddr_idx_reg];
        else
            s_axi_rdata = 32'd0;
    end

endmodule