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
    output wire [1:0]                        s_axi_bresp,
    output reg                               s_axi_bvalid,
    input  wire                              s_axi_bready,

    // -- AXI4-Lite read address channel --
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire                              s_axi_arvalid,
    output reg                               s_axi_arready,

    // -- AXI4-Lite read data channel --
    output reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output wire [1:0]                        s_axi_rresp,
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
    // CHANGED: CONTROL/SHIFTS/RESCALE no longer live inside a uniform
    // 32-bit regs[] array. Each only ever used a handful of low bits
    // (rst=1 bit, fc_shift=5 bits, each rescale word=16 bits) -- the
    // remaining width was still a real flip-flop, reset and written on
    // every access, contributing nothing functionally. Pulling them into
    // their own narrow arrays removes 730 dead FF bits total:
    //   CONTROL:   32->1  bit   (31 saved)
    //   SHIFTS:    32->5  bits  (27 saved)
    //   RESCALE:   32->16 bits x42 words (672 saved)
    // BIAS stays a full 32-bit array -- top_cnn.sv's conv1_bias/
    // conv2_bias/fc_bias ports are declared `signed [31:0]`, so
    // narrowing those would require editing top_cnn.sv itself, which is
    // out of scope here.
    reg        control_reg;
    reg  [4:0] shifts_reg;
    reg [31:0] bias_regs    [0:41];   // CONV1_BIAS[0:15], CONV2_BIAS[0:15], FC_BIAS[0:9]
    reg [15:0] rescale_regs [0:41];   // CONV1_RESC[0:15], CONV2_RESC[0:15], FC_RESC[0:9]
    integer ri;

    wire cnn_rst    = control_reg;
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
            assign conv1_bias[gi] = bias_regs[gi];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV2_BIAS
            assign conv2_bias[gi] = bias_regs[16 + gi];
        end
        for (gi = 0; gi < 10; gi = gi + 1) begin : GEN_FC_BIAS
            assign fc_bias[gi] = bias_regs[32 + gi];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV1_RESC
            assign conv1_rescale[gi] = rescale_regs[gi];
        end
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_CONV2_RESC
            assign conv2_rescale[gi] = rescale_regs[16 + gi];
        end
        for (gi = 0; gi < 10; gi = gi + 1) begin : GEN_FC_RESC
            assign fc_rescale[gi] = rescale_regs[32 + gi];
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
        .fc_shift(shifts_reg),
        .fc_weights_addr(fc_weights_addr),
        .fc_weights_rdata(fc_weights_rdata),
        .fc_bias(fc_bias),
        .final_scores(final_scores_w)
    );

    // ---- AXI4-Lite write channel (standard 2-phase handshake) -----------
    // CHANGED: awaddr_reg captures only the word index (addr[8:2]) now,
    // not the full 9-bit address -- addr[1:0] are always 0 for this
    // slave's 4-byte-aligned-only accesses (documented in the register
    // map above), so those 2 bits were dead weight, same class of fix as
    // the CONTROL/SHIFTS/RESCALE narrowing above.
    reg [6:0] awaddr_reg;
    reg aw_en;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 0;
            awaddr_reg <= 0;
            aw_en <= 1;
        end else begin
            if (~s_axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                s_axi_awready <= 1;
                awaddr_reg <= s_axi_awaddr[8:2];
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

    wire [6:0] waddr_idx = awaddr_reg;  // word index, 0..127 (98 used) -- awaddr_reg now stores this directly
    wire write_en = s_axi_wready && s_axi_wvalid && s_axi_awready && s_axi_awvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            // STATUS and FINAL_SCORES aren't reset here -- they no longer
            // have wrapper-local storage at all (read directly from
            // top_cnn's own done_cnn_w/final_scores_w, which top_cnn
            // resets on its own rst input).
            control_reg <= 1'd0;
            shifts_reg  <= 5'd0;
            for (ri = 0; ri < 42; ri = ri + 1) begin
                bias_regs[ri]    <= 32'd0;
                rescale_regs[ri] <= 16'd0;
            end
        end else if (write_en && waddr_idx < NREGS) begin
            // STATUS (index 1) and FINAL_SCORES (88..97) are read-only --
            // silently ignore writes there, standard AXI-Lite convention.
            //
            // Still ONE write_en condition gating everything below --
            // the branches only choose which D-input the same enable
            // feeds, they don't add new distinct enables, so this stays
            // control-set-safe (the byte-strobe collapse from before is
            // preserved: full-word writes only, no wstrb granularity).
            if (waddr_idx == IDX_CONTROL)
                control_reg <= s_axi_wdata[0];
            else if (waddr_idx == IDX_SHIFTS)
                shifts_reg <= s_axi_wdata[4:0];
            else if (waddr_idx >= IDX_CONV1_BIAS0 && waddr_idx < IDX_CONV1_RESC0)
                bias_regs[waddr_idx - IDX_CONV1_BIAS0] <= s_axi_wdata;
            else if (waddr_idx >= IDX_CONV1_RESC0 && waddr_idx < IDX_FINAL_SCORES0)
                rescale_regs[waddr_idx - IDX_CONV1_RESC0] <= s_axi_wdata[15:0];
        end
    end

    // STATUS register: bit0 = done_cnn. top_cnn.sv declares done_cnn as
    // `output reg`, driven inside its OWN clocked always block -- it's
    // already a real flip-flop, same clock domain as this wrapper
    // (s_axi_aclk feeds top_cnn.clk directly, no CDC). Re-registering it
    // here (status_reg, previously 32 bits for 1 useful bit) was a pure
    // duplicate: same value, one wire hop away, no correctness reason to
    // buffer it again. Removed entirely -- the read mux below now reads
    // done_cnn_w directly.

    // s_axi_bresp is never assigned anything but OKAY -- no error path
    // exists on this slave -- so it's a constant, not a register.
    assign s_axi_bresp = 2'b00;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_bvalid <= 0;
        end else if (write_en && ~s_axi_bvalid) begin
            s_axi_bvalid <= 1;
        end else if (s_axi_bready && s_axi_bvalid) begin
            s_axi_bvalid <= 0;
        end
    end

    // ---- AXI4-Lite read channel (standard 2-phase handshake) ------------
    // CHANGED: araddr_reg removed -- it was written on every read-address
    // handshake but never read anywhere in this module. raddr_idx_reg
    // (below) is the register the read mux actually uses; araddr_reg was
    // pure dead weight (9 bits, no functional role).
    reg [6:0] raddr_idx_reg;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 0;
        end else begin
            if (~s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1;
            end else begin
                s_axi_arready <= 0;
            end
        end
    end

    // s_axi_rresp is never assigned anything but OKAY -- no error path
    // exists on this slave -- so it's a constant, not a register.
    assign s_axi_rresp = 2'b00;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 0;
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
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 0;
        end
    end

    always @(*) begin
        // STATUS and FINAL_SCORES read directly from top_cnn's own
        // registered outputs (done_cnn_w/final_scores_w) -- no wrapper-
        // local mirror needed, since top_cnn already registers both
        // internally in the same clock domain as this module. CONTROL/
        // SHIFTS/RESCALE live in their own narrow registers (control_reg/
        // shifts_reg/rescale_regs) instead of a uniform 32-bit regs[]
        // array -- zero-extended back to 32 bits here on readback,
        // matching the register map's documented word layout exactly.
        if (raddr_idx_reg == IDX_STATUS)
            s_axi_rdata = {31'd0, done_cnn_w};
        else if (raddr_idx_reg == IDX_CONTROL)
            s_axi_rdata = {31'd0, control_reg};
        else if (raddr_idx_reg == IDX_SHIFTS)
            s_axi_rdata = {27'd0, shifts_reg};
        else if (raddr_idx_reg >= IDX_CONV1_BIAS0 && raddr_idx_reg < IDX_CONV1_RESC0)
            s_axi_rdata = bias_regs[raddr_idx_reg - IDX_CONV1_BIAS0];
        else if (raddr_idx_reg >= IDX_CONV1_RESC0 && raddr_idx_reg < IDX_FINAL_SCORES0)
            s_axi_rdata = {16'd0, rescale_regs[raddr_idx_reg - IDX_CONV1_RESC0]};
        else if (raddr_idx_reg >= IDX_FINAL_SCORES0 && raddr_idx_reg < IDX_FINAL_SCORES0 + 10)
            s_axi_rdata = final_scores_w[raddr_idx_reg - IDX_FINAL_SCORES0];
        else
            s_axi_rdata = 32'd0;
    end

endmodule