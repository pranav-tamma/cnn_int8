module top_cnn#(parameter cr1 = 16, cc1 = 8, kr1 = 9, kc1 = 8, rr1 = 26, rc1 = 26, pr1 = 26, pc1 = 26, fcr1 = 400, fcc1 =10, ACT_WIDTH = 8 )(
    input clk,
    input rst,
    output reg done_cnn,
    // --- a_matrix: banked BRAM interface -------------------------------
    // Was: input signed [31:0] a_matrix [0:675][0:15]  (346,112 bits as
    // literal top-level ports -- the actual cause of this design's
    // Bonded IOB explosion in synthesis).
    // Now: a_matrix lives in 16 external BRAMs (one per im2col
    // column/reduction-lane), each 676 words x 32 bits, fed by an AXI
    // BRAM Controller on the other port. top_cnn only sees a plain
    // synchronous read interface: present a row address, get all 16
    // columns back the following cycle. Banking across 16 small BRAMs
    // (rather than one 676x16-wide BRAM) is what preserves today's
    // "one row per cycle" LOAD_TILE1 timing -- a single BRAM port can
    // only return 1 word/cycle, but 16 independent ports can each
    // return 1 word/cycle simultaneously, for 16 total.
    output reg [9:0] a_matrix_addr,             // 0..675, shared across all 16 banks
    input signed [7:0] a_matrix_rdata [0:15],   // int8 pixel/activation input; arrives ONE cycle after a_matrix_addr is presented
    // --- kernel_matrix1: banked BRAM interface (8 banks x 16 words) ---
    // Only 8 banks -- conv1 genuinely only has 8 real output channels,
    // unlike kernel_matrix2/fc_weights below which need all 16 banks
    // (accessed across two c2=8 systolic passes each).
    output reg [3:0] kernel_matrix1_addr,
    input signed [7:0] kernel_matrix1_rdata [0:7],   // int8 weights
    // --- kernel_matrix2: banked BRAM interface (16 banks x 80 words) ---
    output reg [6:0] kernel_matrix2_addr,
    input signed [7:0] kernel_matrix2_rdata [0:15],  // int8 weights
    // Biases stay 32-bit: quantized at the accumulator's scale
    // (S_weight * S_activation), not the int8 output scale -- see relu.sv.
    input signed [31:0] conv1_bias [0:15],
    input signed [31:0] conv2_bias [0:15],
    // Per-layer requantization shifts (see relu.sv) -- calibrated in
    // Python from your chosen weight/activation quantization scales.
    // Runtime ports, not localparams, so you can retune without
    // resynthesizing while you calibrate.
    input [4:0] relu1_shift,
    input [4:0] relu2_shift,
    // --- fc_weights: banked BRAM interface (16 banks x 400 words) ------
    output reg [8:0] fc_weights_addr,
    input signed [7:0] fc_weights_rdata [0:15],  // int8 weights
    input signed [31:0] fc_bias [0:fcc1-1],
    output reg signed [31:0] final_scores [0:fcc1-1]
);

wire signed [31:0] results [0:7][0:15];       // r1 shrunk 16->8, c2 grown 8->16: every layer now fits in ONE systolic pass (conv1: 8 real of 16 cols used; conv2: all 16 real; FC: 10 real of 16)
wire signed [7:0] relu_results1 [0:7][0:15];  // int8; only columns 0:7 are conv1's real channels, 8:15 unused (harmless -- never read downstream)
wire signed [7:0] relu_results2 [0:7][0:15]; // int8; spatial-row dim shrunk 16->8 to match conv2_accumulator's new shape
wire signed [ACT_WIDTH-1:0] pool1_result [0:7][0:12][0:12];
wire signed [ACT_WIDTH-1:0] pool2_result [0:15][0:4][0:4];
wire done_compute;
wire done_pool1;
wire done_pool2;

reg [8:0] tile_count;
reg [8:0] k_tile;
reg signed [7:0] kernel_matrix [0:15][0:15];  // int8 weights; [reduction-row][output-column]. Reduction-row (16, = c1/r2) UNCHANGED -- only output-column (c2) grew back to 16, now covering every layer's real channel count in one pass. For conv1, columns 8:15 are zero-padded (never real data).

// ---------------------------------------------------------------------
// Line buffers replace the old full-feature-map pool1_input/pool2_input
// arrays (173,056 / 61,952 registers). A 2x2 max-pool only ever needs
// the current row and its partner row, so these hold exactly a handful
// of rows:
//   row_buf1: 8 * 4 * 26 * 16b = 13,312 bits  (was 173,056)
// row_buf1 is 4 rows deep, not 2, same reasoning as row_buf2: a tile
// finishing an odd row R (triggering its pool) can, in that same cycle,
// also start writing row R+1 -- which shares a mod-2 slot with row R-1,
// the row the pool is about to read. With 4 slots that collision can't
// happen (R-1 and R+1 always land 2 apart, never the same slot mod 4).
//   row_buf2: 16 * 4 * 11 * 16b = 11,264 bits (was 61,952)
// row_buf2 is 4 rows deep: pool2's row width (11) is smaller than the
// 16-element write-tile, so a single tile write can span 3 distinct
// rows. With only 2 slots, row R and row R+2 alias to the same slot;
// 4 slots (row%4) keeps R and R+2 in different slots.
// WRITE_TILE1/WRITE_TILE2 write into row%4 of the buffer as tile results
// come in. As soon as an odd row completes -- meaning it and its even
// partner are both sitting in the buffer -- that pair is copied out via
// PREP_POOL1/PREP_POOL2 and pooled via START_POOL1/START_POOL2. Nothing
// ever holds the whole image at once.
//
// SERIALIZATION: every state that moves data (tile loads, row-buffer
// writes, pooling-input prep, FC input/weight loads) now does ONE
// spatial row (or one channel, for the pooling-prep copies) per clock
// cycle via the shared `scnt` counter below, instead of the whole
// 128-512-element operation unrolled into a single cycle. This is the
// same fix already applied earlier to pool1_input/conv2_input/etc: a
// synthesis tool can't map "hundreds of simultaneous reads/writes" onto
// real hardware without either massive fanout/routing (flip-flops +
// giant muxes) or an outright failure to infer RAM where one is wanted.
// The compute datapath itself (a_tile/kernel_matrix feeding the
// systolic array every cycle, results/conv2_accumulator/fc_accumulator
// accumulation) is untouched -- those are inherently parallel register
// files tightly coupled to the compute unit, not data-movement/buffer
// staging, and were never flagged by synthesis.
// ---------------------------------------------------------------------
reg signed [ACT_WIDTH-1:0] row_buf1 [0:7][0:3][0:25];
reg signed [ACT_WIDTH-1:0] row_buf2 [0:15][0:3][0:10];
reg [8:0] out_row1, out_row2;
reg [4:0] next_state_after_pool1, next_state_after_pool2;
reg [4:0] completed_row1, completed_row2;
reg row_hit1, row_hit2;

reg signed [31:0] current_conv_bias [0:15];
reg signed [7:0] a_tile [0:7][0:15];  // int8 activations/pixels. Row dim (r1) shrunk 16->8; reduction/column dim (c1) UNCHANGED at 16.
reg signed [31:0] conv1_result_snapshot [0:7][0:15];
reg signed [31:0] conv2_accumulator [0:7][0:15];  // spatial-row dim shrunk 16->8 to match the systolic array's new r1
reg signed [31:0] fc_accumulator [0:15];
reg start_pool1;
reg start_pool2;
reg start_compute;

// conv2_input [0:120][0:79] and fc_input [0:0][0:399] stay removed --
// both were pure index-remapped views of pool1_result/pool2_result,
// which are already valid, registered, and stable by the time conv2/FC
// need them. LOAD_TILE2 and LOAD_FC compute the equivalent im2col/flatten
// index directly and read pool1_result/pool2_result in place.

reg[4:0] state;
localparam IDLE          = 0;
localparam LOAD_TILE1     = 1;
localparam START_COMPUTE1 = 2;
localparam WAIT_COMPUTE1  = 3;
localparam WRITE_TILE1   = 4;
localparam START_POOL1   = 5;
localparam WAIT_POOL1    = 6;
localparam LOAD_TILE2    = 7;
localparam START_COMPUTE2 = 8;
localparam WAIT_COMPUTE2 = 9;
localparam WRITE_TILE2   = 10;
localparam START_POOL2   = 11;
localparam WAIT_POOL2    = 12;
localparam LOAD_FC       = 13;
localparam START_COMPUTE_FC = 14;
localparam WAIT_COMPUTE_FC = 15;
localparam WRITE_FC      = 16;
localparam DONE          = 17;
localparam PREP_POOL2    = 18;
localparam PREP_POOL1    = 19;

integer i,j,k,row,col,index,feature,ch,kr,kc,fc_k,fc_i,fc_j;

// Shared serialization counter -- reused sequentially across
// LOAD_TILE1/WRITE_TILE1/PREP_POOL1/LOAD_TILE2/WRITE_TILE2/PREP_POOL2/
// LOAD_FC. None of these states are ever active at the same time, so a
// single counter is safe to reuse rather than needing a separate one
// per state.
reg [4:0] scnt;
// jcnt: sub-counter used by LOAD_TILE2 and LOAD_FC to serialize their
// reads of pool1_result/pool2_result down to ONE element per cycle.
// Those reads previously pulled 16 different (channel,row,col) addresses
// out of a single unified array in the same clock edge -- that's 16
// simultaneous ports into one array regardless of how the address is
// computed, which a real BRAM (or even a clean register-file mux) can't
// do cheaply; it was almost certainly the dominant remaining LUT driver
// after pooling.sv's own write-side fix. Spreading it to 1 read/cycle
// trades throughput (more cycles) for a real reduction in simultaneous
// mux/decode width.
reg [4:0] jcnt;
reg a_matrix_req_valid;       // outstanding a_matrix BRAM read whose data lands THIS cycle?
reg tile2_km_loaded;         // phase tracker for LOAD_TILE2 (see comment there)
reg fc_weights_req_valid;     // same, for fc_weights (used in LOAD_FC)
// c2 was shrunk from 16 to 8 (see systolic_array instantiation below --
// 256 PEs needs more DSPs than this board has, even in the best case of
// 1 DSP/PE). Conv1 only ever had 8 real output channels, so it needs no
// restructuring. Conv2 (16 channels) and FC (10, padded to 16) each need
// TWO systolic passes to cover their real channel count -- these two
// bits track which half is currently in flight.
// c_half2/c_halfF (conv2/FC channel-half splitting) are no longer
// needed: with c2=16, both conv2 (16 real channels) and FC (10 real,
// padded to 16) now complete in a single systolic pass.

systolic_array#(.r1(8), .c1(16), .r2(16), .c2(16)) conv (
    .a_matrix(a_tile),
    .b_matrix(kernel_matrix),
    .results(results),
    .done_compute(done_compute),
    .clk(clk),
    .rst(rst),
    .start_compute(start_compute)
);

// relu.sv applies its own >>>8 internally, so both instances below get
// the raw, unshifted accumulator -- rel1 gets the systolic array's raw
// results, rel2 gets the raw conv2_accumulator. (A previous version of
// this file added an extra external >>>8 before rel2, guessing that
// relu had no internal shift -- it does, so that was a double-shift bug.)
// rel1 reads a STABLE snapshot of the systolic array's output, captured
// the instant done_compute fires (see WAIT_COMPUTE1) -- not the live
// `results` wire directly. `results` is only valid while start_compute
// is still asserted, or for exactly one cycle after it drops (pe2.sv
// resets its accumulator to 0 on `!start_compute` every cycle it's
// low). WRITE_TILE1 now takes 16 cycles to write row_buf1 one row at a
// time; reading the live wire across all 16 would only get row 0
// right and zero out rows 1-15. conv2's path never had this problem
// since relu_results2 is already sourced from conv2_accumulator, a
// real captured register -- this makes conv1's path the same shape.
relu#(.r1(8), .c1(16)) rel1 (
    .results(conv1_result_snapshot),
    .relu_results(relu_results1),
    .conv_bias(current_conv_bias),
    .shift_amount(relu1_shift)
);

// rel2 deliberately stays c1=16, unchanged: it still operates on the
// FULL, unshrunk conv2_accumulator, recomputed once per completed tile
// (at WRITE_TILE2), not once per c2=8 systolic pass. conv2 needs two
// systolic passes (c_half2 0 and 1) to cover its 16 real channels, but
// they both write into the same full-width conv2_accumulator before
// relu2 ever sees it -- see WAIT_COMPUTE2 below.
relu#(.r1(8), .c1(16)) rel2 (
    .results(conv2_accumulator),
    .relu_results(relu_results2),
    .conv_bias(current_conv_bias),
    .shift_amount(relu2_shift)
);

// row_buf1 is 4 slots deep (see declaration above). completed_row1 is
// always odd, so which pair of slots is live only depends on bit 1 of
// completed_row1 (mod 4 is always 1 or 3 for an odd number) -- PREP_POOL1
// below copies the right pair into this small fixed buffer with a plain
// if/else, same reasoning as PREP_POOL2's case for row_buf2.
reg signed [ACT_WIDTH-1:0] row_pair1_reg [0:7][0:1][0:25];

pooling#(.filters(8), .r1(26), .c1(26), .WIDTH(ACT_WIDTH)) pool1 (
    .clk(clk),
    .rst(rst),
    .start_pooling(start_pool1),
    .done_pooling(done_pool1),
    .row_pair(row_pair1_reg),
    .out_row(out_row1),
    .pooling_results(pool1_result)
);

// row_buf2 is 4 slots deep (see declaration above), but the pooling
// module still just wants 2 consecutive rows. Which 2 of the 4 slots
// are the live pair depends on completed_row2. This used to be a
// continuously-assigned mux that dynamically indexed row_buf2 by
// completed_row2 -- but row_buf2 is *also* written elsewhere using a
// separate variable index (row%4), and two independently-variable
// indices into the same array is exactly the shape that made the
// synthesis tool try (and hang) inferring a RAM for the original
// full-frame pool1_input/pool2_input buffers. completed_row2 only ever
// takes 5 discrete values (1,3,5,7,9), so PREP_POOL2 below copies the
// right pair into this small fixed buffer with a plain case statement
// instead -- an ordinary, unambiguous mux, not an arbitrary-address read.
reg signed [ACT_WIDTH-1:0] row_pair2_reg [0:15][0:1][0:10];

pooling#(.filters(16),.r1(11), .c1(11), .WIDTH(ACT_WIDTH)) pool2 (
    .clk(clk),
    .rst(rst),
    .start_pooling(start_pool2),
    .done_pooling(done_pool2),
    .row_pair(row_pair2_reg),
    .out_row(out_row2),
    .pooling_results(pool2_result)
);

always@(posedge clk) begin
    if(rst) begin
        done_cnn <= 0;
        state <=IDLE;
        tile_count<=0;
        k_tile <=0;
        start_compute<=0;
        start_pool1<=0;
        start_pool2<=0;
        scnt <= 0;
        jcnt <= 0;
        a_matrix_req_valid <= 0;
        a_matrix_addr <= 0;
        kernel_matrix1_addr <= 0;
        tile2_km_loaded <= 0;
        kernel_matrix2_addr <= 0;
        fc_weights_req_valid <= 0;
        fc_weights_addr <= 0;
    end
    else begin
        case(state)

        IDLE : state<= LOAD_TILE1;

        LOAD_TILE1 : begin
            // a_tile now only has 8 rows (r1=8), but kernel_matrix1 still
            // needs all 16 reduction-rows loaded (c1/r2 UNCHANGED at 16)
            // -- these two loads are no longer the same length, so they
            // can't share one counter's range the way they used to when
            // both happened to be 16. scnt now runs the full 0..17 (18
            // cycles -- one more than before).
            //
            // CAPTURE INDEX IS scnt-2, NOT scnt-1: empirically confirmed
            // (via a full 16-row weight comparison against golden) that
            // this BRAM read pipeline has a genuine 2-cycle round trip
            // once the address starts changing every cycle -- the
            // address register itself takes 1 cycle to update, and the
            // testbench's registered read takes a further cycle to
            // reflect that update. scnt-1 was only ever "getting away
            // with it" for the very first captured row, because the
            // address had been sitting idle (unchanging) long enough
            // beforehand for the data to have already caught up by
            // coincidence -- every row after that was silently one step
            // behind, which corner-position spot-checks never exposed
            // since those specific positions happened to multiply
            // against near-zero data.
            // 85 tiles now (was 43): 676 spatial positions / 8 per tile,
            // ceil'd (84*8=672, tile 84 has only 4 valid rows: 672-675).
            if (tile_count<85) begin
                // kernel_matrix1: unconditional every cycle -- always 16
                // real reduction-rows to load, no partial/padding case.
                if (scnt >= 2) begin
                    for(j=0;j<8;j=j+1)
                        kernel_matrix[scnt-2][j] <= kernel_matrix1_rdata[j];
                    for(j=8;j<16;j=j+1)
                        kernel_matrix[scnt-2][j] <= 0;   // conv1 only has 8 real output channels -- pad columns 8:15
                end

                // a_tile: same 2-cycle model, only 8 real rows -- gated
                // by a_matrix_req_valid, which now stays asserted one
                // cycle longer (through scnt==8, not just scnt<8) so the
                // last real row's capture (scnt=9, index 7) still fires.
                // The extra `scnt>=2` guard prevents scnt-2 from
                // underflowing (scnt is unsigned) on the two cycles
                // before any real capture is due.
                if (a_matrix_req_valid && scnt>=2) begin
                    for(j=0;j<16;j=j+1)
                        a_tile[scnt-2][j] <= a_matrix_rdata[j];
                end

                if (scnt == 0)
                    current_conv_bias <= conv1_bias;

                if (scnt == 17) begin
                    scnt <= 0;
                    a_matrix_req_valid <= 0;
                    state <= START_COMPUTE1;
                end
                else begin
                    if (scnt < 16)
                        kernel_matrix1_addr <= scnt;
                    if (scnt < 8) begin
                        a_matrix_addr <= tile_count*8 + scnt;
                        a_matrix_req_valid <= 1;
                    end
                    else if (scnt == 8) begin
                        a_matrix_req_valid <= 1;   // hold one extra cycle so scnt=9's capture (last real row) still fires
                    end
                    else begin
                        a_matrix_req_valid <= 0;
                    end
                    scnt <= scnt + 1;
                end
            end
            else
            state<= IDLE;
        end

        START_COMPUTE1 : begin
             start_compute <=1;
             state <= WAIT_COMPUTE1; 
        end


        WAIT_COMPUTE1 : begin
        if(done_compute) begin
            start_compute <=0;
            for(index=0;index<128;index=index+1) begin
                conv1_result_snapshot[index/16][index%16] <= results[index/16][index%16];
            end
            state<= WRITE_TILE1;
        end
        else
        state<=WAIT_COMPUTE1;
        end

        WRITE_TILE1 : begin
            // Hit-detection scan now only needs 8 elements (i=0..7),
            // matching the new tile size, instead of 16.
            if (scnt == 0) begin
                row_hit1 <= 0;
                completed_row1 <= 0;
                for(i=0;i<8;i=i+1) begin
                    row = ((tile_count*8)+i)/26;
                    col = ((tile_count*8)+i)%26;
                    if(row<26 && col==25 && row[0]==1'b1) begin
                        row_hit1 <= 1;
                        completed_row1 <= row;
                    end
                end
            end

            row = ((tile_count*8)+scnt)/26;
            col = ((tile_count*8)+scnt)%26;
            if(row<26) begin
                for(j=0;j<8;j=j+1)
                    row_buf1[j][row[1:0]][col] <= relu_results1[scnt][j];
            end

            if (scnt == 7) begin
                scnt <= 0;
                if(tile_count==84) begin
                    tile_count<=0;
                    if(row_hit1) begin
                        next_state_after_pool1 <= LOAD_TILE2;
                        state <= PREP_POOL1;
                    end
                    else begin
                        state <= LOAD_TILE2;
                    end
                end
                else begin
                    tile_count <= tile_count + 1;
                    if(row_hit1) begin
                        next_state_after_pool1 <= LOAD_TILE1;
                        state <= PREP_POOL1;
                    end
                    else begin
                        state <= LOAD_TILE1;
                    end
                end
            end
            else begin
                scnt <= scnt + 1;
            end
        end

        PREP_POOL1 : begin
            // One channel (52 elements: 26 cols x 2 rows) per cycle
            // instead of all 8 channels (416 elements) at once.
            // completed_row1 is always odd, so its slot-pair only
            // depends on bit 1 (mod 4 is always 1 or 3 for an odd
            // number): mod4==1 -> slots(0,1), mod4==3 -> slots(2,3).
            if (completed_row1[1] == 1'b0) begin
                for(k=0;k<26;k=k+1) begin
                    row_pair1_reg[scnt][0][k] <= row_buf1[scnt][0][k];
                    row_pair1_reg[scnt][1][k] <= row_buf1[scnt][1][k];
                end
            end
            else begin
                for(k=0;k<26;k=k+1) begin
                    row_pair1_reg[scnt][0][k] <= row_buf1[scnt][2][k];
                    row_pair1_reg[scnt][1][k] <= row_buf1[scnt][3][k];
                end
            end

            if (scnt == 7) begin
                scnt <= 0;
                state <= START_POOL1;
            end
            else begin
                scnt <= scnt + 1;
            end
        end

        START_POOL1 :begin
             start_pool1<=1;
             out_row1 <= completed_row1 >> 1;
             state<= WAIT_POOL1;
        end

        WAIT_POOL1 : begin
             if(done_pool1) begin
                start_pool1<=0;
                state<= next_state_after_pool1;
             end
            else
            state<= WAIT_POOL1;
            end

        LOAD_TILE2 : begin
            current_conv_bias <= conv2_bias;
            // kernel_matrix2 still needs all 16 reduction-rows loaded
            // per k_tile (c1/r2 UNCHANGED at 16), but a_tile only has 8
            // real spatial rows now (r1=8) -- these are no longer the
            // same length, so (like LOAD_TILE1) they can't share one
            // counter's range. tile2_km_loaded explicitly tracks which
            // of the two phases we're in, since jcnt is reused for both
            // (phase A: kernel_matrix2's 16-row load; phase B: a_tile's
            // inner per-row loop) but they run at different times, not
            // simultaneously.
            // 16 tiles now (was 8): 121 spatial positions / 8 per tile,
            // ceil'd (15*8=120, tile 15 has only 1 valid row: index 120).
            if(tile_count<16) begin
                if (!tile2_km_loaded) begin
                    // Phase A: kernel_matrix2, unconditional every
                    // k_tile -- no more c_half2, this ONE pass now
                    // covers all 16 real conv2 output channels.
                    // CAPTURE INDEX IS jcnt-2, NOT jcnt-1 -- same 2-cycle
                    // BRAM read latency correction as LOAD_TILE1 (see
                    // comment there for the full explanation). Loop now
                    // runs through jcnt==17 (was 16).
                    if (jcnt >= 2) begin
                        for(j=0;j<16;j=j+1)
                            kernel_matrix[jcnt-2][j] <= kernel_matrix2_rdata[j];
                    end
                    if (jcnt == 17) begin
                        jcnt <= 0;
                        tile2_km_loaded <= 1;
                    end
                    else begin
                        if (jcnt < 16)
                            kernel_matrix2_addr <= k_tile*16 + jcnt;
                        jcnt <= jcnt + 1;
                    end
                end
                else begin
                    // Phase B: fill a_tile's 8 real spatial rows (scnt
                    // 0..7), one pool1_result element per (scnt,jcnt)
                    // sub-cycle -- same structure as before, just scnt
                    // now only runs 0..7 instead of 0..15.
                    index   = tile_count*8 + scnt;   // spatial position, 0..120 valid (121..127 is last-tile padding)
                    row     = index / 11;
                    col     = index % 11;
                    feature = k_tile*16 + jcnt; // reduction-dim position, 0..79 (72..79 is zero padding)
                    if(index<121 && feature<72) begin
                        ch = feature / 9;
                        kr = (feature % 9) / 3;
                        kc = feature % 3;
                        a_tile[scnt][jcnt] <= pool1_result[ch][row+kr][col+kc];
                    end
                    else begin
                        a_tile[scnt][jcnt] <= 0;
                    end

                    if (jcnt == 15) begin
                        jcnt <= 0;
                        if (scnt == 7) begin
                            scnt <= 0;
                            tile2_km_loaded <= 0;   // reset for the NEXT k_tile's phase A
                            state <= START_COMPUTE2;
                        end
                        else begin
                            scnt <= scnt + 1;
                        end
                    end
                    else begin
                        jcnt <= jcnt + 1;
                    end
                end
            end
        end

        START_COMPUTE2: begin
            start_compute<=1;
            state<= WAIT_COMPUTE2;
        end

        WAIT_COMPUTE2 : begin
            if(done_compute) begin
                start_compute<=0;
                // results is now [8][16] -- all 16 of conv2's real
                // output channels computed in ONE systolic pass, no more
                // channel-half splitting needed. conv2_accumulator
                // matches results' shape exactly now (both [0:7][0:15]),
                // so this is a straight accumulate, no offset.
                if(k_tile == 0)
                for(index=0;index<128;index=index+1) begin
                  conv2_accumulator[index/16][index%16] <= results[index/16][index%16];
                  end
                else
                for(index=0;index<128;index=index+1) begin
                  conv2_accumulator[index/16][index%16] <= conv2_accumulator[index/16][index%16]+results[index/16][index%16];
                  end
                if(k_tile == 4) begin
                    k_tile <= 0;
                    // conv2_accumulator now has all 16 real channels for
                    // this spatial tile after just one pass through the
                    // 5 k_tiles -- no second channel-half pass needed.
                    state <= WRITE_TILE2;
                    end
                    else begin
                        k_tile <= k_tile + 1;
                        state <= LOAD_TILE2;
                        end
            end
            else
            state<=WAIT_COMPUTE2;
        end

        WRITE_TILE2 : begin
            // Hit-detection scan now only needs 8 elements (i=0..7),
            // matching the new tile size, instead of 16.
            if (scnt == 0) begin
                row_hit2 <= 0;
                completed_row2 <= 0;
                for(i=0;i<8;i=i+1) begin
                    row = ((tile_count*8)+i)/11;
                    col = ((tile_count*8)+i)%11;
                    if(row<11 && col==10 && row[0]==1'b1) begin
                        row_hit2 <= 1;
                        completed_row2 <= row;
                    end
                end
            end

            row = ((tile_count*8)+scnt)/11;
            col = ((tile_count*8)+scnt)%11;
            if(row<11) begin
                for(j=0;j<16;j=j+1)
                    row_buf2[j][row[1:0]][col] <= relu_results2[scnt][j];
            end

            if (scnt == 7) begin
                scnt <= 0;
                if(tile_count==15) begin
                    tile_count<=0;
                    for(index=0;index<128;index=index+1) begin
                      conv2_accumulator[index/16][index%16] <= 0;
                      end
                    if(row_hit2) begin
                        next_state_after_pool2 <= LOAD_FC;
                        state <= PREP_POOL2;
                    end
                    else begin
                        state <= LOAD_FC;
                    end
                end
                else begin
                    tile_count <= tile_count+1;
                    if(row_hit2) begin
                        next_state_after_pool2 <= LOAD_TILE2;
                        state <= PREP_POOL2;
                    end
                    else begin
                        state <= LOAD_TILE2;
                    end
                end
            end
            else begin
                scnt <= scnt + 1;
            end
        end

        PREP_POOL2 : begin
            // One channel (22 elements: 11 cols x 2 rows) per cycle
            // instead of all 16 channels (352 elements) at once.
            // completed_row2 only ever equals 1,3,5,7, or 9 here (the
            // odd row of whichever pair just finished) -- a plain
            // enumerable case, not a dynamically-indexed array read.
            case(completed_row2)
                5'd1, 5'd5, 5'd9: begin
                    row_pair2_reg[scnt][0] <= row_buf2[scnt][0];
                    row_pair2_reg[scnt][1] <= row_buf2[scnt][1];
                end
                5'd3, 5'd7: begin
                    row_pair2_reg[scnt][0] <= row_buf2[scnt][2];
                    row_pair2_reg[scnt][1] <= row_buf2[scnt][3];
                end
                default: begin end
            endcase

            if (scnt == 15) begin
                scnt <= 0;
                state <= START_POOL2;
            end
            else begin
                scnt <= scnt + 1;
            end
        end

        START_POOL2 : begin
            start_pool2<=1;
            out_row2 <= completed_row2 >> 1;
            state <= WAIT_POOL2;
        end

        WAIT_POOL2 : begin
            if(done_pool2) begin
                start_pool2<=0;
                state <= next_state_after_pool2;
            end
            else
            state<= WAIT_POOL2;
        end

        LOAD_FC : begin
            // Phase 1 (jcnt 0..15): serially fill a_tile[0][jcnt] from
            // pool2_result, one element per cycle, instead of reading 16
            // different (channel,row,col) addresses out of pool2_result
            // in the same edge. Runs once per tile_count, before phase 2.
            // Phase 2 (scnt 0..16, unchanged -- still tiles fc_weights'
            // 16-deep reduction dimension, independent of r1): zero rows
            // 1..7 (a_tile only has 8 real rows now) and load fc_weights,
            // now directly into all 16 output-columns (no more c_halfF
            // offset -- FC's 10 real classes all fit in one pass, same
            // as conv2 no longer needing a channel-half split). Row 0 is
            // skipped in phase 2's zeroing since phase 1 already filled
            // it with real data.
            if(tile_count<25) begin
                if (jcnt < 16) begin
                    index = tile_count*16 + jcnt;   // 0..399, flat index into the 400-length FC input (exact, 25*16)
                    fc_k = index / 25;
                    fc_i = (index % 25) / 5;
                    fc_j = index % 5;
                    a_tile[0][jcnt] <= pool2_result[fc_k][fc_i][fc_j];
                    jcnt <= jcnt + 1;   // becomes 16 after the jcnt==15 step, which is what moves us into phase 2 below
                end
                else begin
                    // CAPTURE INDEX IS scnt-2, NOT scnt-1 -- same 2-cycle
                    // BRAM read latency correction as LOAD_TILE1/
                    // LOAD_TILE2 (see LOAD_TILE1 for the full
                    // explanation). Loop now runs through scnt==17 (was
                    // 16). fc_weights_req_valid never has an early
                    // cutoff the way a_matrix's did (all 16 rows here
                    // are real, no partial/padding case), so it doesn't
                    // need the extra one-cycle hold a_matrix needed.
                    if (scnt < 8 && scnt > 0) begin
                        for(j=0;j<16;j=j+1)
                            a_tile[scnt][j] <= 0;
                    end

                    if (fc_weights_req_valid && scnt>=2) begin
                        for(j=0;j<16;j=j+1)
                            kernel_matrix[scnt-2][j] <= fc_weights_rdata[j];
                    end

                    if (scnt == 17) begin
                        scnt <= 0;
                        jcnt <= 0;   // reset so the NEXT tile_count's LOAD_FC visit starts phase 1 fresh
                        fc_weights_req_valid <= 0;
                        state<= START_COMPUTE_FC;
                    end
                    else begin
                        if (scnt < 16)
                            fc_weights_addr <= tile_count*16 + scnt;
                        fc_weights_req_valid <= 1;
                        scnt <= scnt + 1;
                    end
                end
            end
        end

        START_COMPUTE_FC : begin
            start_compute<=1;
            state <= WAIT_COMPUTE_FC;
        end

        WAIT_COMPUTE_FC : begin
           if(done_compute) begin
            start_compute<=0;
            state<= WRITE_FC;
           end
           else
           state<= WAIT_COMPUTE_FC;
        end

        WRITE_FC : begin
            // results is now [8][16] -- FC's systolic pass covers all 10
            // real classes directly (columns 10:15 are unused padding,
            // simply never read), no more channel-half splitting needed.
            if(tile_count==0) begin
                for(j=0;j<10;j=j+1) begin
                fc_accumulator[j] <= results[0][j];
                end
                end
            else begin
                for(j=0;j<10;j=j+1) begin
                fc_accumulator[j] <= fc_accumulator[j] + results[0][j];
                end
                end
            if(tile_count==24) begin
                tile_count <= 0;
                // fc_accumulator's tile-24 update (above) hasn't landed
                // yet on THIS edge (nonblocking assignment) -- computing
                // final_scores here would read the pre-tile-24 value,
                // silently dropping the last tile's contribution. DONE
                // (entered next cycle) computes final_scores instead,
                // by which point fc_accumulator is fully correct.
                state<= DONE;
            end
            else begin
                tile_count <= tile_count + 1;
                state <= LOAD_FC;
                end
        end

        DONE: begin
            for(j=0;j<10;j=j+1)
            final_scores[j] <= (fc_accumulator[j] + fc_bias[j]) >>> 8;
            done_cnn <=1;
        end
        endcase
end
end
endmodule