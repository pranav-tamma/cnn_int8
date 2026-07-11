module pooling #(parameter filters = 8, r1 = 26, c1 = 26, WIDTH = 32)(
    input clk,
    input rst,
    input start_pooling,
    output reg done_pooling,
    // Only a 2-row window is ever needed for a 2x2 max-pool. The caller
    // (top_cnn) keeps a small [filters][2][c1] line buffer and pulses
    // start_pooling once a row-pair is complete, telling us which output
    // row it maps to via out_row. This replaces reading/storing the
    // entire [filters][r1][c1] feature map at once.
    input signed [WIDTH-1:0] row_pair [0:filters-1][0:1][0:c1-1],
    input [$clog2((r1/2)>0 ? (r1/2) : 1):0] out_row,
    output reg signed [WIDTH-1:0] pooling_results [0:filters-1][0:(r1/2)-1][0:(c1/2)-1]
);

// Single flattened loop over idx = ch*(c1/2)+k instead of a nested
// ch/k loop -- decoded via / and % ((c1/2) is a compile-time constant
// here). Same hardware either way; purely a "no nested for loops"
// style requirement.
integer idx;
integer ch, k;
reg signed [WIDTH-1:0] m0, m1;

// Pooled row, computed once per trigger -- this part was never the
// resource problem: it's a small, fixed cost (filters * c1/2 compares).
reg signed [WIDTH-1:0] result_row [0:filters-1][0:(c1/2)-1];

reg start_pooling_d;
reg [$clog2((r1/2)>0 ? (r1/2) : 1):0] out_row_d;

// Stage 1: compute the pooled row, and pipeline start_pooling/out_row
// by one cycle so the write stage below reads a result_row that has
// actually finished updating (nonblocking assignments in the block
// below would otherwise still see last cycle's stale value).
always @(posedge clk) begin
    if (rst) begin
        start_pooling_d <= 0;
    end
    else begin
        start_pooling_d <= start_pooling;
        out_row_d <= out_row;
        if (start_pooling) begin
            for (idx = 0; idx < filters*(c1/2); idx = idx + 1) begin
                ch = idx / (c1/2);
                k  = idx % (c1/2);
                m0 = (row_pair[ch][0][2*k] > row_pair[ch][0][2*k+1]) ? row_pair[ch][0][2*k] : row_pair[ch][0][2*k+1];
                m1 = (row_pair[ch][1][2*k] > row_pair[ch][1][2*k+1]) ? row_pair[ch][1][2*k] : row_pair[ch][1][2*k+1];
                result_row[ch][k] <= (m0 > m1) ? m0 : m1;
            end
        end
    end
end

// Stage 2 (one cycle later): route result_row into the correct row of
// pooling_results using a STATIC, per-row generated write-enable
// instead of a runtime-variable array index. The original
// `pooling_results[ch][out_row][k] <= ...` used out_row -- a runtime
// input value -- as an array index inside a loop over every (ch,k)
// combination. That forces synthesis to build a full write-address
// decoder repeated at every one of those combinations (filters * c1/2
// of them), rather than one shared decoder.
//
// This generate block creates exactly ONE always-block PER INDIVIDUAL
// OUTPUT REGISTER (filters * (r1/2) * (c1/2) of them total), each doing
// nothing but a single row-select compare (out_row_d == GR, a simple
// binary-to-one-hot compare) and a single assignment -- no runtime for
// loop anywhere inside the generate loop, so there's no loop nested
// inside a loop. genvar/localparam-derived indices are resolved at
// elaboration time, not synthesized as loop hardware, so this produces
// the same decode-per-row structure as before, just expressed without
// a procedural loop textually inside the generate loop.
genvar gidx;
generate
    for (gidx = 0; gidx < filters*(r1/2)*(c1/2); gidx = gidx + 1) begin : ELEM_WRITE
        localparam GCH = gidx / ((r1/2)*(c1/2));
        localparam GREM = gidx % ((r1/2)*(c1/2));
        localparam GR = GREM / (c1/2);
        localparam GK = GREM % (c1/2);
        always @(posedge clk) begin
            if (start_pooling_d && (out_row_d == GR))
                pooling_results[GCH][GR][GK] <= result_row[GCH][GK];
        end
    end
endgenerate

always @(posedge clk) begin
    if (rst)
        done_pooling <= 0;
    else
        done_pooling <= start_pooling_d;
end

endmodule