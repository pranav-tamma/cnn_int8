module pe(
input signed [7:0] ain,
input signed [7:0] bin,
input clk,
input rst,
input start_compute,
output reg signed [7:0] aout,
output reg signed [7:0] bout,
(* use_dsp = "yes" *) output reg signed [31:0] c
);

always @(posedge clk)
begin
    if(rst || !start_compute)
    begin
        c <= 0;
        aout <= 0;
        bout <= 0;
    end

    else begin
        if(start_compute)
        c <= c + aout*bout;
        aout <= ain;
        bout <= bin;
    end
end
endmodule

