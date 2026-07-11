module systolic_array #(
    parameter r1 = 16,
    parameter c1 = 9,
    parameter r2 = 9,
    parameter c2 = 8
)(
    input clk,
    input rst,
    input start_compute,

    input signed [7:0] a_matrix [0:r1-1][0:c1-1],
    input signed [7:0] b_matrix [0:r2-1][0:c2-1],

    output wire signed [31:0] results [0:r1-1][0:c2-1],
    output reg done_compute
);

  reg signed [31:0]  aside_sched  [0:r1-1][0:c1+r1-2]; // input schedule matrix for A
  reg signed [31:0]  btop_sched  [0:c1+c2-2][0:c2-1]; // input schedule matrix for B

  wire signed [7:0]  alink [0:r1-1][0:c2-1]; // links for A between PEs (horizontal)
  wire signed [7:0]  blink [0:r1-1][0:c2-1]; // links for B between PEs (vertical)

  integer I, J, K; // loop variables for scheduling

  reg [15:0] cycle; // to keep track of the current cycle

  reg signed [7:0] aside [0:r1-1]; // to hold the left inputs for A
  reg signed [7:0] btop [0:c2-1]; // to hold the top inputs for B

  
// initializing schedule matrices with zeros
initial
   begin
    for(I=0;I<r1;I=I+1) begin
      for(J=0;J<c1+r1-1;J=J+1) begin
        aside_sched[I][J] = 0; // initialize schedule A with 0s
      end
    end

    for(K=0; K<c1+c2-1; K=K+1) begin
      for(J=0; J<c2; J=J+1) begin
        btop_sched[K][J] = 0; // initialize schedule B with 0s
      end
    end
    end

// create schedules for feeding A and B into the array
  
  always@(*)
  begin
    for(I=0; I<r1; I=I+1) begin
      for(K=0; K<c1; K=K+1) begin
        aside_sched[I][I+K] = a_matrix[I][K]; // schedule A: shift right every cycle
      end
    end
  end


  always@(*)
  begin
      for(J=0; J<c2; J=J+1) begin
        for(K=0; K<r2; K=K+1) begin
        btop_sched[J+K][J] = b_matrix[K][J]; // schedule B: shift down every cycle
      end
    end
  end

// initializing cycle count
always @(posedge clk or posedge rst) begin
    if (rst) begin
        cycle <= 0;
        done_compute <= 0;
    end
    else if (!start_compute) begin
        // Idle state
        cycle <= 0;
        done_compute <= 0;
    end
    else begin
        if (cycle == c1 + r1 + c2 - 1) begin
            done_compute <= 1;
            // Hold the cycle count here until start_compute goes low
            cycle <= cycle;
        end
        else begin
            done_compute <= 0;
            cycle <= cycle + 1;
        end
    end
end

  // feeders for the left and top inputs to the array based on the schedules

  always@(*)
  begin
  for(I=0;I<r1;I=I+1) begin
    aside[I] = (cycle < c1 + r1 -1) ? aside_sched[I][cycle] : 0; // feed aside from schedule for the first c1+r1-1 cycles, then feed 0
  end
  end


  always@(*)
  begin
   for(J=0;J<c2;J=J+1) begin
    btop[J] = (cycle < c1 + c2 - 1) ? btop_sched[cycle][J] : 0; // feed btop from schedule for the first c1+c2-1 cycles, then feed 0
  end
  end

// instantiate the PEs in a 2D grid and connect them according to the systolic array architecture

  genvar i, j;
  generate
    for (i = 0; i < r1; i = i+1) begin : Row
      for (j = 0; j < c2; j = j+1) begin : Col
        pe PE (.clk(clk), .ain((j==0) ? aside[i] : alink[i][j-1]), .bin((i==0) ? btop[j] : blink[i-1][j]), .aout(alink[i][j]), .bout(blink[i][j]), .c(results[i][j]),.rst(rst), .start_compute(start_compute)); // instantiate PE with appropriate inputs and outputs
      end
    end
  endgenerate 
endmodule