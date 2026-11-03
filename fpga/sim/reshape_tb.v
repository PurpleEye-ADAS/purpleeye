// 姓名：丁黄炎

`timescale 1ns/1ps
module reshape_tb;
reg clk=0,rst_n=0,vs=0,de=0;
reg [15:0] pixel=0;
wire valid;
wire [15:0] out_pixel;
integer x,y,kept=0;
always #5 clk=~clk;
image_reshape dut(clk,rst_n,vs,de,pixel,4'b0000,valid,out_pixel);
initial begin
 $dumpfile("reshape.vcd");$dumpvars(0,clk,vs,de,pixel,valid,out_pixel,dut.col_cnt,dut.row_cnt);
 repeat(5) @(negedge clk);rst_n=1;
 vs=1;repeat(4) @(negedge clk);vs=0;repeat(5) @(negedge clk);
 for(y=0;y<720;y=y+1) begin
  if(y==4) $dumpoff;
  for(x=0;x<1280;x=x+1) begin
   @(negedge clk);de=1;pixel=((y&31)<<11)|((x&63)<<5)|((x>>6)&31);
   @(posedge clk);#1;
   if(valid !== ((x%2==0)&&(y%2==0))) $fatal(1,"reshape enable at %0d,%0d",x,y);
   if(valid) begin
    if(out_pixel !== pixel) $fatal(1,"reshape value at %0d,%0d",x,y);
    kept=kept+1;
   end
  end
  @(negedge clk);de=0; repeat(4) @(negedge clk);
 end
 if(kept!=230400) $fatal(1,"reshape count %0d",kept);
 $display("PASS reshape: 921600 input pixels -> 230400 selected pixels, all coordinates checked");
 $finish;
end
endmodule
