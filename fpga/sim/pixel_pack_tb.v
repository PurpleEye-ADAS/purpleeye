// 姓名：丁黄炎

`timescale 1ns/1ps
module pixel_pack_tb;
reg pclk=0,rst_n=0,de=0,vs=0;
reg [7:0] din=0;
wire pixel_clk,de_o;
wire [15:0] dout;
integer i,seen=0;
reg [15:0] sample;
always #5 pclk=~pclk;
cmos_8_16bit dut(pclk,rst_n,de,din,vs,pixel_clk,de_o,dout);
initial begin
 $dumpfile("pixel_pack.vcd");
 $dumpvars(0,pclk,rst_n,de,vs,din,pixel_clk,de_o,dout);
 repeat(5) @(negedge pclk); rst_n=1;
 repeat(3) @(negedge pclk); vs=1;
 repeat(4) @(negedge pclk); vs=0;
 repeat(10) @(negedge pclk);
 for(i=0;i<16;i=i+1) begin
  sample=16'h1000+i;
  @(negedge pclk); de=1;din=sample[15:8];
  @(negedge pclk); din=sample[7:0];
 end
 @(negedge pclk);de=0;din=0;
 repeat(20) @(negedge pclk);
 if(seen!=16) $fatal(1,"pixel count %0d",seen);
 $display("PASS pixel_pack: 32 input bytes -> 16 ordered RGB565 pixels, vendor divider model");
 $finish;
end
always @(posedge pixel_clk) begin
 #2;
 if(rst_n && de_o) begin
  if(dout !== (16'h1000+seen)) $fatal(1,"pixel %0d got %h",seen,dout);
  seen=seen+1;
 end
end
endmodule
