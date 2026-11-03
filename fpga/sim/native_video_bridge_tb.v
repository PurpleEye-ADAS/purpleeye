// 姓名：丁黄炎
`timescale 1ns/1ps
module native_video_bridge_tb;
reg rst_n=0, clk=0, c0=0, c1=0, c2=0;
reg [1:0] request=2;
wire [2:0] vs, de;
wire [15:0] p0,p1,p2;
wire [1:0] active;
wire ready, out_vs, out_de;
wire [15:0] out_pixel;
wire [31:0] good;
integer checked=0, index=0, previous=2;
always #4 clk=~clk;
always #7 c0=~c0;
always #11 c1=~c1;
always #5 c2=~c2;
native_test_source #(.ID(1)) s0(c0,rst_n,vs[0],de[0],p0);
native_test_source #(.ID(2)) s1(c1,rst_n,vs[1],de[1],p1);
native_test_source #(.ID(3)) s2(c2,rst_n,vs[2],de[2],p2);
native_video_bridge #(.FRAME_PIXELS(16)) dut(
    .rst_n(rst_n),.output_clk(clk),.source_request(request),
    .clk0(c0),.vs0(vs[0]),.de0(de[0]),.pixel0(p0),
    .clk1(c1),.vs1(vs[1]),.de1(de[1]),.pixel1(p1),
    .clk2(c2),.vs2(vs[2]),.de2(de[2]),.pixel2(p2),
    .source_active(active),.source_ready(ready),.good_frames(good),
    .vsync(out_vs),.data_enable(out_de),.pixel(out_pixel)
);
always @(negedge clk) begin
    if(!rst_n || active!=previous) index=0;
    if(out_vs) index=0;
    if(out_de) begin
        if(out_pixel[15:12] !== active+1 || out_pixel[3:0] !== index)
            $fatal(1,"native pixel/source mismatch source=%0d pixel=%h index=%0d",active,out_pixel,index);
        index=index+1; checked=checked+1;
    end
    previous=active;
end
task select_source;
    input [1:0] selected;
    begin
        @(negedge clk); request=selected;
        repeat(3) @(negedge clk);
        if(ready) $fatal(1,"new source inherited old-frame permission");
        wait(ready && good>=3);
        repeat(45) @(negedge clk);
    end
endtask
initial begin
    $dumpfile("native_cdc.vcd");
    $dumpvars(0,clk,c0,c1,c2,request,active,ready,out_vs,out_de,out_pixel,good);
    repeat(4) @(negedge clk); rst_n=1;
    wait(ready && good>=3);
    select_source(0); select_source(1); select_source(2); select_source(0);
    @(negedge clk); rst_n=0;
    repeat(5) @(negedge clk); rst_n=1;
    wait(ready && good>=3);
    if(checked<250) $fatal(1,"insufficient pixel coverage");
    $display("PASS: %0d native pixels, three unrelated source clocks, source swaps, frame-ready isolation and reset",checked);
    $finish;
end
initial begin #100000; $fatal(1,"native bridge timeout"); end
endmodule

module native_test_source #(parameter ID=1)(
    input wire clk,rst_n,
    output reg vs=0,de=0,
    output reg [15:0] pixel=0
);
reg [7:0] cycle=0, frame=0;
always @(negedge clk) begin
    if(!rst_n) begin cycle<=0;frame<=0;vs<=0;de<=0;pixel<=0;end
    else begin
        vs<=cycle==0;de<=cycle>=8 && cycle<24;
        if(cycle>=8 && cycle<24) pixel<=(ID<<12)|(frame<<4)|(cycle-8);
        if(cycle==35) begin cycle<=0;frame<=frame+1'b1;end
        else cycle<=cycle+1'b1;
    end
end
endmodule
