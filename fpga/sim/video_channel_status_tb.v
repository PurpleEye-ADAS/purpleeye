`timescale 1ns / 1ps
module video_channel_status_tb;
reg pixel_clk=0, status_clk=0, reset_n=0, vsync=0, data_valid=0;
reg pixel_running=1;
wire active;
wire [31:0] good_frames, total_frames, last_pixels;
always #7 if (pixel_running) pixel_clk=~pixel_clk;
always #5 status_clk=~status_clk;
video_channel_status #(.FRAME_PIXELS(12), .TIMEOUT_TICKS(80)) dut(
    .pixel_clk(pixel_clk), .pixel_rst_n(reset_n), .vsync(vsync), .data_valid(data_valid),
    .status_clk(status_clk), .status_rst_n(reset_n), .active(active),
    .good_frames(good_frames), .total_frames(total_frames), .last_pixels(last_pixels)
);
task boundary;
begin
    @(negedge pixel_clk); vsync=1;
    repeat(3) @(negedge pixel_clk);
    vsync=0;
end
endtask
task pixels;
input integer n;
begin
    @(negedge pixel_clk); data_valid=1;
    repeat(n) @(negedge pixel_clk);
    data_valid=0;
end
endtask
task check;
input condition;
input [511:0] message;
begin
    if (condition !== 1'b1) begin $display("FAIL: %0s", message); $fatal(1); end
end
endtask
initial begin
    #2; reset_n=0; #40; reset_n=1;
    repeat(6) @(negedge status_clk);
    check(!active && good_frames==0, "no source is invalid");
    boundary; pixels(12); boundary;
    repeat(8) @(negedge status_clk);
    check(active && good_frames==1 && total_frames==1 && last_pixels==12, "complete frame and CDC");
    pixels(11); boundary;
    repeat(8) @(negedge status_clk);
    check(!active && good_frames==1 && total_frames==2 && last_pixels==11, "partial frame invalidates source");
    pixels(13); boundary;
    repeat(8) @(negedge status_clk);
    check(good_frames==1 && total_frames==3 && last_pixels==13, "wrong dimensions not accepted");
    pixel_running=0;
    repeat(85) @(negedge status_clk);
    check(!active && good_frames==1, "unplugged pixel clock expires");
    reset_n=0; repeat(4) @(negedge status_clk); reset_n=1;
    repeat(4) @(negedge status_clk);
    check(!active && good_frames==0 && last_pixels==0, "reset works with absent input clock");
    pixel_running=1;
    boundary; pixels(12); boundary;
    repeat(8) @(negedge status_clk);
    check(active && good_frames==1, "source recovers after reset");
    $display("PASS: complete frames, CDC, wrong size, clock loss, reset without source, recovery");
    $finish;
end
initial begin #20000; $fatal(1,"test timeout"); end
endmodule
