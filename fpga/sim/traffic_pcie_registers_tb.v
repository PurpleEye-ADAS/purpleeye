`timescale 1ns / 1ps
module traffic_pcie_registers_tb;
reg clk=0, rst_n=0, key_n=1, write_enable=0, read_enable=0;
reg [11:0] write_address=0, read_address=0;
reg [127:0] write_data=0;
reg [15:0] write_byte_enable=0;
wire selected, source_sel;
wire capture_full_request, capture_restart;
wire [1:0] native_source_request;
wire [9:0] hdmi_sampling_request;
reg [9:0] hdmi_sampling_active=10'h2d0;
reg capture_full_active=0, capture_ready=0;
wire [127:0] data;
wire [1:0] led;
always #5 clk=~clk;
traffic_pcie_registers #(.CLK_HZ(100), .DEBOUNCE_TICKS(3), .LONG_TICKS(20),
    .AUTO_TICKS(50), .WATCHDOG_TICKS(40)) dut(
    .clk(clk), .rst_n(rst_n), .key_n(key_n), .write_enable(write_enable),
    .write_address(write_address), .write_data(write_data), .write_byte_enable(write_byte_enable),
    .read_enable(read_enable), .read_address(read_address), .read_selected(selected),
    .read_data(data), .led(led), .source_sel(source_sel)
    , .capture_full_request(capture_full_request), .capture_restart(capture_restart)
    , .capture_full_active(capture_full_active), .capture_ready(capture_ready)
    , .native_source_request(native_source_request), .native_source_active(2'd1)
    , .native_source_ready(1'b1), .native_good_frames(32'd19)
    , .hdmi_sampling_request(hdmi_sampling_request), .hdmi_sampling_active(hdmi_sampling_active)
    , .video_valid(4'b0101), .hdmi_flags(3'b011), .video_good_frames({32'd0,32'd24,32'd0,32'd12})
    , .video_pixels({32'd0,32'd230400,32'd0,32'd230400})
    , .video_total_frames({32'd0,32'd25,32'd0,32'd13})
    , .hdmi_debug({32'd3,32'h209f9636,32'h00023abc,32'h00023abc})
    , .hdmi_raw_samples({128'h80708070,128'h807e807e})
);
task check;
    input condition;
    input [511:0] message;
    begin if (condition !== 1'b1) begin $display("FAIL: %0s",message); $fatal(1); end end
endtask
task ticks;
    input integer count;
    begin repeat(count) @(negedge clk); end
endtask
task write;
    input [11:0] address;
    input [31:0] value;
    input [15:0] bytes;
    begin
        write_address=address; write_data={96'd0,value}; write_byte_enable=bytes;
        write_enable=1; ticks(1); write_enable=0; ticks(1);
    end
endtask
task read;
    input [11:0] address;
    begin read_address=address; read_enable=1; ticks(1); read_enable=0; ticks(1); end
endtask
initial begin
    ticks(3); rst_n=1; ticks(3);
    read('h20); check(selected && data=={32'h3b02,32'd5,32'h02d00500,32'h34444956},"VID4 mirrored format and native camera capability");
    check(native_source_request==2,"native source defaults to HDMI");
    write('h270,0,16'h0007); check(native_source_request==2,"partial native selection ignored");
    write('h270,3,16'h000f); check(native_source_request==2,"invalid native source ignored");
    write('h270,4,16'h000f); check(native_source_request==2,"reserved native source bits ignored");
    write('h270,0,16'h000f); check(native_source_request==0,"camera one selected");
    read('h27); check(selected && data=={32'd0,32'd19,32'h14,32'h4e415456},"native active source and completed-frame acknowledgement");
    write('h270,1,16'h000f); check(native_source_request==1,"camera two selected");
    write('h270,2,16'h000f); check(native_source_request==2,"HDMI selected");
    read('h1e); check(selected && data==128'h48444d49,"HDMI mode reset");
    read('h1f); check(selected && data=={32'd0,32'h2d0,32'h2d0,32'h53414d50},"sampling reset and signature");
    write('h1f0,32'h3f7,16'h0007); check(hdmi_sampling_request==10'h2d0,"partial sampling write ignored");
    write('h1f0,32'h3f8,16'h000f); check(hdmi_sampling_request==10'h2d0,"out-of-range tap ignored");
    write('h1f0,32'h400,16'h000f); check(hdmi_sampling_request==10'h2d0,"reserved sampling bits ignored");
    write('h1f0,32'h3f7,16'h000f); check(hdmi_sampling_request==10'h3f7,"maximum valid tap and chain accepted");
    read('h1f); check(data[73:64]==10'h2d0 && data[41:32]==10'h3f7,"sampling request is not active acknowledgement");
    hdmi_sampling_active=10'h3f7;
    read('h1f); check(data[73:64]==10'h3f7,"sampling acknowledgement returned");
    check(led==0 && !source_sel && !capture_full_request,"sampling leaves vehicle and capture control unchanged");
    write('h1e0,1,16'h0007); check(!capture_full_request,"partial mode ignored");
    write('h1e0,2,16'h000f); check(!capture_full_request,"invalid mode ignored");
    write('h1e0,1,16'h000f); check(capture_full_request,"full HDMI requested");
    read('h1e); check(data[34:32]==1,"requested mode is not an active acknowledgement");
    capture_full_active=1; capture_ready=1;
    read('h1e); check(data[34:32]==7,"reader acknowledges full mode and prefetch");
    check(led==0 && !source_sel,"capture mode does not alter vehicle control");
    read('h21); check(selected && data=={32'd0,32'd24,32'd0,32'd12},"independent good frame counts");
    read('h22); check(selected && data=={32'd0,32'd230400,32'd0,32'd230400},"per-source geometry counts");
    read('h23); check(selected && data=={32'd0,32'd25,32'd0,32'd13},"invalid frames remain observable");
    read('h24); check(selected && data=={32'd3,32'h209f9636,32'h00023abc,32'h00023abc},"HDMI diagnostics are read-only");
    read('h25); check(selected && data==128'h807e807e,"raw HDMI first four samples");
    read('h26); check(selected && data==128'h80708070,"raw HDMI last four samples");
    read('h28); check(!selected,"next BAR0 beat remains original DMA RAM");
    read('h1c); check(selected && data[31:0]==32'h54524632,"BAR0 beat address and signature");
    read('h1c0); check(!selected,"BAR0 byte address must not alias status");
    write('h1d0,0,16'h000f);
    write('h1c0,32'ha55a0051,16'h0007); check(led==0,"partial command ignored");
    write('h01c,32'ha55a0051,16'hffff); check(led==0,"BAR1 beat address ignored");
    write('h1c0,32'ha55a0051,16'h000f); check(led==1,"32-bit command enables movement");
    key_n=0; ticks(8); key_n=1; ticks(7); check(led==0,"physical key invalidates prior outputs");
    write('h1c0,32'ha55a00d1,16'hffff); check(led==0,"stale sequence refuses override");
    read('h1c); check(data[33:32]==2 && data[95:64]==1,"physical lane and event returned");
    write('h1d0,data[95:64],16'hffff);
    write('h1c0,32'ha55a0072,16'hffff); check(led==3,"endpoint expanded byte mask accepted");
    read('h1c); check(data[127:96]==32'ha55a0072,"accepted command echo");
    ticks(42); check(led==0,"lost heartbeat clears outputs");
    $display("PASS: BAR addresses, sampling tap limits and acknowledgement, partial writes, physical key race, watchdog");
    $finish;
end
initial begin #20000; $fatal(1,"test timeout"); end
endmodule
