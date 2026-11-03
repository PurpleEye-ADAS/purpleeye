`timescale 1ns / 1ps
// Each instance observes its own input clock. A copied display quadrant cannot
// create another hardware source. Only complete 640x360 frames count as valid.
module video_channel_status #(
    parameter integer FRAME_PIXELS = 640 * 360,
    parameter integer TIMEOUT_TICKS = 125000000 / 2
)(
    input wire pixel_clk, pixel_rst_n, vsync, data_valid,
    input wire status_clk, status_rst_n,
    output reg active,
    output reg [31:0] good_frames, total_frames, last_pixels
);
reg vs_d, frame_started;
reg [31:0] pixels, good_binary, total_binary, pixels_latched;
reg [31:0] good_gray, total_gray;
wire [31:0] next_good = good_binary + 1'b1;
wire [31:0] next_total = total_binary + 1'b1;

always @(posedge pixel_clk or negedge pixel_rst_n) begin
    if (!pixel_rst_n) begin
        vs_d <= 0; frame_started <= 0; pixels <= 0;
        good_binary <= 0; total_binary <= 0; pixels_latched <= 0;
        good_gray <= 0; total_gray <= 0;
    end else begin
        vs_d <= vsync;
        if (vsync && !vs_d) begin
            frame_started <= 1;
            pixels <= 0;
            if (frame_started) begin
                pixels_latched <= pixels;
                total_binary <= next_total;
                total_gray <= (next_total >> 1) ^ next_total;
                if (pixels == FRAME_PIXELS) begin
                    good_binary <= next_good;
                    good_gray <= (next_good >> 1) ^ next_good;
                end
            end
        end else if (data_valid && frame_started && pixels != 32'hffffffff)
            pixels <= pixels + 1'b1;
    end
end

reg [31:0] good_meta, good_sync, total_meta, total_sync;
reg [31:0] pixels_meta, pixels_sync;
reg [31:0] total_seen;
reg pixels_pending;
reg [31:0] idle_ticks;
function [31:0] gray_binary;
    input [31:0] gray;
    integer i;
    begin
        gray_binary[31] = gray[31];
        for (i=30; i>=0; i=i-1) gray_binary[i] = gray_binary[i+1] ^ gray[i];
    end
endfunction

always @(posedge status_clk or negedge status_rst_n) begin
    if (!status_rst_n) begin
        good_meta <= 0; good_sync <= 0; total_meta <= 0; total_sync <= 0;
        pixels_meta <= 0; pixels_sync <= 0; total_seen <= 0; pixels_pending <= 0;
        good_frames <= 0; total_frames <= 0; last_pixels <= 0;
        active <= 0; idle_ticks <= 0;
    end else begin
        good_meta <= good_gray; good_sync <= good_meta;
        total_meta <= total_gray; total_sync <= total_meta;
        // Bundled pixel count stays fixed for the entire next video frame.
        // Capture one status cycle after the two-stage Gray event crossing.
        pixels_meta <= pixels_latched; pixels_sync <= pixels_meta;
        pixels_pending <= total_sync != total_seen;
        total_seen <= total_sync;
        total_frames <= gray_binary(total_sync);
        if (pixels_pending) last_pixels <= pixels_sync;
        good_frames <= gray_binary(good_sync);
        if (gray_binary(good_sync) != good_frames) begin
            idle_ticks <= 0;
            active <= 1;
        end else if (pixels_pending && pixels_sync != FRAME_PIXELS)
            active <= 0;
        else if (idle_ticks < TIMEOUT_TICKS - 1)
            idle_ticks <= idle_ticks + 1'b1;
        else active <= 0;
    end
end
endmodule
