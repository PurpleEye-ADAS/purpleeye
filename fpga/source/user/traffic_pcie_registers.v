`timescale 1ns / 1ps
module traffic_pcie_registers #(
    parameter integer ADDR_WIDTH = 12,
    parameter integer CLK_HZ = 125000000,
    parameter integer DEBOUNCE_TICKS = CLK_HZ / 50,
    parameter integer LONG_TICKS = CLK_HZ,
    parameter integer AUTO_TICKS = CLK_HZ * 3,
    parameter integer WATCHDOG_TICKS = CLK_HZ / 2
)(
    input wire clk, rst_n, key_n,
    input wire [3:0] video_valid,
    input wire [2:0] hdmi_flags,
    input wire [127:0] hdmi_debug,
    input wire [255:0] hdmi_raw_samples,
    input wire [127:0] video_good_frames, video_total_frames, video_pixels,
    input wire write_enable,
    input wire [ADDR_WIDTH-1:0] write_address,
    input wire [127:0] write_data,
    input wire [15:0] write_byte_enable,
    input wire read_enable,
    input wire [ADDR_WIDTH-1:0] read_address,
    output reg read_selected,
    output reg [127:0] read_data,
    output wire [1:0] led,
    output wire source_sel,
    input wire capture_full_active, capture_ready,
    input wire [1:0] native_source_active,
    input wire native_source_ready,
    input wire [31:0] native_good_frames,
    output reg [1:0] native_source_request,
    input wire [9:0] hdmi_sampling_active,
    output reg [9:0] hdmi_sampling_request,
    output reg capture_full_request,
    output reg capture_restart
);
reg [31:0] expected_sequence;
wire [127:0] status;
wire write_word = write_enable && &write_byte_enable[3:0];
// This endpoint's BAR1 addresses are bytes; BAR0 addresses are 16-byte beats.
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        expected_sequence <= 0;
        read_selected <= 0;
        read_data <= 0;
        capture_full_request <= 0;
        native_source_request <= 2;
        capture_restart <= 0;
        hdmi_sampling_request <= 10'h2d0;
    end else begin
        capture_restart <= 0;
        if (write_word && write_address == 'h270 && write_data[31:2] == 0 && write_data[1:0] != 3)
            native_source_request <= write_data[1:0];
        if (write_word && write_address == 'h1f0 && write_data[31:10] == 0 && write_data[7:0] <= 247)
            hdmi_sampling_request <= write_data[9:0];
        if (write_word && write_address == 'h1e0 && write_data[31:1] == 0) begin
            capture_full_request <= write_data[0];
            capture_restart <= 1;
        end
        if (write_word && write_address == 'h1d0)
            expected_sequence <= write_data[31:0];
        if (read_enable) begin
            read_selected <= read_address == 'h1c || read_address == 'h1e || read_address == 'h1f || (read_address >= 'h20 && read_address <= 'h27);
            case (read_address)
                // Format 2 maps display channels 0/1/2/3 to physical sources 0/1/2/2.
                'h20: read_data <= {18'd0, 1'b1, 1'b1, 1'b1, hdmi_flags, 8'd2, 28'd0, video_valid, 32'h02d00500, 32'h34444956};
                'h27: read_data <= {32'd0, native_good_frames, 27'd0, native_source_ready, native_source_active, native_source_request, 32'h4e415456};
                'h1e: read_data <= {64'd0, 29'd0, capture_ready, capture_full_active, capture_full_request, 32'h48444d49};
                'h1f: read_data <= {32'd0, 22'd0, hdmi_sampling_active, 22'd0, hdmi_sampling_request, 32'h53414d50};
                'h21: read_data <= video_good_frames;
                'h22: read_data <= video_pixels;
                'h23: read_data <= video_total_frames;
                'h24: read_data <= hdmi_debug;
                'h25: read_data <= hdmi_raw_samples[127:0];
                'h26: read_data <= hdmi_raw_samples[255:128];
                default: read_data <= status;
            endcase
        end
    end
end
traffic_pcie_control #(
    .CLK_HZ(CLK_HZ), .DEBOUNCE_TICKS(DEBOUNCE_TICKS), .LONG_TICKS(LONG_TICKS),
    .AUTO_TICKS(AUTO_TICKS), .WATCHDOG_TICKS(WATCHDOG_TICKS)
) control(
    .clk(clk), .rst_n(rst_n), .key_n(key_n),
    .command_valid(write_word && write_address == 'h1c0),
    .command(write_data[31:0]), .expected_sequence(expected_sequence),
    .status(status), .led(led), .source_sel(source_sel)
);
endmodule
