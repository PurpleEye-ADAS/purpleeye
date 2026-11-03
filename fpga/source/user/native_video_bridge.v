module native_pixel_fifo #(
    parameter integer ADDRESS_BITS = 6
)(
    input wire rst_n, pixel_clk, vsync, data_enable,
    input wire [15:0] pixel,
    input wire output_clk,
    output reg [17:0] output_data,
    output reg output_valid,
    output reg overflow
);
localparam integer POINTER_BITS = ADDRESS_BITS + 1;
reg [17:0] memory [0:(1 << ADDRESS_BITS)-1];
reg [POINTER_BITS-1:0] write_pointer, read_pointer, write_gray, read_gray;
reg [POINTER_BITS-1:0] read_meta, read_sync, write_meta, write_sync;
reg vsync_delayed;
wire start_frame = vsync && !vsync_delayed;
wire push = start_frame || data_enable;
wire [POINTER_BITS-1:0] write_next = write_pointer + 1'b1;
wire [POINTER_BITS-1:0] write_next_gray = (write_next >> 1) ^ write_next;
wire full = write_next_gray == {~read_sync[POINTER_BITS-1:POINTER_BITS-2], read_sync[POINTER_BITS-3:0]};
wire empty = read_gray == write_sync;
wire [POINTER_BITS-1:0] read_next = read_pointer + 1'b1;

always @(posedge pixel_clk or negedge rst_n) begin
    if (!rst_n) begin
        write_pointer <= 0; write_gray <= 0; read_meta <= 0; read_sync <= 0;
        vsync_delayed <= 0; overflow <= 0;
    end else begin
        read_meta <= read_gray; read_sync <= read_meta;
        vsync_delayed <= vsync;
        if (start_frame) overflow <= 0;
        if (push) begin
            if (full) overflow <= 1;
            else begin
                // SOF carries the preceding frame's loss flag in the same FIFO as its pixels.
                memory[write_pointer[ADDRESS_BITS-1:0]] <= start_frame ? {1'b1, overflow, 16'd0} : {2'b00, pixel};
                write_pointer <= write_next;
                write_gray <= write_next_gray;
            end
        end
    end
end

always @(posedge output_clk or negedge rst_n) begin
    if (!rst_n) begin
        read_pointer <= 0; read_gray <= 0; write_meta <= 0; write_sync <= 0;
        output_data <= 0; output_valid <= 0;
    end else begin
        write_meta <= write_gray; write_sync <= write_meta;
        output_valid <= !empty;
        if (!empty) begin
            output_data <= memory[read_pointer[ADDRESS_BITS-1:0]];
            read_pointer <= read_next;
            read_gray <= (read_next >> 1) ^ read_next;
        end
    end
end
endmodule

module native_video_bridge #(
    parameter integer FRAME_PIXELS = 1280 * 720
)(
    input wire rst_n, output_clk,
    input wire [1:0] source_request,
    input wire clk0, vs0, de0,
    input wire [15:0] pixel0,
    input wire clk1, vs1, de1,
    input wire [15:0] pixel1,
    input wire clk2, vs2, de2,
    input wire [15:0] pixel2,
    output reg [1:0] source_active,
    output reg source_ready,
    output reg [31:0] good_frames,
    output reg vsync, data_enable,
    output reg [15:0] pixel
);
wire [17:0] data0, data1, data2;
wire valid0, valid1, valid2;
wire [2:0] overflow;
reg [2:0] overflow_meta, overflow_sync;
reg receiving_frame;
reg [20:0] pixels;
wire [17:0] selected_data = source_active == 0 ? data0 : source_active == 1 ? data1 : data2;
wire selected_valid = source_active == 0 ? valid0 : source_active == 1 ? valid1 : valid2;

native_pixel_fifo fifo0(.rst_n(rst_n), .pixel_clk(clk0), .vsync(vs0), .data_enable(de0), .pixel(pixel0),
    .output_clk(output_clk), .output_data(data0), .output_valid(valid0), .overflow(overflow[0]));
native_pixel_fifo fifo1(.rst_n(rst_n), .pixel_clk(clk1), .vsync(vs1), .data_enable(de1), .pixel(pixel1),
    .output_clk(output_clk), .output_data(data1), .output_valid(valid1), .overflow(overflow[1]));
native_pixel_fifo fifo2(.rst_n(rst_n), .pixel_clk(clk2), .vsync(vs2), .data_enable(de2), .pixel(pixel2),
    .output_clk(output_clk), .output_data(data2), .output_valid(valid2), .overflow(overflow[2]));

always @(posedge output_clk or negedge rst_n) begin
    if (!rst_n) begin
        source_active <= 2; source_ready <= 0; good_frames <= 0;
        vsync <= 0; data_enable <= 0; pixel <= 0; pixels <= 0; receiving_frame <= 0;
        overflow_meta <= 0; overflow_sync <= 0;
    end else begin
        overflow_meta <= overflow; overflow_sync <= overflow_meta;
        vsync <= 0; data_enable <= 0;
        if (source_request != source_active) begin
            source_active <= source_request;
            source_ready <= 0; good_frames <= 0; pixels <= 0; receiving_frame <= 0;
        end else begin
            if (overflow_sync[source_active]) source_ready <= 0;
            if (selected_valid) begin
                if (selected_data[17]) begin
                    vsync <= 1;
                    source_ready <= receiving_frame && pixels == FRAME_PIXELS && !selected_data[16];
                    if (receiving_frame && pixels == FRAME_PIXELS && !selected_data[16])
                        good_frames <= good_frames + 1'b1;
                    receiving_frame <= 1; pixels <= 0;
                end else if (receiving_frame) begin
                    data_enable <= 1; pixel <= selected_data[15:0];
                    if (pixels <= FRAME_PIXELS) pixels <= pixels + 1'b1;
                end
            end
        end
    end
end
endmodule
