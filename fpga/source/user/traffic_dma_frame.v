// Frame boundaries count 128-bit image words, independent of host DMA chunks.
module traffic_dma_frame #(
    parameter FRAME_WORDS = 115200,
    parameter COUNT_BITS = 17
) (
    input wire clk,
    input wire rst_n,
    input wire restart,
    input wire word_request,
    output reg frame_done
);
reg [COUNT_BITS-1:0] words;
reg last_word;
always @(posedge clk) begin
    if (!rst_n || restart) begin
        words <= 0;
        last_word <= 0;
        frame_done <= 0;
    end else begin
        last_word <= 0;
        frame_done <= last_word;
        if (word_request) begin
            if (words == FRAME_WORDS - 1) begin
                words <= 0;
                last_word <= 1;
            end else begin
                words <= words + 1'b1;
            end
        end
    end
end
endmodule
