// 姓名：丁黄炎
`timescale 1ns / 1ps
module pcie_img_select_tb;
reg clk=0, rst_n=0, frame_start=0, full_request=0, data_req=0;
wire full_active, ready, req0, req1, req2, req3;
reg [127:0] data2=128'h3000;
wire [127:0] data;
reg [127:0] expected_line [0:79];
integer row, col; integer image_file;
always #5 clk=~clk;
always @(posedge clk) if(req2) data2 <= data2+1;
pcie_img_select #(.MIRROR_CHANNEL_3(1)) dut(
    .clk(clk), .rst_n(rst_n), .line_full_flag(1'b1), .dma_sim_vs(frame_start),
    .capture_full_request(full_request), .capture_full_active(full_active), .capture_ready(ready),
    .ch0_data_req(req0), .ch0_data(128'h1000), .ch1_data_req(req1), .ch1_data(128'h2000),
    .ch2_data_req(req2), .ch2_data(data2), .ch3_data_req(req3), .ch3_data(128'h4000),
    .dma_wr_data_req(data_req), .dma_wr_data(data)
);
task start_frame;
    begin
        frame_start=1; repeat(5) @(negedge clk); frame_start=0;
        wait(ready); repeat(3) @(negedge clk);
    end
endtask
initial begin
    $dumpfile("mosaic.vcd");
    $dumpvars(0,clk,data_req,data,req0,req1,req2,req3,dut.row_cnt,dut.col_cnt,full_active,ready);
    image_file=$fopen("mosaic_frame.hex","w");
    repeat(3) @(negedge clk); rst_n=1;
    start_frame;
    for(row=0; row<720; row=row+1) begin
        data_req=1;
        for(col=0; col<160; col=col+1) begin
            @(negedge clk);
            $fwrite(image_file,"%032h\n",data);
            if(row<360 && data !== (col<80 ? 128'h1000 : 128'h2000))
                $fatal(1,"camera tile changed at %0d,%0d",row,col);
            if(row>=360 && col<80) expected_line[col]=data;
            if(row>=360 && col>=80 && data !== expected_line[col-80])
                $fatal(1,"mirrored HDMI word differs at %0d,%0d",row,col);
            if(req3) $fatal(1,"mosaic consumed the independent full-resolution FIFO");
        end
        data_req=0; repeat(25) @(negedge clk);
    end
    $fclose(image_file); $dumpoff;
    full_request=1; repeat(3) @(negedge clk);
    if(full_active) $fatal(1,"mode changed without a frame boundary");
    start_frame;
    if(!full_active) $fatal(1,"full HDMI mode did not latch");
    for(row=0; row<720; row=row+1) begin
        data_req=1;
        repeat(160) begin
            @(negedge clk);
            if(data !== 128'h4000 || req0 || req1 || req2)
                $fatal(1,"full HDMI frame contains mosaic data");
        end
        data_req=0; repeat(25) @(negedge clk);
    end
    full_request=0; start_frame;
    if(full_active) $fatal(1,"mosaic mode did not latch");
    $display("PASS: all 720 mosaic rows, exact mirrored words, frame-boundary mode change, full 1280x720 HDMI");
    $finish;
end
initial begin #4000000; $fatal(1,"test timeout"); end
endmodule
