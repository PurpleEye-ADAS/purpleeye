`timescale 1ns/1ps
module traffic_dma_frame_tb;
reg clk=0, rst_n=0, restart=0, word_request=0;
wire frame_done;
integer pulses=0, frame, chunk, word_index;
always #5 clk=~clk;
traffic_dma_frame dut(clk,rst_n,restart,word_request,frame_done);
always @(negedge clk) if (frame_done) pulses=pulses+1;
task tick;
    begin @(negedge clk); #1; end
endtask
task send_frame;
    input integer chunk_words;
    input integer expected;
    begin
        for (chunk=0; chunk<115200/chunk_words; chunk=chunk+1) begin
            for (word_index=0;word_index<chunk_words;word_index=word_index+1) begin
                word_request=1; tick;
                // Exercise gaps inside a TLP as well as between DMA requests.
                if (word_index%31==0) begin word_request=0; tick; end
            end
            word_request=0; repeat(7) tick;
            if (pulses != expected - (chunk+1==115200/chunk_words ? 0 : 1))
                $fatal(1,"Early/missing frame boundary, chunk=%d pulses=%d",chunk,pulses);
        end
    end
endtask
initial begin
    repeat(3) tick; rst_n=1; tick;
    send_frame(160,1); // Legacy 2560-byte row DMA.
    send_frame(240,2); // 3840-byte DMA spans one and a half rows.
    send_frame(160,3); // Switching back remains aligned.
    word_request=1; repeat(100) tick;
    restart=1; word_request=0; tick; restart=0; tick;
    send_frame(240,4); // Explicit restart drops a partial frame.
    if (pulses!=4) $fatal(1,"Wrong pulse count");
    $display("PASS: DMA frame boundaries, stalls, legacy/chunked DMA and restart");
    $finish;
end
endmodule
