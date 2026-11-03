// 姓名：丁黄炎

`timescale 1ns/1ps
module axi_transfer_tb;
reg clk=0,rst_n=0,wr_start=0,rd_start=0;
reg [31:0] wr_len=2048,rd_len=5120;
wire [31:0] awaddr,araddr;
wire [7:0] awlen,arlen;
wire awvalid,arvalid,wr_re,wr_done,rd_we,rd_done;
wire [255:0] master_wdata,master_rdata;
wire [127:0] memory_wdata=master_wdata[127:0];
reg awready=0,arready=0,wrequest=0,rvalid=0,rlast=0;
reg [127:0] fifo_data=0,response_data=0;
integer cycle=0,aw_count=0,ar_count=0,write_pending=0,read_pending=0;
integer fetched=0,written=0,returned=0,accepted=0;
always #5 clk=~clk;
aq_axi_master_256_burst dut(
 .ARESETN(rst_n),.ACLK(clk),.MASTER_RST(1'b0),
 .DEBUG(),.RD_READY(),.WR_READY(),.M_AXI_RREADY(),.M_AXI_ARUSER(),.M_AXI_ARQOS(),.M_AXI_ARPROT(),.M_AXI_ARCACHE(),.M_AXI_ARLOCK(),.M_AXI_ARBURST(),.M_AXI_ARSIZE(),.M_AXI_ARID(),.M_AXI_BREADY(),.M_AXI_WVALID(),.M_AXI_WUSER(),.M_AXI_WLAST(),.M_AXI_WSTRB(),.M_AXI_AWUSER(),.M_AXI_AWQOS(),.M_AXI_AWPROT(),.M_AXI_AWCACHE(),.M_AXI_AWLOCK(),.M_AXI_AWBURST(),.M_AXI_AWSIZE(),.M_AXI_AWID(),
 .M_AXI_AWADDR(awaddr),.M_AXI_AWLEN(awlen),.M_AXI_AWVALID(awvalid),.M_AXI_AWREADY(awready),
 .M_AXI_WDATA(master_wdata),.M_AXI_WREADY(wrequest),
 .M_AXI_BID(1'b0),.M_AXI_BRESP(2'b0),.M_AXI_BUSER(1'b0),.M_AXI_BVALID(1'b1),
 .M_AXI_ARADDR(araddr),.M_AXI_ARLEN(arlen),.M_AXI_ARVALID(arvalid),.M_AXI_ARREADY(arready),
 .M_AXI_RID(1'b0),.M_AXI_RDATA({128'd0,response_data}),.M_AXI_RRESP(2'b0),
 .M_AXI_RLAST(rlast),.M_AXI_RUSER(1'b0),.M_AXI_RVALID(rvalid),
 .WR_START(wr_start),.WR_ADRS(32'h10000),.WR_LEN(wr_len),
 .WR_FIFO_RE(wr_re),.WR_FIFO_EMPTY(1'b0),.WR_FIFO_AEMPTY(1'b0),
 .WR_FIFO_DATA({128'd0,fifo_data}),.WR_DONE(wr_done),
 .RD_START(rd_start),.RD_ADRS(32'h40000),.RD_LEN(rd_len),
 .RD_FIFO_WE(rd_we),.RD_FIFO_FULL(1'b0),.RD_FIFO_AFULL(1'b0),
 .RD_FIFO_DATA(master_rdata),.RD_DONE(rd_done));
always @(negedge clk) begin
 if(rst_n) begin
  awready=(cycle%7>=3);arready=(cycle%5>=2);
  wrequest=(write_pending>0 && cycle%6!=0 && cycle%6!=1);
  rvalid=(read_pending>0 && cycle%7!=0 && cycle%7!=1);
  response_data=128'hB000+returned;rlast=rvalid&&(returned%16==15);
 end else begin awready=0;arready=0;wrequest=0;rvalid=0;rlast=0;end
end
always @(posedge clk) begin
 if(rst_n) begin
  cycle=cycle+1;
  if(awvalid&&awready) begin
   if(awaddr !== (32'h10000+aw_count*512) || awlen!=15) $fatal(1,"AW sequence");
   aw_count=aw_count+1;write_pending=write_pending+16;
  end
  if(wrequest) begin
   if(memory_wdata !== (128'hA000+written)) $fatal(1,"write beat %0d got %h",written,memory_wdata);
   written=written+1;write_pending=write_pending-1;
  end
  if(wr_re) begin fifo_data<=128'hA000+fetched;fetched=fetched+1;end
  if(arvalid&&arready) begin
   if(araddr !== (32'h40000+ar_count*512)||arlen!=15) $fatal(1,"AR sequence");
   ar_count=ar_count+1;read_pending=read_pending+16;
  end
  if(rvalid) begin
   if(!rd_we || master_rdata[127:0] !== (128'hB000+accepted)) $fatal(1,"read data");
   returned=returned+1;accepted=accepted+1;read_pending=read_pending-1;
  end
 end
end
initial begin
 $dumpfile("axi_transfer.vcd");
 $dumpvars(0,clk,rst_n,wr_start,rd_start,awaddr,awlen,awvalid,awready,wrequest,wr_re,
 fifo_data,memory_wdata,wr_done,araddr,arlen,arvalid,arready,rvalid,rlast,response_data,rd_we,rd_done);
 repeat(5) @(negedge clk);rst_n=1;
 repeat(3) @(negedge clk);wr_start=1;
 @(negedge clk);wr_start=0;
 wait(wr_done); wait(write_pending==0); repeat(3) @(negedge clk);
 if(aw_count!=4||written!=64||fetched!=64) $fatal(1,"write totals %0d %0d %0d",aw_count,written,fetched);
 rd_start=1;@(negedge clk);rd_start=0;
 wait(rd_done);repeat(3) @(negedge clk);
 if(ar_count!=10||returned!=160||accepted!=160) $fatal(1,"read totals");
 $display("PASS axi_transfer: 4 write address requests / 64 payload beats; 10 read requests / 160 payload beats; delayed address ready and gapped data");
 $finish;
end
initial begin #200000;$fatal(1,"AXI timeout");end
endmodule
