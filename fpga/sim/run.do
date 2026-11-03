# 姓名：丁黄炎
onerror {quit -code 1}
if {![file exists work]} {vlib work}
vlog -sv {*}[glob ../source/user/*.v] pcie_img_select_tb.v native_video_bridge_tb.v traffic_dma_frame_tb.v traffic_pcie_registers_tb.v video_channel_status_tb.v
foreach test {pcie_img_select native_video_bridge traffic_dma_frame traffic_pcie_registers video_channel_status} {
    vsim -onfinish stop -t 1ps work.${test}_tb
    run -all
    quit -sim
}
quit -code 0
