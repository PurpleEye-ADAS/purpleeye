# 紫眸智驾 · PurpleEye ADAS

基于紫光同创 PG2L50H FPGA 和 RK3568 的视觉辅助驾驶台架。FPGA 将两路 OV5640 与一路 HDMI 组织成四格，经 PCIe 送往 RK3568。右下格复用 HDMI。RK3568 处理信号灯、车道线、视频导航提示和行人，输出方向箭头、红边及 LED 告警。驾驶和制动以台架状态模拟。

## 团队成员

- 丁黄炎（[dingyan1](https://github.com/dingyan1)）
- 小1671

| 目录 | 内容 |
| --- | --- |
| `fpga/source/user/` | 帧组织、读出拼接、跨时钟换源、按键与寄存器逻辑 |
| `fpga/sim/` | RTL 测试台及 ModelSim 批处理脚本 |
| `rk3568/` | 识别、跟踪、规则、PCIe 取图和板端启动 |
| `tools/` | 电脑 HDMI 测试播放器 |

### 本地验证

Python 3 环境安装依赖后，可运行软件测试与播放器。

```sh
python -m pip install -r requirements-dev.txt
cd rk3568
python -m unittest discover -s tests -p "test_traffic_*.py"
cd ..
python tools/play_hdmi_test.py --source /path/to/video.mp4
```

视频会在电脑端循环播放。用 `--display-x` 和 `--display-y` 设置 HDMI 扩展屏坐标，按 Esc 退出。未提供上游源码、原始摄像头截图时，对应用例会标记为跳过。部分 GUI 测试需要 Python 的 Tkinter，原生预览测试需要对应的 Linux 图形环境。

在 `fpga/sim/` 中运行以下命令。部分 ModelSim 版本需要将仓库检出到纯英文路径。

```sh
vsim -c -do run.do -l simulation.log
```

脚本覆盖四格拼接、原生视频跨域、DMA 帧边界、控制寄存器和视频源状态。另附的像素配对、AXI 和降采样测试台需从有授权的工程中提供对应 RTL。像素配对还需要 Pango 的 GTP_IOCLKDIV_E2 仿真模型。

### RK3568 板端运行

使用已经配好图形桌面的 RK3568 Linux，并准备对应的 VID4 固件、匹配内核的 Pango PCIe 驱动、Rockchip RKNN Runtime、C++ 编译器、pkg-config、OpenCV 开发库，以及 EGL/GLES/X11 开发库。原配置使用 6.1.99 内核。Python 建议使用板上已有 OpenCV、NumPy 和 Pillow。

准备应用所需的五份模型和配套 JSON。RKNN 文件名与放置位置如下。

| 路径（相对 `rk3568/`） | 用途 |
| --- | --- |
| `models/color.rknn` | 信号灯颜色 |
| `direction.rknn` | HDMI 信号灯方向 |
| `camera_direction.rknn` | 摄像头信号灯方向 |
| `lane.rknn` | 车道线 |
| `models/person.rknn` | 行人 |

```sh
cd rk3568
sh pcie_capture/build_capture.sh
sh native_capture/build_capture.sh
sh native_camera/build.sh
export RKNN_RUNTIME=/path/to/librknnrt.so
export PANGO_PCIE_DRIVER=/path/to/pango_pci_driver.ko
bash run_cameras.sh --display-width 1280 --display-height 790
```

已加载 PCIe 驱动时可省略 `PANGO_PCIE_DRIVER`。在板端图形桌面启动。默认入口需要 `xterm` 和 `xdpyinfo`。F_KEY1 切换方向，C_KEY1 短按切换模拟行驶和停止，长按回 AUTO。

### FPGA 整机集成

器件 PG2L50H，FBG484，-6，工具 PDS 2022.2-SP6.4。用有授权的底板参考工程和匹配的 PCIe、DDR3、FIFO、PLL IP，按原项目顶层连接用户逻辑，再按板卡版本复核引脚与时钟约束。此源码目录支持用户逻辑仿真，完整 PDS 编译与烧录还需要配套工程、IP、约束与工具授权。关于代码和外部依赖的发布范围见 [THIRD_PARTY.md](THIRD_PARTY.md)。
