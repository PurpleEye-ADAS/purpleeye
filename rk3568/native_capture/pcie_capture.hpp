#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace plate {

struct CaptureTiming {
  double dma_rows_ms = 0.0;
  double rgb_convert_ms = 0.0;
};

class PcieFrameGrabber {
public:
  static constexpr int kImageWidth = 1280;
  static constexpr int kImageHeight = 720;

  explicit PcieFrameGrabber(
      const std::string &device_path = "/dev/pango_pci_driver",
      int dma_delay_loops = 2500);
  ~PcieFrameGrabber();

  PcieFrameGrabber(const PcieFrameGrabber &) = delete;
  PcieFrameGrabber &operator=(const PcieFrameGrabber &) = delete;

  bool isOpen() const { return fd_ >= 0; }
  const std::string &devicePath() const { return device_path_; }

  void printDeviceInfo();
  void setTransferBytes(int bytes);
  // The returned Mat owns its pixel storage and remains valid across captures.
  cv::Mat captureBgr(CaptureTiming *timing = nullptr, uint8_t *pixels = nullptr);

private:
  struct BarBaseInfo {
    unsigned long bar_base;
    unsigned long bar_len;
  };

  struct CapInfo {
    unsigned char flag;
    unsigned char id;
    unsigned char addr_offset;
    unsigned char next_offset;
  };

  struct CapList {
    unsigned char cap_status;
    unsigned char cap_error;
    CapInfo cap_buf[256];
  };

  struct PciDeviceInfo {
    unsigned int vendor_id;
    unsigned int device_id;
    unsigned int cmd_reg;
    unsigned int status_reg;
    unsigned int revision_id;
    unsigned int class_prog;
    unsigned int class_device;
    BarBaseInfo bar[6];
    unsigned int min_gnt;
    unsigned int max_lat;
    unsigned int link_speed;
    unsigned int link_width;
    unsigned int mps;
    unsigned int mrrs;
    unsigned int data[1024];
  };

  struct LoadDataInfo {
    unsigned int num_words;
    unsigned int block_words[1024];
  };

  struct PciLoadInfo {
    unsigned char link_status;
    unsigned int crc;
    unsigned char axi_direction;
    unsigned char load_status;
    unsigned int total_num_words;
    LoadDataInfo data_block;
  };

  struct CommandOperation {
    unsigned char w_r;
    unsigned char step;
    unsigned int addr;
    unsigned int data;
    unsigned int cnt;
    unsigned int delay;
    PciDeviceInfo get_pci_dev_info;
    CapList cap_info;
    PciLoadInfo load_info;
  };

  struct DmaData {
    unsigned char read_buf[4096];
    unsigned char write_buf[4096];
  };

  struct DmaOperation {
    unsigned int current_len;
    unsigned int offset_addr;
    unsigned int cmd;
    DmaData data;
  };

  static void rgb565ToBgr888(const uint16_t *image565, uint8_t *image_bgr,
                             size_t num_pixels);

  void cmdOperation(unsigned char cmd, CommandOperation *cmd_op);
  void mapOnce();
  void unmapOnce();

  std::string device_path_;
  int dma_delay_loops_ = 2500;
  int transfer_bytes_ = kImageWidth * 2;
  int fd_ = -1;
  bool mapped_ = false;
  CommandOperation command_operation_{};
  DmaOperation dma_operation_{};
};

} // namespace plate
