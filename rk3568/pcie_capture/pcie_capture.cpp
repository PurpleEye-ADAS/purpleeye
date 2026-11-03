#include "pcie_capture.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace plate {

namespace {

constexpr unsigned long kType = 'S';
constexpr unsigned long kInfoNum = 3;
constexpr unsigned long kPciMapAddrCmd = _IOWR(kType, 2, int);
constexpr unsigned long kPciDmaWriteCmd = _IOWR(kType, 5, int);
constexpr unsigned long kPciReadFromKernelCmd = _IOWR(kType, 6, int);
constexpr unsigned long kPciUmapAddrCmd = _IOWR(kType, 7, int);
constexpr int kRowBytes = PcieFrameGrabber::kImageWidth * 2;

} // namespace

PcieFrameGrabber::PcieFrameGrabber(const std::string &device_path,
                                   int dma_delay_loops)
    : device_path_(device_path),
      dma_delay_loops_(std::max(0, dma_delay_loops)) {
  fd_ = open(device_path_.c_str(), O_RDWR);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open PCIe driver: " + device_path_);
  }
}

PcieFrameGrabber::~PcieFrameGrabber() {
  if (mapped_) {
    try {
      unmapOnce();
    } catch (const std::exception &e) {
      std::cerr << "Failed to release PCIe DMA buffers: " << e.what() << '\n';
    }
  }
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void PcieFrameGrabber::rgb565ToBgr888(const uint16_t *image565,
                                       uint8_t *image_bgr,
                                       size_t num_pixels) {
  size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
  // 8 pixels/iteration. This preserves the original bit expansion exactly
  // (5-bit channels <<3, 6-bit green <<2) while avoiding a scalar loop over
  // all 921,600 pixels.
  const uint16x8_t mask5 = vdupq_n_u16(0x1f);
  const uint16x8_t mask6 = vdupq_n_u16(0x3f);
  for (; i + 8 <= num_pixels; i += 8) {
    const uint16x8_t px = vld1q_u16(image565 + i);
    const uint8x8_t b5 = vmovn_u16(vandq_u16(px, mask5));
    const uint8x8_t g6 = vmovn_u16(vandq_u16(vshrq_n_u16(px, 5), mask6));
    const uint8x8_t r5 = vmovn_u16(vandq_u16(vshrq_n_u16(px, 11), mask5));
    // Match the verified 3.zip conversion bit-for-bit: replicate the high bits
    // instead of merely left-shifting. This restores the full 0..255 range.
    const uint8x8_t b = vorr_u8(vshl_n_u8(b5, 3), vshr_n_u8(b5, 2));
    const uint8x8_t g = vorr_u8(vshl_n_u8(g6, 2), vshr_n_u8(g6, 4));
    const uint8x8_t r = vorr_u8(vshl_n_u8(r5, 3), vshr_n_u8(r5, 2));
    uint8x8x3_t bgr{};
    bgr.val[0] = b;
    bgr.val[1] = g;
    bgr.val[2] = r;
    vst3_u8(image_bgr + 3 * i, bgr);
  }
#endif
  const size_t tail = num_pixels - i;
  const uint16_t *tail_src = image565 + i;
  uint8_t *tail_dst = image_bgr + 3 * i;
  for (size_t j = 0; j < tail; ++j) {
    const uint16_t pixel = tail_src[j];
    const uint8_t b5 = static_cast<uint8_t>(pixel & 0x1F);
    const uint8_t g6 = static_cast<uint8_t>((pixel >> 5) & 0x3F);
    const uint8_t r5 = static_cast<uint8_t>((pixel >> 11) & 0x1F);
    tail_dst[3 * j] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
    tail_dst[3 * j + 1] = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
    tail_dst[3 * j + 2] = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
  }
}

void PcieFrameGrabber::cmdOperation(unsigned char cmd,
                                    CommandOperation *cmd_op) {
  switch (cmd) {
  case kInfoNum:
    cmd_op->delay = 0;
    if (read(fd_, cmd_op, sizeof(CommandOperation)) < 0) {
      throw std::runtime_error("Failed to read PCIe info");
    }
    if (cmd_op->cap_info.cap_status == 1) {
      std::cout << "PCIe link successful\n";
    } else {
      throw std::runtime_error(
          "PCIe link failure: device node exists but endpoint is not ready");
    }
    break;
  default:
    break;
  }
}

void PcieFrameGrabber::printDeviceInfo() {
  cmdOperation(static_cast<unsigned char>(kInfoNum), &command_operation_);
  std::cout << "The DMA control base address is [ BAR1 ]\n";
}

void PcieFrameGrabber::mapOnce() {
  dma_operation_.current_len = kRowBytes / 4;
  dma_operation_.offset_addr = 0;
  std::memset(dma_operation_.data.write_buf, 0, sizeof(dma_operation_.data.write_buf));
  std::memset(dma_operation_.data.read_buf, 0, sizeof(dma_operation_.data.read_buf));

  if (ioctl(fd_, kPciMapAddrCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_MAP_ADDR_CMD failed");
  }
  mapped_ = true;
}

void PcieFrameGrabber::unmapOnce() {
  if (ioctl(fd_, kPciUmapAddrCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_UMAP_ADDR_CMD failed");
  }
  mapped_ = false;
}

cv::Mat PcieFrameGrabber::captureBgr(CaptureTiming *timing) {
  using Clock = std::chrono::steady_clock;
  auto elapsed_ms = [](const Clock::time_point &a, const Clock::time_point &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  if (timing != nullptr) {
    *timing = {};
  }
  if (!mapped_) {
    mapOnce();
  }

  // Pipeline the line DMA with RGB565 -> BGR conversion. The original code
  // waited for one row, copied it into a 1.84 MB intermediate RGB565 frame,
  // repeated that 720 times, and only then converted the whole frame. Here the
  // next row DMA is started immediately after reading the current row, while
  // the CPU converts the already-copied user-space row. This removes the extra
  // full-frame RGB565 memcpy and overlaps useful CPU work with the next DMA.
  cv::Mat image_bgr(kImageHeight, kImageWidth, CV_8UC3);
  const auto capture_start = Clock::now();
  double convert_cpu_ms = 0.0;

  if (ioctl(fd_, kPciDmaWriteCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_DMA_WRITE_CMD failed");
  }

  for (int i = 0; i < kImageHeight; ++i) {
    for (volatile int k = 0; k < dma_delay_loops_; ++k) {
    }
    if (ioctl(fd_, kPciReadFromKernelCmd, &dma_operation_) < 0) {
      throw std::runtime_error("PCI_READ_FROM_KERNEL_CMD failed");
    }

    // IMPORTANT: snapshot the current row before issuing another ioctl with
    // the same _IOWR dma_operation_ object.  The pango driver is allowed to
    // copy data back through that object on PCI_DMA_WRITE_CMD, so starting the
    // next DMA before consuming data.read_buf can overwrite/corrupt the row.
    // The verified 3.zip path always consumed/copied the row before the next
    // DMA command.  Keeping a 2560-byte private row snapshot preserves that
    // correctness while still allowing the next DMA to overlap RGB conversion.
    alignas(16) unsigned char row_snapshot[kRowBytes];
    std::memcpy(row_snapshot, dma_operation_.data.read_buf, kRowBytes);

    // Once the row is private, it is safe to launch the next hardware transfer
    // immediately and convert the snapshot while DMA progresses.
    if (i + 1 < kImageHeight) {
      if (ioctl(fd_, kPciDmaWriteCmd, &dma_operation_) < 0) {
        throw std::runtime_error("PCI_DMA_WRITE_CMD failed");
      }
    }

    const auto convert_start = Clock::now();
    rgb565ToBgr888(reinterpret_cast<const uint16_t *>(row_snapshot),
                   image_bgr.ptr<uint8_t>(i),
                   static_cast<size_t>(kImageWidth));
    convert_cpu_ms += elapsed_ms(convert_start, Clock::now());
  }

  const auto capture_end = Clock::now();
  if (timing != nullptr) {
    // dma_rows_ms is the wall-clock line-pipeline duration. rgb_convert_ms is
    // accumulated CPU conversion work and is partly overlapped with DMA, so
    // these two fields should not be added together in V4.
    timing->dma_rows_ms = elapsed_ms(capture_start, capture_end);
    timing->rgb_convert_ms = convert_cpu_ms;
  }
  return image_bgr;
}

} // namespace plate

namespace {
thread_local std::string traffic_capture_error_text;
}

extern "C" {

void *traffic_capture_open(const char *device_path, int dma_delay_loops) {
  traffic_capture_error_text.clear();
  try {
    if (device_path == nullptr || dma_delay_loops < 0) {
      throw std::invalid_argument("Invalid PCIe capture arguments");
    }
    return new plate::PcieFrameGrabber(device_path, dma_delay_loops);
  } catch (const std::exception &error) {
    traffic_capture_error_text = error.what();
    return nullptr;
  }
}

int traffic_capture_read(void *handle, unsigned char *pixels, size_t size,
                         double *dma_ms, double *convert_ms) {
  traffic_capture_error_text.clear();
  try {
    constexpr size_t bytes = plate::PcieFrameGrabber::kImageWidth *
                             plate::PcieFrameGrabber::kImageHeight * 3;
    if (handle == nullptr || pixels == nullptr || size != bytes) {
      throw std::invalid_argument("Expected a 1280x720 BGR capture buffer");
    }
    plate::CaptureTiming timing;
    const cv::Mat image = static_cast<plate::PcieFrameGrabber *>(handle)->captureBgr(&timing);
    std::memcpy(pixels, image.data, bytes);
    if (dma_ms != nullptr) *dma_ms = timing.dma_rows_ms;
    if (convert_ms != nullptr) *convert_ms = timing.rgb_convert_ms;
    return 1;
  } catch (const std::exception &error) {
    traffic_capture_error_text = error.what();
    return 0;
  }
}

void traffic_capture_close(void *handle) {
  delete static_cast<plate::PcieFrameGrabber *>(handle);
}

const char *traffic_capture_error() {
  return traffic_capture_error_text.c_str();
}

}
