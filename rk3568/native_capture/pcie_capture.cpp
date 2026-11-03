#include "pcie_capture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

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
  dma_operation_.current_len = transfer_bytes_ / 4;
  dma_operation_.offset_addr = 0;
  std::memset(dma_operation_.data.write_buf, 0, sizeof(dma_operation_.data.write_buf));
  std::memset(dma_operation_.data.read_buf, 0, sizeof(dma_operation_.data.read_buf));

  if (ioctl(fd_, kPciMapAddrCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_MAP_ADDR_CMD failed");
  }
  mapped_ = true;
}

void PcieFrameGrabber::setTransferBytes(int bytes) {
  if (mapped_ || (bytes != kRowBytes && bytes != 3840)) {
    throw std::invalid_argument("DMA chunk must be 2560 or 3840 bytes, set before capture");
  }
  transfer_bytes_ = bytes;
}

void PcieFrameGrabber::unmapOnce() {
  if (ioctl(fd_, kPciUmapAddrCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_UMAP_ADDR_CMD failed");
  }
  mapped_ = false;
}

cv::Mat PcieFrameGrabber::captureBgr(CaptureTiming *timing, uint8_t *pixels) {
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

  // Overlap each DMA chunk with conversion of the previous private snapshot.
  // Chunk boundaries may span rows; the complete RGB565 frame remains intact.
  cv::Mat image_bgr = pixels == nullptr ? cv::Mat(kImageHeight, kImageWidth, CV_8UC3)
                                      : cv::Mat(kImageHeight, kImageWidth, CV_8UC3, pixels);
  const auto capture_start = Clock::now();
  double convert_cpu_ms = 0.0;

  if (ioctl(fd_, kPciDmaWriteCmd, &dma_operation_) < 0) {
    throw std::runtime_error("PCI_DMA_WRITE_CMD failed");
  }

  constexpr int frame_bytes = kImageWidth * kImageHeight * 2;
  for (int offset = 0; offset < frame_bytes; offset += transfer_bytes_) {
    for (volatile int k = 0; k < dma_delay_loops_; ++k) {
    }
    if (ioctl(fd_, kPciReadFromKernelCmd, &dma_operation_) < 0) {
      throw std::runtime_error("PCI_READ_FROM_KERNEL_CMD failed");
    }

    // IMPORTANT: snapshot the current chunk before issuing another ioctl with
    // the same _IOWR dma_operation_ object.  The pango driver is allowed to
    // copy data back through that object on PCI_DMA_WRITE_CMD, so starting the
    // next DMA before consuming data.read_buf can overwrite/corrupt the data.
    // The private snapshot preserves ownership across the next DMA command.
    alignas(16) unsigned char row_snapshot[3840];
    std::memcpy(row_snapshot, dma_operation_.data.read_buf, transfer_bytes_);

    // Once the row is private, it is safe to launch the next hardware transfer
    // immediately and convert the snapshot while DMA progresses.
    if (offset + transfer_bytes_ < frame_bytes) {
      if (ioctl(fd_, kPciDmaWriteCmd, &dma_operation_) < 0) {
        throw std::runtime_error("PCI_DMA_WRITE_CMD failed");
      }
    }

    const auto convert_start = Clock::now();
    rgb565ToBgr888(reinterpret_cast<const uint16_t *>(row_snapshot),
                   image_bgr.data + offset * 3 / 2,
                   static_cast<size_t>(transfer_bytes_ / 2));
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

struct SignalAlignment {
  uint64_t source=0;int width=0,height=0,original_width=0,original_height=0;
  cv::Point start,end;cv::Mat previous,points;std::vector<double> boxes;
  std::vector<cv::Mat> previous_pyramid;
};
}

extern "C" {

void *traffic_alignment_open(){return new SignalAlignment;}
void traffic_alignment_close(void *handle){delete static_cast<SignalAlignment *>(handle);}

int traffic_align_boxes(void *handle,const unsigned char *source,const unsigned char *current,
                       int width,int height,int source_stride,int current_stride,uint64_t source_id,uint64_t current_id,
                       const double *boxes,int count,int original_width,int original_height,
                       double *output,int *kept,double *motion) {
  try {
    auto &state=*static_cast<SignalAlignment *>(handle);
    const float gx=float(width)/original_width,gy=float(height)/original_height;
    if(state.source!=source_id||state.width!=width||state.height!=height||
       state.original_width!=original_width||state.original_height!=original_height||
       state.boxes!=std::vector<double>(boxes,boxes+4*count)) {
      std::vector<std::array<float,4>> scaled(count);float largest=0;
      float low_x=1e30f,low_y=1e30f,high_x=-1e30f,high_y=-1e30f;
      for(int i=0;i<count;++i) {
        auto &b=scaled[i];for(int c=0;c<4;++c)b[c]=float(boxes[4*i+c])*(c%2?gy:gx);
        largest=std::max({largest,b[2]-b[0],b[3]-b[1]});
        low_x=std::min(low_x,b[0]);low_y=std::min(low_y,b[1]);
        high_x=std::max(high_x,b[2]);high_y=std::max(high_y,b[3]);
      }
      const int padding=std::max(24,int(largest*2));
      state.start=cv::Point(std::max(0,int(std::floor(low_x))-padding),std::max(0,int(std::floor(low_y))-padding));
      state.end=cv::Point(std::min(width,int(std::ceil(high_x))+padding),std::min(height,int(std::ceil(high_y))+padding));
      if(state.end.x<=state.start.x||state.end.y<=state.start.y)return 0;
      cv::Mat input(height,width,CV_8UC1,const_cast<unsigned char *>(source),source_stride);
      state.previous=input(cv::Rect(state.start,state.end)).clone();
      state.previous_pyramid.clear();
      cv::Mat mask(state.previous.size(),CV_8UC1,cv::Scalar(0));
      for(const auto &b:scaled) {
        const int margin=std::max(5,int((b[3]-b[1])/2));
        cv::rectangle(mask,cv::Point(int(std::floor(double(b[0])-state.start.x-margin)),int(std::floor(double(b[1])-state.start.y-margin))),
                      cv::Point(int(std::ceil(double(b[2])-state.start.x+margin)),int(std::ceil(double(b[3])-state.start.y+margin))),cv::Scalar(255),-1);
      }
      for(const auto &b:scaled)
        cv::rectangle(mask,cv::Point(int(std::floor(double(b[0])-state.start.x)),int(std::floor(double(b[1])-state.start.y))),
                      cv::Point(int(std::ceil(double(b[2])-state.start.x)),int(std::ceil(double(b[3])-state.start.y))),cv::Scalar(0),-1);
      cv::goodFeaturesToTrack(state.previous,state.points,30,.02,4,mask,3);
      state.source=source_id;state.width=width;state.height=height;
      state.original_width=original_width;state.original_height=original_height;
      state.boxes.assign(boxes,boxes+count*4);
    }
    if(state.points.total()<6)return 0;
    cv::Mat image(height,width,CV_8UC1,const_cast<unsigned char *>(current),current_stride);
    cv::Mat live=image(cv::Rect(state.start,state.end)).clone(),moved,returned,forward,backward,error;
    auto &first=state.previous_pyramid;
    std::vector<cv::Mat> second;
    if(first.empty())cv::buildOpticalFlowPyramid(state.previous,first,cv::Size(15,15),3);
    cv::buildOpticalFlowPyramid(live,second,cv::Size(15,15),3);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,15,.02);
    cv::calcOpticalFlowPyrLK(first,second,state.points,moved,forward,error,cv::Size(15,15),3,criteria);
    cv::calcOpticalFlowPyrLK(second,first,moved,returned,backward,error,cv::Size(15,15),3,criteria);
    std::vector<cv::Point2f> old_points,new_points;
    for(int i=0;i<int(state.points.total());++i) {
      const auto p=state.points.at<cv::Point2f>(i),q=moved.at<cv::Point2f>(i),back=returned.at<cv::Point2f>(i);
      const float dx=back.x-p.x,dy=back.y-p.y,distance=std::sqrt(dx*dx+dy*dy);
      if(forward.at<unsigned char>(i)&&backward.at<unsigned char>(i)&&
         std::isfinite(q.x)&&std::isfinite(q.y)&&std::isfinite(back.x)&&std::isfinite(back.y)&&distance<1.f) {
        old_points.push_back(p);new_points.push_back(q);
      }
    }
    if(old_points.size()<6)return 0;
    cv::Mat inliers,matrix=cv::estimateAffinePartial2D(old_points,new_points,inliers,cv::RANSAC,1.5,100,.95,5);
    if(matrix.empty()||!cv::checkRange(matrix)||cv::countNonZero(inliers)<std::max(6.,old_points.size()*.6))return 0;
    const double a=matrix.at<double>(0,0),b=matrix.at<double>(0,1),c=matrix.at<double>(1,0),d=matrix.at<double>(1,1);
    const double x=matrix.at<double>(0,2)+state.start.x-(a*state.start.x+b*state.start.y);
    const double y=matrix.at<double>(1,2)+state.start.y-(c*state.start.x+d*state.start.y);
    const float inverse_x=1.f/gx,inverse_y=1.f/gy;
    const double m[9]={inverse_x*a*gx,inverse_x*b*gy,inverse_x*x,
                       inverse_y*c*gx,inverse_y*d*gy,inverse_y*y,0,0,1};
    std::copy(m,m+9,motion);int found=0;
    for(int i=0;i<count;++i) {
      const float x1=float(boxes[i*4]),y1=float(boxes[i*4+1]),x2=float(boxes[i*4+2]),y2=float(boxes[i*4+3]);
      const std::array<cv::Point2f,4> corners={cv::Point2f(x1,y1),cv::Point2f(x2,y1),cv::Point2f(x2,y2),cv::Point2f(x1,y2)};
      double left=1e100,top=1e100,right=-1e100,bottom=-1e100;bool visible=true;
      for(auto p:corners) {
        const double px=(p.x*m[0]+p.y*m[1])+m[2],py=(p.x*m[3]+p.y*m[4])+m[5];
        visible=visible&&std::isfinite(px)&&std::isfinite(py);
        left=std::min(left,px);top=std::min(top,py);right=std::max(right,px);bottom=std::max(bottom,py);
      }
      if(visible&&left>=0&&top>=0&&right<=original_width&&bottom<=original_height&&
         right-left>=2.f/gx&&bottom-top>=2.f/gy) {
        output[4*found]=left;output[4*found+1]=top;output[4*found+2]=right;output[4*found+3]=bottom;
        kept[found++]=i;
      }
    }
    if(found) {
      std::vector<cv::Point2f> next;
      for(size_t i=0;i<new_points.size();++i)if(inliers.at<unsigned char>(int(i)))next.push_back(new_points[i]);
      state.points=cv::Mat(next).clone();state.previous=live;state.source=current_id;
      state.previous_pyramid.swap(second);
      state.boxes.assign(output,output+found*4);
    }
    return found;
  } catch(const std::exception &error) {
    traffic_capture_error_text=error.what();return -1;
  }
}

int traffic_sparse_flow(const unsigned char *previous,const unsigned char *current,int width,int height,
                        const float *points,int count,float *moved,float *returned,
                        unsigned char *forward,unsigned char *backward) {
  try {
    cv::Mat a(height,width,CV_8UC1,const_cast<unsigned char *>(previous));
    cv::Mat b(height,width,CV_8UC1,const_cast<unsigned char *>(current));
    cv::Mat p(count,1,CV_32FC2,const_cast<float *>(points));
    cv::Mat q(count,1,CV_32FC2,moved),back(count,1,CV_32FC2,returned);
    cv::Mat valid_forward(count,1,CV_8UC1,forward),valid_backward(count,1,CV_8UC1,backward),error;
    std::vector<cv::Mat> first,second;
    cv::buildOpticalFlowPyramid(a,first,cv::Size(15,15),3);
    cv::buildOpticalFlowPyramid(b,second,cv::Size(15,15),3);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,15,.02);
    cv::calcOpticalFlowPyrLK(first,second,p,q,valid_forward,error,cv::Size(15,15),3,criteria);
    cv::calcOpticalFlowPyrLK(second,first,q,back,valid_backward,error,cv::Size(15,15),3,criteria);
    return 1;
  } catch(const std::exception &error) {
    traffic_capture_error_text=error.what();return -1;
  }
}

int traffic_light_color(const unsigned char *pixels,int width,int height,int stride,double *confidence) {
  try {
    *confidence=0;
    cv::Mat image(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride),hsv;
    cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;cv::split(hsv,channels);
    const cv::Mat &value=channels[2];cv::Mat saturated;
    cv::inRange(channels[1],60,255,saturated);
    double peak;cv::minMaxLoc(value,nullptr,&peak,nullptr,nullptr,saturated);
    if(peak<140)return -1;
    if(peak<180) {
      std::array<int,256> histogram{};
      for(int y=0;y<height;++y)for(int x=0;x<width;++x)++histogram[value.at<unsigned char>(y,x)];
      const int n=width*height;int total=0,lo=-1,hi=-1;
      for(int v=0;v<256;++v) {
        total+=histogram[v];if(lo<0&&total>(n-1)/2)lo=v;
        if(total>n/2){hi=v;break;}
      }
      if(peak-(lo+hi)/2.<55)return -1;
    }
    const int minimum=std::max(140,(int(peak)*4+4)/5),component_minimum=std::max(140,cvRound(peak*.6));
    std::array<cv::Mat,3> masks;cv::Mat extra;
    cv::inRange(hsv,cv::Scalar(0,60,component_minimum),cv::Scalar(12,255,255),masks[0]);
    cv::inRange(hsv,cv::Scalar(165,60,component_minimum),cv::Scalar(179,255,255),extra);
    cv::bitwise_or(masks[0],extra,masks[0]);
    cv::inRange(hsv,cv::Scalar(13,60,component_minimum),cv::Scalar(35,255,255),masks[1]);
    cv::inRange(hsv,cv::Scalar(36,60,component_minimum),cv::Scalar(100,255,255),masks[2]);
    std::array<std::array<int,6>,3> components{};std::array<int,3> scores{};
    for(int i=0;i<3;++i) {
      cv::Mat labels,stats,centroids;
      const int count=cv::connectedComponentsWithStats(masks[i],labels,stats,centroids,8);
      std::vector<int> seeds(count);
      for(int y=0;y<height;++y)for(int x=0;x<width;++x)
        if(value.at<unsigned char>(y,x)>=minimum)++seeds[labels.at<int>(y,x)];
      for(int label=1;label<count;++label) {
        const int x=stats.at<int>(label,cv::CC_STAT_LEFT),y=stats.at<int>(label,cv::CC_STAT_TOP);
        const int w=stats.at<int>(label,cv::CC_STAT_WIDTH),h=stats.at<int>(label,cv::CC_STAT_HEIGHT);
        const int area=stats.at<int>(label,cv::CC_STAT_AREA),seed=seeds[label];const double center=x+w/2.;
        if(center>=.2*width-.5&&center<=.8*width+.5&&h<=4*w&&seed&&
           (scores[i]==0||std::make_pair(seed,area)>std::make_pair(scores[i],components[i][4]))) {
          components[i]={x,y,w,h,area,seed};scores[i]=seed;
        }
      }
    }
    int best=0;for(int i=1;i<3;++i)if(scores[i]>scores[best])best=i;
    const auto &component=components[best];const int sum=scores[0]+scores[1]+scores[2];
    const double ratio=double(scores[best])/std::max(1,sum);
    if(!scores[best]||component[4]<std::max(best==2?3.:2.,height*width*.004)||
       (best==2&&peak<180)||(component[4]==2&&component[5]<2)||ratio<.75)return -1;
    if(best==0&&height>=width*1.7) {
      const double center=(component[1]+component[3]/2.)/height;
      if(center>=.35&&center<=.65) {
        double green=0,red=0;int count=0;
        for(int y=std::max(0,component[1]-1);y<std::min(height,component[1]+component[3]+1);++y)
          for(int x=std::max(0,component[0]-1);x<std::min(width,component[0]+component[2]+1);++x) {
            auto color=image.at<cv::Vec3b>(y,x);
            if(std::max({color[0],color[1],color[2]})>=std::max(140.,peak*.8)) {
              green+=color[1];red+=color[2];++count;
            }
          }
        if(count&&green/count>=(red/count)*.5)best=1;
      }
    }
    *confidence=ratio;return best;
  } catch(const std::exception &error) {
    traffic_capture_error_text=error.what();return -2;
  }
}

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
    static_cast<plate::PcieFrameGrabber *>(handle)->captureBgr(&timing, pixels);
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

int traffic_capture_set_chunk(void *handle, int bytes) {
  traffic_capture_error_text.clear();
  try {
    if (handle == nullptr) throw std::invalid_argument("Missing capture handle");
    static_cast<plate::PcieFrameGrabber *>(handle)->setTransferBytes(bytes);
    return 1;
  } catch (const std::exception &error) {
    traffic_capture_error_text = error.what();
    return 0;
  }
}

const char *traffic_capture_error() {
  return traffic_capture_error_text.c_str();
}

int traffic_resize_bgr(unsigned char *src, int width, int height,
                       unsigned char *dst, int target_width, int target_height) {
  if (width == 2*target_width && height == 2*target_height) {
    for (int y = 0; y < target_height; ++y) {
      const unsigned char *a=src+2*y*width*3,*b=a+width*3;
      unsigned char *out=dst+y*target_width*3;
      int x=0;
#if defined(__aarch64__) || defined(__ARM_NEON)
      for (; x+8<=target_width; x+=8) {
        const uint8x16x3_t p=vld3q_u8(a+6*x),q=vld3q_u8(b+6*x);
        uint8x8x3_t average{};
        for (int c=0;c<3;++c)
          average.val[c]=vrshrn_n_u16(vaddq_u16(vpaddlq_u8(p.val[c]),vpaddlq_u8(q.val[c])),2);
        vst3_u8(out+3*x,average);
      }
#endif
      for (;x<target_width;++x)
        for (int c=0;c<3;++c)out[3*x+c]=(a[6*x+c]+a[6*x+3+c]+b[6*x+c]+b[6*x+3+c]+2)/4;
    }
  } else {
    cv::Mat input(height,width,CV_8UC3,src),output(target_height,target_width,CV_8UC3,dst);
    cv::resize(input,output,output.size());
  }
  return 1;
}

int traffic_gray_flow(const unsigned char *previous, const unsigned char *pixels,
                      int width, int height, int stride, unsigned char *gray_output, float *flow_output) {
  try {
    const cv::Mat input(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride);
    cv::Mat gray(height/2,width/2,CV_8UC1,gray_output),converted;
    cv::cvtColor(input,converted,cv::COLOR_BGR2GRAY);
    cv::resize(converted,gray,gray.size());
    if (previous == nullptr)return 0;
    const cv::Mat before(gray.rows,gray.cols,CV_8UC1,const_cast<unsigned char *>(previous));
    if (cv::norm(before,gray,cv::NORM_L1)/gray.total()>35)return 2;
    thread_local cv::Ptr<cv::DISOpticalFlow> optical_flow=[] {
      auto algorithm=cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
      algorithm->setFinestScale(0);algorithm->setPatchStride(6);return algorithm;
    }();
    cv::Mat result;optical_flow->calc(before,gray,result);
    std::memcpy(flow_output,result.data,result.total()*result.elemSize());
    return 1;
  } catch (const std::exception &error) {
    traffic_capture_error_text=error.what();return -1;
  }
}

int traffic_paint_lane(unsigned char *pixels,int width,int height,int stride,
                       const unsigned char *mask,int mask_width,int mask_height,int blue,int green,int red) {
  try {
    cv::Mat frame(height,width,CV_8UC3,pixels,stride);
    cv::Mat selected(mask_height,mask_width,CV_8UC1,const_cast<unsigned char *>(mask)),resized;
    if (selected.size()!=frame.size()) {
      cv::resize(selected,resized,frame.size(),0,0,cv::INTER_NEAREST);selected=resized;
    }
    int left=width,right=0,top=height,bottom=0;
    const cv::Rect bounds=cv::boundingRect(selected);
    if(bounds.empty())return 1;
    const double colors[3]={.6*blue,.6*green,.6*red};
    for (int y=bounds.y;y<bounds.y+bounds.height;++y) {
      auto *row=frame.ptr<cv::Vec3b>(y);const auto *chosen=selected.ptr<unsigned char>(y);
      for (int x=bounds.x;x<bounds.x+bounds.width;++x)if(chosen[x]) {
        for (int c=0;c<3;++c)row[x][c]=static_cast<unsigned char>(.4*row[x][c]+colors[c]);
        left=std::min(left,x);right=std::max(right,x+1);top=std::min(top,y);bottom=std::max(bottom,y+1);
      }
    }
    if(right>left) {
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(selected(cv::Rect(left,top,right-left,bottom-top)),contours,
                       cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE,cv::Point(left,top));
      cv::drawContours(frame,contours,-1,cv::Scalar(blue,green,red),1);
    }
    return 1;
  } catch(const std::exception &error) {
    traffic_capture_error_text=error.what();return -1;
  }
}

int traffic_lane_evidence(const unsigned char *pixels, const unsigned char *gray_pixels,
                          const unsigned char *hsv_pixels, int width, int height, int stride,
                          double slope, double base_x, int paint_top,
                          unsigned char *mask_pixels, double *metrics) {
  try {
    if (!pixels || !gray_pixels || !mask_pixels || !metrics || width<=0 || height<4 ||
        stride<width*3 || !std::isfinite(slope) || !std::isfinite(base_x))
      throw std::invalid_argument("Invalid lane evidence input");
    const cv::Mat image(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride);
    const cv::Mat gray(height,width,CV_8UC1,const_cast<unsigned char *>(gray_pixels));
    cv::Mat supplied_hsv;
    if(hsv_pixels)supplied_hsv=cv::Mat(height,width,CV_8UC3,const_cast<unsigned char *>(hsv_pixels));
    cv::Mat mask(height,width,CV_8UC1,mask_pixels);mask.setTo(0);
    const int start=cvRound(height*.45),end=cvRound(height*.92),rows=end-start;
    cv::Mat gathered(rows,17,CV_8UC3),colors;
    std::vector<int> positions(rows*17),centers_radius(rows);
    std::vector<unsigned char> inside(rows*17);
    const std::array<int,6> flank_offsets={-15,-12,-10,10,12,15};
    std::vector<double> background(rows);
    for(int row=0;row<rows;++row) {
      const int y=start+row;
      const double center=(base_x+slope*(double(y)/height-.86))*width;
      centers_radius[row]=std::max(2,cvRound(width*(.002+.007*y/height)));
      std::array<unsigned char,6> flank;
      for(int i=0;i<6;++i)flank[i]=gray.at<unsigned char>(y,
          std::clamp(cvRound(center+flank_offsets[i]),0,width-1));
      std::sort(flank.begin(),flank.end());background[row]=(int(flank[2])+int(flank[3]))/2.;
      for(int j=0;j<17;++j) {
        const int x=cvRound(center+(j-8)),clipped=std::clamp(x,0,width-1);
        positions[row*17+j]=clipped;inside[row*17+j]=x>=0&&x<width;
        gathered.at<cv::Vec3b>(row,j)=(hsv_pixels?supplied_hsv:image).at<cv::Vec3b>(y,clipped);
      }
    }
    if(hsv_pixels)colors=gathered;
    else cv::cvtColor(gathered,colors,cv::COLOR_BGR2HSV);
    int left=width,right=0;
    for(int row=0;row<rows;++row)for(int j=0;j<17;++j) {
      const int x=positions[row*17+j],y=start+row;
      const auto color=colors.at<cv::Vec3b>(row,j);
      const bool white=color[1]<120&&color[2]>60;
      const bool yellow=color[0]>=12&&color[0]<=42&&color[1]>65&&color[2]>60;
      const bool paint=inside[row*17+j]&&std::abs(j-8)<=centers_radius[row]&&(white||yellow)&&
                       double(gray.at<unsigned char>(y,x))-background[row]>=16;
      // Preserve advanced-index assignment order, including repeated clipped columns.
      mask.at<unsigned char>(y,x)=paint;
      if(paint){left=std::min(left,x);right=std::max(right,x+1);}
    }
    if(right>left) {
      const cv::Rect bounds(left,start,right-left,rows);
      cv::Mat labels,statistics,centroids;
      cv::connectedComponentsWithStats(mask(bounds),labels,statistics,centroids,8,CV_32S);
      std::vector<unsigned char> keep(statistics.rows,0);
      const double expected=slope*width/height;
      for(int label=1;label<statistics.rows;++label) {
        const int x=statistics.at<int>(label,cv::CC_STAT_LEFT),y=statistics.at<int>(label,cv::CC_STAT_TOP);
        const int w=statistics.at<int>(label,cv::CC_STAT_WIDTH),h=statistics.at<int>(label,cv::CC_STAT_HEIGHT);
        const int area=statistics.at<int>(label,cv::CC_STAT_AREA);
        if(h<std::max(3,cvRound(height*.008))||area<std::max(4,cvRound(height*width*.000018)))continue;
        double sx=0,sy=0;
        for(int yy=0;yy<h;++yy)for(int xx=0;xx<w;++xx)
          if(labels.at<int>(y+yy,x+xx)==label){sx+=xx;sy+=yy;}
        const double mx=sx/area,my=sy/area;
        double covariance=0,variance=0;
        for(int yy=0;yy<h;++yy)for(int xx=0;xx<w;++xx)if(labels.at<int>(y+yy,x+xx)==label) {
          const double dy=yy-my;covariance+=dy*(xx-mx);variance+=dy*dy;
        }
        const double measured=covariance/variance;
        keep[label]=(1+measured*expected)/std::sqrt((1+measured*measured)*(1+expected*expected))>=.9;
      }
      for(int y=0;y<rows;++y)for(int x=left;x<right;++x)
        mask.at<unsigned char>(start+y,x)=keep[labels.at<int>(y,x-left)];
    }
    std::vector<std::pair<int,int>> runs;
    int occupied_count=0,run_start=-1;
    for(int row=0;row<=rows;++row) {
      const bool occupied=row<rows&&cv::countNonZero(mask.row(start+row))!=0;
      if(occupied){++occupied_count;if(run_start<0)run_start=row;}
      else if(run_start>=0) {
        if(row-run_start>=std::max(3,cvRound(height*.02)))runs.emplace_back(run_start,row);
        run_start=-1;
      }
    }
    const bool visible=occupied_count>=std::max(6,cvRound(height*.017));
    const double coverage=double(occupied_count)/rows;
    bool long_run=false;int gaps=0;
    for(size_t i=0;i<runs.size();++i) {
      long_run|=runs[i].second-runs[i].first>=height*.20;
      if(i+1<runs.size()&&runs[i+1].first-runs[i].second>=height*.018)++gaps;
    }
    const int clear_rows=std::clamp(paint_top<0?height+paint_top:paint_top,0,height);
    if(clear_rows>0)mask.rowRange(0,clear_rows).setTo(0);
    metrics[0]=visible;metrics[1]=visible&&coverage>=.74&&long_run;
    metrics[2]=visible&&gaps>=2;metrics[3]=coverage;
    return 1;
  } catch(const std::exception &error) {
    traffic_capture_error_text=error.what();return -1;
  }
}

int traffic_move_people(const unsigned char *previous, const unsigned char *current,
                        const float *motion_field, int width, int height,
                        const double *people, int count, int columns, double fraction,
                        int frame_width, int frame_height, double *output) {
  try {
    const cv::Mat before(height, width, CV_8UC1, const_cast<unsigned char *>(previous));
    const cv::Mat after(height, width, CV_8UC1, const_cast<unsigned char *>(current));
    const cv::Mat flow(height, width, CV_32FC2, const_cast<float *>(motion_field));
    int kept = 0;
    for (int index = 0; index < count; ++index) {
      const double *person = people + index * columns;
      double box[4] = {person[0]/2, person[1]/2, person[2]/2, person[3]/2};
      const int left = std::max(0, cvRound(box[0]+(box[2]-box[0])*.15));
      const int top = std::max(0, cvRound(box[1]+(box[3]-box[1])*.08));
      const int right = std::min(width, cvRound(box[2]-(box[2]-box[0])*.15));
      const int bottom = std::min(height, cvRound(box[3]-(box[3]-box[1])*.4));
      if (right > left && bottom > top) {
        std::vector<float> dx, dy;
        dx.reserve((right-left)*(bottom-top));dy.reserve(dx.capacity());
        for (int y = top; y < bottom; ++y) {
          const auto *row = flow.ptr<cv::Vec2f>(y);
          for (int x = left; x < right; ++x) {dx.push_back(row[x][0]);dy.push_back(row[x][1]);}
        }
        auto median = [](std::vector<float> &values) {
          const size_t middle = values.size()/2;
          std::nth_element(values.begin(), values.begin()+middle, values.end());
          const float high = values[middle];
          return values.size()%2 ? high : (high+*std::max_element(values.begin(),values.begin()+middle))/2.f;
        };
        const float mx = median(dx)*static_cast<float>(fraction);
        const float my = median(dy)*static_cast<float>(fraction);
        box[0]+=mx;box[2]+=mx;box[1]+=my;box[3]+=my;
        const cv::Mat core = before(cv::Rect(left,top,right-left,bottom-top));
        cv::Mat patch, score;
        cv::getRectSubPix(after, core.size(), cv::Point2f((left+right-1)/2.0+mx,(top+bottom-1)/2.0+my), patch);
        cv::matchTemplate(patch,core,score,cv::TM_CCOEFF_NORMED);
        cv::Scalar mean, deviation;cv::meanStdDev(core,mean,deviation);
        if (deviation[0]>3 && score.at<float>(0,0)<.2 && cv::norm(patch,core,cv::NORM_L1)/core.total()>50)
          continue;
      }
      for (double &value : box) value*=2;
      const double clipped[4] = {std::clamp(box[0],0.,static_cast<double>(frame_width)),
                                std::clamp(box[1],0.,static_cast<double>(frame_height)),
                                std::clamp(box[2],0.,static_cast<double>(frame_width)),
                                std::clamp(box[3],0.,static_cast<double>(frame_height))};
      if (clipped[2]-clipped[0]<std::max(2.,(box[2]-box[0])*.4) || clipped[3]-clipped[1]<8)
        continue;
      std::memcpy(output+kept*columns, person, columns*sizeof(double));
      std::memcpy(output+kept*columns, clipped, sizeof(clipped));
      ++kept;
    }
    return kept;
  } catch (const std::exception &error) {
    traffic_capture_error_text = error.what();
    return -1;
  }
}

}
