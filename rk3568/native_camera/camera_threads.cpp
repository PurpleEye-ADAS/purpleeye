// 姓名：丁黄炎
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <array>
#include <vector>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#undef None
#undef Status

namespace {
const bool x11_threads_ready=XInitThreads()!=0;
thread_local std::string camera_error;
struct WarpTarget {
  GLuint source=0,texture=0,framebuffer=0;
  cv::Mat rgba;
};
struct CameraWarp {
  EGLDisplay display=EGL_NO_DISPLAY;
  EGLContext context=EGL_NO_CONTEXT;
  EGLSurface surface=EGL_NO_SURFACE;
  GLuint program=0;
  std::map<std::pair<int,int>,WarpTarget> targets;
  static void require(bool condition,const char *message) {
    if(!condition)throw std::runtime_error(message);
  }
  GLuint shader(GLenum kind,const char *source) {
    GLuint value=glCreateShader(kind);glShaderSource(value,1,&source,nullptr);glCompileShader(value);
    GLint ready=0;glGetShaderiv(value,GL_COMPILE_STATUS,&ready);
    if(!ready) {
      char log[2048];glGetShaderInfoLog(value,sizeof(log),nullptr,log);glDeleteShader(value);
      throw std::runtime_error(log);
    }
    return value;
  }
  void initialize() {
    display=eglGetDisplay(EGL_DEFAULT_DISPLAY);EGLint major,minor,count;
    require(eglInitialize(display,&major,&minor),"Camera EGL initialization failed");
    require(eglBindAPI(EGL_OPENGL_ES_API),"Camera EGL API selection failed");
    const EGLint attributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_NONE};
    EGLConfig config;
    require(eglChooseConfig(display,attributes,&config,1,&count)&&count,"Camera EGL configuration failed");
    const EGLint size[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
    surface=eglCreatePbufferSurface(display,config,size);
    const EGLint version[]={EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE};
    context=eglCreateContext(display,config,EGL_NO_CONTEXT,version);
    require(eglMakeCurrent(display,surface,surface,context),"Camera EGL context failed");
    const char *renderer=reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    require(renderer&&std::string(renderer).find("Mali")!=std::string::npos,"Camera GPU is not the verified Mali renderer");
    const char *vertex="attribute vec2 position;void main(){gl_Position=vec4(position,0.0,1.0);}";
    const char *fragment=R"(
      precision highp float;
      uniform sampler2D image;uniform mat3 inverse_map;uniform vec2 dimensions;
      void main(){
        vec3 point=inverse_map*vec3(gl_FragCoord.xy-vec2(0.5),1.0);
        vec2 source=floor((point.xy/point.z)*32.0+vec2(0.5))/32.0;
        vec2 weight=clamp(source+vec2(1.0),0.0,1.0)*clamp(dimensions-source,0.0,1.0);
        gl_FragColor=vec4(texture2D(image,(source+vec2(0.5))/dimensions).rgb*weight.x*weight.y,1.0);
      })";
    GLuint vs=shader(GL_VERTEX_SHADER,vertex),fs=shader(GL_FRAGMENT_SHADER,fragment);
    program=glCreateProgram();glAttachShader(program,vs);glAttachShader(program,fs);
    glBindAttribLocation(program,0,"position");glLinkProgram(program);glDeleteShader(vs);glDeleteShader(fs);
    GLint linked=0;glGetProgramiv(program,GL_LINK_STATUS,&linked);require(linked,"Camera GPU program link failed");
    glUseProgram(program);glUniform1i(glGetUniformLocation(program,"image"),0);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);glPixelStorei(GL_PACK_ALIGNMENT,1);
    glDisable(GL_DITHER);glDisable(GL_BLEND);glDisable(GL_DEPTH_TEST);
    eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
  }
  void render(const unsigned char *pixels,int width,int height,int stride,const double *matrix,unsigned char *output) {
    require(eglMakeCurrent(display,surface,surface,context),"Camera GPU context activation failed");
    struct CurrentGuard {
      EGLDisplay display;
      ~CurrentGuard(){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);}
    } guard{display};
    auto key=std::make_pair(width,height);auto iterator=targets.find(key);
    if(iterator==targets.end()) {
      WarpTarget target;target.rgba.create(height,width,CV_8UC4);
      glGenTextures(1,&target.source);glBindTexture(GL_TEXTURE_2D,target.source);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
      glGenTextures(1,&target.texture);glBindTexture(GL_TEXTURE_2D,target.texture);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
      glGenFramebuffers(1,&target.framebuffer);glBindFramebuffer(GL_FRAMEBUFFER,target.framebuffer);
      glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,target.texture,0);
      require(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"Camera GPU framebuffer failed");
      iterator=targets.emplace(key,std::move(target)).first;
    }
    auto &target=iterator->second;
    cv::Mat source(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride);
    if(!source.isContinuous())source=source.clone();
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,target.source);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,source.data);
    glBindFramebuffer(GL_FRAMEBUFFER,target.framebuffer);glViewport(0,0,width,height);
    GLfloat inverse[9];for(int row=0;row<3;++row)for(int col=0;col<3;++col)inverse[col*3+row]=matrix[row*3+col];
    glUseProgram(program);glUniformMatrix3fv(glGetUniformLocation(program,"inverse_map"),1,GL_FALSE,inverse);
    glUniform2f(glGetUniformLocation(program,"dimensions"),width,height);
    const GLfloat vertices[]={-1,-1,1,-1,-1,1,1,1};
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,vertices);glEnableVertexAttribArray(0);
    glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,target.rgba.data);
    require(glGetError()==GL_NO_ERROR,"Camera GPU pixel transfer failed");
    cv::Mat destination(height,width,CV_8UC3,output);
    cv::cvtColor(target.rgba,destination,cv::COLOR_RGBA2RGB);
  }
  ~CameraWarp() {
    if(display==EGL_NO_DISPLAY)return;
    if(context!=EGL_NO_CONTEXT) {
      eglMakeCurrent(display,surface,surface,context);
      for(auto &entry:targets) {
        glDeleteTextures(1,&entry.second.source);glDeleteTextures(1,&entry.second.texture);
        glDeleteFramebuffers(1,&entry.second.framebuffer);
      }
      if(program)glDeleteProgram(program);
      eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);eglDestroyContext(display,context);
    }
    if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);
    eglTerminate(display);
  }
};
}

extern "C" const char *traffic_camera_error(){return camera_error.c_str();}
extern "C" int traffic_camera_configure_threads(){
  cv::setNumThreads(1);return cv::getNumThreads();
}
extern "C" void *traffic_camera_warp_open(){
  try {auto renderer=std::make_unique<CameraWarp>();renderer->initialize();return renderer.release();}
  catch(const std::exception &error){camera_error=error.what();return nullptr;}
}
extern "C" void traffic_camera_warp_close(void *handle){delete static_cast<CameraWarp *>(handle);}
extern "C" int traffic_camera_warp(void *handle,const unsigned char *pixels,int width,int height,int stride,
                                    const double *inverse,unsigned char *output){
  try {static_cast<CameraWarp *>(handle)->render(pixels,width,height,stride,inverse,output);return 1;}
  catch(const std::exception &error){camera_error=error.what();return -1;}
}

extern "C" int traffic_camera_color(const unsigned char *pixels, int width, int height,
                                    int stride, double *confidence) {
  try {
    *confidence=0;
    cv::Mat image(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride),hsv;
    cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;cv::split(hsv,channels);
    const cv::Mat &value=channels[2];cv::Mat saturated;
    cv::inRange(channels[1],60,255,saturated);
    double peak;cv::minMaxLoc(value,nullptr,&peak,nullptr,nullptr,saturated);
    if(peak<100)return -1;
    std::array<int,256> histogram{};
    for(int y=0;y<height;++y)for(int x=0;x<width;++x)++histogram[value.at<unsigned char>(y,x)];
    const int n=width*height;int total=0,lo=-1,hi=-1;
    for(int v=0;v<256;++v) {
      total+=histogram[v];if(lo<0&&total>(n-1)/2)lo=v;
      if(total>n/2){hi=v;break;}
    }
    const double contrast=peak-(lo+hi)/2.;
    if(contrast<30||(peak<140&&contrast<55))return -1;
    const int minimum=std::max(100,(int(peak)*4+4)/5);
    const int component_minimum=std::max(100,cvRound(peak*.6));
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
           (!scores[i]||std::make_pair(seed,area)>std::make_pair(scores[i],components[i][4]))) {
          components[i]={x,y,w,h,area,seed};scores[i]=seed;
        }
      }
    }
    int best=0;for(int i=1;i<3;++i)if(scores[i]>scores[best])best=i;
    const auto &component=components[best];const int sum=scores[0]+scores[1]+scores[2];
    const double ratio=double(scores[best])/std::max(1,sum);
    if(!scores[best]||component[4]<std::max(best==2?3.:2.,height*width*.004)||
       (best==2&&peak<140)||(component[4]==2&&component[5]<2)||ratio<.75)return -1;
    if(best==0&&height>=width*1.7) {
      const double center=(component[1]+component[3]/2.)/height;
      if(center>=.35&&center<=.65) {
        double green=0,red=0;int count=0;
        for(int y=std::max(0,component[1]-1);y<std::min(height,component[1]+component[3]+1);++y)
          for(int x=std::max(0,component[0]-1);x<std::min(width,component[0]+component[2]+1);++x) {
            auto color=image.at<cv::Vec3b>(y,x);
            if(std::max({color[0],color[1],color[2]})>=std::max(100.,peak*.8)) {
              green+=color[1];red+=color[2];++count;
            }
          }
        if(count&&green>=red*.5)best=1;
      }
    }
    *confidence=ratio;return best;
  } catch(...) {return -2;}
}

extern "C" int traffic_camera_move_people(const unsigned char *previous,const unsigned char *current,
    const float *motion_field,int width,int height,const double *people,int count,int columns,
    double fraction,int frame_width,int frame_height,double *output) {
  try {
    const cv::Mat before(height,width,CV_8UC1,const_cast<unsigned char *>(previous));
    const cv::Mat after(height,width,CV_8UC1,const_cast<unsigned char *>(current));
    const cv::Mat flow(height,width,CV_32FC2,const_cast<float *>(motion_field));
    int kept=0;
    for(int index=0;index<count;++index) {
      const double *person=people+index*columns;
      double box[4]={person[0]/2,person[1]/2,person[2]/2,person[3]/2};
      const int left=std::max(0,cvRound(box[0]+(box[2]-box[0])*.15));
      const int top=std::max(0,cvRound(box[1]+(box[3]-box[1])*.08));
      const int right=std::min(width,cvRound(box[2]-(box[2]-box[0])*.15));
      const int bottom=std::min(height,cvRound(box[3]-(box[3]-box[1])*.4));
      if(right>left && bottom>top) {
        std::vector<float> dx,dy;
        for(int y=top;y<bottom;++y)for(int x=left;x<right;++x) {
          const auto value=flow.at<cv::Vec2f>(y,x);dx.push_back(value[0]);dy.push_back(value[1]);
        }
        auto median=[](std::vector<float> &values) {
          const auto middle=values.size()/2;
          std::nth_element(values.begin(),values.begin()+middle,values.end());
          const float high=values[middle];
          return values.size()%2?high:(high+*std::max_element(values.begin(),values.begin()+middle))/2.f;
        };
        float mx=median(dx)*fraction,my=median(dy)*fraction;
        const cv::Mat core=before(cv::Rect(left,top,right-left,bottom-top));
        cv::Scalar mean,deviation;cv::meanStdDev(core,mean,deviation);
        cv::Mat patch,score;
        auto compare=[&]() {
          cv::getRectSubPix(after,core.size(),cv::Point2f((left+right-1)/2.0+mx,(top+bottom-1)/2.0+my),patch);
          cv::matchTemplate(patch,core,score,cv::TM_CCOEFF_NORMED);
          return score.at<float>(0,0);
        };
        float agreement=compare();
        // Correct flow underestimation only with a distinct match in the current image.
        if(fraction>=.999 && deviation[0]>3 && core.cols>=4 && core.rows>=4 && agreement<.9f) {
          const int radius=std::clamp(core.rows/2,4,12);
          const int x1=std::max(0,cvRound(left+mx)-radius),y1=std::max(0,cvRound(top+my)-radius);
          const int x2=std::min(width,cvRound(right+mx)+radius),y2=std::min(height,cvRound(bottom+my)+radius);
          if(x2-x1>=core.cols && y2-y1>=core.rows) {
            cv::Mat matches;cv::matchTemplate(after(cv::Rect(x1,y1,x2-x1,y2-y1)),core,matches,cv::TM_CCOEFF_NORMED);
            double best;cv::Point position;cv::minMaxLoc(matches,nullptr,&best,nullptr,&position);
            cv::Mat rivals=matches.clone();
            cv::rectangle(rivals,cv::Rect(std::max(0,position.x-2),std::max(0,position.y-2),
                std::min(matches.cols,position.x+3)-std::max(0,position.x-2),
                std::min(matches.rows,position.y+3)-std::max(0,position.y-2)),cv::Scalar(-1),cv::FILLED);
            double second;cv::minMaxLoc(rivals,nullptr,&second);
            if(best>=.65 && best>=agreement+.08 && best-second>=.05) {
              mx=x1+position.x-left;my=y1+position.y-top;agreement=compare();
            }
          }
        }
        if(deviation[0]>3 && agreement<.2 && cv::norm(patch,core,cv::NORM_L1)/core.total()>50)continue;
        box[0]+=mx;box[2]+=mx;box[1]+=my;box[3]+=my;
      }
      for(double &value:box)value*=2;
      const double clipped[4]={std::clamp(box[0],0.,double(frame_width)),std::clamp(box[1],0.,double(frame_height)),
        std::clamp(box[2],0.,double(frame_width)),std::clamp(box[3],0.,double(frame_height))};
      if(clipped[2]-clipped[0]<std::max(2.,(box[2]-box[0])*.4)||clipped[3]-clipped[1]<8)continue;
      std::memcpy(output+kept*columns,person,columns*sizeof(double));
      std::memcpy(output+kept*columns,clipped,sizeof(clipped));++kept;
    }
    return kept;
  } catch(const std::exception &error) {camera_error=error.what();return -1;}
}

extern "C" int traffic_camera_match_boxes(const unsigned char *previous,const unsigned char *current,
    int width,int height,const double *people,int count,int columns,double fraction,
    int frame_width,int frame_height,double *output,double minimum_score) {
  try {
    if(fraction<.999)return -2;
    const cv::Mat before(height,width,CV_8UC1,const_cast<unsigned char *>(previous));
    const cv::Mat after(height,width,CV_8UC1,const_cast<unsigned char *>(current));
    for(int index=0;index<count;++index) {
      const double *person=people+index*columns;
      const double left=person[0]/2,top=person[1]/2,right=person[2]/2,bottom=person[3]/2;
      if(!std::isfinite(left)||!std::isfinite(top)||!std::isfinite(right)||!std::isfinite(bottom))return -2;
      const int x1=std::max(0,cvRound(left+(right-left)*.15));
      const int y1=std::max(0,cvRound(top+(bottom-top)*.08));
      const int x2=std::min(width,cvRound(right-(right-left)*.15));
      const int y2=std::min(height,cvRound(bottom-(bottom-top)*.4));
      if(x2-x1<4||y2-y1<4)return -2;
      const cv::Mat core=before(cv::Rect(x1,y1,x2-x1,y2-y1));
      cv::Scalar mean,deviation;cv::meanStdDev(core,mean,deviation);
      if(deviation[0]<=3)return -2;
      const int radius=std::clamp(cvRound(core.rows*.6),6,16);
      const int sx=std::max(0,x1-radius),sy=std::max(0,y1-radius);
      const int ex=std::min(width,x2+radius),ey=std::min(height,y2+radius);
      cv::Mat matches;cv::matchTemplate(after(cv::Rect(sx,sy,ex-sx,ey-sy)),core,matches,cv::TM_CCOEFF_NORMED);
      double best;cv::Point position;cv::minMaxLoc(matches,nullptr,&best,nullptr,&position);
      cv::Mat rivals=matches.clone();
      cv::rectangle(rivals,cv::Rect(std::max(0,position.x-2),std::max(0,position.y-2),
          std::min(matches.cols,position.x+3)-std::max(0,position.x-2),
          std::min(matches.rows,position.y+3)-std::max(0,position.y-2)),cv::Scalar(-1),cv::FILLED);
      double second;cv::minMaxLoc(rivals,nullptr,&second);
      if(best<minimum_score||best-second<.12)return -2;
      const double dx=2*(sx+position.x-x1),dy=2*(sy+position.y-y1);
      double moved[4]={std::clamp(person[0]+dx,0.,double(frame_width)),std::clamp(person[1]+dy,0.,double(frame_height)),
          std::clamp(person[2]+dx,0.,double(frame_width)),std::clamp(person[3]+dy,0.,double(frame_height))};
      if(moved[2]-moved[0]<2||moved[3]-moved[1]<8)return -2;
      std::memcpy(output+index*columns,person,columns*sizeof(double));
      std::memcpy(output+index*columns,moved,sizeof(moved));
    }
    return count;
  } catch(const std::exception &error) {camera_error=error.what();return -1;}
}

extern "C" int traffic_camera_match_people(const unsigned char *previous,const unsigned char *current,
    int width,int height,const double *people,int count,int columns,double fraction,
    int frame_width,int frame_height,double *output) {
  return traffic_camera_match_boxes(previous,current,width,height,people,count,columns,fraction,
      frame_width,frame_height,output,.85);
}

extern "C" int traffic_camera_dense_flow(const unsigned char *previous,const unsigned char *current,
    int width,int height,float *flow_output) {
  try {
    const cv::Mat before(height,width,CV_8UC1,const_cast<unsigned char *>(previous));
    const cv::Mat gray(height,width,CV_8UC1,const_cast<unsigned char *>(current));
    thread_local cv::Ptr<cv::DISOpticalFlow> optical_flow=[] {
      auto algorithm=cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
      algorithm->setFinestScale(0);algorithm->setPatchStride(6);return algorithm;
    }();
    cv::Mat result;optical_flow->calc(before,gray,result);
    std::memcpy(flow_output,result.data,result.total()*result.elemSize());
    return 1;
  } catch(const std::exception &error) {camera_error=error.what();return -1;}
}

namespace {
std::atomic<Display *> preview_error_display{nullptr};
std::atomic<int> preview_x_error{0};
std::atomic<XErrorHandler> preview_previous_handler{nullptr};
int preview_attach_error(Display *display,XErrorEvent *event) {
  if(display==preview_error_display.load()) {preview_x_error=event->error_code;return 0;}
  auto previous=preview_previous_handler.load();return previous?previous(display,event):0;
}
struct NativePreview {
  Display *xdisplay=nullptr;Window window=0;Atom close_atom=0;GC gc=nullptr;
  XImage *image=nullptr;XShmSegmentInfo shared{};bool attached=false,visible=true,pending=false;
  int width=0,height=0,completion_type=0;
  NativePreview(){shared.shmid=-1;}
  static void require(bool condition,const char *message) {
    if(!condition)throw std::runtime_error(message);
  }
  void wait_pending() {
    if(!pending)return;
    XEvent event;XIfEvent(xdisplay,&event,completed,reinterpret_cast<XPointer>(this));pending=false;
  }
  void destroy_image() {
    wait_pending();
    if(attached) {XShmDetach(xdisplay,&shared);XSync(xdisplay,False);attached=false;}
    if(image) {if(shared.shmid>=0)image->data=nullptr;XDestroyImage(image);image=nullptr;}
    if(shared.shmaddr&&shared.shmaddr!=reinterpret_cast<char *>(-1))shmdt(shared.shmaddr);
    if(shared.shmid>=0)shmctl(shared.shmid,IPC_RMID,nullptr);
    shared={};shared.shmid=-1;
  }
  static void check_format(const XImage *value) {
    require(value&&value->bits_per_pixel==32&&value->byte_order==LSBFirst&&
      value->red_mask==0xff0000&&value->green_mask==0xff00&&value->blue_mask==0xff,
      "Preview requires verified RGB32 little-endian X11 pixels");
  }
  void create_image() {
    int screen=DefaultScreen(xdisplay);Visual *visual=DefaultVisual(xdisplay,screen);
    unsigned int depth=DefaultDepth(xdisplay,screen);
    struct ucred peer{};socklen_t peer_size=sizeof(peer);
    bool local=getsockopt(ConnectionNumber(xdisplay),SOL_SOCKET,SO_PEERCRED,&peer,&peer_size)==0;
    if(XShmQueryExtension(xdisplay)&&local&&(peer.uid==geteuid()||geteuid()==0)) {
      image=XShmCreateImage(xdisplay,visual,depth,ZPixmap,nullptr,&shared,width,height);check_format(image);
      shared.shmid=shmget(IPC_PRIVATE,size_t(image->bytes_per_line)*height,IPC_CREAT|0600);
      if(shared.shmid>=0) {
        shared.shmaddr=static_cast<char *>(shmat(shared.shmid,nullptr,0));shared.readOnly=True;
        if(shared.shmaddr!=reinterpret_cast<char *>(-1)) {
          image->data=shared.shmaddr;bool permitted=true;
          if(peer.uid!=geteuid()) {
            struct shmid_ds permissions{};
            permitted=shmctl(shared.shmid,IPC_STAT,&permissions)==0;
            permissions.shm_perm.uid=peer.uid;permissions.shm_perm.gid=peer.gid;permissions.shm_perm.mode=0600;
            permitted=permitted&&shmctl(shared.shmid,IPC_SET,&permissions)==0;
          }
          if(permitted) {
            XSync(xdisplay,False);preview_error_display=xdisplay;preview_x_error=0;
            auto previous=XSetErrorHandler(preview_attach_error);preview_previous_handler=previous;
            bool requested=XShmAttach(xdisplay,&shared);XSync(xdisplay,False);
            XSetErrorHandler(previous);preview_error_display=nullptr;preview_previous_handler=nullptr;
            attached=requested&&preview_x_error.load()==0;
            if(attached) {
              require(shmctl(shared.shmid,IPC_RMID,nullptr)==0,"Preview cannot mark shared memory for cleanup");
              completion_type=XShmGetEventBase(xdisplay)+ShmCompletion;return;
            }
          }
        }
      }
      destroy_image();
    }
    // Unavailable or inaccessible MIT-SHM uses Xlib's copying path, never wider permissions.
    image=XCreateImage(xdisplay,visual,depth,ZPixmap,0,nullptr,width,height,32,0);check_format(image);
    image->data=static_cast<char *>(std::malloc(size_t(image->bytes_per_line)*height));
    require(image->data,"Preview image allocation failed");
  }
  static Bool completed(Display *,XEvent *event,XPointer argument) {
    auto &preview=*reinterpret_cast<NativePreview *>(argument);
    return event->type==preview.completion_type&&
      reinterpret_cast<XShmCompletionEvent *>(event)->shmseg==preview.shared.shmseg;
  }
  void initialize(const char *title,int frame_width,int frame_height) {
    require(x11_threads_ready,"Preview X11 thread initialization failed");
    width=frame_width;height=frame_height;xdisplay=XOpenDisplay(nullptr);
    require(xdisplay,"Preview cannot open X11 display");
    int screen=DefaultScreen(xdisplay);
    window=XCreateSimpleWindow(xdisplay,RootWindow(xdisplay,screen),0,0,width,height,0,0,0);
    require(window,"Preview X11 window creation failed");
    XSelectInput(xdisplay,window,KeyPressMask|ButtonPressMask|ButtonReleaseMask|PointerMotionMask|StructureNotifyMask);
    gc=XCreateGC(xdisplay,window,0,nullptr);require(gc,"Preview X11 drawing context failed");
    XStoreName(xdisplay,window,title);close_atom=XInternAtom(xdisplay,"WM_DELETE_WINDOW",False);
    XSetWMProtocols(xdisplay,window,&close_atom,1);set_size();XMapWindow(xdisplay,window);
    XEvent mapped;XIfEvent(xdisplay,&mapped,[](Display *,XEvent *event,XPointer argument)->Bool {
      return event->type==MapNotify&&event->xmap.window==reinterpret_cast<NativePreview *>(argument)->window;
    },reinterpret_cast<XPointer>(this));
    create_image();
  }
  void set_size() {
    XSizeHints hints{};hints.flags=PMinSize|PMaxSize;
    hints.min_width=hints.max_width=width;hints.min_height=hints.max_height=height;
    XSetWMNormalHints(xdisplay,window,&hints);
  }
  void show(const unsigned char *pixels,int frame_width,int frame_height,int stride,unsigned char *output) {
    require(visible&&window,"Preview window is closed");
    wait_pending();
    if(width!=frame_width||height!=frame_height) {
      destroy_image();
      width=frame_width;height=frame_height;set_size();XResizeWindow(xdisplay,window,width,height);XSync(xdisplay,False);
      create_image();
    }
    cv::Mat source(height,width,CV_8UC3,const_cast<unsigned char *>(pixels),stride);
    cv::Mat destination(height,width,CV_8UC4,image->data,image->bytes_per_line);
    cv::cvtColor(source,destination,cv::COLOR_BGR2BGRA);
    if(attached) {
      require(XShmPutImage(xdisplay,window,gc,image,0,0,0,0,width,height,True),"Preview MIT-SHM transfer failed");
      pending=true;XFlush(xdisplay);
    } else {
      XPutImage(xdisplay,window,gc,image,0,0,0,0,width,height);XFlush(xdisplay);
    }
    if(output) {
      wait_pending();
      XImage *actual=XGetImage(xdisplay,window,0,0,width,height,AllPlanes,ZPixmap);
      require(actual,"Preview server pixel readback failed");
      for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
        unsigned long pixel=XGetPixel(actual,x,y);size_t offset=(size_t(y)*width+x)*3;
        output[offset]=pixel&255;output[offset+1]=(pixel>>8)&255;output[offset+2]=(pixel>>16)&255;
      }
      XDestroyImage(actual);
    }
  }
  int poll(int *fields) {
    fields[0]=fields[4]=-1;fields[1]=fields[2]=fields[3]=0;fields[5]=visible;
    if(!XPending(xdisplay))return 0;
    XEvent event;XNextEvent(xdisplay,&event);
    if(pending&&completed(xdisplay,&event,reinterpret_cast<XPointer>(this)))pending=false;
    if(event.type==ClientMessage&&Atom(event.xclient.data.l[0])==close_atom)visible=false;
    if(event.type==DestroyNotify) {visible=false;window=0;}
    if(event.type==KeyPress) {
      char text[16];KeySym symbol;int size=XLookupString(&event.xkey,text,sizeof(text),&symbol,nullptr);
      if(size==1&&static_cast<unsigned char>(text[0])<128)fields[4]=static_cast<unsigned char>(text[0]);
    }
    if(event.type==MotionNotify||event.type==ButtonPress||event.type==ButtonRelease) {
      unsigned int state=(event.type==MotionNotify)?event.xmotion.state:event.xbutton.state;
      fields[1]=(event.type==MotionNotify)?event.xmotion.x:event.xbutton.x;
      fields[2]=(event.type==MotionNotify)?event.xmotion.y:event.xbutton.y;
      fields[3]=((state&Button1Mask)?1:0)|((state&Button3Mask)?2:0)|((state&Button2Mask)?4:0)|
        ((state&ControlMask)?8:0)|((state&ShiftMask)?16:0)|((state&Mod1Mask)?32:0);
      if(event.type==MotionNotify)fields[0]=0;
      else if(event.xbutton.button>=1&&event.xbutton.button<=3) {
        const int buttons[]={0,1,3,2};fields[0]=buttons[event.xbutton.button]+(event.type==ButtonRelease?3:0);
        const int flag=(event.xbutton.button==1?1:event.xbutton.button==2?4:2);
        fields[3]=event.type==ButtonPress?(fields[3]|flag):(fields[3]&~flag);
      }
    }
    fields[5]=visible;return 1;
  }
  ~NativePreview() {
    if(xdisplay) {destroy_image();if(gc)XFreeGC(xdisplay,gc);if(window)XDestroyWindow(xdisplay,window);XCloseDisplay(xdisplay);}
  }
};
}

extern "C" void *traffic_preview_open(const char *title,int width,int height) {
  try {auto preview=std::make_unique<NativePreview>();preview->initialize(title,width,height);return preview.release();}
  catch(const std::exception &error){camera_error=error.what();return nullptr;}
}
extern "C" int traffic_preview_show(void *handle,const unsigned char *pixels,int width,int height,int stride,unsigned char *output) {
  try {static_cast<NativePreview *>(handle)->show(pixels,width,height,stride,output);return 1;}
  catch(const std::exception &error){camera_error=error.what();return -1;}
}
extern "C" int traffic_preview_poll(void *handle,int *fields) {
  try {return static_cast<NativePreview *>(handle)->poll(fields);}
  catch(const std::exception &error){camera_error=error.what();return -1;}
}
extern "C" int traffic_preview_move(void *handle,int x,int y) {
  auto &preview=*static_cast<NativePreview *>(handle);XMoveWindow(preview.xdisplay,preview.window,x,y);XFlush(preview.xdisplay);return 1;
}
extern "C" void traffic_preview_close(void *handle){delete static_cast<NativePreview *>(handle);}
