# 姓名：丁黄炎
from pathlib import Path
import time
import argparse
import os
import ctypes
if os.name == 'nt':
    desktop = ctypes.WinDLL('user32', use_last_error=True)
    desktop.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    desktop.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
    desktop.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--display-x', type=int, default=0)
parser.add_argument('--display-y', type=int, default=0)
parser.add_argument('--color-bars', action='store_true', help='Static diagnostic patches instead of video')
args = parser.parse_args()
source = args.source
capture = cv2.VideoCapture(str(source))
if not capture.isOpened():
    raise RuntimeError(f'Cannot open {source}')
fps = capture.get(cv2.CAP_PROP_FPS)
window = 'HDMI traffic test - Esc exits'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
next_frame = time.monotonic()
first = True
try:
    while True:
        ok, frame = capture.read()
        if not ok:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError('Cannot restart the test video')
            continue
        height, width = frame.shape[:2]
        gain = min(1280 / width, 720 / height)
        scaled_width, scaled_height = round(width * gain), round(height * gain)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        x, y = (1280 - scaled_width) // 2, (720 - scaled_height) // 2
        canvas[y:y + scaled_height, x:x + scaled_width] = cv2.resize(frame, (scaled_width, scaled_height))
        if args.color_bars:
            patches = [(255,255,255),(0,255,255),(255,255,0),(0,255,0),
                       (255,0,255),(0,0,255),(255,0,0),(0,0,0)]
            for column, color in enumerate(patches):
                canvas[:360,column*160:(column+1)*160] = color
                level = (96,112,126,127,128,129,144,192)[column]
                canvas[360:,column*160:(column+1)*160] = level
        cv2.imshow(window, canvas)
        if first:
            cv2.waitKey(1)
            cv2.moveWindow(window, args.display_x, args.display_y)
            cv2.waitKey(1)
            if os.name == 'nt':
                desktop.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
                desktop.FindWindowW.restype = ctypes.c_void_p
                desktop.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
                desktop.SetWindowPos.restype = ctypes.c_bool
                native_window = desktop.FindWindowW(None, window)
                if not native_window or not desktop.SetWindowPos(native_window, ctypes.c_void_p(-1),
                        args.display_x, args.display_y, 1280, 720, 0x0040):
                    raise ctypes.WinError(ctypes.get_last_error())
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if os.name == 'nt':
                cv2.waitKey(1)
                if not desktop.SetWindowPos(native_window, ctypes.c_void_p(-1),
                        args.display_x, args.display_y, 1280, 720, 0x0040):
                    raise ctypes.WinError(ctypes.get_last_error())
            first = False
        delay = max(1, round((next_frame - time.monotonic()) * 1000))
        if cv2.waitKey(delay) & 255 in (27, ord('q')):
            break
        next_frame += 1 / fps
finally:
    capture.release()
    cv2.destroyAllWindows()
