#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


# DMM-1118 MakerBot Digitizer

#                  brightness 0x00980900 (int)    : min=-10 max=10 step=1 default=-10 value=10
#                    contrast 0x00980901 (int)    : min=0 max=20 step=1 default=10 value=9
#                  saturation 0x00980902 (int)    : min=0 max=10 step=1 default=6 value=2
#     white_balance_automatic 0x0098090c (bool)   : default=1 value=1
#                       gamma 0x00980910 (int)    : min=100 max=200 step=1 default=150 value=100
#                        gain 0x00980913 (int)    : min=32 max=48 step=1 default=32 value=32
#        power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2 (60 Hz)
# 			0: Disabled
# 			1: 50 Hz
# 			2: 60 Hz
#   white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=6500 value=5000 flags=inactive
#                   sharpness 0x0098091b (int)    : min=0 max=10 step=1 default=7 value=7

#   auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=1 (Manual Mode)
# 				1: Manual Mode
# 				3: Aperture Priority Mode
#   exposure_time_absolute 0x009a0902 (int)    : min=8 max=16384 step=1 default=256 value=3940

# Kamera - ustawienia domyślne

CAMERA_DEFAULT_BRIGHTNESS = -10      # int, min=-10, max=10
CAMERA_DEFAULT_CONTRAST = 10         # int, min=0, max=20
CAMERA_DEFAULT_SATURATION = 6        # int, min=0, max=10
CAMERA_DEFAULT_WHITE_BALANCE_AUTO = True  # bool, default=1 (enabled)
CAMERA_DEFAULT_GAMMA = 150           # int, min=100, max=200
CAMERA_DEFAULT_GAIN = 32             # int, min=32, max=48
CAMERA_DEFAULT_POWER_LINE_FREQUENCY = 1  # Menu: 0 = Disabled, 1 = 50 Hz, 2 = 60 Hz
CAMERA_DEFAULT_WHITE_BALANCE_TEMP = 6500 # int, min=2800, max=6500 (inactive if auto enabled)
CAMERA_DEFAULT_SHARPNESS = 10        # int, min=0, max=10
CAMERA_AUTO_EXPOSURE = 1             # manual mode
CAMERA_EXPOSURE_TIME_ABSOLUTE = 256        # int, min=8 max=16384


import cv2
import math
import time
import glob
import platform

import logging
logger = logging.getLogger(__name__)

system = platform.system()

if system == 'Darwin':
    from . import uvc
    from .uvc.mac import *


class WrongCamera(Exception):

    def __init__(self):
        pass
        Exception.__init__(self, "Wrong Camera")


class CameraNotConnected(Exception):

    def __init__(self):
        Exception.__init__(self, "Camera Not Connected")


class InvalidVideo(Exception):

    def __init__(self):
        Exception.__init__(self, "Invalid Video")


class WrongDriver(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Driver")


class InputOutputError(Exception):

    def __init__(self):
        Exception.__init__(self, "V4L2 Input/Output Error")


class Camera(object):

    """Camera class. For accessing to the scanner camera"""

    def __init__(self, parent=None, camera_id=0):
        self.parent = parent
        self.camera_id = camera_id
        self.unplug_callback = None

        self._capture = None
        self._is_connected = False
        self._reading = False
        self._updating = False
        self._last_image = None
        self._video_list = None
        self._tries = 0  # Check if command fails
        self._luminosity = 1.0

        self.initialize()

        if system == 'Windows':
            self._number_frames_fail = 3
            self._max_brightness = 1.
            self._max_contrast = 1.
            self._max_saturation = 1.
        elif system == 'Darwin':
            self._number_frames_fail = 3
            self._max_brightness = 255.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._rel_exposure = 10.
        else:
            self._number_frames_fail = 3
            self._max_brightness = 8.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._max_exposure = 16384.
            self._min_exposure = 8.
            self._min_contrast = 0.
            self._min_saturation = 0.

    def initialize(self):
        # Ustawienie wszystkich wartości domyślnych zgodnie ze stałymi CAMERA_DEFAULT_*
        self._brightness = CAMERA_DEFAULT_BRIGHTNESS
        self._contrast = CAMERA_DEFAULT_CONTRAST
        self._saturation = CAMERA_DEFAULT_SATURATION
        self._exposure = CAMERA_EXPOSURE_TIME_ABSOLUTE
        self._frame_rate = 0
        self._width = 0
        self._height = 0
        self._rotate = True
        self._hflip = True
        self._vflip = False

    def connect(self):
        logger.info("Connecting camera {0}".format(self.camera_id))
        self._is_connected = False
        self.initialize()
        if system == 'Darwin':
            for device in uvc.mac.Camera_List():
                if device.src_id == self.camera_id:
                    self.controls = uvc.mac.Controls(device.uId)
        if self._capture is not None:
            print('self._capture is not None')
            self._capture.release()
        self._capture = cv2.VideoCapture(self.camera_id)
        
        # Ustawienie domyślnego trybu exposure
        self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
        print(self._capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        
        print(('Camera ID %s' % self.camera_id))
        time.sleep(0.2)
        if not self._capture.isOpened():
            time.sleep(1)
            self._capture.open(self.camera_id)
        if self._capture.isOpened():
            self._is_connected = True
            self._check_video()
            self._check_camera()
            self._check_driver()
            
            # Zastosowanie wszystkich domyślnych ustawień po połączeniu
            self._apply_default_settings()
            
            logger.info(" Done")
        else:
            raise CameraNotConnected()

    def _apply_default_settings(self):
        """Zastosowanie wszystkich domyślnych ustawień kamery"""
        logger.info("Applying default camera settings...")
        
        try:
            # Ustawienie brightness
            self.set_brightness(CAMERA_DEFAULT_BRIGHTNESS)
            logger.info(f"Set brightness to: {CAMERA_DEFAULT_BRIGHTNESS}")
            
            # Ustawienie contrast
            self.set_contrast(CAMERA_DEFAULT_CONTRAST)
            logger.info(f"Set contrast to: {CAMERA_DEFAULT_CONTRAST}")
            
            # Ustawienie saturation
            self.set_saturation(CAMERA_DEFAULT_SATURATION)
            logger.info(f"Set saturation to: {CAMERA_DEFAULT_SATURATION}")
            
            # Ustawienie exposure
            self.set_exposure(CAMERA_EXPOSURE_TIME_ABSOLUTE)
            logger.info(f"Set exposure to: {CAMERA_EXPOSURE_TIME_ABSOLUTE}")
            
            # Dodatkowe ustawienia specyficzne dla systemu
            if system == 'Darwin':
                # Ustawienia dla macOS przez UVC controls
                if hasattr(self, 'controls'):
                    # White balance automatic
                    if 'UVCC_REQ_WHITE_BALANCE_AUTO' in self.controls:
                        self.controls['UVCC_REQ_WHITE_BALANCE_AUTO'].set_val(1 if CAMERA_DEFAULT_WHITE_BALANCE_AUTO else 0)
                        logger.info(f"Set white balance auto to: {CAMERA_DEFAULT_WHITE_BALANCE_AUTO}")
                    
                    # Gamma
                    if 'UVCC_REQ_GAMMA' in self.controls:
                        gamma_ctl = self.controls['UVCC_REQ_GAMMA']
                        gamma_val = self._line(CAMERA_DEFAULT_GAMMA, 100, 200, gamma_ctl.min, gamma_ctl.max)
                        gamma_ctl.set_val(gamma_val)
                        logger.info(f"Set gamma to: {CAMERA_DEFAULT_GAMMA}")
                    
                    # Gain
                    if 'UVCC_REQ_GAIN' in self.controls:
                        gain_ctl = self.controls['UVCC_REQ_GAIN']
                        gain_val = self._line(CAMERA_DEFAULT_GAIN, 32, 48, gain_ctl.min, gain_ctl.max)
                        gain_ctl.set_val(gain_val)
                        logger.info(f"Set gain to: {CAMERA_DEFAULT_GAIN}")
                    
                    # Sharpness
                    if 'UVCC_REQ_SHARPNESS_ABS' in self.controls:
                        sharp_ctl = self.controls['UVCC_REQ_SHARPNESS_ABS']
                        sharp_val = self._line(CAMERA_DEFAULT_SHARPNESS, 0, 10, sharp_ctl.min, sharp_ctl.max)
                        sharp_ctl.set_val(sharp_val)
                        logger.info(f"Set sharpness to: {CAMERA_DEFAULT_SHARPNESS}")
            
            elif system == 'Linux':
                # Dodatkowe ustawienia dla Linux przez V4L2
                # Gain (jeśli dostępne)
                try:
                    gain_val = self._line(CAMERA_DEFAULT_GAIN, 32, 48, 0, 255)
                    self._capture.set(cv2.CAP_PROP_GAIN, gain_val)
                    logger.info(f"Set gain to: {CAMERA_DEFAULT_GAIN}")
                except:
                    logger.warning("Could not set gain")
                
                # Gamma (jeśli dostępne)
                try:
                    gamma_val = self._line(CAMERA_DEFAULT_GAMMA, 100, 200, 0, 255)
                    self._capture.set(cv2.CAP_PROP_GAMMA, gamma_val)
                    logger.info(f"Set gamma to: {CAMERA_DEFAULT_GAMMA}")
                except:
                    logger.warning("Could not set gamma")
                
                # White balance (jeśli dostępne)
                try:
                    if CAMERA_DEFAULT_WHITE_BALANCE_AUTO:
                        self._capture.set(cv2.CAP_PROP_AUTO_WB, 1)
                    else:
                        self._capture.set(cv2.CAP_PROP_AUTO_WB, 0)
                        self._capture.set(cv2.CAP_PROP_WB_TEMPERATURE, CAMERA_DEFAULT_WHITE_BALANCE_TEMP)
                    logger.info(f"Set white balance auto to: {CAMERA_DEFAULT_WHITE_BALANCE_AUTO}")
                except:
                    logger.warning("Could not set white balance")
            
            logger.info("Default camera settings applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying default settings: {e}")

    def disconnect(self):
        tries = 0
        if self._is_connected:
            logger.info("Disconnecting camera {0}".format(self.camera_id))
            if self._capture is not None:
                if self._capture.isOpened():
                    self._is_connected = False
                    while tries < 10:
                        tries += 1
                        if not self._reading:
                            self._capture.release()
                logger.info(" Done")

    def set_unplug_callback(self, value):
        self.unplug_callback = value

    def _check_video(self):
        """Check correct video"""
        frame = self.capture_image(flush=1)
        if frame is None or (frame == 0).all():
            raise InvalidVideo()

    def _check_camera(self):
        """Check correct camera"""
        print("Checking camera...")        
        c_exp = False
        c_bri = False

        
        try:
            # Check exposure
            if system == 'Darwin':
                print("Darwin")
                self.controls['UVCC_REQ_EXPOSURE_AUTOMODE'].set_val(1)
            print("Setting exposure = 2")
            self.set_exposure(2)
            exposure = self.get_exposure()
            print("Received exposure = %s" % exposure)
            if exposure is not None:
                c_exp = exposure >= 1.9

            # Check brightness
            print("Setting test brightness = 2")
            self.set_brightness(2)
            brightness = self.get_brightness()
            print("Received test brightness = %s" % brightness)
            if brightness is not None:
                c_bri = brightness >= 2
        except:
            raise WrongCamera()

        if not c_exp or not c_bri:
            raise WrongCamera()

    def _check_driver(self):
        """Check correct driver: only for Windows"""
        if system == 'Windows':
            print("Windows")
            self.set_exposure(10)
            frame = self.capture_image(flush=1)
            mean = sum(cv2.mean(frame)) / 3.0
            if mean > 200:
                raise WrongDriver()

    def capture_image(self, flush=0, auto=False):
        """Capture image from camera"""
        if self._is_connected:
            if self._updating:
                return self._last_image
            else:
                self._reading = True
                if auto:
                    b, e = 0, 0
                    while e - b < (0.030):
                        b = time.time()
                        self._capture.grab()
                        e = time.time()
                else:
                    if flush > 0:
                        for i in range(flush):
                            self._capture.read()
                            # Note: Windows needs read() to perform
                            #       the flush instead of grab()
                ret, image = self._capture.read()
                self._reading = False
                if ret:
                    if self._rotate:
                        image = cv2.transpose(image)
                    if self._hflip:
                        image = cv2.flip(image, 1)
                    if self._vflip:
                        image = cv2.flip(image, 0)
                    self._success()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self._last_image = image
                    return image
                else:
                    self._fail()
                    return None
        else:
            return None

    def save_image(self, filename, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)

    def set_rotate(self, value):
        self._rotate = value

    def set_hflip(self, value):
        self._hflip = value

    def set_vflip(self, value):
        self._vflip = value

    def set_brightness(self, value):
        if self._is_connected:
            if self._brightness != value:
                self._updating = True
                self._brightness = value
                if system == 'Darwin':
                    ctl = self.controls['UVCC_REQ_BRIGHTNESS_ABS']
                    ctl.set_val(self._line(value, -10, 10, ctl.min, ctl.max))
                else:
                    print("requested brightness %s" % value)
                    value = int(value)
                    print("calculated brightness %s" % value)                    
                    ret = self._capture.set(cv2.CAP_PROP_BRIGHTNESS, value)

                    if system == 'Linux' and not ret:
                        raise InputOutputError()
                self._updating = False

    def set_contrast(self, value):
        if self._is_connected:
            if self._contrast != value:
                self._updating = True
                self._contrast = value
                if system == 'Darwin':
                    ctl = self.controls['UVCC_REQ_CONTRAST_ABS']
                    ctl.set_val(self._line(value, 0, 20, ctl.min, ctl.max))
                else:
                    value = int(value)
                    ret = self._capture.set(cv2.CAP_PROP_CONTRAST, value)
                    if system == 'Linux' and not ret:
                        raise InputOutputError()
                self._updating = False

    def set_saturation(self, value):
        if self._is_connected:
            if self._saturation != value:
                self._updating = True
                self._saturation = value
                if system == 'Darwin':
                    ctl = self.controls['UVCC_REQ_SATURATION_ABS']
                    ctl.set_val(self._line(value, 0, 10, ctl.min, ctl.max))
                else:
                    value = int(value) / self._max_saturation
                    ret = self._capture.set(cv2.CAP_PROP_SATURATION, value)
                    if system == 'Linux' and not ret:
                        raise InputOutputError()
                self._updating = False

    def set_exposure(self, value, force=False):
        print("Entering set_exposure")
        print("System = %s" % system)
        if self._is_connected:
            print("Camera _is_connected")
            if self._exposure != value or force:
                self._updating = True
                self._exposure = value
                value *= self._luminosity
                if value < 1:
                    value = 1
                if system == 'Darwin':
                    print("Darwin")
                    ctl = self.controls['UVCC_REQ_EXPOSURE_ABS']
                    value = int(value * self._rel_exposure)
                    ctl.set_val(value)
                elif system == 'Windows':
                    print("Windows")
                    value = int(round(-math.log(value) / math.log(2)))
                    self._capture.set(cv2.CAP_PROP_EXPOSURE, value)
                else:
                    print("Calculating value from %s" % value)
                    self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
                    print("before setting to %s get(cv2.CAP_PROP_EXPOSURE)=%s)" % (value, self._capture.get(cv2.CAP_PROP_EXPOSURE)))
                    ret = self._capture.set(cv2.CAP_PROP_EXPOSURE, value)
                    print("after setting to %s get(cv2.CAP_PROP_EXPOSURE)=%s)" % (value, self._capture.get(cv2.CAP_PROP_EXPOSURE)))

                    if system == 'Linux' and not ret:
                        print("Raising error - Linux ret=%s" % ret)
                        raise InputOutputError()
                self._updating = False

    def set_luminosity(self, value):
        possible_values = {
            "High": 0.5,
            "Medium": 1.0,
            "Low": 2.0
        }
        self._luminosity = possible_values[value]
        self.set_exposure(self._exposure, force=True)

    def set_frame_rate(self, value):
        print("Entering set_frame_rate")
        print("requested value = %s" % value)
        if self._is_connected:
            print("Camera _is_connected")
            if self._frame_rate != value:
                self._frame_rate = value
                self._updating = True
                print("set CAP_PROP_FPS to value = %s" % value)
                self._capture.set(cv2.CAP_PROP_FPS, value)
                print("CAP_PROP_FPS set to %s" % self._capture.get(cv2.CAP_PROP_FPS))
                self._updating = False

    def set_resolution(self, width, height):
        if self._is_connected:
            if self._width != width or self._height != height:
                self._updating = True
                self._set_width(width)
                self._set_height(height)
                self._update_resolution()
                self._updating = False

    def _set_width(self, value):
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)

    def _set_height(self, value):
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    def _update_resolution(self):
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_brightness(self):
        if self._is_connected:
            if system == 'Darwin':
                ctl = self.controls['UVCC_REQ_BRIGHTNESS_ABS']
                value = ctl.get_val()
            else:
                value = self._capture.get(cv2.CAP_PROP_BRIGHTNESS)
            return value

    def get_exposure(self):
        if self._is_connected:
            if system == 'Darwin':
                ctl = self.controls['UVCC_REQ_EXPOSURE_ABS']
                value = ctl.get_val()
                value /= self._rel_exposure
            elif system == 'Windows':
                value = self._capture.get(cv2.CAP_PROP_EXPOSURE)
                value = 2 ** -value
            else:
                value = self._capture.get(cv2.CAP_PROP_EXPOSURE)
            return value

    def get_resolution(self):
        if self._rotate:
            return int(self._height), int(self._width)
        else:
            return int(self._width), int(self._height)

    def _success(self):
        self._tries = 0

    def _fail(self):
        logger.debug("Camera fail")
        self._tries += 1
        if self._tries >= self._number_frames_fail:
            self._tries = 0
            if self.unplug_callback is not None and \
               self.parent is not None and \
               not self.parent.unplugged:
                self.parent.unplugged = True
                self.unplug_callback()

    def _line(self, value, imin, imax, omin, omax):
        ret = 0
        if omin is not None and omax is not None:
            if (imax - imin) != 0:
                ret = int((value - imin) * (omax - omin) / (imax - imin) + omin)
        return ret

    def _count_cameras(self):
        for i in range(5):
            cap = cv2.VideoCapture(i)
            res = not cap.isOpened()
            cap.release()
            if res:
                return i
        return 5

    def get_video_list(self):
        baselist = []
        if system == 'Windows':
            if not self._is_connected:
                count = self._count_cameras()
                for i in range(count):
                    baselist.append(str(i))
                self._video_list = baselist
        elif system == 'Darwin':
            for device in uvc.mac.Camera_List():
                baselist.append(str(device.src_id))
            self._video_list = baselist
        else:
            for device in ['/dev/video*']:
                baselist = baselist + glob.glob(device)
            self._video_list = baselist
        return self._video_list