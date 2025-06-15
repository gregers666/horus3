#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
from scipy.optimize import least_squares
import warnings

from horus import Singleton
from horus.engine.calibration.pattern import Pattern
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class ImageDetection(object):

    def __init__(self):
        self.pattern = Pattern()
        self.calibration_data = CalibrationData()

        # Ulepszone kryteria zbieżności
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        
        # Nowe parametry dla lepszej detekcji
        self.preprocessing_enabled = True
        self.adaptive_threshold = True
        self.morphological_cleanup = True
        self.subpixel_refinement = True
        self.outlier_detection = True
        self.pose_validation = True
        
        # Parametry filtracji
        self.min_pattern_area = 100  # minimalna powierzchnia wzorca w pikselach^2
        self.max_pattern_area = 500000  # maksymalna powierzchnia wzorca
        self.corner_quality_threshold = 0.01
        self.reprojection_error_threshold = 2.0  # piksele

    def set_preprocessing(self, enabled):
        self.preprocessing_enabled = enabled

    def set_adaptive_threshold(self, enabled):
        self.adaptive_threshold = enabled

    def set_subpixel_refinement(self, enabled):
        self.subpixel_refinement = enabled

    def set_outlier_detection(self, enabled):
        self.outlier_detection = enabled

    def detect_pattern(self, image):
        """Ulepszone wykrywanie wzorca z lepszą wizualizacją"""
        if image is None:
            return None
            
        try:
            corners = self._detect_chessboard(image)
            if corners is not None:
                # Walidacja corners
                if self._validate_corners(corners, image.shape[:2]):
                    image_with_pattern = self.draw_pattern(image, corners)
                    return image_with_pattern
                else:
                    print("Detected corners failed validation")
            return image
            
        except Exception as e:
            print(f"Error in detect_pattern: {e}")
            return image

    def draw_pattern(self, image, corners):
        """Ulepszone rysowanie wzorca z lepszymi kolorami i grubością linii"""
        if image is None or corners is None:
            return image
            
        try:
            # Konwersja kolorów dla lepszej wizualizacji
            if len(image.shape) == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Adaptacyjna grubość linii na podstawie rozmiaru obrazu
            line_thickness = max(1, min(image.shape[:2]) // 200)
            
            # Rysuj szachownicę z lepszymi parametrami
            pattern_found = cv2.drawChessboardCorners(
                display_image, 
                (self.pattern.columns, self.pattern.rows), 
                corners, 
                True
            )
            
            if pattern_found:
                # Dodaj informacje o jakości wykrycia
                self._draw_pattern_info(display_image, corners)
            
            # Konwersja z powrotem
            if len(image.shape) == 3:
                result = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            else:
                result = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
                
            return result
            
        except Exception as e:
            print(f"Error in draw_pattern: {e}")
            return image