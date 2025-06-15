#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import math
import numpy as np
import scipy.ndimage
from scipy import signal
from sklearn.linear_model import RANSACRegressor
import warnings

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData
from horus.engine.algorithms.point_cloud_roi import PointCloudROI


@Singleton
class LaserSegmentation(object):

    def __init__(self):
        self.calibration_data = CalibrationData()
        self.point_cloud_roi = PointCloudROI()

        self.red_channel = 'R (RGB)'
        self.threshold_enable = True  # Domyślnie włączone
        self.threshold_value = 15  # Lepszy domyślny próg
        self.blur_enable = True   # Domyślnie włączone
        self.blur_value = 3       # Lepszy domyślny rozmiar
        self.window_enable = True # Domyślnie włączone
        self.window_value = 5     # Lepszy domyślny rozmiar okna
        self.refinement_method = 'SGF'
        
        # Nowe parametry dla lepszej segmentacji
        self.adaptive_threshold = True
        self.morphological_ops = True
        self.edge_detection = False
        self.min_laser_width = 2
        self.max_laser_width = 50
        self.intensity_weighting = True
        self.outlier_removal = True
        self.gaussian_sigma = 1.5

    def set_red_channel(self, value):
        self.red_channel = value

    def set_threshold_enable(self, value):
        self.threshold_enable = value

    def set_threshold_value(self, value):
        self.threshold_value = max(0, min(255, value))

    def set_blur_enable(self, value):
        self.blur_enable = value

    def set_blur_value(self, value):
        self.blur_value = 2 * max(1, value) + 1  # Zawsze nieparzyste

    def set_window_enable(self, value):
        self.window_enable = value

    def set_window_value(self, value):
        self.window_value = max(1, value)

    def set_refinement_method(self, value):
        self.refinement_method = value

    def set_adaptive_threshold(self, value):
        self.adaptive_threshold = value

    def set_morphological_ops(self, value):
        self.morphological_ops = value

    def set_edge_detection(self, value):
        self.edge_detection = value

    def set_intensity_weighting(self, value):
        self.intensity_weighting = value

    def set_outlier_removal(self, value):
        self.outlier_removal = value

    def compute_2d_points(self, image):
        """Ulepszone wykrywanie punktów 2D z lepszą segmentacją"""
        if image is None:
            return (None, None), None
            
        try:
            # Segmentacja linii lasera
            segmented_image = self.compute_line_segmentation(image)
            if segmented_image is None:
                return (None, None), None
            
            # Wykrywanie maksimów z wagami intensywności
            points_2d = self._advanced_peak_detection(segmented_image)
            
            if points_2d is None or len(points_2d[0]) == 0:
                return (None, None), segmented_image
            
            u, v = points_2d
            
            # Usuwanie outlierów
            if self.outlier_removal and len(u) > 10:
                u, v = self._remove_outliers(u, v)
            
            # Refinement
            if len(u) > 1:
                if self.refinement_method == 'SGF':
                    u, v = self._improved_sgf(u, v, segmented_image)
                elif self.refinement_method == 'RANSAC':
                    u, v = self._improved_ransac(u, v)
                elif self.refinement_method == 'POLYNOMIAL':
                    u, v = self._polynomial_fitting(u, v)
            
            return (u, v), segmented_image
            
        except Exception as e:
            print(f"Error in compute_2d_points: {e}")
            return (None, None), None

    def compute_line_segmentation(self, image, roi_mask=False):
        """Ulepszona segmentacja linii lasera"""
        if image is None:
            return None
            
        try:
            # Apply ROI mask
            if roi_mask:
                image = self.point_cloud_roi.mask_image(image)
            
            # Preprocessing - redukcja szumów
            if len(image.shape) == 3:
                image = cv2.bilateralFilter(image, 5, 80, 80)
            
            # Obtain red channel
            red_channel = self._obtain_red_channel(image)
            if red_channel is None:
                return None
            
            # Adaptacyjne progowanie
            if self.adaptive_threshold:
                red_channel = self._adaptive_threshold_processing(red_channel)
            else:
                red_channel = self._threshold_image(red_channel)
            
            # Operacje morfologiczne
            if self.morphological_ops:
                red_channel = self._morphological_operations(red_channel)
            
            # Window mask z ulepszoną logiką
            red_channel = self._improved_window_mask(red_channel)
            
            return red_channel
            
        except Exception as e:
            print(f"Error in line segmentation: {e}")
            return None

    def _obtain_red_channel(self, image):
        """Ulepszone pozyskiwanie kanału czerwonego"""
        try:
            if self.red_channel == 'R (RGB)':
                if len(image.shape) == 3:
                    return cv2.split(image)[2]  # Kanał R w OpenCV to indeks 2
                return image
            elif self.red_channel == 'Cr (YCrCb)':
                if len(image.shape) == 3:
                    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
                    return cv2.split(ycrcb)[1]
                return image
            elif self.red_channel == 'U (YUV)':
                if len(image.shape) == 3:
                    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    return cv2.split(yuv)[1]
                return image
            elif self.red_channel == 'HSV_V':
                if len(image.shape) == 3:
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    return cv2.split(hsv)[2]
                return image
            else:
                return cv2.split(image)[0] if len(image.shape) == 3 else image
        except Exception as e:
            print(f"Error obtaining red channel: {e}")
            return None

    def _adaptive_threshold_processing(self, image):
        """Adaptacyjne progowanie dla lepszej segmentacji"""
        try:
            # Otsu's thresholding dla automatycznego progu
            _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Kombinacja z manualnym progiem
            if self.threshold_enable:
                manual_thresh = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_TOZERO)[1]
                # Użyj lepszego z dwóch progów
                combined = cv2.bitwise_or(otsu_thresh, manual_thresh)
            else:
                combined = otsu_thresh
            
            # Blur jeśli włączony
            if self.blur_enable:
                combined = cv2.GaussianBlur(combined, (self.blur_value, self.blur_value), 0)
                # Ponowne progowanie po blur
                combined = cv2.threshold(combined, self.threshold_value // 2, 255, cv2.THRESH_TOZERO)[1]
            
            return combined
            
        except Exception as e:
            print(f"Error in adaptive threshold: {e}")
            return self._threshold_image(image)

    def _threshold_image(self, image):
        """Oryginalna metoda progowania z poprawkami"""
        try:
            if self.threshold_enable:
                image = cv2.threshold(
                    image, self.threshold_value, 255, cv2.THRESH_TOZERO)[1]
                if self.blur_enable:
                    image = cv2.GaussianBlur(image, (self.blur_value, self.blur_value), 0)
                image = cv2.threshold(
                    image, self.threshold_value, 255, cv2.THRESH_TOZERO)[1]
            return image
        except Exception as e:
            print(f"Error in threshold: {e}")
            return image

    def _morphological_operations(self, image):
        """Operacje morfologiczne dla oczyszczenia linii lasera"""
        try:
            # Kernel dla operacji morfologicznych
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Zamknięcie małych dziur w linii
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_line)
            
            # Usunięcie małych szumów
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_noise)
            
            return image
        except Exception as e:
            print(f"Error in morphological operations: {e}")
            return image

    def _improved_window_mask(self, image):
        """Ulepszona maska okna z lepszą logiką"""
        if not self.window_enable or image is None:
            return image
            
        try:
            h, w = image.shape
            result = np.zeros_like(image)
            
            for i in range(h):
                row = image[i, :]
                if np.max(row) > 0:
                    # Znajdź wszystkie maksima lokalne
                    peaks, properties = signal.find_peaks(row, 
                                                        height=self.threshold_value,
                                                        width=self.min_laser_width,
                                                        distance=10)
                    
                    for peak in peaks:
                        # Określ szerokość okna na podstawie intensywności
                        intensity = row[peak]
                        adaptive_window = max(self.window_value, 
                                            min(self.max_laser_width, 
                                                int(self.window_value * intensity / 255.0 * 2)))
                        
                        left = max(0, peak - adaptive_window)
                        right = min(w, peak + adaptive_window + 1)
                        result[i, left:right] = image[i, left:right]
            
            return result
            
        except Exception as e:
            print(f"Error in window mask: {e}")
            return self._original_window_mask(image)

    def _original_window_mask(self, image):
        """Oryginalna maska okna jako fallback"""
        if self.window_enable:
            peak = image.argmax(axis=1)
            _min = np.maximum(0, peak - self.window_value)
            _max = np.minimum(image.shape[1], peak + self.window_value + 1)
            mask = np.zeros_like(image)
            h = image.shape[0]
            for i in range(h):
                if peak[i] > 0:  # Tylko jeśli znaleziono peak
                    mask[i, _min[i]:_max[i]] = 255
            return cv2.bitwise_and(image, mask)
        return image

    def _advanced_peak_detection(self, image):
        """Zaawansowane wykrywanie maksimów z wagami"""
        try:
            h, w = image.shape
            
            # Suma intensywności w każdym wierszu
            row_sums = image.sum(axis=1)
            valid_rows = np.where(row_sums > 0)[0]
            
            if len(valid_rows) == 0:
                return None
            
            if self.intensity_weighting:
                # Wagi oparte na intensywności
                weights = np.arange(w, dtype=np.float32)
                weighted_sums = (image * weights).sum(axis=1)
                u_coords = weighted_sums[valid_rows] / row_sums[valid_rows]
            else:
                # Standardowe centrum masy
                u_coords = (self.calibration_data.weight_matrix * image).sum(axis=1)[valid_rows] / row_sums[valid_rows]
            
            return u_coords, valid_rows
            
        except Exception as e:
            print(f"Error in peak detection: {e}")
            # Fallback do oryginalnej metody
            s = image.sum(axis=1)
            v = np.where(s > 0)[0]
            if len(v) == 0:
                return None
            u = (self.calibration_data.weight_matrix * image).sum(axis=1)[v] / s[v]
            return u, v

    def _remove_outliers(self, u, v):
        """Usuwanie outlierów z wykorzystaniem IQR"""
        try:
            if len(u) < 10:
                return u, v
                
            # Oblicz IQR dla współrzędnych u
            q1, q3 = np.percentile(u, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filtruj outlierów
            mask = (u >= lower_bound) & (u <= upper_bound)
            
            # Dodatkowe filtrowanie na podstawie ciągłości
            if len(v) > 2:
                v_diff = np.diff(v)
                median_diff = np.median(v_diff)
                diff_mask = np.abs(v_diff - median_diff) < 3 * median_diff
                # Rozszerz maskę o pierwszy element
                diff_mask = np.concatenate([[True], diff_mask])
                mask = mask & diff_mask
            
            return u[mask], v[mask]
            
        except Exception as e:
            print(f"Error removing outliers: {e}")
            return u, v

    def _improved_sgf(self, u, v, image):
        """Ulepszona segmentowana filtracja Gaussa"""
        try:
            if len(u) <= 1:
                return u, v
            
            # Dynamiczny dobór parametrów
            adaptive_sigma = max(0.5, min(3.0, len(u) / 100.0))
            
            # Wykryj segmenty na podstawie przerw w danych
            v_diff = np.diff(v)
            gap_threshold = np.median(v_diff) * 3 if len(v_diff) > 0 else 5
            gap_indices = np.where(v_diff > gap_threshold)[0] + 1
            
            # Dodaj początek i koniec
            segment_boundaries = np.concatenate([[0], gap_indices, [len(v)]])
            
            filtered_u = np.array([])
            filtered_v = np.array([])
            
            for i in range(len(segment_boundaries) - 1):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i + 1]
                
                if end_idx - start_idx > 2:  # Minimum 3 punkty dla segmentu
                    segment_u = u[start_idx:end_idx]
                    segment_v = v[start_idx:end_idx]
                    
                    # Filtracja Gaussa z adaptacyjnym sigma
                    segment_length = end_idx - start_idx
                    if segment_length > 5:
                        sigma = min(adaptive_sigma, segment_length / 6.0)
                        filtered_segment_u = scipy.ndimage.gaussian_filter1d(segment_u, sigma=sigma)
                    else:
                        filtered_segment_u = segment_u
                    
                    filtered_u = np.concatenate([filtered_u, filtered_segment_u])
                    filtered_v = np.concatenate([filtered_v, segment_v])
            
            return filtered_u, filtered_v.astype(int)
            
        except Exception as e:
            print(f"Error in improved SGF: {e}")
            return self._original_sgf(u, v, image)

    def _original_sgf(self, u, v, image):
        """Oryginalna metoda SGF jako fallback"""
        try:
            if len(u) > 1:
                i = 0
                sigma = self.gaussian_sigma
                f = np.array([])
                s = image.sum(axis=1)
                segments = [s[_r] for _r in np.ma.clump_unmasked(np.ma.masked_equal(s, 0))]
                
                for segment in segments:
                    j = len(segment)
                    if j > 0:
                        fseg = scipy.ndimage.gaussian_filter(u[i:i + j], sigma=sigma)
                        f = np.concatenate((f, fseg))
                        i += j
                return f, v
            return u, v
        except Exception as e:
            print(f"Error in original SGF: {e}")
            return u, v

    def _improved_ransac(self, u, v):
        """Ulepszona implementacja RANSAC z sklearn"""
        try:
            if len(u) <= 3:
                return u, v
            
            # Przygotuj dane
            X = v.reshape(-1, 1).astype(np.float32)
            y = u.astype(np.float32)
            
            # RANSAC z lepszymi parametrami
            ransac = RANSACRegressor(
                estimator=None,  # Używa LinearRegression
                min_samples=max(2, len(u) // 10),
                residual_threshold=2.0,
                max_trials=100,
                random_state=42
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ransac.fit(X, y)
            
            # Przewiduj dla wszystkich punktów
            u_fitted = ransac.predict(X)
            
            # Zachowaj tylko inliers z pewną tolerancją
            residuals = np.abs(y - u_fitted)
            threshold = np.percentile(residuals, 85)  # 85% najlepszych punktów
            mask = residuals <= threshold
            
            return u_fitted[mask], v[mask]
            
        except Exception as e:
            print(f"Error in improved RANSAC: {e}")
            return self._original_ransac(u, v)

    def _original_ransac(self, u, v):
        """Oryginalna implementacja RANSAC jako fallback"""
        try:
            if len(u) > 1:
                data = np.vstack((v.ravel(), u.ravel())).T
                dr, thetar = self.ransac(data, self.LinearLeastSquares2D(), 2, 2)
                if dr is not None and thetar is not None:
                    u_fitted = (dr - v * math.sin(thetar)) / math.cos(thetar)
                    return u_fitted, v
            return u, v
        except Exception as e:
            print(f"Error in original RANSAC: {e}")
            return u, v

    def _polynomial_fitting(self, u, v, degree=2):
        """Dopasowanie wielomianowe dla gładkich krzywych"""
        try:
            if len(u) < degree + 1:
                return u, v
            
            # Dopasuj wielomian
            coeffs = np.polyfit(v, u, degree)
            u_fitted = np.polyval(coeffs, v)
            
            # Filtruj punkty na podstawie błędu dopasowania
            residuals = np.abs(u - u_fitted)
            threshold = np.percentile(residuals, 90)
            mask = residuals <= threshold
            
            return u_fitted[mask], v[mask]
            
        except Exception as e:
            print(f"Error in polynomial fitting: {e}")
            return u, v

    def compute_hough_lines(self, image):
        """Ulepszone wykrywanie linii Hough z lepszymi parametrami"""
        if image is None:
            return None
            
        try:
            # Segmentacja z ulepszeniami
            segmented = self.compute_line_segmentation(image)
            if segmented is None:
                return None
            
            # Preprocessing dla Hough
            # Wykrywanie krawędzi
            edges = cv2.Canny(segmented, 50, 150, apertureSize=3)
            
            # Ulepszone parametry Hough
            lines = cv2.HoughLines(edges, 
                                 rho=1, 
                                 theta=np.pi/180, 
                                 threshold=max(50, segmented.shape[0]//4),
                                 min_theta=0,
                                 max_theta=np.pi)
            
            return lines
            
        except Exception as e:
            print(f"Error in Hough lines: {e}")
            return None

    # Zachowaj oryginalną klasę LinearLeastSquares2D
    class LinearLeastSquares2D(object):
        '''
        2D linear least squares using the hesse normal form:
            d = x*sin(theta) + y*cos(theta)
        which allows you to have vertical lines.
        '''

        def fit(self, data):
            try:
                data_mean = data.mean(axis=0)
                x0, y0 = data_mean
                if data.shape[0] > 2:  # over determined
                    u, v, w = np.linalg.svd(data - data_mean)
                    vec = w[0]
                    theta = math.atan2(vec[0], vec[1])
                elif data.shape[0] == 2:  # well determined
                    theta = math.atan2(data[1, 0] - data[0, 0], data[1, 1] - data[0, 1])
                else:
                    return None, None
                    
                theta = (theta + math.pi * 5 / 2) % (2 * math.pi)
                d = x0 * math.sin(theta) + y0 * math.cos(theta)
                return d, theta
            except Exception as e:
                print(f"Error in LinearLeastSquares2D fit: {e}")
                return None, None

        def residuals(self, model, data):
            try:
                d, theta = model
                if d is None or theta is None:
                    return np.full(data.shape[0], float('inf'))
                dfit = data[:, 0] * math.sin(theta) + data[:, 1] * math.cos(theta)
                return np.abs(d - dfit)
            except Exception as e:
                print(f"Error in residuals calculation: {e}")
                return np.full(data.shape[0], float('inf'))

        def is_degenerate(self, sample):
            try:
                # Sprawdź czy punkty nie są współliniowe w sposób zdegenerowany
                if sample.shape[0] < 2:
                    return True
                if sample.shape[0] == 2:
                    return np.allclose(sample[0], sample[1])
                return False
            except:
                return True

    def ransac(self, data, model_class, min_samples, threshold, max_trials=100):
        '''
        Ulepszona implementacja RANSAC z lepszą obsługą błędów
        '''
        try:
            if data.shape[0] < min_samples:
                return None, None
                
            best_model = None
            best_inlier_num = 0
            best_inliers = None
            data_idx = np.arange(data.shape[0])
            
            for trial in range(max_trials):
                # Losowy wybór próbek
                sample_indices = np.random.choice(data.shape[0], min_samples, replace=False)
                sample = data[sample_indices]
                
                if model_class.is_degenerate(sample):
                    continue
                    
                sample_model = model_class.fit(sample)
                if sample_model[0] is None or sample_model[1] is None:
                    continue
                    
                sample_model_residuals = model_class.residuals(sample_model, data)
                sample_model_inliers = data_idx[sample_model_residuals < threshold]
                inlier_num = sample_model_inliers.shape[0]
                
                if inlier_num > best_inlier_num:
                    best_inlier_num = inlier_num
                    best_inliers = sample_model_inliers
                    best_model = sample_model
            
            if best_inliers is not None and len(best_inliers) >= min_samples:
                # Ponowne dopasowanie z wszystkimi inliers
                final_model = model_class.fit(data[best_inliers])
                return final_model
            
            return best_model
            
        except Exception as e:
            print(f"Error in RANSAC: {e}")
            return None, None