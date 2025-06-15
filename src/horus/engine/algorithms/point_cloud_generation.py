#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import numpy as np
from scipy.spatial.distance import cdist
from scipy import interpolate
import warnings

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class PointCloudGeneration(object):

    def __init__(self):
        self.calibration_data = CalibrationData()
        
        # Nowe parametry dla lepszej jakości
        self.noise_reduction_enabled = True
        self.outlier_removal_enabled = True
        self.interpolation_enabled = True
        self.smoothing_enabled = True
        
        # Parametry filtracji
        self.max_distance_threshold = 5.0  # mm
        self.min_points_per_line = 5
        self.interpolation_factor = 1.0
        self.smoothing_sigma = 0.5

    def set_noise_reduction(self, enabled):
        self.noise_reduction_enabled = enabled

    def set_outlier_removal(self, enabled):
        self.outlier_removal_enabled = enabled

    def set_interpolation(self, enabled):
        self.interpolation_enabled = enabled

    def set_smoothing(self, enabled):
        self.smoothing_enabled = enabled

    def compute_point_cloud(self, theta, points_2d, index):
        """Ulepszone generowanie chmury punktów z filtracją i poprawkami"""
        try:
            if points_2d is None or len(points_2d) != 2:
                return None
                
            u, v = points_2d
            if u is None or v is None or len(u) == 0 or len(v) == 0:
                return None
                
            # Walidacja danych wejściowych
            if len(u) != len(v):
                min_len = min(len(u), len(v))
                u = u[:min_len]
                v = v[:min_len]
            
            # Sprawdź czy mamy wystarczająco punktów
            if len(u) < self.min_points_per_line:
                return None
            
            # Load calibration values z walidacją
            if not self._validate_calibration(index):
                return None
                
            R = np.matrix(self.calibration_data.platform_rotation)
            t = np.matrix(self.calibration_data.platform_translation).T
            
            # Pre-processing punktów 2D
            if self.noise_reduction_enabled:
                u, v = self._reduce_noise_2d(u, v)
                
            if len(u) == 0:
                return None
            
            # Compute platform transformation
            Xwo = self.compute_platform_point_cloud(points_2d=(u, v), R=R, t=t, index=index)
            
            if Xwo is None or Xwo.size == 0:
                return None
            
            # Rotate to world coordinates
            c, s = np.cos(-theta), np.sin(-theta)
            Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            Xw = Rz * Xwo
            
            # Post-processing chmury punktów
            if Xw.size > 0:
                Xw_array = np.array(Xw)
                
                # Usuwanie outlierów
                if self.outlier_removal_enabled:
                    Xw_array = self._remove_outliers_3d(Xw_array)
                
                # Wygładzanie
                if self.smoothing_enabled and Xw_array.shape[1] > 10:
                    Xw_array = self._smooth_point_cloud(Xw_array)
                
                # Interpolacja dla gęstszej chmury punktów
                if self.interpolation_enabled and Xw_array.shape[1] > 5:
                    Xw_array = self._interpolate_point_cloud(Xw_array)
                
                return Xw_array if Xw_array.shape[1] > 0 else None
            else:
                return None
                
        except Exception as e:
            print(f"Error in compute_point_cloud: {e}")
            return None

    def compute_platform_point_cloud(self, points_2d, R, t, index):
        """Ulepszone obliczanie chmury punktów w układzie platformy"""
        try:
            # Walidacja danych kalibracji lasera
            if (index >= len(self.calibration_data.laser_planes) or 
                self.calibration_data.laser_planes[index] is None):
                print(f"Invalid laser plane index: {index}")
                return None
                
            # Load calibration values
            laser_plane = self.calibration_data.laser_planes[index]
            n = laser_plane.normal
            d = laser_plane.distance
            
            if n is None or d is None:
                print("Invalid laser plane parameters")
                return None
            
            # Camera system with validation
            Xc = self.compute_camera_point_cloud(points_2d, d, n)
            
            if Xc is None or Xc.size == 0:
                return None
            
            # Compute platform transformation with better numerical stability
            try:
                R_T = R.T
                result = R_T * Xc - R_T * t
                return result
            except np.linalg.LinAlgError as e:
                print(f"Linear algebra error in platform transformation: {e}")
                return None
                
        except Exception as e:
            print(f"Error in compute_platform_point_cloud: {e}")
            return None

    def compute_camera_point_cloud(self, points_2d, d, n):
        """Ulepszone obliczanie chmury punktów w układzie kamery"""
        try:
            # Load calibration values z walidacją
            camera_matrix = self.calibration_data.camera_matrix
            if camera_matrix is None or camera_matrix.shape != (3, 3):
                print("Invalid camera matrix")
                return None
                
            fx = camera_matrix[0][0]
            fy = camera_matrix[1][1]
            cx = camera_matrix[0][2]
            cy = camera_matrix[1][2]
            
            # Walidacja parametrów kalibracji
            if fx <= 0 or fy <= 0:
                print("Invalid focal length parameters")
                return None
            
            # Compute projection point
            u, v = points_2d
            
            # Walidacja danych wejściowych
            u = np.asarray(u, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            
            if len(u) == 0 or len(v) == 0:
                return None
            
            # Filtruj punkty poza zakresem obrazu
            if hasattr(self.calibration_data, 'width') and hasattr(self.calibration_data, 'height'):
                valid_mask = ((u >= 0) & (u < self.calibration_data.width) & 
                             (v >= 0) & (v < self.calibration_data.height))
                u = u[valid_mask]
                v = v[valid_mask]
                
                if len(u) == 0:
                    return None
            
            # Przekształć do współrzędnych znormalizowanych
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
            z_norm = np.ones(len(u))
            
            # Utwórz macierz punktów
            x = np.vstack([x_norm, y_norm, z_norm])
            
            # Walidacja normalnej płaszczyzny
            n = np.asarray(n).flatten()
            if len(n) != 3:
                print("Invalid normal vector")
                return None
                
            n_norm = np.linalg.norm(n)
            if n_norm == 0:
                print("Zero normal vector")
                return None
                
            n = n / n_norm  # Normalizacja
            
            # Compute laser intersection z lepszą stabilnością numeryczną
            denominator = np.dot(n, x)
            
            # Sprawdź czy denominatory nie są zbyt małe (prawie równoległe promienie)
            min_denominator = 1e-6
            valid_intersections = np.abs(denominator) > min_denominator
            
            if not np.any(valid_intersections):
                print("No valid laser intersections found")
                return None
            
            # Filtruj tylko poprawne przecięcia
            x_filtered = x[:, valid_intersections]
            denominator_filtered = denominator[valid_intersections]
            
            # Oblicz punkty przecięcia
            distances = d / denominator_filtered
            
            # Filtruj punkty zbyt daleko od kamery
            max_distance = 1000.0  # mm
            distance_mask = (distances > 0) & (distances < max_distance)
            
            if not np.any(distance_mask):
                return None
            
            final_points = distances[distance_mask] * x_filtered[:, distance_mask]
            
            return np.matrix(final_points)
            
        except Exception as e:
            print(f"Error in compute_camera_point_cloud: {e}")
            return None

    def _validate_calibration(self, laser_index):
        """Walidacja danych kalibracji"""
        try:
            # Sprawdź kalibrację kamery
            if (self.calibration_data.camera_matrix is None or
                self.calibration_data.camera_matrix.shape != (3, 3)):
                return False
            
            # Sprawdź kalibrację platformy
            if (self.calibration_data.platform_rotation is None or
                self.calibration_data.platform_translation is None):
                return False
            
            # Sprawdź kalibrację lasera
            if (laser_index >= len(self.calibration_data.laser_planes) or
                self.calibration_data.laser_planes[laser_index] is None):
                return False
            
            return True
            
        except Exception:
            return False

    def _reduce_noise_2d(self, u, v):
        """Redukcja szumów w punktach 2D"""
        try:
            if len(u) < 5:
                return u, v
            
            # Użyj mediany do filtracji outlierów
            u_median = np.median(u)
            v_median = np.median(v)
            
            # Oblicz odchylenie bezwzględne od mediany
            u_mad = np.median(np.abs(u - u_median))
            v_mad = np.median(np.abs(v - v_median))
            
            # Filtruj punkty oddalone o więcej niż 3*MAD od mediany
            threshold_factor = 3.0
            u_mask = np.abs(u - u_median) <= threshold_factor * u_mad
            v_mask = np.abs(v - v_median) <= threshold_factor * v_mad
            
            combined_mask = u_mask & v_mask
            
            return u[combined_mask], v[combined_mask]
            
        except Exception as e:
            print(f"Error in 2D noise reduction: {e}")
            return u, v

    def _remove_outliers_3d(self, points):
        """Usuwanie outlierów z chmury punktów 3D"""
        try:
            if points.shape[1] < 10:
                return points
            
            # Oblicz odległości między sąsiednimi punktami
            distances = []
            for i in range(points.shape[1] - 1):
                dist = np.linalg.norm(points[:, i+1] - points[:, i])
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Użyj IQR do identyfikacji outlierów
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            
            # Znajdź punkty do zachowania
            valid_indices = [0]  # Zawsze zachowaj pierwszy punkt
            for i, dist in enumerate(distances):
                if dist <= threshold:
                    valid_indices.append(i + 1)
            
            return points[:, valid_indices]
            
        except Exception as e:
            print(f"Error in 3D outlier removal: {e}")
            return points

    def _smooth_point_cloud(self, points):
        """Wygładzanie chmury punktów"""
        try:
            if points.shape[1] < 5:
                return points
            
            from scipy.ndimage import gaussian_filter1d
            
            # Wygładź każdą współrzędną osobno
            smoothed_points = np.zeros_like(points)
            for i in range(3):  # x, y, z
                smoothed_points[i, :] = gaussian_filter1d(
                    points[i, :], 
                    sigma=self.smoothing_sigma,
                    mode='nearest'
                )
            
            return smoothed_points
            
        except Exception as e:
            print(f"Error in point cloud smoothing: {e}")
            return points

    def _interpolate_point_cloud(self, points):
        """Interpolacja chmury punktów dla większej gęstości"""
        try:
            if points.shape[1] < 3 or self.interpolation_factor <= 1.0:
                return points
            
            # Liczba nowych punktów
            n_original = points.shape[1]
            n_new = int(n_original * self.interpolation_factor)
            
            if n_new <= n_original:
                return points
            
            # Parametr t dla interpolacji
            t_original = np.linspace(0, 1, n_original)
            t_new = np.linspace(0, 1, n_new)
            
            # Interpoluj każdą współrzędną
            interpolated_points = np.zeros((3, n_new))
            
            for i in range(3):
                # Użyj interpolacji sześciennej
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    interpolator = interpolate.interp1d(
                        t_original, 
                        points[i, :], 
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    interpolated_points[i, :] = interpolator(t_new)
            
            return interpolated_points
            
        except Exception as e:
            print(f"Error in point cloud interpolation: {e}")
            return points

    def merge_point_clouds(self, point_clouds):
        """Łączenie chmur punktów z różnych laserów"""
        try:
            valid_clouds = [pc for pc in point_clouds if pc is not None and pc.shape[1] > 0]
            
            if not valid_clouds:
                return None
            
            if len(valid_clouds) == 1:
                return valid_clouds[0]
            
            # Połącz wszystkie chmury punktów
            merged = np.concatenate(valid_clouds, axis=1)
            
            # Opcjonalne sortowanie według współrzędnej z
            if merged.shape[1] > 1:
                z_coords = merged[2, :]
                sort_indices = np.argsort(z_coords)
                merged = merged[:, sort_indices]
            
            return merged
            
        except Exception as e:
            print(f"Error merging point clouds: {e}")
            return None

    def compute_point_cloud_quality_metrics(self, point_cloud):
        """Obliczanie metryk jakości chmury punktów"""
        try:
            if point_cloud is None or point_cloud.shape[1] < 2:
                return None
            
            metrics = {}
            
            # Liczba punktów
            metrics['point_count'] = point_cloud.shape[1]
            
            # Gęstość punktów (średnia odległość między sąsiednimi punktami)
            distances = []
            for i in range(point_cloud.shape[1] - 1):
                dist = np.linalg.norm(point_cloud[:, i+1] - point_cloud[:, i])
                distances.append(dist)
            
            metrics['mean_point_density'] = np.mean(distances)
            metrics['std_point_density'] = np.std(distances)
            
            # Zasięg w każdej osi
            metrics['x_range'] = np.ptp(point_cloud[0, :])
            metrics['y_range'] = np.ptp(point_cloud[1, :])
            metrics['z_range'] = np.ptp(point_cloud[2, :])
            
            # Gładkość (wariancja drugich pochodnych)
            if point_cloud.shape[1] > 4:
                smoothness = []
                for i in range(3):
                    second_diff = np.diff(point_cloud[i, :], 2)
                    smoothness.append(np.var(second_diff))
                metrics['smoothness'] = np.mean(smoothness)
            
            return metrics
            
        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            return None