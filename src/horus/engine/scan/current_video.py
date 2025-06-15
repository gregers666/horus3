#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
import threading
from collections import deque

from horus import Singleton


@Singleton
class CurrentVideo(object):
    """Zoptymalizowana klasa zarządzania bieżącymi obrazami video"""

    def __init__(self):
        self.mode = 'Texture'
        self._lock = threading.Lock()
        
        # Cache obrazów z historią
        self.images = {}
        self.images['Texture'] = None
        self.images['Laser'] = None
        self.images['Gray'] = None
        self.images['Line'] = None
        
        # Bufor dla smooth playback
        self._image_buffer = {
            'Texture': deque(maxlen=3),
            'Laser': deque(maxlen=3),
            'Gray': deque(maxlen=3),
            'Line': deque(maxlen=3)
        }
        
        # Cache dla operacji kombinowania
        self._combine_cache = {}
        self._cache_size_limit = 10
        
        # Optymalizacje
        self._use_buffer = True
        self._cache_enabled = True
        self._async_processing = True
        
        # Statystyki wydajności
        self._frame_count = 0
        self._processing_times = deque(maxlen=20)

    def set_use_buffer(self, enabled):
        """Włącza/wyłącza buforowanie obrazów"""
        self._use_buffer = enabled

    def set_cache_enabled(self, enabled):
        """Włącza/wyłącza cache operacji"""
        self._cache_enabled = enabled
        if not enabled:
            self._combine_cache.clear()

    def set_texture(self, image):
        """Zoptymalizowane ustawianie tekstury"""
        if image is None:
            return
            
        with self._lock:
            self.images['Texture'] = image
            
            if self._use_buffer:
                self._image_buffer['Texture'].append(image.copy())

    def set_laser(self, images):
        """Zoptymalizowane ustawianie obrazów laserów"""
        start_time = cv2.getTickCount()
        
        try:
            combined_image = self._combine_images_optimized(images)
            
            with self._lock:
                self.images['Laser'] = combined_image
                
                if self._use_buffer and combined_image is not None:
                    self._image_buffer['Laser'].append(combined_image.copy())
        
        except Exception as e:
            print(f"Error setting laser images: {e}")
        finally:
            self._update_performance_stats(start_time)

    def set_gray(self, images):
        """Zoptymalizowane ustawianie obrazów w skali szarości"""
        start_time = cv2.getTickCount()
        
        try:
            combined_image = self._combine_images_optimized(images)
            
            if combined_image is not None:
                # Optymalizowana konwersja do RGB
                if len(combined_image.shape) == 2:
                    gray_rgb = cv2.merge((combined_image, combined_image, combined_image))
                else:
                    gray_rgb = combined_image
                
                with self._lock:
                    self.images['Gray'] = gray_rgb
                    
                    if self._use_buffer:
                        self._image_buffer['Gray'].append(gray_rgb.copy())
        
        except Exception as e:
            print(f"Error setting gray images: {e}")
        finally:
            self._update_performance_stats(start_time)

    def set_line(self, points, image):
        """Zoptymalizowane ustawianie obrazów linii"""
        if image is None:
            return
            
        start_time = cv2.getTickCount()
        
        try:
            images = [None, None]
            
            # Równoległe przetwarzanie punktów jeśli możliwe
            for i in range(2):
                if points[i] is not None:
                    images[i] = self._compute_line_image_optimized(points[i], image)
            
            combined_image = self._combine_images_optimized(images)
            
            if combined_image is not None:
                # Optymalizowana konwersja do RGB
                if len(combined_image.shape) == 2:
                    line_rgb = cv2.merge((combined_image, combined_image, combined_image))
                else:
                    line_rgb = combined_image
                
                with self._lock:
                    self.images['Line'] = line_rgb
                    
                    if self._use_buffer:
                        self._image_buffer['Line'].append(line_rgb.copy())
        
        except Exception as e:
            print(f"Error setting line images: {e}")
        finally:
            self._update_performance_stats(start_time)

    def _combine_images_optimized(self, images):
        """Zoptymalizowane kombinowanie obrazów z cache"""
        if images is None or len(images) != 2:
            return None
        
        # Sprawdź cache jeśli włączony
        if self._cache_enabled:
            cache_key = self._generate_cache_key(images)
            if cache_key in self._combine_cache:
                return self._combine_cache[cache_key].copy()
        
        result = None
        
        try:
            if images[0] is not None and images[1] is not None:
                # Sprawdź kompatybilność rozmiarów
                if images[0].shape == images[1].shape:
                    result = np.maximum(images[0], images[1])
                else:
                    # Resize do mniejszego rozmiaru
                    min_shape = self._get_min_shape(images[0].shape, images[1].shape)
                    img0_resized = self._resize_image_safe(images[0], min_shape)
                    img1_resized = self._resize_image_safe(images[1], min_shape)
                    result = np.maximum(img0_resized, img1_resized)
                    
            elif images[0] is not None:
                result = images[0].copy()
            elif images[1] is not None:
                result = images[1].copy()
            
            # Zapisz do cache jeśli włączony
            if result is not None and self._cache_enabled:
                cache_key = self._generate_cache_key(images)
                self._update_cache(cache_key, result)
                
        except Exception as e:
            print(f"Error combining images: {e}")
            # Fallback - zwróć pierwszy dostępny obraz
            if images[0] is not None:
                result = images[0].copy()
            elif images[1] is not None:
                result = images[1].copy()
        
        return result

    def _compute_line_image_optimized(self, points, reference_image):
        """Zoptymalizowane obliczanie obrazu linii"""
        if points is None or reference_image is None:
            return None
            
        try:
            u, v = points
            
            if len(u) == 0 or len(v) == 0:
                return None
            
            # Utwórz obraz o tym samym rozmiarze co referencyjny
            line_image = np.zeros_like(reference_image)
            
            # Bezpieczne indeksowanie
            h, w = reference_image.shape[:2]
            v_safe = np.clip(v.astype(int), 0, h - 1)
            u_safe = np.clip(np.around(u).astype(int), 0, w - 1)
            
            # Ustaw piksele linii
            if len(line_image.shape) == 3:
                line_image[v_safe, u_safe] = [255, 255, 255]
            else:
                line_image[v_safe, u_safe] = 255
                
            return line_image
            
        except Exception as e:
            print(f"Error computing line image: {e}")
            return np.zeros_like(reference_image)

    def _generate_cache_key(self, images):
        """Generuje klucz cache dla obrazów"""
        try:
            key_parts = []
            for i, img in enumerate(images):
                if img is not None:
                    # Użyj hash z kilku pikseli jako klucz
                    if img.size > 0:
                        sample_pixels = img.flat[::max(1, img.size // 100)]
                        key_parts.append(f"{i}_{hash(sample_pixels.tobytes())}")
                    else:
                        key_parts.append(f"{i}_empty")
                else:
                    key_parts.append(f"{i}_none")
            return "_".join(key_parts)
        except:
            return f"fallback_{id(images)}"

    def _update_cache(self, key, image):
        """Aktualizuje cache z ograniczeniem rozmiaru"""
        if len(self._combine_cache) >= self._cache_size_limit:
            # Usuń najstarszy wpis
            oldest_key = next(iter(self._combine_cache))
            del self._combine_cache[oldest_key]
        
        self._combine_cache[key] = image.copy()

    def _get_min_shape(self, shape1, shape2):
        """Zwraca mniejszy z dwóch kształtów"""
        return tuple(min(s1, s2) for s1, s2 in zip(shape1, shape2))

    def _resize_image_safe(self, image, target_shape):
        """Bezpieczne przeskalowanie obrazu"""
        try:
            if len(target_shape) >= 2:
                h, w = target_shape[:2]
                return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            return image
        except:
            return image

    def _update_performance_stats(self, start_time):
        """Aktualizuje statystyki wydajności"""
        try:
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
            self._processing_times.append(processing_time)
            self._frame_count += 1
        except:
            pass

    def get_performance_stats(self):
        """Zwraca statystyki wydajności"""
        if not self._processing_times:
            return {"avg_time": 0, "frames_processed": self._frame_count}
        
        avg_time = sum(self._processing_times) / len(self._processing_times)
        return {
            "avg_time": avg_time,
            "frames_processed": self._frame_count,
            "recent_times": list(self._processing_times)
        }

    def capture(self):
        """Zwraca bieżący obraz z optymalizacjami"""
        with self._lock:
            current_image = self.images.get(self.mode)
            
            # Jeśli brak obrazu, spróbuj z bufora
            if current_image is None and self._use_buffer:
                buffer = self._image_buffer.get(self.mode)
                if buffer and len(buffer) > 0:
                    current_image = buffer[-1]  # Najnowszy obraz z bufora
            
            return current_image

    def get_buffered_image(self, mode, index=-1):
        """Zwraca obraz z bufora (index: -1 = najnowszy, 0 = najstarszy)"""
        if not self._use_buffer:
            return self.images.get(mode)
        
        with self._lock:
            buffer = self._image_buffer.get(mode)
            if buffer and len(buffer) > abs(index):
                return buffer[index]
            return None

    def clear_buffers(self):
        """Czyści wszystkie bufory"""
        with self._lock:
            for buffer in self._image_buffer.values():
                buffer.clear()
            self._combine_cache.clear()
            self._processing_times.clear()
            self._frame_count = 0

    def set_mode(self, mode):
        """Ustawia tryb z walidacją"""
        valid_modes = ['Texture', 'Laser', 'Gray', 'Line']
        if mode in valid_modes:
            self.mode = mode
        else:
            print(f"Warning: Invalid mode '{mode}'. Valid modes: {valid_modes}")

    def get_available_modes(self):
        """Zwraca dostępne tryby"""
        return list(self.images.keys())

    def get_image_info(self, mode=None):
        """Zwraca informacje o obrazie"""
        target_mode = mode or self.mode
        
        with self._lock:
            image = self.images.get(target_mode)
            if image is None:
                return None
            
            return {
                "mode": target_mode,
                "shape": image.shape,
                "dtype": str(image.dtype),
                "size_bytes": image.nbytes,
                "has_buffer": len(self._image_buffer.get(target_mode, [])) > 0
            }

    def optimize_memory(self):
        """Optymalizuje użycie pamięci"""
        with self._lock:
            # Ogranicz bufory do minimum
            for buffer in self._image_buffer.values():
                while len(buffer) > 1:
                    buffer.popleft()
            
            # Wyczyść cache
            self._combine_cache.clear()
            
            print("Memory optimization completed")

    def __del__(self):
        """Cleanup"""
        try:
            self.clear_buffers()
        except:
            pass