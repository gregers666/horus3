#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import time
import numpy as np
import threading
import gc


class ScanCapture(object):
    """
    Zoptymalizowana klasa przechowująca dane z pojedynczego capture skanowania.
    """
    
    def __init__(self, theta=0.0, texture=None, lasers=None, timestamp=0.0, 
                 capture_id=0, quality_score=0.0, processing_flags=None):
        self.theta = theta
        self.texture = texture
        self.lasers = lasers if lasers is not None else [None, None]
        self.timestamp = timestamp if timestamp != 0.0 else time.time()
        self.capture_id = capture_id
        self.quality_score = quality_score
        
        if processing_flags is None:
            self.processing_flags = {
                'texture_processed': False,
                'laser_0_processed': False,
                'laser_1_processed': False,
                'validation_passed': False
            }
        else:
            self.processing_flags = processing_flags
    
    @classmethod
    def create_optimized(cls, theta=0.0, capture_id=0):
        """Factory method dla zoptymalizowanego tworzenia obiektu"""
        return cls(
            theta=theta,
            texture=None,
            lasers=[None, None],
            timestamp=time.time(),
            capture_id=capture_id,
            quality_score=0.0,
            processing_flags={
                'texture_processed': False,
                'laser_0_processed': False,
                'laser_1_processed': False,
                'validation_passed': False
            }
        )
    
    def set_texture(self, texture, copy_data=True):
        """Bezpieczne ustawianie tekstury z opcjonalnym kopiowaniem"""
        if texture is not None:
            self.texture = texture.copy() if copy_data else texture
            self.processing_flags['texture_processed'] = True
            self._update_quality_score()
    
    def set_laser(self, index, laser_image, copy_data=True):
        """Bezpieczne ustawianie obrazu lasera"""
        if 0 <= index < len(self.lasers) and laser_image is not None:
            self.lasers[index] = laser_image.copy() if copy_data else laser_image
            self.processing_flags['laser_{}_processed'.format(index)] = True
            self._update_quality_score()
    
    def set_lasers(self, laser_images, copy_data=True):
        """Ustawia wszystkie obrazy laserów jednocześnie"""
        if laser_images and len(laser_images) >= 2:
            for i in range(min(2, len(laser_images))):
                if laser_images[i] is not None:
                    self.set_laser(i, laser_images[i], copy_data)
    
    def get_laser(self, index):
        """Bezpieczne pobieranie obrazu lasera"""
        if 0 <= index < len(self.lasers):
            return self.lasers[index]
        return None
    
    def has_texture(self):
        """Sprawdza czy ma teksturę"""
        return self.texture is not None and self.processing_flags.get('texture_processed', False)
    
    def has_laser(self, index):
        """Sprawdza czy ma obraz lasera"""
        if 0 <= index < len(self.lasers):
            return (self.lasers[index] is not None and 
                   self.processing_flags.get('laser_{}_processed'.format(index), False))
        return False
    
    def has_any_laser(self):
        """Sprawdza czy ma jakikolwiek obraz lasera"""
        return any(self.has_laser(i) for i in range(len(self.lasers)))
    
    def get_data_size(self):
        """Zwraca informacje o rozmiarze danych"""
        size_info = {
            'texture_size': 0,
            'laser_sizes': [0, 0],
            'total_size': 0
        }
        
        if self.texture is not None:
            size_info['texture_size'] = self.texture.nbytes
        
        for i, laser in enumerate(self.lasers):
            if laser is not None and i < 2:
                size_info['laser_sizes'][i] = laser.nbytes
        
        size_info['total_size'] = (size_info['texture_size'] + 
                                  sum(size_info['laser_sizes']))
        
        return size_info
    
    def validate(self):
        """Waliduje dane capture"""
        is_valid = True
        
        # Sprawdź czy theta jest w sensownym zakresie
        if not (-720 <= self.theta <= 720):  # 2 pełne obroty w każdą stronę
            is_valid = False
        
        # Sprawdź czy mamy jakieś dane do przetworzenia
        if not self.has_texture() and not self.has_any_laser():
            is_valid = False
        
        # Sprawdź integralność obrazów
        if self.texture is not None:
            if self.texture.size == 0 or not np.any(self.texture):
                is_valid = False
        
        for i, laser in enumerate(self.lasers):
            if laser is not None:
                if laser.size == 0:
                    is_valid = False
        
        self.processing_flags['validation_passed'] = is_valid
        return is_valid
    
    def _update_quality_score(self):
        """Aktualizuje score jakości na podstawie dostępnych danych"""
        score = 0.0
        
        # Punkty za teksturę
        if self.has_texture():
            # Sprawdź kontrast i szczegóły tekstury
            if self.texture.size > 0:
                texture_std = np.std(self.texture)
                score += min(0.3, texture_std / 255.0 * 0.3)
        
        # Punkty za lasery
        for i in range(len(self.lasers)):
            if self.has_laser(i):
                laser = self.lasers[i]
                if laser.size > 0:
                    # Sprawdź intensywność lasera
                    laser_max = np.max(laser)
                    score += min(0.35, laser_max / 255.0 * 0.35)
        
        # Bonus za kompletność danych
        if self.has_texture() and self.has_any_laser():
            score += 0.2
        
        self.quality_score = min(1.0, score)
    
    def get_processing_summary(self):
        """Zwraca podsumowanie przetwarzania"""
        return {
            'capture_id': self.capture_id,
            'timestamp': self.timestamp,
            'theta_degrees': np.rad2deg(self.theta),
            'quality_score': self.quality_score,
            'has_texture': self.has_texture(),
            'has_laser_0': self.has_laser(0),
            'has_laser_1': self.has_laser(1),
            'validation_passed': self.processing_flags.get('validation_passed', False),
            'data_size': self.get_data_size()
        }
    
    def optimize_memory(self):
        """Optymalizuje użycie pamięci"""
        # Konwertuj do optymalnych typów danych jeśli możliwe
        if self.texture is not None:
            # Sprawdź czy można użyć uint8 zamiast większych typów
            if self.texture.dtype != np.uint8:
                if np.max(self.texture) <= 255:
                    self.texture = self.texture.astype(np.uint8)
        
        for i, laser in enumerate(self.lasers):
            if laser is not None:
                if laser.dtype != np.uint8:
                    if np.max(laser) <= 255:
                        self.lasers[i] = laser.astype(np.uint8)
    
    def create_lightweight_copy(self):
        """Tworzy lekką kopię bez danych obrazów (tylko metadane)"""
        return ScanCapture(
            theta=self.theta,
            texture=None,
            lasers=[None, None],
            timestamp=self.timestamp,
            capture_id=self.capture_id,
            quality_score=self.quality_score,
            processing_flags=self.processing_flags.copy()
        )
    
    def clear_data(self):
        """Czyści dane obrazów zwalniając pamięć"""
        self.texture = None
        self.lasers = [None, None]
        self.processing_flags = {
            'texture_processed': False,
            'laser_0_processed': False,
            'laser_1_processed': False,
            'validation_passed': False
        }
        self.quality_score = 0.0
        
        # Wymuś garbage collection
        gc.collect()
    
    def to_dict(self):
        """Konwertuje do słownika (bez danych obrazów)"""
        return {
            'theta': self.theta,
            'timestamp': self.timestamp,
            'capture_id': self.capture_id,
            'quality_score': self.quality_score,
            'processing_flags': self.processing_flags.copy(),
            'has_texture': self.has_texture(),
            'has_laser_0': self.has_laser(0),
            'has_laser_1': self.has_laser(1),
            'data_size': self.get_data_size()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Tworzy obiekt ze słownika"""
        return cls(
            theta=data.get('theta', 0.0),
            timestamp=data.get('timestamp', time.time()),
            capture_id=data.get('capture_id', 0),
            quality_score=data.get('quality_score', 0.0),
            processing_flags=data.get('processing_flags', {})
        )
    
    def __repr__(self):
        """String representation dla debugging"""
        return ("ScanCapture(id={}, theta={:.1f}°, quality={:.2f}, "
                "texture={}, lasers=[{}, {}])".format(
                    self.capture_id, np.rad2deg(self.theta), self.quality_score,
                    self.has_texture(), self.has_laser(0), self.has_laser(1)))
    
    def __del__(self):
        """Destruktor - zwalnia pamięć"""
        try:
            self.clear_data()
        except:
            pass


class ScanCaptureManager(object):
    """Manager do zarządzania kolekcją ScanCapture obiektów"""
    
    def __init__(self, max_captures=1000):
        self.captures = []
        self.max_captures = max_captures
        self._lock = threading.Lock()
        self._current_id = 0
    
    def add_capture(self, capture):
        """Dodaje capture z automatycznym zarządzaniem pamięcią"""
        with self._lock:
            # Sprawdź czy nie przekraczamy limitu
            if len(self.captures) >= self.max_captures:
                # Usuń najstarszy capture
                old_capture = self.captures.pop(0)
                old_capture.clear_data()
            
            # Ustaw ID jeśli nie ustawione
            if capture.capture_id == 0:
                capture.capture_id = self._get_next_id()
            
            # Waliduj przed dodaniem
            if capture.validate():
                self.captures.append(capture)
                return True
            else:
                return False
    
    def _get_next_id(self):
        """Zwraca kolejny ID"""
        self._current_id += 1
        return self._current_id
    
    def get_capture(self, capture_id):
        """Pobiera capture po ID"""
        with self._lock:
            for capture in self.captures:
                if capture.capture_id == capture_id:
                    return capture
        return None
    
    def get_captures_in_range(self, theta_min, theta_max):
        """Pobiera captures w określonym zakresie theta"""
        with self._lock:
            return [c for c in self.captures 
                   if theta_min <= c.theta <= theta_max]
    
    def get_quality_stats(self):
        """Zwraca statystyki jakości"""
        with self._lock:
            if not self.captures:
                return {}
            
            qualities = [c.quality_score for c in self.captures]
            return {
                'count': len(qualities),
                'avg_quality': np.mean(qualities),
                'min_quality': np.min(qualities),
                'max_quality': np.max(qualities),
                'std_quality': np.std(qualities)
            }
    
    def optimize_memory(self):
        """Optymalizuje pamięć wszystkich captures"""
        with self._lock:
            for capture in self.captures:
                capture.optimize_memory()
    
    def clear_all(self):
        """Czyści wszystkie captures"""
        with self._lock:
            for capture in self.captures:
                capture.clear_data()
            self.captures.clear()
            self._current_id = 0
    
    def get_summary(self):
        """Zwraca podsumowanie managera"""
        with self._lock:
            total_size = sum(c.get_data_size()['total_size'] for c in self.captures)
            
            return {
                'total_captures': len(self.captures),
                'max_captures': self.max_captures,
                'total_memory_bytes': total_size,
                'total_memory_mb': total_size / (1024 * 1024),
                'quality_stats': self.get_quality_stats()
            }