#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import threading
import time
import logging

from horus.engine.driver.driver import Driver
from horus.engine.algorithms.image_capture import ImageCapture
from horus.engine.algorithms.image_detection import ImageDetection
from horus.engine.algorithms.laser_segmentation import LaserSegmentation
from horus.engine.algorithms.point_cloud_generation import PointCloudGeneration
from horus.engine.algorithms.point_cloud_roi import PointCloudROI

logger = logging.getLogger(__name__)


class Scan(object):
    """Zoptymalizowana klasa bazowa dla skanowania z threading"""

    def __init__(self):
        # Komponenty skanowania
        self.driver = Driver()
        self.image_capture = ImageCapture()
        self.image_detection = ImageDetection()
        self.laser_segmentation = LaserSegmentation()
        self.point_cloud_generation = PointCloudGeneration()
        self.point_cloud_roi = PointCloudROI()
        
        # Stan skanowania
        self.is_scanning = False
        self._inactive = False
        self._paused = False
        
        # Callbacks - Observer pattern
        self._before_callback = None
        self._progress_callback = None
        self._after_callback = None
        
        # Progress tracking
        self._progress = 0
        self._range = 0
        
        # Threading
        self._capture_thread = None
        self._process_thread = None
        self._thread_lock = threading.Lock()
        
        # Performance monitoring
        self._start_time = None
        self._last_progress_update = 0
        self._performance_stats = {
            "captures_completed": 0,
            "processes_completed": 0,
            "errors_count": 0,
            "avg_capture_time": 0,
            "avg_process_time": 0
        }
        
        # Configuration
        self._enable_performance_monitoring = True
        self._progress_update_interval = 1.0  # seconds
        self._graceful_shutdown_timeout = 5.0  # seconds

    def set_callbacks(self, before, progress, after):
        """Ustawia callback functions"""
        self._before_callback = before
        self._progress_callback = progress
        self._after_callback = after

    def set_performance_monitoring(self, enabled):
        """Włącza/wyłącza monitorowanie wydajności"""
        self._enable_performance_monitoring = enabled

    def start(self):
        """Rozpoczyna skanowanie z lepszą obsługą threading"""
        with self._thread_lock:
            if self.is_scanning:
                logger.warning("Scan already in progress")
                return False
                
            logger.info("Starting scan...")
            
            # Wywołaj callback przed rozpoczęciem
            if self._before_callback is not None:
                try:
                    self._before_callback()
                except Exception as e:
                    logger.error(f"Before callback error: {e}")
                    return False

            # Zainicjalizuj progress
            if self._progress_callback is not None:
                try:
                    self._progress_callback(0)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

            # Przygotuj skanowanie
            try:
                self._initialize()
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                self._handle_scan_error(e)
                return False

            # Ustaw stan
            self.is_scanning = True
            self._inactive = False
            self._paused = False
            self._start_time = time.time()
            self._reset_performance_stats()

            # Uruchom wątki
            self._start_threads()
            
            logger.info("Scan started successfully")
            return True

    def stop(self):
        """Zatrzymuje skanowanie z graceful shutdown"""
        logger.info("Stopping scan...")
        
        with self._thread_lock:
            if not self.is_scanning:
                logger.warning("No scan in progress")
                return
                
            # Ustaw flagi zatrzymania
            self._inactive = False
            self.is_scanning = False

        # Poczekaj na zakończenie wątków
        self._wait_for_threads()
        
        # Wyczyść stan
        self._cleanup_after_scan()
        
        logger.info("Scan stopped")

    def pause(self):
        """Wstrzymuje skanowanie"""
        if self.is_scanning and not self._paused:
            logger.info("Pausing scan...")
            self._inactive = True
            self._paused = True

    def resume(self):
        """Wznawia skanowanie"""
        if self.is_scanning and self._paused:
            logger.info("Resuming scan...")
            self._inactive = False
            self._paused = False

    def is_paused(self):
        """Sprawdza czy skanowanie jest wstrzymane"""
        return self._paused

    def get_progress(self):
        """Zwraca aktualny postęp"""
        if self._range > 0:
            return min(100, int(100 * self._progress / self._range))
        return 0

    def get_performance_stats(self):
        """Zwraca statystyki wydajności"""
        if not self._enable_performance_monitoring:
            return None
            
        stats = self._performance_stats.copy()
        if self._start_time:
            stats["elapsed_time"] = time.time() - self._start_time
        return stats

    def _start_threads(self):
        """Uruchamia wątki capture i process"""
        try:
            # Wątek przechwytywania
            self._capture_thread = threading.Thread(
                target=self._safe_capture,
                name="ScanCapture",
                daemon=True
            )
            
            # Wątek przetwarzania
            self._process_thread = threading.Thread(
                target=self._safe_process,
                name="ScanProcess", 
                daemon=True
            )
            
            # Uruchom wątki
            self._capture_thread.start()
            self._process_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting threads: {e}")
            self.is_scanning = False
            raise

    def _wait_for_threads(self):
        """Czeka na zakończenie wątków z timeout"""
        threads = [self._capture_thread, self._process_thread]
        
        for thread in threads:
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=self._graceful_shutdown_timeout)
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not finish gracefully")
                except Exception as e:
                    logger.error(f"Error joining thread {thread.name}: {e}")

    def _safe_capture(self):
        """Bezpieczny wrapper dla _capture"""
        try:
            self._capture()
        except Exception as e:
            logger.error(f"Capture thread error: {e}")
            self._handle_scan_error(e)

    def _safe_process(self):
        """Bezpieczny wrapper dla _process"""
        try:
            self._process()
        except Exception as e:
            logger.error(f"Process thread error: {e}")
            self._handle_scan_error(e)

    def _handle_scan_error(self, error):
        """Obsługuje błędy skanowania"""
        logger.error(f"Scan error: {error}")
        
        with self._thread_lock:
            self.is_scanning = False
            
        if self._enable_performance_monitoring:
            self._performance_stats["errors_count"] += 1
        
        # Wywołaj callback błędu
        if self._after_callback is not None:
            try:
                self._after_callback((False, error))
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _update_progress_callback(self):
        """Aktualizuje progress callback z throttling"""
        if self._progress_callback is None:
            return
            
        current_time = time.time()
        if (current_time - self._last_progress_update) >= self._progress_update_interval:
            try:
                progress = self.get_progress()
                self._progress_callback(progress)
                self._last_progress_update = current_time
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _reset_performance_stats(self):
        """Resetuje statystyki wydajności"""
        self._performance_stats = {
            "captures_completed": 0,
            "processes_completed": 0,
            "errors_count": 0,
            "avg_capture_time": 0,
            "avg_process_time": 0
        }

    def _update_performance_stats(self, operation_type, elapsed_time):
        """Aktualizuje statystyki wydajności"""
        if not self._enable_performance_monitoring:
            return
            
        try:
            if operation_type == "capture":
                count = self._performance_stats["captures_completed"]
                avg_key = "avg_capture_time"
                count_key = "captures_completed"
            elif operation_type == "process":
                count = self._performance_stats["processes_completed"]
                avg_key = "avg_process_time"
                count_key = "processes_completed"
            else:
                return
            
            # Aktualizuj średnią (running average)
            current_avg = self._performance_stats[avg_key]
            new_avg = (current_avg * count + elapsed_time) / (count + 1)
            
            self._performance_stats[avg_key] = new_avg
            self._performance_stats[count_key] = count + 1
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    def _cleanup_after_scan(self):
        """Czyści zasoby po skanowaniu"""
        try:
            # Reset stream states
            if hasattr(self.image_capture, 'stream'):
                self.image_capture.stream = True
            if hasattr(self.image_detection, 'stream'):
                self.image_detection.stream = True
                
            # Clear thread references
            self._capture_thread = None
            self._process_thread = None
            
            # Reset state
            self._inactive = False
            self._paused = False
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def get_scan_info(self):
        """Zwraca informacje o skanowaniu"""
        info = {
            "is_scanning": self.is_scanning,
            "is_paused": self._paused,
            "progress": self.get_progress(),
            "progress_raw": self._progress,
            "range": self._range
        }
        
        if self._start_time:
            info["elapsed_time"] = time.time() - self._start_time
            
        if self._enable_performance_monitoring:
            info["performance"] = self.get_performance_stats()
            
        return info

    def _initialize(self):
        """Metoda do przeciążenia w klasach pochodnych"""
        pass

    def _capture(self):
        """Metoda do przeciążenia w klasach pochodnych"""
        pass

    def _process(self):
        """Metoda do przeciążenia w klasach pochodnych"""
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        if self.is_scanning:
            self.stop()