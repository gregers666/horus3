#!/usr/bin/env python3
"""
Horus3 OpenGL Configuration
Globalny plik konfiguracji dla renderowania 3D
"""

# Główna flaga kontrolująca renderowanie OpenGL
# Ustaw na True aby włączyć renderowanie 3D
# Ustaw na False aby wyłączyć renderowanie 3D (gdy brak sterowników)
DISABLE_OPENGL = False  # ZMIENIONE: Włączamy OpenGL

# Szczegółowe ustawienia debugowania
DEBUG_OPENGL = True  # Pokaż komunikaty o stanie OpenGL
OPENGL_ERROR_HANDLING = True  # Obsługa błędów OpenGL

# Informacje o systemie (automatycznie wykrywane)
import sys
import platform

SYSTEM_INFO = {
    'platform': platform.system(),
    'architecture': platform.architecture()[0],
    'python_version': sys.version,
    'opengl_available': None  # Będzie wykryte automatycznie
}

def is_opengl_enabled():
    """Sprawdza czy OpenGL jest włączony"""
    return not DISABLE_OPENGL

def enable_opengl():
    """Włącza OpenGL (wymaga ponownego uruchomienia aplikacji)"""
    global DISABLE_OPENGL
    DISABLE_OPENGL = False
    if DEBUG_OPENGL:
        print("OpenGL enabled - restart application to take effect")

def disable_opengl():
    """Wyłącza OpenGL"""
    global DISABLE_OPENGL
    DISABLE_OPENGL = True  # POPRAWIONE: było False
    if DEBUG_OPENGL:
        print("OpenGL disabled")

def check_opengl_support():
    """Sprawdza dostępność OpenGL w systemie"""
    try:
        import OpenGL
        from OpenGL.GL import glGetString, GL_VERSION
        # Nie sprawdzamy wersji tutaj, bo wymaga aktywnego kontekstu
        SYSTEM_INFO['opengl_available'] = True
        if DEBUG_OPENGL:
            print("OpenGL biblioteka dostępna")
        return True
    except Exception as e:
        SYSTEM_INFO['opengl_available'] = False
        if DEBUG_OPENGL:
            print(f"OpenGL not available: {e}")
        return False

def safe_gl_check():
    """Bezpieczne sprawdzenie OpenGL z aktywnym kontekstem"""
    try:
        from OpenGL.GL import glGetString, GL_VERSION
        version = glGetString(GL_VERSION)
        if version:
            if DEBUG_OPENGL:
                print(f"OpenGL version: {version.decode() if isinstance(version, bytes) else version}")
            return True
        return False
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"OpenGL context error: {e}")
        return False

# Sprawdź OpenGL przy imporcie
if __name__ != "__main__":
    check_opengl_support()