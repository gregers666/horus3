#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt awaryjnej naprawy problemu z kontekstem OpenGL w Horus3
Uruchom ten skrypt w katalogu głównym projektu Horus
"""

import os
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Tworzy kopię zapasową pliku"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Kopia zapasowa: {backup_path}")
        return True
    return False

def write_opengl_config():
    """Tworzy naprawiony plik opengl_config.py"""
    content = '''#!/usr/bin/env python3
"""
Horus3 OpenGL Configuration - NAPRAWIONA WERSJA
"""

# GŁÓWNA FLAGA - USTAW NA TRUE JEŚLI MASZ PROBLEMY Z OPENGL
DISABLE_OPENGL = False

# Debug
DEBUG_OPENGL = True
OPENGL_ERROR_HANDLING = True

import sys
import platform

SYSTEM_INFO = {
    'platform': platform.system(),
    'architecture': platform.architecture()[0],
    'python_version': sys.version,
    'opengl_available': None
}

def is_opengl_enabled():
    return not DISABLE_OPENGL

def enable_opengl():
    global DISABLE_OPENGL
    DISABLE_OPENGL = False
    if DEBUG_OPENGL:
        print("OpenGL enabled")

def disable_opengl():
    global DISABLE_OPENGL
    DISABLE_OPENGL = True
    if DEBUG_OPENGL:
        print("OpenGL disabled")

def check_opengl_support():
    try:
        import OpenGL
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
        from OpenGL.GL import glGetString, GL_VERSION, glGetError, GL_NO_ERROR
        glGetError()  # Wyczyść błędy
        version = glGetString(GL_VERSION)
        error = glGetError()
        if error == GL_NO_ERROR and version:
            if DEBUG_OPENGL:
                print(f"OpenGL version: {version.decode() if isinstance(version, bytes) else version}")
            return True
        return False
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"OpenGL context error: {e}")
        return False

check_opengl_support()
'''
    
    config_path = "horus/gui/util/opengl_config.py"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if backup_file(config_path):
        print(f"📄 Aktualizuję {config_path}")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Napisano {config_path}")

def patch_opengl_gui():
    """Naprawia opengl_gui.py"""
    gui_path = "horus/gui/util/opengl_gui.py"
    
    if not os.path.exists(gui_path):
        print(f"❌ Nie znaleziono {gui_path}")
        return
        
    backup_file(gui_path)
    
    with open(gui_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dodaj sprawdzenie kontekstu do activate_context
    if "def activate_context(self):" in content:
        old_method = '''def activate_context(self):
        """Bezpiecznie aktywuje kontekst OpenGL"""
        try:
            if self._context and self.IsShownOnScreen():
                self.SetCurrent(self._context)
                return True
            else:
                if DEBUG_OPENGL:
                    print("❌ Nie można aktywować kontekstu OpenGL")
                return False
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd aktywacji kontekstu: {e}")
            return False'''
            
        new_method = '''def activate_context(self):
        """Bezpiecznie aktywuje kontekst OpenGL"""
        if DISABLE_OPENGL:
            return False
            
        # Sprawdź czy okno jest gotowe
        if not self.IsShownOnScreen():
            return False
            
        size = self.GetSize()
        if size.GetWidth() <= 0 or size.GetHeight() <= 0:
            return False
        
        # Sprawdź/utwórz kontekst
        if self._context is None:
            try:
                self._context = glcanvas.GLContext(self)
            except Exception as e:
                if DEBUG_OPENGL:
                    print(f"❌ Nie można utworzyć kontekstu: {e}")
                return False
                
        try:
            self.SetCurrent(self._context)
            # Test kontekstu
            from .opengl_config import safe_gl_check
            if not hasattr(self, '_context_tested'):
                if safe_gl_check():
                    self._context_tested = True
                    if DEBUG_OPENGL:
                        print("✅ Kontekst OpenGL gotowy")
                else:
                    return False
            return True
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd aktywacji kontekstu: {e}")
            return False'''
            
        content = content.replace(old_method, new_method)
    
    # Napraw _on_gui_paint
    if "def _on_gui_paint(self, e):" in content:
        # Dodaj sprawdzenie kontekstu na początku
        paint_check = '''def _on_gui_paint(self, e):
        wx.PaintDC(self)
        
        if DISABLE_OPENGL:
            # Rysuj placeholder
            dc = wx.PaintDC(self)
            dc.SetBrush(wx.Brush(wx.Colour(200, 200, 200)))
            size = self.GetSize()
            dc.DrawRectangle(0, 0, size.GetWidth(), size.GetHeight())
            dc.DrawText("Renderowanie 3D wyłączone", 10, 10)
            return
            
        if not self.activate_context():
            return'''
            
        # Znajdź i zastąp początek metody
        lines = content.split('\n')
        new_lines = []
        in_paint_method = False
        method_indent = 0
        
        for line in lines:
            if "def _on_gui_paint(self, e):" in line:
                in_paint_method = True
                method_indent = len(line) - len(line.lstrip())
                new_lines.extend(paint_check.split('\n'))
                continue
            elif in_paint_method and line.strip() and not line.startswith(' ' * (method_indent + 1)):
                in_paint_method = False
            elif in_paint_method and "wx.PaintDC(self)" in line:
                continue  # Pomiń, już mamy
            elif in_paint_method and "try:" in line and "self.SetCurrent" in lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else False:
                continue  # Pomiń stary kod aktywacji
                
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
    
    with open(gui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Poprawiono {gui_path}")

def patch_opengl_helpers():
    """Naprawia opengl_helpers.py"""
    helpers_path = "horus/gui/util/opengl_helpers.py"
    
    if not os.path.exists(helpers_path):
        print(f"❌ Nie znaleziono {helpers_path}")
        return
        
    backup_file(helpers_path)
    
    with open(helpers_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dodaj funkcję sprawdzania kontekstu
    context_check_func = '''
def check_opengl_context():
    """Globalna funkcja sprawdzająca kontekst OpenGL"""
    if DISABLE_OPENGL:
        return False
    try:
        from OpenGL.GL import glGetError, glGetString, GL_VERSION, GL_NO_ERROR
        glGetError()  # Wyczyść błędy
        version = glGetString(GL_VERSION)
        error = glGetError()
        return error == GL_NO_ERROR and version is not None
    except Exception:
        return False
'''
    
    # Dodaj na końcu pliku przed ostatnią linią
    if not "def check_opengl_context():" in content:
        content = content.rstrip() + context_check_func + '\n'
    
    # Napraw metodę render w GLVBO
    if "def render(self):" in content and "class GLVBO" in content:
        # Znajdź klasę GLVBO i dodaj sprawdzenie kontekstu
        lines = content.split('\n')
        new_lines = []
        in_render_method = False
        
        for i, line in enumerate(lines):
            if "def render(self):" in line and any("class GLVBO" in prev_line for prev_line in lines[max(0, i-50):i]):
                in_render_method = True
                new_lines.append(line)
                new_lines.append("        if DISABLE_OPENGL:")
                new_lines.append("            return")
                new_lines.append("        if not check_opengl_context():")
                new_lines.append("            if DEBUG_OPENGL:")
                new_lines.append("                print('❌ Brak kontekstu OpenGL - nie można renderować VBO')")
                new_lines.append("            return")
                continue
            elif in_render_method and line.strip() and not line.startswith('        '):
                in_render_method = False
                
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
    
    with open(helpers_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Poprawiono {helpers_path}")

def patch_scene_view():
    """Naprawia scene_view.py"""
    scene_path = "horus/gui/util/scene_view.py"
    
    if not os.path.exists(scene_path):
        print(f"❌ Nie znaleziono {scene_path}")
        return
        
    backup_file(scene_path)
    
    with open(scene_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Usuń konfliktującą definicję DISABLE_OPENGL
    if "DISABLE_OPENGL = False" in content:
        content = content.replace("DISABLE_OPENGL = False", "# DISABLE_OPENGL = False  # używamy z opengl_config")
    
    # Dodaj sprawdzenie kontekstu w append_point_cloud
    if "def append_point_cloud(self, point, color):" in content:
        old_append = '''def append_point_cloud(self, point, color):
        self._object_point_cloud.append(point)
        self._object_texture.append(color)'''
        
        new_append = '''def append_point_cloud(self, point, color):
        if DISABLE_OPENGL:
            return
        # Sprawdź czy kontekst jest gotowy
        if not hasattr(self, 'activate_context') or not self.activate_context():
            # Zapisz punkty do bufora tymczasowego
            if not hasattr(self, '_pending_points'):
                self._pending_points = []
            self._pending_points.append((point.copy(), color.copy()))
            return
            
        self._object_point_cloud.append(point)
        self._object_texture.append(color)'''
        
        content = content.replace(old_append, new_append)
    
    # Dodaj sprawdzenie w _render_object
    if "def _render_object(self, obj, brightness=0):" in content:
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            if "def _render_object(self, obj, brightness=0):" in line:
                # Dodaj sprawdzenia kontekstu na początku metody
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * (indent + 4) + "if DISABLE_OPENGL:")
                new_lines.append(' ' * (indent + 8) + "return")
                new_lines.append(' ' * (indent + 4) + "if not hasattr(self, 'activate_context') or not self.activate_context():")
                new_lines.append(' ' * (indent + 8) + "return")
        
        content = '\n'.join(new_lines)
    
    with open(scene_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Poprawiono {scene_path}")

def create_test_script():
    """Tworzy skrypt testowy"""
    test_content = '''#!/usr/bin/env python3
"""Test kontekstu OpenGL po naprawach"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_opengl():
    try:
        # Test podstawowych importów
        print("🔧 Test importów...")
        from horus.gui.util.opengl_config import DISABLE_OPENGL, DEBUG_OPENGL, safe_gl_check
        print(f"   DISABLE_OPENGL = {DISABLE_OPENGL}")
        
        if DISABLE_OPENGL:
            print("✅ OpenGL wyłączony - aplikacja powinna działać w trybie 2D")
            return True
            
        import wx
        from wx import glcanvas
        from OpenGL.GL import *
        print("✅ Importy przeszły")
        
        # Test tworzenia kontekstu
        print("🔧 Test kontekstu...")
        app = wx.App(False)
        frame = wx.Frame(None, title="Test")
        
        try:
            canvas = glcanvas.GLCanvas(frame)
            context = glcanvas.GLContext(canvas)
            print("✅ Kontekst utworzony")
            
            frame.Show()
            canvas.SetCurrent(context)
            
            if safe_gl_check():
                print("✅ Kontekst działa poprawnie")
                result = True
            else:
                print("⚠️ Kontekst ma problemy")
                result = False
                
        except Exception as e:
            print(f"❌ Błąd kontekstu: {e}")
            result = False
            
        frame.Destroy()
        app.Destroy()
        return result
        
    except Exception as e:
        print(f"❌ Błąd testu: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TEST NAPRAWEK OPENGL")
    print("=" * 50)
    
    if test_opengl():
        print("\\n✅ Test przeszedł - OpenGL powinien działać")
    else:
        print("\\n❌ Test nie przeszedł - ustaw DISABLE_OPENGL = True")
        print("   w pliku horus/gui/util/opengl_config.py")
'''
    
    with open("test_opengl_fix.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    print("✅ Utworzono test_opengl_fix.py")

def main():
    print("=" * 60)
    print("SKRYPT AWARYJNEJ NAPRAWY KONTEKSTU OPENGL")
    print("=" * 60)
    
    if not os.path.exists("horus"):
        print("❌ Nie znaleziono katalogu 'horus'. Uruchom skrypt w katalogu głównym projektu.")
        return False
    
    print("🔧 Tworzenie kopii zapasowych i naprawianie plików...")
    
    # Napraw pliki
    write_opengl_config()
    patch_opengl_gui()
    patch_opengl_helpers()
    patch_scene_view()
    create_test_script()
    
    print("\n" + "=" * 60)
    print("PODSUMOWANIE NAPRAW")
    print("=" * 60)
    print("✅ opengl_config.py - główna konfiguracja")
    print("✅ opengl_gui.py - naprawiono aktywację kontekstu")
    print("✅ opengl_helpers.py - dodano sprawdzenie kontekstu")
    print("✅ scene_view.py - naprawiono renderowanie")
    print("✅ test_opengl_fix.py - skrypt testowy")
    
    print("\n🚀 NASTĘPNE KROKI:")
    print("1. Uruchom: python test_opengl_fix.py")
    print("2. Jeśli test przejdzie - uruchom Horus3")
    print("3. Jeśli nie - ustaw DISABLE_OPENGL = True w opengl_config.py")
    
    return True

if __name__ == "__main__":
    main()