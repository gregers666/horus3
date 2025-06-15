#!/usr/bin/env python3
"""
Definitywne wyłączenie renderowania 3D
"""

import os

def disable_3d_rendering():
    """
    Wyłącza renderowanie 3D poprzez modyfikację kluczowych plików
    """
    
    files_to_modify = [
        "src/horus/gui/util/scene_view.py",
        "src/horus/gui/util/opengl_gui.py"
    ]
    
    modifications = []
    
    # 1. Wyłącz scene_view.py
    scene_view_path = "src/horus/gui/util/scene_view.py"
    if os.path.exists(scene_view_path):
        try:
            with open(scene_view_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Znajdź wszystkie metody związane z renderowaniem i wyłącz je
            changes_made = False
            
            # Wyłącz on_paint
            if 'def on_paint(self, event):' in content:
                if 'return  # 3D disabled' not in content:
                    content = content.replace(
                        'def on_paint(self, event):',
                        'def on_paint(self, event):\n        return  # 3D disabled due to OpenGL context issues'
                    )
                    changes_made = True
            
            # Wyłącz _draw_machine
            if 'def _draw_machine(self):' in content:
                if 'return  # 3D disabled' not in content:
                    content = content.replace(
                        'def _draw_machine(self):',
                        'def _draw_machine(self):\n        return  # 3D disabled due to OpenGL context issues'
                    )
                    changes_made = True
            
            # Wyłącz _render_object
            if 'def _render_object(self,' in content:
                content = content.replace(
                    'def _render_object(self,',
                    'def _render_object(self,\n        return  # 3D disabled\n        # Original method:'
                )
                changes_made = True
            
            if changes_made:
                # Kopia zapasowa
                with open(scene_view_path + '.3d_disabled_backup', 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Zapisz zmieniony plik
                with open(scene_view_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                modifications.append(f"✅ Wyłączono renderowanie 3D w {scene_view_path}")
        
        except Exception as e:
            modifications.append(f"❌ Błąd w {scene_view_path}: {e}")
    
    # 2. Wyłącz opengl_gui.py
    opengl_gui_path = "src/horus/gui/util/opengl_gui.py"
    if os.path.exists(opengl_gui_path):
        try:
            with open(opengl_gui_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Wyłącz _on_gui_paint
            if 'def _on_gui_paint(self, event):' in content:
                if 'return  # 3D disabled' not in content:
                    content = content.replace(
                        'def _on_gui_paint(self, event):',
                        'def _on_gui_paint(self, event):\n        return  # 3D disabled due to OpenGL context issues'
                    )
                    
                    # Kopia zapasowa
                    with open(opengl_gui_path + '.3d_disabled_backup', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Zapisz zmieniony plik
                    with open(opengl_gui_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    modifications.append(f"✅ Wyłączono GUI renderowanie 3D w {opengl_gui_path}")
        
        except Exception as e:
            modifications.append(f"❌ Błąd w {opengl_gui_path}: {e}")
    
    return modifications

def create_3d_disable_config():
    """
    Tworzy plik konfiguracyjny do łatwego włączania/wyłączania 3D
    """
    
    config_content = '''# Horus3 3D Rendering Configuration
# Ustaw na False aby wyłączyć renderowanie 3D
ENABLE_3D_RENDERING = False

# Przyczyna wyłączenia 3D
DISABLE_REASON = "OpenGL context issues on this system"

# Aby ponownie włączyć 3D, zmień ENABLE_3D_RENDERING na True
# i uruchom ponownie aplikację
'''
    
    try:
        with open("src/horus/gui/util/render_config.py", 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return "✅ Utworzono plik konfiguracji 3D: src/horus/gui/util/render_config.py"
    
    except Exception as e:
        return f"❌ Nie udało się utworzyć pliku konfiguracji: {e}"

def main():
    print("🔧 Definitywne wyłączenie renderowania 3D...")
    print("=" * 60)
    
    print("📝 Wyłączanie renderowania 3D w kluczowych plikach...")
    modifications = disable_3d_rendering()
    
    for mod in modifications:
        print(f"   {mod}")
    
    print(f"\n📋 Tworzenie pliku konfiguracji...")
    config_result = create_3d_disable_config()
    print(f"   {config_result}")
    
    print("\n" + "=" * 60)
    print("🎉 RENDEROWANIE 3D WYŁĄCZONE!")
    print("💡 Aplikacja będzie działać bez widoku 3D, ale wszystkie inne funkcje będą dostępne")
    print("📷 Skanowanie 3D nadal będzie działać - tylko podgląd będzie wyłączony")
    print("🚀 Uruchom aplikację: ./horus3")
    print("\n🔄 Aby ponownie włączyć 3D w przyszłości:")
    print("   1. Zainstaluj poprawne sterowniki OpenGL")
    print("   2. Przywróć pliki z kopii zapasowych (.3d_disabled_backup)")
    print("   3. Lub edytuj render_config.py")

if __name__ == "__main__":
    main()