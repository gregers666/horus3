#!/usr/bin/env python3
"""
Prosty skrypt do przełączania flag OpenGL w Horus3
"""

import sys
import os

def update_opengl_flags(disable_opengl=None, debug_opengl=None):
    """
    Aktualizuje flagi OpenGL w pliku konfiguracji
    
    Args:
        disable_opengl: True/False lub None (nie zmieniaj)
        debug_opengl: True/False lub None (nie zmieniaj)
    """
    
    config_path = "src/horus/gui/util/opengl_config.py"
    
    if not os.path.exists(config_path):
        print(f"❌ Plik konfiguracji nie istnieje: {config_path}")
        print("💡 Utwórz go najpierw poprzez uruchomienie głównego skryptu systemu")
        return False
    
    try:
        # Wczytaj plik
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Zmień DISABLE_OPENGL
        if disable_opengl is not None:
            old_value = "True" if "DISABLE_OPENGL = True" in content else "False"
            new_value = "True" if disable_opengl else "False"
            
            content = content.replace(f"DISABLE_OPENGL = {old_value}", f"DISABLE_OPENGL = {new_value}")
            
            if old_value != new_value:
                status = "WYŁĄCZONY" if disable_opengl else "WŁĄCZONY"
                changes.append(f"DISABLE_OPENGL: {old_value} → {new_value} (OpenGL {status})")
        
        # Zmień DEBUG_OPENGL
        if debug_opengl is not None:
            old_value = "True" if "DEBUG_OPENGL = True" in content else "False"
            new_value = "True" if debug_opengl else "False"
            
            content = content.replace(f"DEBUG_OPENGL = {old_value}", f"DEBUG_OPENGL = {new_value}")
            
            if old_value != new_value:
                status = "WŁĄCZONY" if debug_opengl else "WYŁĄCZONY"
                changes.append(f"DEBUG_OPENGL: {old_value} → {new_value} (Debug {status})")
        
        # Zapisz jeśli były zmiany
        if content != original_content:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Zaktualizowano flagi OpenGL:")
            for change in changes:
                print(f"   {change}")
            return True
        else:
            print("ℹ️ Brak zmian w flagach")
            return False
            
    except Exception as e:
        print(f"❌ Błąd aktualizacji: {e}")
        return False

def show_current_status():
    """Pokazuje aktualny stan flag OpenGL"""
    
    config_path = "src/horus/gui/util/opengl_config.py"
    
    if not os.path.exists(config_path):
        print(f"❌ Plik konfiguracji nie istnieje: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sprawdź DISABLE_OPENGL
        if "DISABLE_OPENGL = True" in content:
            opengl_status = "WYŁĄCZONY ❌"
        elif "DISABLE_OPENGL = False" in content:
            opengl_status = "WŁĄCZONY ✅"
        else:
            opengl_status = "NIEZNANY ❓"
        
        # Sprawdź DEBUG_OPENGL
        if "DEBUG_OPENGL = True" in content:
            debug_status = "WŁĄCZONY ✅"
        elif "DEBUG_OPENGL = False" in content:
            debug_status = "WYŁĄCZONY ❌"
        else:
            debug_status = "NIEZNANY ❓"
        
        print("📊 Aktualny stan flag OpenGL:")
        print(f"   🎮 DISABLE_OPENGL: {opengl_status}")
        print(f"   🐛 DEBUG_OPENGL: {debug_status}")
        
    except Exception as e:
        print(f"❌ Błąd odczytu stanu: {e}")

def print_usage():
    """Pokazuje instrukcję użycia"""
    
    print("🎮 Horus3 OpenGL Toggle - Prosty przełącznik flag")
    print("=" * 50)
    print("📋 Użycie:")
    print("  python opengl_toggle.py status           - pokaż stan flag")
    print("  python opengl_toggle.py on               - włącz OpenGL")
    print("  python opengl_toggle.py off              - wyłącz OpenGL")
    print("  python opengl_toggle.py debug-on         - włącz debug OpenGL")
    print("  python opengl_toggle.py debug-off        - wyłącz debug OpenGL")
    print("  python opengl_toggle.py toggle           - przełącz OpenGL")
    print("  python opengl_toggle.py debug-toggle     - przełącz debug")
    print("")
    print("🔄 Kombinacje:")
    print("  python opengl_toggle.py on debug-on      - włącz OpenGL i debug")
    print("  python opengl_toggle.py off debug-off    - wyłącz OpenGL i debug")
    print("")
    print("💡 Po zmianie flag uruchom ponownie aplikację: ./horus3")

def main():
    if len(sys.argv) < 2:
        print_usage()
        print("")
        show_current_status()
        return
    
    commands = sys.argv[1:]
    
    # Parsuj komendy
    disable_opengl = None
    debug_opengl = None
    show_status = False
    
    for cmd in commands:
        cmd = cmd.lower()
        
        if cmd == "status":
            show_status = True
        elif cmd == "on":
            disable_opengl = False  # Włącz OpenGL (disable=False)
        elif cmd == "off":
            disable_opengl = True   # Wyłącz OpenGL (disable=True)
        elif cmd == "toggle":
            # Sprawdź aktualny stan i przełącz
            config_path = "src/horus/gui/util/opengl_config.py"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read()
                disable_opengl = False if "DISABLE_OPENGL = True" in content else True
        elif cmd == "debug-on":
            debug_opengl = True
        elif cmd == "debug-off":
            debug_opengl = False
        elif cmd == "debug-toggle":
            # Sprawdź aktualny stan debug i przełącz
            config_path = "src/horus/gui/util/opengl_config.py"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read()
                debug_opengl = False if "DEBUG_OPENGL = True" in content else True
        else:
            print(f"❌ Nieznana komenda: {cmd}")
            print_usage()
            return
    
    # Wykonaj akcje
    if show_status:
        show_current_status()
    
    if disable_opengl is not None or debug_opengl is not None:
        success = update_opengl_flags(disable_opengl, debug_opengl)
        if success:
            print("🔄 Uruchom ponownie aplikację: ./horus3")
        print("")
        show_current_status()

if __name__ == "__main__":
    main()