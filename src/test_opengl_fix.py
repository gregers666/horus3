#!/usr/bin/env python3
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
        print("\n✅ Test przeszedł - OpenGL powinien działać")
    else:
        print("\n❌ Test nie przeszedł - ustaw DISABLE_OPENGL = True")
        print("   w pliku horus/gui/util/opengl_config.py")
