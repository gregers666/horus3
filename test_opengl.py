#!/usr/bin/env python3
import sys
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import wx
    import wx.glcanvas
    
    class TestFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, title="OpenGL Test")
            self.canvas = wx.glcanvas.GLCanvas(self)
            self.context = wx.glcanvas.GLContext(self.canvas)
            
    app = wx.App()
    frame = TestFrame()
    frame.Show()
    print("✅ OpenGL + wxPython działa!")
    app.MainLoop()
    
except Exception as e:
    print(f"❌ Błąd OpenGL: {e}")
    sys.exit(1)
