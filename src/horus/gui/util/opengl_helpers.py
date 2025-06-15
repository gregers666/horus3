#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.\
                 Copyright (C) 2013 David Braam from Cura Project'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx
import numpy

from horus.util.resources import get_path_for_image

import OpenGL

OpenGL.ERROR_CHECKING = False
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GL import shaders

import logging
logger = logging.getLogger(__name__)

from sys import platform as _platform

# OpenGL global disable flag
from .opengl_config import DISABLE_OPENGL, DEBUG_OPENGL

if _platform != 'darwin':
    try:
        glutInit()  # Hack; required before glut can be called. Not required for all OS.
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"GLUT initialization failed: {e}")

def check_opengl_context():
    """Sprawdza czy kontekst OpenGL jest dostępny"""
    if DISABLE_OPENGL:
        return False
    try:
        # Próba wywołania prostej funkcji OpenGL
        version = glGetString(GL_VERSION)
        return version is not None
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"Brak kontekstu OpenGL: {e}")
        return False

def ensure_context_active():
    """Sprawdza i raportuje stan kontekstu OpenGL bez przerywania wykonania"""
    if DISABLE_OPENGL:
        return False
    try:
        # Nie wywołujemy żadnych funkcji OpenGL które wymagają kontekstu
        # Tylko sprawdzamy czy możemy wywołać glGetString
        version = glGetString(GL_VERSION)
        if version:
            return True
        else:
            if DEBUG_OPENGL:
                print("⚠️ Kontekst OpenGL nieaktywny ale bez błędu")
            return False
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"⚠️ Kontekst OpenGL problem: {e}")
        return False

class GLReferenceCounter(object):

    def __init__(self):
        self._ref_counter = 1

    def inc_ref(self):
        self._ref_counter += 1

    def dec_ref(self):
        self._ref_counter -= 1
        return self._ref_counter <= 0


def has_shader_support():
    """Sprawdza czy shadery są obsługiwane"""
    if DISABLE_OPENGL:
        return False
    try:
        if not check_opengl_context():
            return False
        return bool(glCreateShader)
    except:
        return False


class GLShader(GLReferenceCounter):

    def __init__(self, vertex_program, fragment_program):
        super(GLShader, self).__init__()
        self._vertex_program = vertex_program
        self._fragment_program = fragment_program
        self._program = None
        
        if DISABLE_OPENGL:
            if DEBUG_OPENGL:
                print("OpenGL wyłączony - nie tworzę shaderów")
            return
            
        if not check_opengl_context():
            if DEBUG_OPENGL:
                print("Brak kontekstu OpenGL - nie można utworzyć shadera")
            return
            
        try:
            vertex_shader = shaders.compileShader(vertex_program, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_program, GL_FRAGMENT_SHADER)

            # shader.compileProgram tries to return the shader program as a overloaded int.
            # But the return value of a shader does not always fit in a int (needs to be a long).
            # So we do raw OpenGL calls.
            # This is to ensure that this works on intel GPU's
            self._program = glCreateProgram()
            glAttachShader(self._program, vertex_shader)
            glAttachShader(self._program, fragment_shader)
            glLinkProgram(self._program)
            # Validation has to occur *after* linking
            glValidateProgram(self._program)
            if glGetProgramiv(self._program, GL_VALIDATE_STATUS) == GL_FALSE:
                raise RuntimeError("Validation failure: %s" % (glGetProgramInfoLog(self._program)))
            if glGetProgramiv(self._program, GL_LINK_STATUS) == GL_FALSE:
                raise RuntimeError("Link failure: %s" % (glGetProgramInfoLog(self._program)))
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            
            if DEBUG_OPENGL:
                print("✅ Shader utworzony pomyślnie")
                
        except RuntimeError as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd tworzenia shadera: {e}")
            self._program = None
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Nieoczekiwany błąd shadera: {e}")
            self._program = None

    def bind(self):
        if DISABLE_OPENGL or self._program is None:
            return
        # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
        try:
            shaders.glUseProgram(self._program)
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"Błąd bindowania shadera: {e}")

    def unbind(self):
        if DISABLE_OPENGL:
            return
        # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
        try:
            shaders.glUseProgram(0)
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"Błąd unbindowania shadera: {e}")

    def release(self):
        if self._program is not None and not DISABLE_OPENGL:
            try:
                glDeleteProgram(self._program)
                self._program = None
                if DEBUG_OPENGL:
                    print("✅ Shader zwolniony")
            except Exception as e:
                if DEBUG_OPENGL:
                    print(f"Błąd zwalniania shadera: {e}")

    def set_uniform(self, name, value):
        if DISABLE_OPENGL or self._program is None:
            return
        # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
        try:
            if type(value) is float:
                glUniform1f(glGetUniformLocation(self._program, name), value)
            elif type(value) is numpy.matrix:
                glUniformMatrix3fv(
                    glGetUniformLocation(self._program, name), 1, False,
                    value.getA().astype(numpy.float32))
            else:
                logger.warning('Unknown type for setUniform: %s' % (str(type(value))))
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"Błąd ustawiania uniform: {e}")

    def is_valid(self):
        return self._program is not None and not DISABLE_OPENGL

    def get_vertex_shader(self):
        return self._vertex_program

    def get_fragment_shader(self):
        return self._fragment_program

    def __del__(self):
        if self._program is not None and bool(glDeleteProgram) and not DISABLE_OPENGL:
            logger.warning("Shader was not properly released!")


class GLFakeShader(GLReferenceCounter):
    """
    A Class that acts as an OpenGL shader, but in reality is not one.
    Used if shaders are not supported.
    """

    def __init__(self):
        super(GLFakeShader, self).__init__()

    def bind(self):
        if DISABLE_OPENGL:
            return
        # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
        try:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0, 0, 0, 0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0, 0, 0, 0])
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"Błąd fake shadera bind: {e}")

    def unbind(self):
        if DISABLE_OPENGL:
            return
        # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
        try:
            glDisable(GL_LIGHTING)
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"Błąd fake shadera unbind: {e}")

    def release(self):
        pass

    def set_uniform(self, name, value):
        pass

    def is_valid(self):
        return True

    def get_vertex_shader(self):
        return ''

    def get_fragment_shader(self):
        return ''


class GLVBO(GLReferenceCounter):
    """
    Vertex buffer object. Used for faster rendering.
    """
    
    # Statyczna zmienna do zapamiętania czy VBO działa
    _vbo_support_tested = False
    _vbo_works = False

    def __init__(self, render_type, vertex_array,
                 normal_array=None, indices_array=None, color_array=None, point_size=2):
                 
        super(GLVBO, self).__init__()
        self._render_type = render_type
        self._point_size = point_size
        self._buffer = None
        self._buffer_indices = None
        
        # ZAWSZE przechowuj kopię danych dla fallback
        self._vertex_array = vertex_array
        self._normal_array = normal_array
        self._indices_array = indices_array
        self._color_array = color_array
        self._size = len(vertex_array) if vertex_array is not None else 0
        self._has_normals = self._normal_array is not None
        self._has_indices = self._indices_array is not None
        self._has_color = self._color_array is not None
        if self._has_indices:
            self._size = len(indices_array)
        
        if DISABLE_OPENGL:
            if DEBUG_OPENGL:
                print("OpenGL wyłączony - używam fallback dla VBO")
            return
        
        # Sprawdź czy VBO jest dostępne (tylko raz)
        if not GLVBO._vbo_support_tested:
            GLVBO._test_vbo_support()
            
        # Jeśli VBO nie działa, nie próbuj go tworzyć
        if not GLVBO._vbo_works:
            if DEBUG_OPENGL:
                print("VBO nie działa w tym systemie - używam fallback")
            return
            
        if not check_opengl_context() or not bool(glGenBuffers):  # Fallback if buffers are not supported.
            if DEBUG_OPENGL:
                print("Brak VBO support - używam fallback")
            return
        else:
            try:
                if self._has_normals:
                    self._buffer = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, self._buffer)
                    glBufferData(GL_ARRAY_BUFFER, numpy.concatenate(
                        (vertex_array, normal_array), 1), GL_STATIC_DRAW)
                else:
                    if self._has_color:
                        glPointSize(self._point_size)
                        self._buffer = glGenBuffers(2)
                        glBindBuffer(GL_ARRAY_BUFFER, self._buffer[0])
                        glBufferData(GL_ARRAY_BUFFER, vertex_array, GL_STATIC_DRAW)
                        glBindBuffer(GL_ARRAY_BUFFER, self._buffer[1])
                        glBufferData(
                            GL_ARRAY_BUFFER, numpy.array(color_array, numpy.uint8), GL_STATIC_DRAW)
                    else:
                        self._buffer = glGenBuffers(1)
                        glBindBuffer(GL_ARRAY_BUFFER, self._buffer)
                        glBufferData(GL_ARRAY_BUFFER, vertex_array, GL_STATIC_DRAW)

                glBindBuffer(GL_ARRAY_BUFFER, 0)
                if self._has_indices:
                    self._buffer_indices = glGenBuffers(1)
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._buffer_indices)
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numpy.array(
                        indices_array, numpy.uint32), GL_STATIC_DRAW)
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                        
                if DEBUG_OPENGL:
                    print(f"✅ VBO utworzony: {self._size} elementów, kolory: {self._has_color}")
                    
            except Exception as e:
                if DEBUG_OPENGL:
                    print(f"❌ Błąd tworzenia VBO: {e}")
                # VBO się nie udało ale mamy fallback data
                self._buffer = None

    @staticmethod
    def _test_vbo_support():
        """Test czy VBO działa w tym systemie"""
        GLVBO._vbo_support_tested = True
        
        if DISABLE_OPENGL:
            GLVBO._vbo_works = False
            return
            
        try:
            if DEBUG_OPENGL:
                print("🧪 Testowanie obsługi VBO...")
                
            # Sprawdź czy funkcje VBO są dostępne
            if not bool(glGenBuffers):
                if DEBUG_OPENGL:
                    print("❌ VBO Test: Brak funkcji glGenBuffers")
                GLVBO._vbo_works = False
                return
                
            # Sprawdź kontekst
            if not ensure_context_active():
                if DEBUG_OPENGL:
                    print("❌ VBO Test: Brak aktywnego kontekstu")
                GLVBO._vbo_works = False
                return
            
            # Stwórz mały test VBO
            test_vertices = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=numpy.float32)
            
            test_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, test_buffer)
            glBufferData(GL_ARRAY_BUFFER, test_vertices, GL_STATIC_DRAW)
            
            # Test renderowania
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            # Sprawdź błędy
            error = glGetError()
            if error != GL_NO_ERROR:
                if DEBUG_OPENGL:
                    print(f"❌ VBO Test: Błąd OpenGL: {error}")
                GLVBO._vbo_works = False
            else:
                if DEBUG_OPENGL:
                    print("✅ VBO Test: VBO działa poprawnie!")
                GLVBO._vbo_works = True
            
            # Sprzątanie
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDeleteBuffers(1, [test_buffer])
            
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ VBO Test: Wyjątek: {e}")
            GLVBO._vbo_works = False

    def render(self):
        if DISABLE_OPENGL:
            if DEBUG_OPENGL:
                print("OpenGL wyłączony - pomijanie renderowania VBO")
            return
        
        # Jeśli wiemy że VBO nie działa, od razu idź na fallback
        if GLVBO._vbo_support_tested and not GLVBO._vbo_works:
            if DEBUG_OPENGL:
                print("🔄 VBO nie działa w systemie - używam immediate mode")
            self._render_simple_fallback()
            return
        
        # Całkowicie zrezygnuj z operacji OpenGL jeśli kontekst jest niestabilny
        # Zamiast tego użyj bardzo prostego sprawdzenia
        try:
            # Test czy możemy wykonać najprostszą operację OpenGL
            glGetError()  # Wyczyść poprzednie błędy
        except:
            if DEBUG_OPENGL:
                print("⚠️ VBO render: Kontekst całkowicie niestabilny, pomijanie")
            return
        
        # Sprawdź kontekst i spróbuj VBO, jeśli nie działa to fallback
        context_ok = ensure_context_active()
        use_vbo = context_ok and self._buffer is not None
        
        try:
            # BARDZO prosta aktywacja stanów - bez VBO
            if use_vbo:
                # Próba VBO - ale z bardzo ostrożnym podejściem
                try:
                    glEnableClientState(GL_VERTEX_ARRAY)
                    
                    if self._has_normals:
                        glBindBuffer(GL_ARRAY_BUFFER, self._buffer)
                        glEnableClientState(GL_NORMAL_ARRAY)
                        glVertexPointer(3, GL_FLOAT, 2 * 3 * 4, None)
                        glNormalPointer(GL_FLOAT, 2 * 3 * 4, None)
                    else:
                        if self._has_color:
                            glEnableClientState(GL_COLOR_ARRAY)
                            glBindBuffer(GL_ARRAY_BUFFER, self._buffer[1])
                            glColorPointer(3, GL_UNSIGNED_BYTE, 0, None)
                            glBindBuffer(GL_ARRAY_BUFFER, self._buffer[0])
                            glVertexPointer(3, GL_FLOAT, 0, None)
                        else:
                            glBindBuffer(GL_ARRAY_BUFFER, self._buffer)
                            glVertexPointer(3, GL_FLOAT, 3 * 4, None)

                    if self._has_indices:
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._buffer_indices)

                    # Renderowanie VBO
                    if self._has_indices:
                        glDrawElements(self._render_type, self._size, GL_UNSIGNED_INT, None)
                    else:
                        if self._size > 0:
                            glDrawArrays(self._render_type, 0, self._size)

                    # Cleanup VBO
                    if self._buffer is not None:
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                    if self._has_indices:
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                        
                    # Cleanup stanów
                    if self._has_normals:
                        glDisableClientState(GL_NORMAL_ARRAY)
                    if self._has_color:
                        glDisableClientState(GL_COLOR_ARRAY)
                    glDisableClientState(GL_VERTEX_ARRAY)
                        
                    if DEBUG_OPENGL:
                        print(f"✅ VBO renderowanie zakończone: {self._size} elementów")
                    return  # Sukces VBO
                        
                except Exception as e:
                    if DEBUG_OPENGL:
                        print(f"❌ VBO nie działa, oznaczam jako niedziałające i przełączam na immediate mode: {e}")
                    # Oznacz VBO jako niedziałające
                    GLVBO._vbo_works = False
                    # Wyczyść stany po błędzie VBO
                    try:
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                        glDisableClientState(GL_NORMAL_ARRAY)
                        glDisableClientState(GL_COLOR_ARRAY)
                        glDisableClientState(GL_VERTEX_ARRAY)
                    except:
                        pass
            
            # BARDZO PROSTY fallback rendering - tylko podstawowe funkcje
            if DEBUG_OPENGL:
                print("🔄 Używam immediate mode rendering")
            self._render_simple_fallback()
            
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd ogólny renderowania VBO: {e}")

    def _render_simple_fallback(self):
        """BARDZO PROSTY fallback rendering - tylko najbardziej podstawowe funkcje"""
        try:
            if self._vertex_array is None or len(self._vertex_array) == 0:
                if DEBUG_OPENGL:
                    print("⚠️ Brak danych dla simple fallback rendering")
                return
                
            # Bardzo prosta implementacja - bez sprawdzania kontekstu
            # Użyj tylko najważniejszych funkcji OpenGL
            
            # Dla punktów - po prostu narysuj jako GL_POINTS
            if self._render_type == GL_POINTS and self._size > 0:
                try:
                    # Podziel na mniejsze fragmenty żeby nie przeciążać OpenGL
                    batch_size = 1000
                    total_rendered = 0
                    
                    for batch_start in range(0, self._size, batch_size):
                        batch_end = min(batch_start + batch_size, self._size)
                        batch_count = batch_end - batch_start
                        
                        glBegin(GL_POINTS)
                        for i in range(batch_start, batch_end):
                            if i < len(self._vertex_array):
                                vertex = self._vertex_array[i]
                                if len(vertex) >= 3:
                                    if self._has_color and i < len(self._color_array):
                                        color = self._color_array[i] 
                                        if len(color) >= 3:
                                            glColor3ub(color[0], color[1], color[2])
                                    glVertex3f(vertex[0], vertex[1], vertex[2])
                        glEnd()
                        total_rendered += batch_count
                        
                    if DEBUG_OPENGL:
                        print(f"✅ BARDZO PROSTY fallback punkty: {total_rendered}/{self._size} punktów")
                    return
                except Exception as e:
                    if DEBUG_OPENGL:
                        print(f"❌ Błąd immediate mode points: {e}")
            
            # Dla trójkątów - renderuj jako GL_TRIANGLES
            elif self._render_type == GL_TRIANGLES and self._size > 0:
                try:
                    # Renderuj trójkąty w mniejszych fragmentach
                    batch_size = 999  # Musi być wielokrotnością 3 dla trójkątów
                    total_rendered = 0
                    
                    for batch_start in range(0, self._size, batch_size):
                        batch_end = min(batch_start + batch_size, self._size)
                        # Upewnij się że mamy pełne trójkąty
                        batch_end = batch_start + ((batch_end - batch_start) // 3) * 3
                        
                        if batch_end > batch_start:
                            glBegin(GL_TRIANGLES)
                            for i in range(batch_start, batch_end):
                                if i < len(self._vertex_array):
                                    vertex = self._vertex_array[i]
                                    if len(vertex) >= 3:
                                        if self._has_normals and i < len(self._normal_array):
                                            normal = self._normal_array[i]
                                            if len(normal) >= 3:
                                                glNormal3f(normal[0], normal[1], normal[2])
                                        if self._has_color and i < len(self._color_array):
                                            color = self._color_array[i] 
                                            if len(color) >= 3:
                                                glColor3ub(color[0], color[1], color[2])
                                        glVertex3f(vertex[0], vertex[1], vertex[2])
                            glEnd()
                            total_rendered += (batch_end - batch_start)
                        
                    if DEBUG_OPENGL:
                        print(f"✅ BARDZO PROSTY fallback trójkąty: {total_rendered}/{self._size} wierzchołków")
                    return
                except Exception as e:
                    if DEBUG_OPENGL:
                        print(f"❌ Błąd immediate mode triangles: {e}")
            
            # Inne typy primitywów
            else:
                try:
                    # Ogólny fallback dla innych typów
                    batch_size = 1000
                    total_rendered = 0
                    
                    for batch_start in range(0, min(self._size, 3000), batch_size):  # Max 3000 wierzchołków
                        batch_end = min(batch_start + batch_size, self._size, 3000)
                        
                        glBegin(self._render_type)
                        for i in range(batch_start, batch_end):
                            if i < len(self._vertex_array):
                                vertex = self._vertex_array[i]
                                if len(vertex) >= 3:
                                    if self._has_color and i < len(self._color_array):
                                        color = self._color_array[i] 
                                        if len(color) >= 3:
                                            glColor3ub(color[0], color[1], color[2])
                                    glVertex3f(vertex[0], vertex[1], vertex[2])
                        glEnd()
                        total_rendered += (batch_end - batch_start)
                        
                    if DEBUG_OPENGL:
                        print(f"✅ BARDZO PROSTY fallback ogólny: {total_rendered}/{self._size} ({self._render_type})")
                    return
                except Exception as e:
                    if DEBUG_OPENGL:
                        print(f"❌ Błąd immediate mode ogólny: {e}")
            
            if DEBUG_OPENGL:
                print(f"⚠️ Nie udało się wyrenderować typu: {self._render_type}")
                
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd simple fallback rendering: {e}")

    def _render_fallback(self):
        """USUŃ tę metodę - zbyt skomplikowana"""
        if DEBUG_OPENGL:
            print("⚠️ Stara metoda fallback - przekierowuję do simple")
        self._render_simple_fallback()

    def release(self):
        if DISABLE_OPENGL or self._buffer is None:
            return
        
        # Nie sprawdzamy kontekstu przy release - może być wywoływane podczas zamykania
        try:
            if self._has_color and isinstance(self._buffer, (list, tuple)):
                glBindBuffer(GL_ARRAY_BUFFER, self._buffer[0])
                glBufferData(GL_ARRAY_BUFFER, None, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [self._buffer[0]])
                glBindBuffer(GL_ARRAY_BUFFER, self._buffer[1])
                glBufferData(GL_ARRAY_BUFFER, None, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [self._buffer[1]])
            else:
                glBindBuffer(GL_ARRAY_BUFFER, self._buffer)
                glBufferData(GL_ARRAY_BUFFER, None, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [self._buffer])
                
            if self._has_indices and self._buffer_indices is not None:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._buffer_indices)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, None, GL_STATIC_DRAW)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [self._buffer_indices])
                
            self._buffer = None
            self._buffer_indices = None
            if DEBUG_OPENGL:
                print("✅ VBO zwolniony")
                
        except Exception as e:
            if DEBUG_OPENGL:
                print(f"❌ Błąd zwalniania VBO: {e}")
                
        self._vertex_array = None
        self._normal_array = None

    def __del__(self):
        if self._buffer is not None and bool(glDeleteBuffers) and not DISABLE_OPENGL:
            logger.warning("VBO was not properly released!")


def unproject(winx, winy, winz, model_matrix, proj_matrix, viewport):
    """
    Projects window position to 3D space. (gluUnProject).
    Reimplentation as some drivers crash with the original.
    """
    if DISABLE_OPENGL:
        return numpy.array([0, 0, 0], numpy.float32)
        
    try:
        np_model_matrix = numpy.matrix(numpy.array(model_matrix, numpy.float64).reshape((4, 4)))
        np_proj_matrix = numpy.matrix(numpy.array(proj_matrix, numpy.float64).reshape((4, 4)))
        final_matrix = np_model_matrix * np_proj_matrix
        final_matrix = numpy.linalg.inv(final_matrix)

        viewport = list(map(float, viewport))
        if viewport[2] > 0 and viewport[3] > 0:
            vector = numpy.array([(winx - viewport[0]) / viewport[2] * 2.0 - 1.0,
                                  (winy - viewport[1]) / viewport[3] * 2.0 - 1.0,
                                  winz * 2.0 - 1.0, 1]).reshape((1, 4))
            vector = (numpy.matrix(vector) * final_matrix).getA().flatten()
            ret = list(vector)[0:3] / vector[3]
            return ret
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"Błąd unproject: {e}")
        return numpy.array([0, 0, 0], numpy.float32)


def convert_3x3_matrix_to_4x4(matrix):
    """Konwertuje macierz 3x3 na 4x4"""
    try:
        return list(matrix.getA()[0]) + [0] + list(matrix.getA()[1]) + \
            [0] + list(matrix.getA()[2]) + [0, 0, 0, 0, 1]
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"Błąd konwersji macierzy: {e}")
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # Macierz jednostkowa


def load_gl_texture(filename):
    """Bezpieczne ładowanie tekstur OpenGL"""
    if DISABLE_OPENGL:
        if DEBUG_OPENGL:
            print("OpenGL wyłączony - pomijanie ładowania tekstury")
        return None
        
    # Nie sprawdzamy kontekstu - zakładamy że został już zweryfikowany
    try:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        img = wx.Bitmap(get_path_for_image(filename)).ConvertToImage()
        rgb_data = img.GetData()
        
        # Sprawdź czy obraz ma dane alfa
        if img.HasAlpha():
            alpha_data = img.GetAlpha()
            # Kombinuj RGB z alfa
            rgba_data = bytearray()
            for i in range(0, len(rgb_data), 3):
                rgba_data.extend(rgb_data[i:i+3])
                rgba_data.append(alpha_data[i//3])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.GetWidth(),
                         img.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(rgba_data))
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.GetWidth(),
                         img.GetHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_data)
        
        if DEBUG_OPENGL:
            print(f"✅ Tekstura załadowana: {filename}")
        return tex
        
    except Exception as e:
        if DEBUG_OPENGL:
            print(f"❌ Błąd ładowania tekstury {filename}: {e}")
        return None