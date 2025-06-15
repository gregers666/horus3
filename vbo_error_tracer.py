#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracer błędów VBO - znajduje dokładnie gdzie powstaje błąd kontekstu
"""

import os
import re
import ast
import sys
from pathlib import Path

class VBOErrorTracer:
    def __init__(self):
        self.critical_functions = [
            'glGenBuffers', 'glBindBuffer', 'glBufferData', 'glDeleteBuffers',
            'glDrawArrays', 'glDrawElements', 'glVertexPointer', 'glColorPointer'
        ]
        
        self.call_stack = []
        self.problematic_files = []

    def trace_function_calls(self, filepath):
        """Śledzi wywołania funkcji w pliku"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []
        
        # Parse AST dla dokładnej analizy
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self.trace_regex_fallback(filepath, content)
        
        calls = []
        
        class CallVisitor(ast.NodeVisitor):
            def __init__(self, tracer):
                self.tracer = tracer
                self.current_class = None
                self.current_function = None
            
            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                
                # Sprawdź czy funkcja używa OpenGL
                func_source = ast.get_source_segment(content, node) or ""
                gl_calls = [func for func in self.tracer.critical_functions if func in func_source]
                
                if gl_calls:
                    location = f"{self.current_class}.{node.name}" if self.current_class else node.name
                    calls.append({
                        'location': location,
                        'line': node.lineno,
                        'gl_functions': gl_calls,
                        'has_context_check': self._has_context_check(func_source),
                        'function_name': node.name,
                        'class_name': self.current_class
                    })
                
                self.generic_visit(node)
                self.current_function = old_function
            
            def _has_context_check(self, source):
                checks = [
                    'DISABLE_OPENGL', 'activate_context', 'SetCurrent', 
                    'check_opengl_context', 'GLContext', '_context'
                ]
                return any(check in source for check in checks)
        
        visitor = CallVisitor(self)
        visitor.visit(tree)
        
        return calls

    def trace_regex_fallback(self, filepath, content):
        """Fallback regex parsing gdy AST nie działa"""
        calls = []
        lines = content.split('\n')
        
        current_class = None
        current_function = None
        
        for i, line in enumerate(lines):
            # Wykryj klasę
            class_match = re.match(r'^\s*class\s+(\w+)', line)
            if class_match:
                current_class = class_match.group(1)
                continue
            
            # Wykryj funkcję
            func_match = re.match(r'^\s*def\s+(\w+)', line)
            if func_match:
                current_function = func_match.group(1)
                
                # Sprawdź następne 50 linii tej funkcji
                func_end = i + 50
                func_lines = lines[i:min(func_end, len(lines))]
                func_content = '\n'.join(func_lines)
                
                # Znajdź wywołania OpenGL
                gl_calls = [func for func in self.critical_functions if func in func_content]
                
                if gl_calls:
                    has_context = any(check in func_content for check in [
                        'DISABLE_OPENGL', 'activate_context', 'SetCurrent', 
                        'check_opengl_context', 'GLContext', '_context'
                    ])
                    
                    location = f"{current_class}.{current_function}" if current_class else current_function
                    calls.append({
                        'location': location,
                        'line': i + 1,
                        'gl_functions': gl_calls,
                        'has_context_check': has_context,
                        'function_name': current_function,
                        'class_name': current_class
                    })
        
        return calls

    def analyze_vbo_lifecycle(self, filepath):
        """Analizuje cykl życia VBO - gdzie są tworzone i używane"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return {}
        
        analysis = {
            'creation_points': [],
            'usage_points': [],
            'destruction_points': [],
            'problematic_patterns': []
        }
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # VBO creation
            if 'glGenBuffers' in line:
                context_check = self._check_context_in_vicinity(lines, i)
                analysis['creation_points'].append({
                    'line': line_num,
                    'code': stripped,
                    'has_context_check': context_check,
                    'risk_level': 'HIGH' if not context_check else 'LOW'
                })
            
            # VBO usage
            if any(func in line for func in ['glBindBuffer', 'glBufferData', 'glDrawArrays', 'glDrawElements']):
                context_check = self._check_context_in_vicinity(lines, i)
                analysis['usage_points'].append({
                    'line': line_num,
                    'code': stripped,
                    'has_context_check': context_check,
                    'risk_level': 'HIGH' if not context_check else 'LOW'
                })
            
            # VBO destruction
            if 'glDeleteBuffers' in line:
                context_check = self._check_context_in_vicinity(lines, i)
                analysis['destruction_points'].append({
                    'line': line_num,
                    'code': stripped,
                    'has_context_check': context_check,
                    'risk_level': 'MEDIUM' if not context_check else 'LOW'
                })
            
            # Problematic patterns
            if '__init__' in line and any(func in content[content.find(line):content.find(line) + 500] for func in self.critical_functions):
                analysis['problematic_patterns'].append({
                    'type': 'OPENGL_IN_INIT',
                    'line': line_num,
                    'description': 'OpenGL calls in __init__ - możliwe tworzenie VBO za wcześnie'
                })
        
        return analysis

    def _check_context_in_vicinity(self, lines, center_line, window=15):
        """Sprawdza czy w pobliżu są sprawdzenia kontekstu"""
        start = max(0, center_line - window)
        end = min(len(lines), center_line + window)
        
        vicinity_content = '\n'.join(lines[start:end])
        
        context_patterns = [
            r'DISABLE_OPENGL',
            r'activate_context\(\)',
            r'SetCurrent\(',
            r'check.*context',
            r'GLContext\(',
            r'if.*not.*context',
            r'try:.*except.*OpenGL',
            r'glGetError\(\)'
        ]
        
        return any(re.search(pattern, vicinity_content, re.IGNORECASE) for pattern in context_patterns)

    def find_import_order_issues(self, filepath):
        """Znajduje problemy z kolejnością importów"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return []
        
        issues = []
        opengl_imported = False
        wx_imported = False
        first_gl_call = None
        
        for i, line in enumerate(lines):
            if 'from OpenGL' in line or 'import OpenGL' in line:
                opengl_imported = i
            
            if 'import wx' in line or 'from wx' in line:
                wx_imported = i
            
            if not first_gl_call and any(func in line for func in self.critical_functions):
                first_gl_call = i
        
        if opengl_imported and first_gl_call:
            if first_gl_call - opengl_imported < 10:
                issues.append({
                    'type': 'QUICK_GL_USAGE',
                    'description': f'OpenGL używany {first_gl_call - opengl_imported} linii po imporcie',
                    'risk': 'HIGH'
                })
        
        return issues

    def create_call_graph(self, calls_data):
        """Tworzy graf wywołań do analizy zależności"""
        graph = {}
        
        for filepath, calls in calls_data.items():
            for call in calls:
                if not call['has_context_check']:
                    key = f"{os.path.basename(filepath)}:{call['location']}"
                    graph[key] = {
                        'file': filepath,
                        'line': call['line'],
                        'gl_functions': call['gl_functions'],
                        'risk_score': len(call['gl_functions']) * (2 if 'render' in call['function_name'].lower() else 1)
                    }
        
        return graph

    def scan_project(self, root_path):
        """Skanuje cały projekt"""
        print(f"🔍 Skanowanie projektu: {root_path}")
        
        python_files = []
        for root, dirs, files in os.walk(root_path):
            # Pomiń __pycache__ i .git
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        print(f"📄 Znaleziono {len(python_files)} plików Python")
        
        all_calls = {}
        all_vbo_analysis = {}
        
        # Priorytetyzuj pliki związane z OpenGL
        priority_files = [f for f in python_files if any(keyword in f.lower() for keyword in 
                         ['opengl', 'scene', 'render', 'vbo', 'gl', '3d'])]
        
        other_files = [f for f in python_files if f not in priority_files]
        
        print(f"🎯 Pliki priorytetowe: {len(priority_files)}")
        print(f"📁 Inne pliki: {len(other_files)}")
        
        # Skanuj pliki priorytetowe najpierw
        for filepath in priority_files + other_files:
            rel_path = os.path.relpath(filepath, root_path)
            print(f"  🔍 {rel_path}")
            
            calls = self.trace_function_calls(filepath)
            if calls:
                all_calls[filepath] = calls
            
            vbo_analysis = self.analyze_vbo_lifecycle(filepath)
            if any(vbo_analysis.values()):
                all_vbo_analysis[filepath] = vbo_analysis
        
        return all_calls, all_vbo_analysis

    def generate_priority_report(self, calls_data, vbo_analysis):
        """Generuje raport z priorytetami napraw"""
        print("\n" + "=" * 80)
        print("🎯 RAPORT PRIORYTETOWY - PRAWDOPODOBNE PRZYCZYNY BŁĘDU VBO")
        print("=" * 80)
        
        # Znajdź najgroźniejsze miejsca
        critical_issues = []
        high_risk_issues = []
        medium_risk_issues = []
        
        # Analizuj VBO lifecycle
        for filepath, analysis in vbo_analysis.items():
            filename = os.path.basename(filepath)
            
            # Sprawdź creation points
            for point in analysis['creation_points']:
                if point['risk_level'] == 'HIGH':
                    critical_issues.append({
                        'file': filename,
                        'line': point['line'],
                        'type': 'VBO_CREATION_NO_CONTEXT',
                        'code': point['code'],
                        'description': 'glGenBuffers wywoływany bez sprawdzenia kontekstu'
                    })
            
            # Sprawdź usage points
            for point in analysis['usage_points']:
                if point['risk_level'] == 'HIGH':
                    if 'render' in point['code'].lower() or 'draw' in point['code'].lower():
                        critical_issues.append({
                            'file': filename,
                            'line': point['line'],
                            'type': 'VBO_RENDER_NO_CONTEXT',
                            'code': point['code'],
                            'description': 'Renderowanie VBO bez sprawdzenia kontekstu'
                        })
                    else:
                        high_risk_issues.append({
                            'file': filename,
                            'line': point['line'],
                            'type': 'VBO_USAGE_NO_CONTEXT', 
                            'code': point['code'],
                            'description': 'Używanie VBO bez sprawdzenia kontekstu'
                        })
        
        # Analizuj function calls
        for filepath, calls in calls_data.items():
            filename = os.path.basename(filepath)
            
            for call in calls:
                if not call['has_context_check']:
                    if call['function_name'] in ['__init__', 'render', 'draw', 'paint']:
                        critical_issues.append({
                            'file': filename,
                            'line': call['line'],
                            'type': 'CRITICAL_FUNCTION_NO_CONTEXT',
                            'code': f"{call['location']}() używa {', '.join(call['gl_functions'])}",
                            'description': f'Krytyczna funkcja {call["function_name"]} bez sprawdzenia kontekstu'
                        })
                    else:
                        medium_risk_issues.append({
                            'file': filename,
                            'line': call['line'],
                            'type': 'FUNCTION_NO_CONTEXT',
                            'code': f"{call['location']}() używa {', '.join(call['gl_functions'])}",
                            'description': 'Funkcja używa OpenGL bez sprawdzenia kontekstu'
                        })
        
        # Sortuj według ważności
        critical_issues.sort(key=lambda x: (x['file'], x['line']))
        high_risk_issues.sort(key=lambda x: (x['file'], x['line']))
        
        print(f"\n🔴 KRYTYCZNE ({len(critical_issues)} problemów) - NAJPRAWDOPODOBNIEJ TU JEST BŁĄD:")
        print("-" * 80)
        
        if not critical_issues:
            print("✅ Nie znaleziono krytycznych problemów!")
        else:
            for issue in critical_issues[:10]:  # Pokaż maksymalnie 10
                print(f"\n📁 {issue['file']} : linia {issue['line']}")
                print(f"   ❌ {issue['description']}")
                print(f"   📝 {issue['code'][:100]}{'...' if len(issue['code']) > 100 else ''}")
                
                # Sugeruj konkretną naprawę
                if issue['type'] == 'VBO_CREATION_NO_CONTEXT':
                    print(f"   🔧 NAPRAWA: Dodaj na początku funkcji:")
                    print(f"      if DISABLE_OPENGL or not check_opengl_context(): return None")
                elif issue['type'] == 'VBO_RENDER_NO_CONTEXT':
                    print(f"   🔧 NAPRAWA: Dodaj na początku metody render:")
                    print(f"      if DISABLE_OPENGL or not check_opengl_context(): return")
        
        if high_risk_issues:
            print(f"\n🟡 WYSOKIE RYZYKO ({len(high_risk_issues)} problemów):")
            print("-" * 80)
            for issue in high_risk_issues[:5]:
                print(f"📁 {issue['file']}:{issue['line']} - {issue['description']}")
        
        # Sprawdź specyficzne wzorce błędów
        self.check_specific_patterns(vbo_analysis)
        
        return critical_issues

    def check_specific_patterns(self, vbo_analysis):
        """Sprawdza specyficzne wzorce powodujące błąd kontekstu"""
        print(f"\n🔍 ANALIZA SPECYFICZNYCH WZORCÓW:")
        print("-" * 80)
        
        patterns_found = []
        
        for filepath, analysis in vbo_analysis.items():
            filename = os.path.basename(filepath)
            
            # Wzorzec 1: VBO tworzony w __init__
            if any('__init__' in pattern.get('description', '') for pattern in analysis.get('problematic_patterns', [])):
                patterns_found.append(f"❌ {filename}: VBO prawdopodobnie tworzony w __init__ zanim kontekst jest gotowy")
            
            # Wzorzec 2: Wiele wywołań bez sprawdzenia
            high_risk_count = len([p for p in analysis.get('creation_points', []) + analysis.get('usage_points', []) 
                                 if p.get('risk_level') == 'HIGH'])
            if high_risk_count > 3:
                patterns_found.append(f"⚠️ {filename}: {high_risk_count} wywołań OpenGL bez sprawdzenia kontekstu")
            
            # Wzorzec 3: Brak sprawdzenia w destrukcji
            destruction_issues = [p for p in analysis.get('destruction_points', []) if p.get('risk_level') != 'LOW']
            if destruction_issues:
                patterns_found.append(f"🗑️ {filename}: Problemy ze zwalnianiem VBO - może powodować wycieki")
        
        if patterns_found:
            for pattern in patterns_found:
                print(f"  {pattern}")
        else:
            print("  ✅ Nie znaleziono typowych wzorców błędów")

    def create_fix_script(self, critical_issues):
        """Tworzy skrypt z konkretnymi naprawami"""
        fix_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatyczne naprawy dla błędów kontekstu OpenGL
Wygenerowane przez VBOErrorTracer
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    if os.path.exists(filepath):
        backup = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup)
        print(f"📁 Kopia zapasowa: {backup}")

def fix_vbo_context_issues():
    """Naprawia najczęstsze problemy z kontekstem VBO"""
    
    fixes_applied = []
'''
        
        # Grupuj naprawy według plików
        files_to_fix = {}
        for issue in critical_issues:
            if issue['file'] not in files_to_fix:
                files_to_fix[issue['file']] = []
            files_to_fix[issue['file']].append(issue)
        
        for filename, issues in files_to_fix.items():
            fix_script += f'''
    # Napraw {filename}
    print(f"🔧 Naprawianie: {filename}")
    backup_file("{filename}")
    
    try:
        with open("{filename}", 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
'''
            
            for issue in issues:
                if issue['type'] == 'VBO_CREATION_NO_CONTEXT':
                    fix_script += f'''
        # Naprawa VBO creation na linii {issue['line']}
        if "glGenBuffers" in content and "check_opengl_context" not in content:
            # Dodaj import sprawdzania kontekstu
            if "from .opengl_config import" in content:
                content = content.replace(
                    "from .opengl_config import", 
                    "from .opengl_config import DISABLE_OPENGL, DEBUG_OPENGL, check_opengl_context")
            else:
                content = "from .opengl_config import DISABLE_OPENGL, DEBUG_OPENGL\\n" + content
            
            # Dodaj sprawdzenie przed glGenBuffers
            content = re.sub(
                r'(\\s+)(glGenBuffers)',
                r'\\1if DISABLE_OPENGL or not check_opengl_context(): return None\\n\\1\\2',
                content
            )
            modified = True
'''
                
                elif issue['type'] == 'VBO_RENDER_NO_CONTEXT':
                    fix_script += f'''
        # Naprawa VBO render na linii {issue['line']}
        render_pattern = r'(def render\\(.*?\\):[^\\n]*\\n)(\\s+)'
        if re.search(render_pattern, content):
            content = re.sub(
                render_pattern,
                r'\\1\\2if DISABLE_OPENGL or not check_opengl_context(): return\\n\\2',
                content
            )
            modified = True
'''
            
            fix_script += f'''
        if modified:
            with open("{filename}", 'w', encoding='utf-8') as f:
                f.write(content)
            fixes_applied.append("{filename}")
            print(f"✅ Naprawiono: {filename}")
        else:
            print(f"ℹ️ Brak zmian w: {filename}")
            
    except Exception as e:
        print(f"❌ Błąd naprawiania {filename}: {{e}}")
'''
        
        fix_script += '''
    print(f"\\n🎉 Naprawiono {len(fixes_applied)} plików:")
    for file in fixes_applied:
        print(f"  ✅ {file}")

if __name__ == "__main__":
    fix_vbo_context_issues()
'''
        
        with open("auto_fix_vbo_context.py", "w", encoding="utf-8") as f:
            f.write(fix_script)
        
        print(f"\n💾 Utworzono skrypt napraw: auto_fix_vbo_context.py")
        print("   Uruchom: python auto_fix_vbo_context.py")


def main():
    tracer = VBOErrorTracer()
    
    print("🔍 VBO ERROR TRACER - PRECYZYJNE WYKRYWANIE BŁĘDÓW KONTEKSTU")
    print("=" * 70)
    
    # Sprawdź argumenty
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "horus"
    
    if not os.path.exists(target):
        print(f"❌ Nie znaleziono: {target}")
        print("Użycie: python vbo_error_tracer.py [ścieżka]")
        return
    
    # Skanuj projekt
    calls_data, vbo_analysis = tracer.scan_project(target)
    
    # Generuj raport priorytetowy
    critical_issues = tracer.generate_priority_report(calls_data, vbo_analysis)
    
    # Utwórz call graph
    call_graph = tracer.create_call_graph(calls_data)
    
    print(f"\n📊 STATYSTYKI:")
    print(f"   📄 Plików z wywołaniami OpenGL: {len(calls_data)}")
    print(f"   🔍 Plików z analizą VBO: {len(vbo_analysis)}")
    print(f"   ❌ Krytycznych problemów: {len(critical_issues)}")
    print(f"   📈 Węzłów w grafie wywołań: {len(call_graph)}")
    
    # Zapisz szczegółowy raport
    with open("vbo_error_detailed_report.txt", "w", encoding="utf-8") as f:
        f.write("VBO ERROR TRACER - SZCZEGÓŁOWY RAPORT\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write("KRYTYCZNE PROBLEMY:\\n")
        f.write("-" * 30 + "\\n")
        for issue in critical_issues:
            f.write(f"{issue['file']}:{issue['line']} - {issue['description']}\\n")
            f.write(f"  Kod: {issue['code']}\\n\\n")
        
        f.write("\\nANALIZA VBO PER PLIK:\\n")
        f.write("-" * 30 + "\\n")
        for filepath, analysis in vbo_analysis.items():
            f.write(f"\\nPLIK: {filepath}\\n")
            f.write(f"  Creation points: {len(analysis['creation_points'])}\\n")
            f.write(f"  Usage points: {len(analysis['usage_points'])}\\n")
            f.write(f"  Destruction points: {len(analysis['destruction_points'])}\\n")
            
            high_risk = [p for p in analysis['creation_points'] + analysis['usage_points'] 
                        if p.get('risk_level') == 'HIGH']
            if high_risk:
                f.write(f"  WYSOKIE RYZYKO ({len(high_risk)} miejsc):\\n")
                for point in high_risk:
                    f.write(f"    Linia {point['line']}: {point['code'][:80]}\\n")
    
    print(f"\\n💾 Szczegółowy raport: vbo_error_detailed_report.txt")
    
    # Utwórz skrypt napraw jeśli są krytyczne problemy
    if critical_issues:
        tracer.create_fix_script(critical_issues)
        
        print(f"\\n🎯 REKOMENDACJA:")
        print("1. ❗ Sprawdź krytyczne problemy powyżej - to prawdopodobnie przyczyna błędu")
        print("2. 🔧 Uruchom: python auto_fix_vbo_context.py (automatyczne naprawy)")
        print("3. 🧪 Przetestuj aplikację")
        print("4. 📋 Jeśli to nie pomoże, ustaw DISABLE_OPENGL = True")
    else:
        print(f"\\n✅ Nie znaleziono oczywistych przyczyn błędu kontekstu VBO")
        print("Błąd może wynikać z:")
        print("- Problemów ze sterownikami graficznymi")
        print("- Niewłaściwej wersji PyOpenGL")
        print("- Problemów z wxPython")


if __name__ == "__main__":
    main()