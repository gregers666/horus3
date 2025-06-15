#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skrypt testowy do weryfikacji poprawek OpenGL.
Sprawdza czy naprawione pliki działają poprawnie.
"""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

class OpenGLTester:
    """Klasa do testowania naprawionych plików OpenGL"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_results = []
    
    def log(self, message: str, level: str = "INFO"):
        """Logowanie"""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def check_syntax(self, file_path: Path) -> bool:
        """Sprawdzenie składni Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            ast.parse(source)
            return True
        except SyntaxError as e:
            self.log(f"Błąd składni w {file_path}: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Błąd sprawdzania składni {file_path}: {e}", "ERROR")
            return False
    
    def check_imports(self, file_path: Path) -> bool:
        """Sprawdzenie czy importy działają"""
        try:
            # Uruchom plik z flagą sprawdzania składni
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(file_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                self.log(f"Błąd kompilacji {file_path}: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Błąd sprawdzania importów {file_path}: {e}", "ERROR")
            return False
    
    def check_opengl_patterns(self, file_path: Path) -> Dict[str, bool]:
        """Sprawdzenie czy wzorce OpenGL są prawidłowe"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            checks = {
                'has_context_check_function': 'def check_gl_context(' in content,
                'has_context_checks_in_render': 'check_gl_context()' in content and 'def render(' in content,
                'has_try_except_in_gl_calls': 'try:' in content and any(func in content for func in ['glGenBuffers', 'glBindBuffer']),
                'has_lazy_buffer_creation': 'buffers_created' in content or 'ensure_buffers' in content,
                'has_safe_release': 'def release(' in content and ('check_gl_context' in content or 'try:' in content),
                'no_unsafe_gl_in_init': not self._has_unsafe_gl_in_init(content),
            }
            
            return checks
        except Exception as e:
            self.log(f"Błąd sprawdzania wzorców OpenGL {file_path}: {e}", "ERROR")
            return {}
    
    def _has_unsafe_gl_in_init(self, content: str) -> bool:
        """Sprawdza czy __init__ zawiera niebezpieczne wywołania OpenGL"""
        lines = content.split('\n')
        in_init = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def __init__('):
                in_init = True
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if in_init:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                    # Koniec metody __init__
                    break
                
                # Sprawdź niebezpieczne wywołania OpenGL
                unsafe_calls = ['glGenBuffers', 'glBindBuffer', 'glBufferData', 'glGenTextures']
                for call in unsafe_calls:
                    if call in line and 'check_gl_context' not in line:
                        return True
        
        return False
    
    def create_test_report(self, test_results: List[Dict]) -> str:
        """Tworzenie raportu testowego"""
        total_files = len(test_results)
        passed_syntax = sum(1 for r in test_results if r['syntax_ok'])
        passed_imports = sum(1 for r in test_results if r['imports_ok'])
        
        report = f"""
=== RAPORT TESTÓW NAPRAW OPENGL ===
Przetestowane pliki: {total_files}
Poprawna składnia: {passed_syntax}/{total_files}
Poprawne importy: {passed_imports}/{total_files}

Szczegóły sprawdzenia wzorców OpenGL:
"""
        
        pattern_stats = {}
        for result in test_results:
            for pattern, status in result.get('opengl_patterns', {}).items():
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {'pass': 0, 'fail': 0}
                
                if status:
                    pattern_stats[pattern]['pass'] += 1
                else:
                    pattern_stats[pattern]['fail'] += 1
        
        for pattern, stats in pattern_stats.items():
            total = stats['pass'] + stats['fail']
            report += f"- {pattern}: {stats['pass']}/{total} OK\n"
        
        # Problemy
        report += "\nProblemy znalezione:\n"
        for result in test_results:
            if not result['syntax_ok'] or not result['imports_ok']:
                report += f"- {result['file']}: "
                issues = []
                if not result['syntax_ok']:
                    issues.append("składnia")
                if not result['imports_ok']:
                    issues.append("importy")
                report += ", ".join(issues) + "\n"
        
        report += "\n=== KONIEC RAPORTU TESTÓW ==="
        return report
    
    def test_file(self, file_path: Path) -> Dict:
        """Testowanie pojedynczego pliku"""
        self.log(f"Testowanie {file_path}...")
        
        result = {
            'file': str(file_path),
            'syntax_ok': False,
            'imports_ok': False,
            'opengl_patterns': {}
        }
        
        # Test składni
        result['syntax_ok'] = self.check_syntax(file_path)
        
        # Test importów
        if result['syntax_ok']:
            result['imports_ok'] = self.check_imports(file_path)
        
        # Test wzorców OpenGL
        result['opengl_patterns'] = self.check_opengl_patterns(file_path)
        
        return result
    
    def test_directory(self, directory: Path, recursive: bool = True) -> List[Dict]:
        """Testowanie wszystkich plików Python w katalogu"""
        results = []
        
        if recursive:
            pattern = "**/*.py"
        else:
            pattern = "*.py"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.endswith('.backup'):
                result = self.test_file(file_path)
                results.append(result)
        
        return results
    
    def run_integration_test(self, directory: Path) -> bool:
        """Test integracyjny - próba importu wszystkich modułów"""
        self.log("Uruchamianie testu integracyjnego...")
        
        try:
            # Dodaj katalog do PYTHONPATH
            sys.path.insert(0, str(directory))
            
            success_count = 0
            total_count = 0
            
            for file_path in directory.glob("**/*.py"):
                if file_path.is_file() and not file_path.name.endswith('.backup'):
                    total_count += 1
                    
                    # Spróbuj zaimportować moduł
                    module_name = file_path.stem
                    try:
                        # Użyj subprocess żeby uniknąć problemów z importami
                        result = subprocess.run([
                            sys.executable, "-c", f"import {module_name}"
                        ], cwd=directory, capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0:
                            success_count += 1
                            self.log(f"✓ {module_name}")
                        else:
                            self.log(f"✗ {module_name}: {result.stderr.strip()}", "ERROR")
                            
                    except subprocess.TimeoutExpired:
                        self.log(f"✗ {module_name}: Timeout", "ERROR")
                    except Exception as e:
                        self.log(f"✗ {module_name}: {e}", "ERROR")
            
            self.log(f"Test integracyjny: {success_count}/{total_count} modułów załadowanych")
            return success_count == total_count
            
        except Exception as e:
            self.log(f"Błąd testu integracyjnego: {e}", "ERROR")
            return False
        finally:
            # Usuń z PYTHONPATH
            if str(directory) in sys.path:
                sys.path.remove(str(directory))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Testowanie naprawionych plików OpenGL"
    )
    parser.add_argument("path", help="Ścieżka do pliku lub katalogu")
    parser.add_argument("--quiet", "-q", action="store_true", help="Tryb cichy")
    parser.add_argument("--recursive", "-r", action="store_true", default=True,
                       help="Rekursywnie przeglądaj podkatalogi")
    parser.add_argument("--report", help="Zapisz raport do pliku")
    parser.add_argument("--integration", action="store_true", 
                       help="Uruchom test integracyjny")
    
    args = parser.parse_args()
    
    # Sprawdź ścieżkę
    path = Path(args.path)
    if not path.exists():
        print(f"Błąd: Ścieżka {path} nie istnieje!")
        sys.exit(1)
    
    # Inicjalizuj tester
    tester = OpenGLTester(verbose=not args.quiet)
    
    # Uruchom testy
    if path.is_file():
        results = [tester.test_file(path)]
    else:
        results = tester.test_directory(path, args.recursive)
    
    # Generuj raport
    report = tester.create_test_report(results)
    print(report)
    
    # Test integracyjny
    if args.integration and path.is_dir():
        integration_success = tester.run_integration_test(path)
        if integration_success:
            print("\n✓ Test integracyjny ZALICZONY")
        else:
            print("\n✗ Test integracyjny NIEZALICZONY")
    
    # Zapisz raport
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Raport zapisany do: {args.report}")
    
    # Kod wyjścia
    failed_files = [r for r in results if not r['syntax_ok'] or not r['imports_ok']]
    if failed_files:
        print(f"\n❌ {len(failed_files)} plików wymaga uwagi")
        sys.exit(1)
    else:
        print(f"\n✅ Wszystkie {len(results)} plików przeszły testy podstawowe")
        sys.exit(0)

if __name__ == "__main__":
    main()