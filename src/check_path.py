import sys
import os

print(f"Ejecutable de Python: {sys.executable}")
print("Rutas de búsqueda (sys.path):")
for p in sys.path:
    print(p)

try:
    import langchain
    print(f"\nSUCCESS: Langchain encontrado en: {langchain.__file__}")
except ImportError as e:
    print(f"\nERROR: {e}")