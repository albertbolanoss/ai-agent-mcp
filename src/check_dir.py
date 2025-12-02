import os
import langchain

# Obtenemos la ruta donde está instalado langchain
ruta_langchain = os.path.dirname(langchain.__file__)
ruta_chains = os.path.join(ruta_langchain, "chains")

print(f"Buscando en: {ruta_chains}")

if os.path.exists(ruta_chains):
    print("✅ La carpeta 'chains' EXISTE.")
    print("Contenido:", os.listdir(ruta_chains)[:5]) # Muestra los primeros 5 archivos
else:
    print("❌ La carpeta 'chains' NO EXISTE. Tu instalación está corrupta.")