import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno antes de importar librerías pesadas
load_dotenv()

# Verificación simple para evitar errores silenciosos
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("Error: No se encontró HUGGINGFACEHUB_API_TOKEN en el archivo .env o variables de entorno.")
    print("Por favor, asegúrate de tener un archivo .env con tu token.")
    # sys.exit(1) # Descomentar para detener la ejecución si es crítico

# --- IMPORTACIONES DE LANGCHAIN ---
# Nota: LangChain ahora es modular. Si alguna falla, revisa requirements.txt

try:
    # Cadenas para MapReduce y combinación de documentos
    from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    
    # Prompts y Modelos
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    
    # Splitters (División de texto)
    from langchain_text_splitters import CharacterTextSplitter
    
    # Integración con Hugging Face (Paquete separado desde LangChain v0.2)
    from langchain_huggingface import HuggingFaceEndpoint

except ImportError as e:
    print(f"\n❌ Error de Importación: {e}")
    print("💡 Solución: Ejecuta 'pip install -r requirements.txt'")
    sys.exit(1)

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DEL LLM (Hugging Face)
# ---------------------------------------------------------

# Usaremos Mistral-7B.
# Nota: Si el modelo da timeout, intenta con un modelo más ligero o aumenta el timeout.
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"🔄 Conectando con Hugging Face Hub ({repo_id})...")

try:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.1,
        max_new_tokens=512,
        timeout=300 # Aumentamos el timeout para evitar errores de red
    )
except Exception as e:
    print(f"Error al conectar con el modelo: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. GENERACIÓN DE DATOS DE PRUEBA
# ---------------------------------------------------------
print("📄 Generando documento de prueba...")
texto_largo = """
La inteligencia artificial (IA) ha transformado múltiples industrias en la última década. 
En el sector salud, se utiliza para diagnósticos tempranos mediante análisis de imágenes.
La telemedicina ha avanzado gracias a la IA, permitiendo triajes automáticos.
Por otro lado, en el sector financiero, la IA detecta fraudes en tiempo real analizando patrones de transacciones.
Los algoritmos de Machine Learning permiten a los bancos predecir riesgos crediticios con mayor precisión.
Finalmente, en la educación, la IA personalizada adapta el contenido al ritmo de aprendizaje de cada estudiante.
Las plataformas educativas utilizan procesamiento de lenguaje natural para evaluar ensayos automáticamente.
""" * 40 # Aumentamos el tamaño para justificar el MapReduce

docs = [Document(page_content=texto_largo)]

# ---------------------------------------------------------
# 3. CHUNKING (DIVISIÓN DE DOCUMENTOS)
# ---------------------------------------------------------
# Usamos CharacterTextSplitter. Nota: tiktoken debe estar instalado.
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, 
    chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

print(f"✂️  Documento original dividido en {len(split_docs)} trozos.")

# ---------------------------------------------------------
# 4. DEFINICIÓN DE CADENAS (MAP Y REDUCE)
# ---------------------------------------------------------

# Paso MAP: Resume cada trozo individual
map_template = """The following is a set of documents:
{docs}
Based on this list of docs, please identify the main themes concisely.
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Paso REDUCE: Combina los resúmenes
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes in Spanish.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Cadena para combinar documentos (Stuff)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, 
    document_variable_name="docs"
)

# Cadena de reducción inteligente (Iterativa)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=3000, 
)

# Cadena FINAL MapReduce
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

# ---------------------------------------------------------
# 5. EJECUCIÓN
# ---------------------------------------------------------
print("🚀 Ejecutando MapReduce (esto puede tardar un poco)...")

try:
    # Invocación correcta para LangChain moderno
    resultado = map_reduce_chain.invoke({"docs": split_docs})
    
    print("\n✅ Resultados Finales:")
    print("-" * 20)
    print(resultado['output_text'])
    
except Exception as e:
    print(f"\n❌ Error durante la ejecución: {e}")