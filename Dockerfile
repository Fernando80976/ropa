# Imagen para Hugging Face Spaces.
#
# NOTA sobre la regla de CLAUDE.md ("Docker no esta instalado y no se va a
# instalar"): ese fichero se construye en los servidores de Hugging Face, no
# aqui. En este equipo no hace falta Docker ni para desplegar ni para
# desarrollar; el flujo local sigue siendo .venv + uvicorn. Spaces solo ofrece
# SDK propio para Gradio y Streamlit, asi que una app FastAPI con su front
# propio solo se puede servir por esta via.

FROM python:3.13-slim

# Spaces ejecuta el contenedor con el UID 1000. Si los ficheros son de root,
# la app no puede escribir ni leer su propia cache.
RUN useradd -m -u 1000 usuario

# HF_HOME dentro del home del usuario: si apunta a un sitio sin permisos,
# sentence-transformers falla al arrancar intentando escribir la cache.
ENV HF_HOME=/home/usuario/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/home/usuario/app/ropa.db

USER usuario
WORKDIR /home/usuario/app

# Las dependencias van antes que el codigo para que un cambio en un .py no
# invalide la capa de instalacion, que es la que tarda.
COPY --chown=usuario requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

ENV PATH=/home/usuario/.local/bin:$PATH

# El modelo de embeddings (~450 MB) se descarga durante el build y queda
# dentro de la imagen. Si se dejara para el arranque, cada reinicio del Space
# volveria a bajarlo y la primera busqueda tardaria minutos.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=usuario . .

# 7860 es el puerto que Spaces espera por defecto.
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
