# Imagen del buscador.
#
# NOTA sobre la regla de CLAUDE.md ("Docker no esta instalado y no se va a
# instalar"): este fichero lo construye la plataforma de hosting en sus
# servidores, no este equipo. Aqui no hace falta Docker ni para desarrollar
# ni para desplegar; el flujo local sigue siendo .venv + uvicorn.
#
# La imagen sale de unos 350 MB porque no lleva torch: el modelo va en su
# version ONNX cuantizada, dentro de modelo_onnx/. Ver embeddings.py.

FROM python:3.13-slim

# Muchas plataformas ejecutan el contenedor con un UID sin privilegios. Si los
# ficheros son de root, la app no puede leer ni su propio modelo.
RUN useradd -m -u 1000 usuario

ENV PYTHONUNBUFFERED=1 \
    DB_PATH=/home/usuario/app/ropa.db

USER usuario
WORKDIR /home/usuario/app

# Las dependencias van antes que el codigo para que un cambio en un .py no
# invalide la capa de instalacion, que es la que tarda.
COPY --chown=usuario requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

ENV PATH=/home/usuario/.local/bin:$PATH

# El modelo se baja durante el build y queda dentro de la imagen. No esta en
# el repositorio (son 113 MB y Hugging Face ya lo publica) ni se descarga en
# el arranque, que haria lenta la primera consulta tras cada reinicio.
COPY --chown=usuario preparar_modelo.py .
RUN python preparar_modelo.py

# El codigo y la BD ya indexada (ropa.db).
COPY --chown=usuario . .

# PORT lo inyectan Render, Koyeb y Cloud Run; 7860 es el que espera Hugging
# Face Spaces. El valor por defecto cubre el caso de ejecutarlo a mano.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
