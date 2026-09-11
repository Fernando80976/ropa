"""
Despliegue a Hugging Face Spaces.

    python desplegar.py              # crea el Space si no existe y sube todo
    python desplegar.py --solo-codigo  # sube el codigo, no la BD (~120 MB)

Lee HF_TOKEN y GROQ_API_KEY de .env. Ninguna de las dos llega al repo: el
token solo se usa para autenticar, y la de Groq se sube al gestor de secretos
del Space, que es una via aparte del contenido del repositorio.

Por que Spaces y no Render: el modelo de embeddings ocupa unos 450 MB en RAM
y el plan gratuito de Render son 512 MB en total, donde no cabe con torch.
Spaces da 16 GB en CPU gratis.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

NOMBRE_SPACE = os.getenv("HF_SPACE", "buscador-ropa")

# Lo que NO se sube. El codigo y los tests si: en un proyecto de portfolio los
# tests son parte de lo que se quiere ensenar.
NO_SUBIR = [
    ".env",                  # la API key va como secreto, nunca como fichero
    ".venv/**",
    ".git/**",
    "articles.csv",          # 36 MB y ya esta volcado en la BD
    "**/__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    # Documentos de planificacion interna: no son parte del proyecto.
    "CLAUDE.md",
    "prompts.md",
    "columnas.txt",
    "desplegar.py",
]


def exigir(nombre: str) -> str:
    """Lee una variable de .env o corta con un mensaje util."""
    valor = os.getenv(nombre)
    if not valor:
        sys.exit(
            f"Falta {nombre} en .env\n"
            f"  HF_TOKEN: crealo en https://huggingface.co/settings/tokens "
            f"con permiso Write.\n"
            f"  GROQ_API_KEY: la que ya usa el extractor."
        )
    return valor


def main() -> None:
    parser = argparse.ArgumentParser(description="Despliega el Space")
    parser.add_argument(
        "--solo-codigo",
        action="store_true",
        help="No sube ropa.db. Util para iterar sobre el codigo sin resubir 120 MB.",
    )
    argumentos = parser.parse_args()

    token = exigir("HF_TOKEN")
    clave_groq = exigir("GROQ_API_KEY")

    api = HfApi(token=token)
    usuario = api.whoami()["name"]
    repo_id = f"{usuario}/{NOMBRE_SPACE}"

    print(f"Usuario: {usuario}")
    print(f"Space:   {repo_id}")

    # exist_ok para que el script sirva igual para crear y para actualizar.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("Space listo (creado o ya existente)")

    # El secreto se manda por la API de secretos, no en el repo. Spaces lo
    # inyecta como variable de entorno, que es justo de donde lo lee
    # extractor.py con os.getenv.
    api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=clave_groq)
    print("Secreto GROQ_API_KEY subido")

    ignorar = list(NO_SUBIR)
    if argumentos.solo_codigo:
        ignorar.append("*.db")
        print("Sin la BD: el Space usara la que ya tenga subida")
    else:
        tamano = os.path.getsize("ropa.db") / 1024 / 1024
        print(f"Subiendo tambien ropa.db ({tamano:.0f} MB, por LFS). Esto tarda.")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=".",
        ignore_patterns=ignorar,
        commit_message="Despliegue del buscador de ropa",
    )

    print(f"\nListo: https://huggingface.co/spaces/{repo_id}")
    print("El primer build tarda unos minutos: instala torch y descarga el modelo.")


if __name__ == "__main__":
    main()
