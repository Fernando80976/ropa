"""
Descarga la base de datos ya indexada.

    python preparar_datos.py

Hermano de preparar_modelo.py: los dos bajan un artefacto grande que no vive
en el repositorio. Lo usan tanto el equipo de desarrollo como el Dockerfile.

Por que la BD no esta en el repositorio
---------------------------------------
Son 106 MB, por encima del limite de 100 MB por fichero de GitHub, asi que
solo cabria via Git LFS. Se descarto por dos motivos:

1. CLAUDE.md excluye `*.db` del control de versiones. La BD es un artefacto
   reconstruible (ingest.py + embeddings.py), no codigo fuente.
2. Git LFS y los despliegues se llevan mal. Muchas plataformas clonan en
   shallow o descargan un tarball del repositorio, y en los dos casos lo que
   llega es el fichero puntero de texto en vez del binario. El fallo aparece
   en tiempo de ejecucion y con un mensaje que no apunta a la causa.

Reconstruirla en el build tampoco sirve: reindexar 46.922 productos son unos
20 minutos en cada despliegue.
"""

import urllib.request
from pathlib import Path

# Dataset publico con el catalogo ya ingestado e indexado. Se genera con:
#     python ingest.py --limite 0 && python embeddings.py
REPO = "Kirito741/buscador-ropa-datos"

DESTINO = Path(__file__).parent / "ropa.db"

URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/ropa.db"

# Si el fichero pesa menos que esto, no es la BD: es una pagina de error o una
# descarga cortada. Mas vale enterarse aqui que al ejecutar el primer SELECT.
TAMANO_MINIMO = 50 * 1024 * 1024


def main() -> None:
    if DESTINO.exists():
        print(f"ropa.db ya esta ({DESTINO.stat().st_size / 1e6:.0f} MB)")
        return

    print(f"Descargando ropa.db de {REPO}...")
    urllib.request.urlretrieve(URL, DESTINO)

    tamano = DESTINO.stat().st_size
    if tamano < TAMANO_MINIMO:
        DESTINO.unlink()
        raise SystemExit(
            f"La descarga son solo {tamano / 1e6:.1f} MB, no puede ser la BD. "
            f"Comprueba {URL}"
        )

    print(f"Listo: {tamano / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
