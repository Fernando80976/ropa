"""
Descarga el modelo de embeddings a `modelo_onnx/`.

    python preparar_modelo.py

Lo usan tanto el equipo de desarrollo como el Dockerfile, para que los dos
partan exactamente del mismo fichero.

Por que no esta el modelo en el repositorio: Hugging Face ya lo publica, y
duplicarlo en git no aporta nada.

Por que no se descarga en el arranque de la app: un servidor que baja 113 MB
cada vez que se reinicia tarda demasiado en responder la primera consulta.
Aqui se baja una vez, durante el build de la imagen, y queda dentro.
"""

import urllib.request
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"

# Revision fijada, no "main": si el repo de origen publicase otro fichero con
# el mismo nombre, los vectores dejarian de corresponderse con los que ya
# estan indexados en la BD y el ranking se volveria ruido, sin ningun error.
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

DESTINO = Path(__file__).parent / "modelo_onnx"

# model_quint8_avx2: la variante cuantizada a int8 que publica el propio
# repositorio. avx2 y no avx512_vnni porque avx2 lo soporta practicamente
# cualquier x86 desde 2013, y no se sabe que CPU tocara en el servidor.
FICHEROS = {
    "modelo.onnx": "onnx/model_quint8_avx2.onnx",
    "tokenizer.json": "tokenizer.json",
}


def url_de(ruta: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{ruta}"


def main() -> None:
    DESTINO.mkdir(exist_ok=True)

    for nombre_local, ruta_remota in FICHEROS.items():
        destino = DESTINO / nombre_local

        if destino.exists():
            print(f"  {nombre_local}: ya esta ({destino.stat().st_size / 1e6:.0f} MB)")
            continue

        print(f"  {nombre_local}: descargando...")
        urllib.request.urlretrieve(url_de(ruta_remota), destino)
        print(f"  {nombre_local}: listo ({destino.stat().st_size / 1e6:.0f} MB)")

    print(f"\nModelo en {DESTINO}")


if __name__ == "__main__":
    main()
