"""
Calculo y almacenamiento de embeddings.

Un embedding por `product_code`, no por fila: el catalogo repite cada producto
una vez por variante de color (4959 filas dan 1242 productos en la muestra de
desarrollo), y las variantes comparten prod_name y detail_desc. Embeber por
fila seria calcular cuatro veces el mismo vector.
"""

import sqlite3
import struct

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from db import DIMENSIONES

# paraphrase-multilingual-MiniLM-L12-v2: 384 dimensiones y multilingue.
#
# Las dos condiciones son obligatorias aqui. Multilingue porque el catalogo
# esta en ingles ("Jersey top with narrow shoulder straps") y las consultas
# llegan en espanol ("algo abrigado para el invierno"): el modelo tiene que
# colocar las dos frases cerca en el mismo espacio vectorial. Y 384 dims
# porque es el tamano declarado en la tabla vec_productos.
#
# La alternativa obvia era all-MiniLM-L6-v2, mas rapido y mas citado, pero es
# solo ingles: con consultas en espanol devuelve ruido.
NOMBRE_MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# A nivel de modulo: cargar el modelo tarda unos segundos y ocupa ~450 MB.
# Hacerlo dentro de embeder_consulta() lo repetiria en cada peticion HTTP.
modelo = SentenceTransformer(NOMBRE_MODELO)

# Textos por lote en el encode. 64 va bien en CPU; subirlo mucho no acelera y
# dispara la memoria.
LOTE = 64


def texto_de(fila) -> str:
    """
    Texto que representa a un producto.

    Se concatenan los tres campos porque cada uno aporta algo distinto:
    prod_name es como lo llama H&M ("Mr Harrington w/hood"), a menudo un
    nombre propio inutil por si solo; product_type_name es la prenda en
    abstracto ("Hoodie"); detail_desc es lo unico que habla de tejido, corte
    y ocasion, que es por donde busca el usuario.
    """
    partes = [
        fila["prod_name"],
        fila["product_type_name"],
        fila["detail_desc"],
    ]
    return ". ".join(parte for parte in partes if parte)


def _a_bytes(vector) -> bytes:
    """Empaqueta un vector como los floats de 4 bytes que espera sqlite-vec."""
    return struct.pack(f"{DIMENSIONES}f", *vector)


def indexar(con: sqlite3.Connection, recalcular: bool = False) -> int:
    """
    Calcula y guarda un embedding por producto. Devuelve cuantos indexo.

    Por defecto solo procesa los productos que aun no tienen vector, para que
    ejecutarlo despues de ampliar el ingest no rehaga el trabajo ya hecho.
    """
    # min(article_id) para que el texto venga siempre de la misma variante
    # (misma regla de columnas sueltas que en buscador.py).
    consulta = """
        SELECT product_code,
               min(article_id),
               prod_name,
               product_type_name,
               detail_desc
        FROM articulos
        {donde}
        GROUP BY product_code
        ORDER BY product_code
    """
    donde = "" if recalcular else (
        "WHERE product_code NOT IN (SELECT product_code FROM vec_productos)"
    )
    filas = con.execute(consulta.format(donde=donde)).fetchall()

    if not filas:
        print("Nada que indexar: todos los productos tienen ya su embedding.")
        return 0

    print(f"Indexando {len(filas)} productos con {NOMBRE_MODELO}")

    for inicio in tqdm(range(0, len(filas), LOTE), desc="lotes", unit="lote"):
        lote = filas[inicio:inicio + LOTE]
        vectores = modelo.encode(
            [texto_de(fila) for fila in lote],
            batch_size=LOTE,
            show_progress_bar=False,
            # Normalizados: con vectores unitarios la distancia L2 de
            # sqlite-vec ordena igual que la similitud coseno, que es la
            # metrica con la que se entreno el modelo.
            normalize_embeddings=True,
        )
        con.executemany(
            "INSERT OR REPLACE INTO vec_productos(product_code, embedding) "
            "VALUES (?, ?)",
            [(fila["product_code"], _a_bytes(vector))
             for fila, vector in zip(lote, vectores)],
        )
        con.commit()

    return len(filas)


def embeder_consulta(texto: str) -> bytes:
    """Vectoriza la consulta del usuario, lista para pasarla a sqlite-vec."""
    vector = modelo.encode(texto, normalize_embeddings=True)
    return _a_bytes(vector)


if __name__ == "__main__":
    # Comprobacion a mano: python embeddings.py
    from db import conectar

    con = conectar()
    indexados = indexar(con)

    total = con.execute("SELECT count(*) FROM vec_productos").fetchone()[0]
    print(f"\nIndexados ahora: {indexados}. Total en vec_productos: {total}")
    con.close()
