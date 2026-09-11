"""
Calculo y almacenamiento de embeddings.

Un embedding por `product_code`, no por fila: el catalogo repite cada producto
una vez por variante de color (104.828 filas dan 46.922 productos), y las
variantes comparten prod_name y detail_desc. Embeber por fila seria calcular
varias veces el mismo vector.

Por que onnxruntime y no sentence-transformers
----------------------------------------------
El modelo es el mismo (paraphrase-multilingual-MiniLM-L12-v2), pero aqui se
ejecuta su version ONNX cuantizada a int8 en lugar de la de PyTorch:

    modelo            torch fp32   ->  onnx int8
    peso en disco       470 MB     ->   113 MB
    libreria            torch          onnxruntime
    peso libreria       490 MB     ->    50 MB

El motivo no es la velocidad, es que quepa: los planes gratuitos de hosting
dan 512 MB de RAM y con torch no entra.

La cuantizacion no sale gratis, y conviene saber exactamente que se pierde.
Medido sobre el catalogo completo (46.922 productos) con diez consultas en
espanol, comparando contra el mismo indice calculado en fp32:

    coseno entre el vector int8 y el fp32     0,988 en el peor caso
    el resultado top-1 de fp32 sigue saliendo 10 de 10 veces en el top-10
    los 5 mejores de fp32 siguen saliendo     94% de las veces en el top-20
    el orden exacto coincide                  solo 6 de 10 veces en el top-1

O sea: el ranking SI se reordena, pero no se pierden resultados buenos. Tiene
sentido en este catalogo, donde hay miles de prendas casi identicas ("camiseta
basica de algodon" tiene cientos de candidatas validas) y una perturbacion
minima basta para intercambiar casi-empates. Lo que importa es que los
resultados relevantes siguen estando, no en que posicion exacta.

A cambio hay que escribir a mano las dos operaciones que sentence-transformers
hacia por su cuenta, que son pocas lineas y estan abajo: media de los tokens
ponderada por la mascara, y normalizado L2.

El modelo vive en `modelo_onnx/`, que no esta en el repositorio: lo baja
`preparar_modelo.py`, una vez en desarrollo y otra durante el build de la
imagen. En el arranque no se descarga nada, porque un servidor que baja
113 MB en cada reinicio tarda demasiado en responder la primera consulta.
"""

import sqlite3
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from tqdm import tqdm

from db import DIMENSIONES

DIRECTORIO_MODELO = Path(__file__).parent / "modelo_onnx"

# 128 es el max_seq_length con el que se entreno el modelo (viene en su
# sentence_bert_config.json). Subirlo no aporta: el modelo no aprendio
# posiciones mas alla de ahi.
TOKENS_MAXIMOS = 128

# Textos por lote. 64 va bien en CPU; subirlo mucho no acelera y dispara la
# memoria, que es justo lo que se esta intentando ahorrar.
LOTE = 64

# A nivel de modulo: cargar el modelo tarda unos segundos. Hacerlo dentro de
# embeder_consulta() lo repetiria en cada peticion HTTP.
tokenizador = Tokenizer.from_file(str(DIRECTORIO_MODELO / "tokenizer.json"))
tokenizador.enable_truncation(max_length=TOKENS_MAXIMOS)
tokenizador.enable_padding()

sesion = ort.InferenceSession(
    str(DIRECTORIO_MODELO / "modelo.onnx"),
    providers=["CPUExecutionProvider"],
)

# El grafo pide token_type_ids aunque este modelo no los use (van todos a
# cero). Se consulta una vez en vez de asumirlo.
_ENTRADAS = {entrada.name for entrada in sesion.get_inputs()}


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


def vectorizar(textos: list[str]) -> np.ndarray:
    """
    Textos -> matriz (n, 384) de vectores unitarios.

    Reimplementa lo que hacia sentence-transformers: pasar los tokens por el
    modelo, promediar los vectores de cada token y normalizar.
    """
    codificados = tokenizador.encode_batch(textos)
    ids = np.array([c.ids for c in codificados], dtype=np.int64)
    mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)

    entradas = {"input_ids": ids, "attention_mask": mascara}
    if "token_type_ids" in _ENTRADAS:
        entradas["token_type_ids"] = np.zeros_like(ids)

    # (n, tokens, 384): un vector por token.
    por_token = sesion.run(None, entradas)[0]

    # Media ponderada por la mascara. El padding tiene mascara 0, asi que no
    # entra en la suma ni en el divisor: sin esto, una frase corta quedaria
    # diluida por sus propios tokens de relleno.
    pesos = mascara[..., None].astype(np.float32)
    sumas = (por_token * pesos).sum(axis=1)
    vectores = sumas / np.clip(pesos.sum(axis=1), 1e-9, None)

    # Normalizado L2. Con vectores unitarios, la distancia euclidea que usa
    # sqlite-vec ordena igual que la similitud coseno, que es la metrica con
    # la que se entreno el modelo.
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / np.clip(normas, 1e-12, None)


def _a_bytes(vector) -> bytes:
    """Empaqueta un vector como los floats de 4 bytes que espera sqlite-vec."""
    return struct.pack(f"{DIMENSIONES}f", *vector)


def indexar(con: sqlite3.Connection, recalcular: bool = False) -> int:
    """
    Calcula y guarda un embedding por producto. Devuelve cuantos indexo.

    Por defecto solo procesa los productos que aun no tienen vector, para que
    ejecutarlo despues de ampliar el ingest no rehaga el trabajo ya hecho.
    `recalcular` fuerza la pasada entera, que es lo que hace falta al cambiar
    de modelo: mezclar vectores de dos modelos distintos da un ranking sin
    sentido, porque no comparten espacio vectorial.
    """
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
    if recalcular:
        # Las tablas virtuales de sqlite-vec no admiten INSERT OR REPLACE
        # (dan "UNIQUE constraint failed on primary key"), asi que la pasada
        # completa empieza por vaciar. Sin esto, cambiar de modelo obligaria
        # a borrar el fichero de BD entero y repetir el ingest.
        con.execute("DELETE FROM vec_productos")
        con.commit()

    donde = "" if recalcular else (
        "WHERE product_code NOT IN (SELECT product_code FROM vec_productos)"
    )
    filas = con.execute(consulta.format(donde=donde)).fetchall()

    if not filas:
        print("Nada que indexar: todos los productos tienen ya su embedding.")
        return 0

    print(f"Indexando {len(filas)} productos")

    for inicio in tqdm(range(0, len(filas), LOTE), desc="lotes", unit="lote"):
        lote = filas[inicio:inicio + LOTE]
        vectores = vectorizar([texto_de(fila) for fila in lote])
        con.executemany(
            "INSERT INTO vec_productos(product_code, embedding) VALUES (?, ?)",
            [(fila["product_code"], _a_bytes(vector))
             for fila, vector in zip(lote, vectores)],
        )
        con.commit()

    return len(filas)


def embeder_consulta(texto: str) -> bytes:
    """Vectoriza la consulta del usuario, lista para pasarla a sqlite-vec."""
    return _a_bytes(vectorizar([texto])[0])


if __name__ == "__main__":
    # Comprobacion a mano: python embeddings.py [--recalcular]
    import sys

    from db import conectar

    con = conectar()
    indexados = indexar(con, recalcular="--recalcular" in sys.argv)

    total = con.execute("SELECT count(*) FROM vec_productos").fetchone()[0]
    print(f"\nIndexados ahora: {indexados}. Total en vec_productos: {total}")
    con.close()
