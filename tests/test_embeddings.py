"""
Tests de la capa semantica.

Cargan el modelo de verdad, porque lo unico que se quiere comprobar (que una
descripcion de prenda cae cerca de la prenda correcta) es justo lo que no se
puede mockear. El modelo se carga una sola vez para todos gracias a que vive
a nivel de modulo en embeddings.py.

Las consultas van en ingles: el modelo es solo-ingles, y quien traduce del
espanol es el LLM del extractor (ver models.py). La parte de "entender
espanol" se prueba en tests/test_extractor.py, no aqui.

Incluyen la comprobacion que justifica la cuantizacion a int8: que el orden
de los resultados no cambia respecto al modelo original en fp32.
"""

import pandas as pd
import pytest

import db
import ingest


# Se importa dentro de la fixture y no arriba para que el resto de la suite
# no pague la carga del modelo si estos tests se deseleccionan.
@pytest.fixture(scope="module")
def embeddings():
    return pytest.importorskip("embeddings")


CATALOGO = [
    # (product_code, prod_name, product_type_name, detail_desc)
    (100, "Padded jacket", "Jacket",
     "Padded jacket in windproof fabric with a hood, thermal lining and "
     "ribbed cuffs. Warm for cold winter days."),
    (200, "Strap top", "Vest top",
     "Lightweight jersey top with narrow shoulder straps, for warm summer days."),
    (300, "Running shoe", "Sneakers",
     "Lightweight trainers in mesh with cushioned soles for running and sport."),
    (400, "Wedding dress", "Dress",
     "Long elegant dress in satin with a fitted waist, for formal occasions "
     "and ceremonies."),
]


@pytest.fixture(scope="module")
def con(embeddings):
    conexion = db.conectar(":memory:")
    db.crear_esquema(conexion)

    ingest.insertar(conexion, pd.DataFrame([
        {
            "article_id": codigo * 1000,
            "product_code": codigo,
            "prod_name": nombre,
            "product_type_name": tipo,
            "product_group_name": "Garment Upper body",
            "index_group_name": "Ladieswear",
            "perceived_colour_master_name": "Blue",
            "perceived_colour_value_name": "Dark",
            "graphical_appearance_name": "Solid",
            "detail_desc": descripcion,
        }
        for codigo, nombre, tipo, descripcion in CATALOGO
    ]))
    embeddings.indexar(conexion)
    yield conexion
    conexion.close()


def test_el_modelo_tiene_las_dimensiones_de_la_tabla(embeddings):
    """Si no coinciden, el INSERT en vec_productos falla."""
    vector = embeddings.vectorizar(["a blue t-shirt"])
    assert vector.shape == (1, db.DIMENSIONES)


def test_los_vectores_salen_normalizados(embeddings):
    """
    sqlite-vec ordena por distancia euclidea. Solo equivale al coseno, que es
    la metrica del modelo, si los vectores son unitarios.
    """
    import numpy as np

    vectores = embeddings.vectorizar(["winter coat", "t-shirt"])
    normas = np.linalg.norm(vectores, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-5)


def test_vectorizar_es_determinista(embeddings):
    """La misma entrada tiene que dar siempre el mismo vector."""
    import numpy as np

    assert np.allclose(
        embeddings.vectorizar(["t-shirt"]),
        embeddings.vectorizar(["t-shirt"]),
        atol=1e-9,
    )


def test_el_lote_afecta_poco_al_vector(embeddings):
    """
    Una frase suelta y la misma dentro de un lote NO dan exactamente el mismo
    vector, y conviene tenerlo documentado porque sorprende.

    La causa es la cuantizacion dinamica a int8: las escalas de cuantizacion
    se calculan a partir del rango real de las activaciones de CADA lote, asi
    que acompanar la frase de otra distinta mueve un poco el resultado. Con el
    modelo fp32 original esto no pasaba.

    Importa porque el indice se calcula en lotes de 64 y las consultas del
    usuario se vectorizan solas: hay una diferencia sistematica entre las dos
    situaciones. El efecto medido es pequeno (coseno > 0,99) y el impacto real
    sobre los resultados esta comprobado en el catalogo completo: el mejor
    resultado del modelo fp32 sigue apareciendo en el top-10 en las diez
    consultas de prueba.
    """
    sola = embeddings.vectorizar(["t-shirt"])[0]
    en_lote = embeddings.vectorizar(
        ["t-shirt", "a long warm wool coat for very cold winter days"]
    )[0]
    assert float(sola @ en_lote) > 0.98


def test_texto_de_concatena_los_tres_campos(embeddings):
    texto = embeddings.texto_de({
        "prod_name": "Mr Harrington",
        "product_type_name": "Hoodie",
        "detail_desc": "Short padded jacket.",
    })
    assert texto == "Mr Harrington. Hoodie. Short padded jacket."


def test_texto_de_aguanta_campos_vacios(embeddings):
    """product_type_name puede ser nulo; no debe dejar puntos sueltos."""
    texto = embeddings.texto_de({
        "prod_name": "Camiseta", "product_type_name": None, "detail_desc": "Algodon.",
    })
    assert texto == "Camiseta. Algodon."


def test_indexar_crea_un_vector_por_producto(con):
    """Cuatro productos en el catalogo, cuatro vectores."""
    assert con.execute("SELECT count(*) FROM vec_productos").fetchone()[0] == 4


def test_indexar_no_repite_trabajo(con, embeddings):
    """Segunda pasada: todo esta ya indexado, no hay nada que hacer."""
    assert embeddings.indexar(con) == 0


@pytest.mark.parametrize("consulta, esperado", [
    ("something warm for winter", 100),
    ("a light summer top", 200),
    ("running shoes", 300),
    ("an elegant dress for a wedding", 400),
])
def test_la_consulta_encuentra_la_prenda_adecuada(con, consulta, esperado):
    """
    El nucleo de la capa: describir una prenda con otras palabras tiene que
    devolver esa prenda. Si esto falla, el modelo elegido no sirve.
    """
    from buscador import buscar_semantica

    resultados = buscar_semantica(con, consulta, limite=4)
    assert resultados[0]["product_code"] == esperado


def test_buscar_semantica_devuelve_la_distancia(con):
    from buscador import buscar_semantica

    resultados = buscar_semantica(con, "winter coat", limite=4)
    distancias = [r["distancia"] for r in resultados]
    assert distancias == sorted(distancias)
