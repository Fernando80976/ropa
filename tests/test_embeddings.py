"""
Tests de la capa semantica.

Cargan el modelo de verdad (no se puede mockear lo unico que se quiere
comprobar: que un texto en espanol cae cerca de una descripcion en ingles),
asi que son los tests lentos de la suite. El modelo se carga una sola vez
para todos gracias a que vive a nivel de modulo en embeddings.py.
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
    assert embeddings.modelo.get_embedding_dimension() == db.DIMENSIONES


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
    ("algo abrigado para el invierno", 100),
    ("una camiseta fresca de verano", 200),
    ("zapatillas para correr", 300),
    ("un vestido elegante para una boda", 400),
])
def test_consultas_en_espanol_encuentran_descripciones_en_ingles(
    con, consulta, esperado
):
    """
    El nucleo de la etapa: el catalogo esta en ingles y el usuario escribe en
    espanol. Si esto falla, el modelo elegido no sirve para el proyecto.
    """
    from buscador import buscar_semantica

    resultados = buscar_semantica(con, consulta, limite=4)
    assert resultados[0]["product_code"] == esperado


def test_buscar_semantica_devuelve_la_distancia(con):
    from buscador import buscar_semantica

    resultados = buscar_semantica(con, "abrigo de invierno", limite=4)
    distancias = [r["distancia"] for r in resultados]
    assert distancias == sorted(distancias)
