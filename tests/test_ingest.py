"""
Tests del ingest.

Se prueban las dos decisiones que tiene el modulo (que se descarta y que no se
duplica) sobre un DataFrame de mentira, sin tocar articles.csv.
"""

import pandas as pd
import pytest

import db
import ingest


@pytest.fixture
def con():
    conexion = db.conectar(":memory:")
    db.crear_esquema(conexion)
    yield conexion
    conexion.close()


def fila(article_id, product_code, grupo="Garment Upper body", desc="Una camiseta."):
    """Fila minima con las 10 columnas que espera el ingest."""
    return {
        "article_id": article_id,
        "product_code": product_code,
        "prod_name": "Camiseta",
        "product_type_name": "T-shirt",
        "product_group_name": grupo,
        "index_group_name": "Menswear",
        "perceived_colour_master_name": "Blue",
        "perceived_colour_value_name": "Dark",
        "graphical_appearance_name": "Solid",
        "detail_desc": desc,
    }


def test_descarta_lo_que_no_es_ropa():
    df = pd.DataFrame([
        fila(1, 100),
        fila(2, 200, grupo="Furniture"),
        fila(3, 300, grupo="Cosmetic"),
    ])
    assert list(ingest.limpiar(df)["article_id"]) == [1]


def test_descarta_underwear_nightwear():
    """Existe en el CSV pero no en models.Categoria: la lista blanca lo tira."""
    df = pd.DataFrame([fila(1, 100), fila(2, 200, grupo="Underwear/nightwear")])
    assert list(ingest.limpiar(df)["article_id"]) == [1]


def test_descarta_filas_sin_descripcion():
    """Sin detail_desc no hay texto que embeber, asi que no serviria de nada."""
    df = pd.DataFrame([fila(1, 100), fila(2, 200, desc=None)])
    assert list(ingest.limpiar(df)["article_id"]) == [1]


def test_insertar_es_idempotente(con):
    """Ejecutar el ingest dos veces no puede duplicar filas."""
    df = pd.DataFrame([fila(1, 100), fila(2, 100), fila(3, 200)])

    ingest.insertar(con, df)
    ingest.insertar(con, df)

    total, productos = con.execute(
        "SELECT count(*), count(DISTINCT product_code) FROM articulos"
    ).fetchone()
    assert (total, productos) == (3, 2)


def test_insertar_guarda_los_nulos_como_null(con):
    """Las columnas opcionales vienen como NaN de pandas, no como None."""
    df = pd.DataFrame([fila(1, 100)])
    df.loc[0, "graphical_appearance_name"] = None

    ingest.insertar(con, df)

    guardada = con.execute(
        "SELECT graphical_appearance_name FROM articulos"
    ).fetchone()
    assert guardada["graphical_appearance_name"] is None
