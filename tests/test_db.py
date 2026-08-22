"""
Tests del esquema.

Todos usan una BD en memoria: son rapidos y no dependen de que ropa.db exista
ni de en que estado este.
"""

import struct

import pytest

import db


@pytest.fixture
def con():
    """Conexion en memoria con el esquema ya creado."""
    conexion = db.conectar(":memory:")
    db.crear_esquema(conexion)
    yield conexion
    conexion.close()


def test_conectar_carga_sqlite_vec(con):
    """Si la extension no cargara, vec_version() no existiria como funcion."""
    version, = con.execute("SELECT vec_version()").fetchone()
    assert version.startswith("v")


def test_filas_accesibles_por_nombre(con):
    """row_factory: el resto del proyecto asume fila['columna']."""
    fila = con.execute("SELECT 1 AS uno").fetchone()
    assert fila["uno"] == 1


def test_crear_esquema_es_idempotente(con):
    """Se ejecuta en cada arranque, asi que llamarlo dos veces no puede fallar."""
    db.crear_esquema(con)


def test_articulos_tiene_las_columnas_del_contrato(con):
    """Las columnas que buscador.py usara como WHERE, segun models.py."""
    columnas = {fila["name"] for fila in con.execute("PRAGMA table_info(articulos)")}
    assert columnas == {
        "article_id",
        "product_code",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "index_group_name",
        "perceived_colour_master_name",
        "perceived_colour_value_name",
        "graphical_appearance_name",
        "detail_desc",
    }


@pytest.mark.parametrize(
    "columna",
    [
        "product_code",
        "product_group_name",
        "index_group_name",
        "perceived_colour_master_name",
        "perceived_colour_value_name",
        "graphical_appearance_name",
    ],
)
def test_las_columnas_filtrables_estan_indexadas(con, columna):
    """Sin indice, cada filtro seria un scan completo de 105k filas."""
    indices = {fila["name"] for fila in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    )}
    assert f"idx_articulos_{columna}" in indices


def test_vec_productos_guarda_y_devuelve_un_vector(con):
    """Ida y vuelta de un embedding, unido por product_code."""
    vector = struct.pack(f"{db.DIMENSIONES}f", *([0.1] * db.DIMENSIONES))
    con.execute(
        "INSERT INTO vec_productos(product_code, embedding) VALUES (?, ?)",
        (108775, vector),
    )
    fila = con.execute("SELECT product_code FROM vec_productos").fetchone()
    assert fila["product_code"] == 108775


def test_vec_productos_rechaza_dimensiones_erroneas(con):
    """Un vector de otro modelo debe fallar aqui, no dar resultados absurdos."""
    vector_corto = struct.pack("100f", *([0.1] * 100))
    with pytest.raises(Exception):
        con.execute(
            "INSERT INTO vec_productos(product_code, embedding) VALUES (?, ?)",
            (108775, vector_corto),
        )
