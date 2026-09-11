"""
Tests de la busqueda estructurada.

Cada test monta su propio catalogo minimo en memoria: asi se sabe exactamente
que deberia salir, en vez de depender de que ropa.db tenga cargadas unas
filas concretas.
"""

import pandas as pd
import pytest

import db
import ingest
from buscador import buscar_por_filtros, construir_where, contar_por_filtros
from models import SearchFilters


def prenda(article_id, product_code, **campos):
    base = {
        "article_id": article_id,
        "product_code": product_code,
        "prod_name": f"Prenda {product_code}",
        "product_type_name": "T-shirt",
        "product_group_name": "Garment Upper body",
        "index_group_name": "Menswear",
        "perceived_colour_master_name": "Blue",
        "perceived_colour_value_name": "Dark",
        "graphical_appearance_name": "Solid",
        "detail_desc": "Descripcion.",
    }
    return {**base, **campos}


@pytest.fixture
def con():
    conexion = db.conectar(":memory:")
    db.crear_esquema(conexion)
    ingest.insertar(conexion, pd.DataFrame([
        # Un producto con tres variantes de color.
        prenda(1, 100, perceived_colour_master_name="Blue"),
        prenda(2, 100, perceived_colour_master_name="Black"),
        prenda(3, 100, perceived_colour_master_name="Red"),
        # Ropa de mujer, vestido rojo claro de rayas.
        prenda(4, 200, index_group_name="Ladieswear",
               product_group_name="Garment Full body",
               perceived_colour_master_name="Red",
               perceived_colour_value_name="Light",
               graphical_appearance_name="Stripe"),
        # Ropa de nino.
        prenda(5, 300, index_group_name="Baby/Children"),
    ]))
    yield conexion
    conexion.close()


def filtros(**campos) -> SearchFilters:
    """consulta_semantica es obligatoria en el modelo pero esta capa la ignora."""
    return SearchFilters(consulta_semantica="x", **campos)


def test_sin_filtros_devuelve_un_where_neutro():
    where, parametros = construir_where(filtros())
    assert (where, parametros) == ("1 = 1", [])


def test_los_valores_van_como_parametros_no_interpolados():
    """Ningun valor del usuario puede acabar dentro del string de la consulta."""
    where, parametros = construir_where(filtros(publico="Menswear", tono="Dark"))
    assert "Menswear" not in where and "Dark" not in where
    assert parametros == ["Menswear", "Dark"]


def test_agrupa_las_variantes_de_color(con):
    """Tres variantes del producto 100 tienen que salir como un solo resultado."""
    resultados = buscar_por_filtros(con, filtros())

    assert [r["product_code"] for r in resultados] == [100, 200, 300]
    assert next(r for r in resultados if r["product_code"] == 100)["variantes"] == 3


def test_el_representante_del_grupo_es_coherente(con):
    """
    Las columnas sueltas deben venir todas de la MISMA fila del grupo.

    Es la regla de min() de SQLite: si se rompiera, un producto podria salir
    con el article_id de una variante y el color de otra.
    """
    resultado = next(
        r for r in buscar_por_filtros(con, filtros()) if r["product_code"] == 100
    )
    esperado = con.execute(
        "SELECT perceived_colour_master_name FROM articulos WHERE article_id = ?",
        (resultado["article_id"],),
    ).fetchone()
    assert resultado["perceived_colour_master_name"] == esperado[0]


def test_filtra_por_publico(con):
    resultados = buscar_por_filtros(con, filtros(publico="Baby/Children"))
    assert [r["product_code"] for r in resultados] == [300]


def test_colores_es_un_in(con):
    """Varias familias de color a la vez."""
    resultados = buscar_por_filtros(con, filtros(colores=["Red", "Black"]))
    assert {r["product_code"] for r in resultados} == {100, 200}


def test_colores_excluidos_es_un_not_in(con):
    """El producto 200 es rojo, asi que 'que no sea rojo' debe dejarlo fuera."""
    resultados = buscar_por_filtros(con, filtros(colores_excluidos=["Red"]))
    assert 200 not in {r["product_code"] for r in resultados}


def test_combina_varios_filtros(con):
    resultados = buscar_por_filtros(
        con,
        filtros(publico="Ladieswear", categoria="Garment Full body",
                colores=["Red"], tono="Light", estampado="Stripe"),
    )
    assert [r["product_code"] for r in resultados] == [200]


def test_filtros_imposibles_devuelven_lista_vacia(con):
    """Cero resultados es un caso normal: lo resuelve el relajado de la etapa 6."""
    assert buscar_por_filtros(con, filtros(publico="Sport", colores=["Metal"])) == []


def test_respeta_el_limite(con):
    assert len(buscar_por_filtros(con, filtros(), limite=2)) == 2


def test_contar_cuenta_productos_no_filas(con):
    """5 filas en la BD, pero solo 3 productos distintos."""
    assert contar_por_filtros(con, filtros()) == 3
