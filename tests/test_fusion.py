"""
Tests de la fusion (filtros duros + ranking semantico) y del relajado.

El extractor va mockeado: aqui no se prueba si el LLM entiende la frase (eso
es test_extractor.py), sino que el orden filtrar-luego-ordenar y el relajado
se comportan. Los embeddings si son reales, porque el ranking es justo lo que
se quiere comprobar.
"""

import pandas as pd
import pytest

import db
import ingest
from buscador import ORDEN_DE_RELAJADO, buscar, relajar
from models import SearchFilters


CATALOGO = [
    # (codigo, nombre, tipo, publico, categoria, color, tono, estampado, desc)
    (100, "Padded jacket", "Jacket", "Menswear", "Garment Upper body",
     "Blue", "Dark", "Solid",
     "Padded winter jacket with a hood and thermal lining. Very warm."),
    (200, "Summer top", "Vest top", "Ladieswear", "Garment Upper body",
     "White", "Light", "Stripe",
     "Lightweight striped jersey top with narrow straps for warm days."),
    (300, "Kids pyjama", "Pyjama set", "Baby/Children", "Nightwear",
     "Blue", "Light", "Dot",
     "Soft cotton pyjama set for children, with long sleeves."),
    (400, "Party dress", "Dress", "Ladieswear", "Garment Full body",
     "Red", "Bright", "Sequin",
     "Long sequinned dress for parties and evening occasions."),
]


@pytest.fixture(scope="module")
def con():
    import embeddings

    conexion = db.conectar(":memory:")
    db.crear_esquema(conexion)
    ingest.insertar(conexion, pd.DataFrame([
        {
            "article_id": codigo * 1000,
            "product_code": codigo,
            "prod_name": nombre,
            "product_type_name": tipo,
            "product_group_name": categoria,
            "index_group_name": publico,
            "perceived_colour_master_name": color,
            "perceived_colour_value_name": tono,
            "graphical_appearance_name": estampado,
            "detail_desc": descripcion,
        }
        for (codigo, nombre, tipo, publico, categoria,
             color, tono, estampado, descripcion) in CATALOGO
    ]))
    embeddings.indexar(conexion)
    yield conexion
    conexion.close()


@pytest.fixture
def extrae(monkeypatch):
    """Fija lo que 'extrae' el LLM, para probar la fusion sin llamar a Groq."""
    def _extrae(**campos):
        import extractor

        campos.setdefault("consulta_semantica", "una prenda")
        monkeypatch.setattr(
            extractor, "extraer_filtros", lambda frase: SearchFilters(**campos)
        )
    return _extrae


def filtros(**campos) -> SearchFilters:
    campos.setdefault("consulta_semantica", "x")
    return SearchFilters(**campos)


# ---------------------------------------------------------------------------
# Filtrar primero, ordenar despues
# ---------------------------------------------------------------------------

def test_el_filtro_manda_sobre_la_semantica(con, extrae):
    """
    La consulta semantica pide un abrigo, pero el filtro dice Baby/Children.

    El abrigo no puede aparecer: si apareciera, el filtro seria una sugerencia
    y no un filtro, que es justo lo que se quiere evitar.
    """
    extrae(publico="Baby/Children", consulta_semantica="abrigo de invierno")
    respuesta = buscar(con, "abrigo de invierno para nino")

    assert [r["product_code"] for r in respuesta["resultados"]] == [300]


def test_la_semantica_ordena_lo_que_sobrevive(con, extrae):
    """Sin filtros, el orden lo decide la distancia."""
    extrae(consulta_semantica="algo abrigado para el invierno")
    resultados = buscar(con, "algo abrigado para el invierno")["resultados"]

    assert resultados[0]["product_code"] == 100
    distancias = [r["distancia"] for r in resultados]
    assert distancias == sorted(distancias)


def test_devuelve_los_tres_bloques_de_informacion(con, extrae):
    """El front necesita los tres para explicar por que sale lo que sale."""
    extrae(publico="Ladieswear", consulta_semantica="vestido")
    respuesta = buscar(con, "un vestido")

    assert set(respuesta) == {
        "frase", "total", "filtros_pedidos", "filtros_aplicados",
        "filtros_relajados", "resultados",
    }
    assert respuesta["frase"] == "un vestido"


def test_respeta_el_limite(con, extrae):
    extrae(consulta_semantica="ropa")
    assert len(buscar(con, "ropa", limite=2)["resultados"]) == 2


# ---------------------------------------------------------------------------
# Relajado
# ---------------------------------------------------------------------------

def test_no_relaja_nada_si_ya_hay_resultados(con):
    resultantes, relajados = relajar(con, filtros(publico="Menswear"))

    assert relajados == []
    assert resultantes.publico == "Menswear"


def test_relaja_hasta_encontrar_algo(con):
    """Ningun articulo es Menswear + Sequin, pero si hay Menswear."""
    resultantes, relajados = relajar(
        con, filtros(publico="Menswear", estampado="Sequin")
    )

    assert relajados == ["estampado"]
    assert resultantes.publico == "Menswear"
    assert resultantes.estampado is None


def test_relaja_en_orden_de_rigidez(con):
    """
    Con varios filtros imposibles, cae primero el estampado y despues el tono.

    Es el orden que define ORDEN_DE_RELAJADO: se suelta antes lo accesorio.
    """
    _, relajados = relajar(
        con, filtros(publico="Menswear", estampado="Sequin", tono="Bright")
    )

    assert relajados == ["estampado", "tono"]


def test_publico_es_el_ultimo_en_caer(con):
    """
    Un tercio del catalogo real es ropa de bebe: soltar `publico` antes de
    tiempo es el fallo que mas se nota.
    """
    assert ORDEN_DE_RELAJADO[-1] == "publico"


def test_no_anuncia_como_relajado_un_filtro_que_estaba_vacio(con):
    """Solo se anuncia lo que el usuario habia pedido de verdad."""
    _, relajados = relajar(con, filtros(publico="Sport"))

    assert relajados == ["publico"]


def test_los_colores_excluidos_no_se_relajan_nunca(con):
    """
    'Que no sea rojo' es lo unico que el usuario ha pedido en negativo.

    Devolverle justo eso es peor que no devolverle nada, asi que el campo no
    esta en ORDEN_DE_RELAJADO.
    """
    assert "colores_excluidos" not in ORDEN_DE_RELAJADO

    resultantes, _ = relajar(
        con, filtros(colores_excluidos=["Red"], estampado="Melange")
    )
    assert resultantes.colores_excluidos == ["Red"]


def test_la_busqueda_avisa_de_lo_que_relajo(con, extrae):
    """Un buscador que amplia la busqueda en silencio esta mintiendo."""
    extrae(publico="Menswear", estampado="Sequin",
           consulta_semantica="chaqueta de lentejuelas")
    respuesta = buscar(con, "chaqueta de lentejuelas de hombre")

    assert respuesta["filtros_relajados"] == ["estampado"]
    assert respuesta["filtros_pedidos"]["estampado"] == "Sequin"
    assert respuesta["filtros_aplicados"]["estampado"] is None
    assert respuesta["total"] > 0


def test_una_consulta_imposible_acaba_devolviendo_algo(con, extrae):
    """
    Todos los filtros a la vez y ninguno compatible: se sueltan todos.

    Cero resultados es un buscador roto; devolver lo mas parecido con un aviso
    no lo es.
    """
    extrae(publico="Sport", categoria="Swimwear", colores=["Metal"],
           tono="Medium Dusty", estampado="Lace",
           consulta_semantica="banador metalico de encaje")
    respuesta = buscar(con, "banador metalico de encaje deportivo")

    assert respuesta["total"] > 0
    assert respuesta["filtros_relajados"] == ORDEN_DE_RELAJADO
