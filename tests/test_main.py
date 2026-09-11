"""
Tests de las rutas.

El extractor va mockeado: probar HTTP y Groq a la vez haria que un fallo de
red se pareciera a un fallo de las rutas. La BD si es la de verdad (ropa.db),
porque montar una en memoria no sirve: las rutas abren su propia conexion.
"""

import pytest
from fastapi.testclient import TestClient

from models import SearchFilters


@pytest.fixture(scope="module")
def cliente():
    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def extrae(monkeypatch):
    """Fija los filtros que 'devuelve' el LLM."""
    def _extrae(**campos):
        import extractor

        campos.setdefault("consulta_semantica", "una prenda")
        monkeypatch.setattr(
            extractor, "extraer_filtros", lambda frase: SearchFilters(**campos)
        )
    return _extrae


@pytest.fixture(scope="module", autouse=True)
def hay_datos(cliente):
    """Estos tests necesitan el catalogo cargado e indexado."""
    from db import conectar

    con = conectar()
    articulos = con.execute("SELECT count(*) FROM articulos").fetchone()[0]
    vectores = con.execute("SELECT count(*) FROM vec_productos").fetchone()[0]
    con.close()

    if not articulos or not vectores:
        pytest.skip("ropa.db vacia: ejecuta ingest.py y embeddings.py")


# ---------------------------------------------------------------------------
# Pagina y estaticos
# ---------------------------------------------------------------------------

def test_la_pagina_carga(cliente):
    respuesta = cliente.get("/")

    assert respuesta.status_code == 200
    assert "Buscador de ropa" in respuesta.text


def test_la_pagina_trae_htmx_y_la_hoja_de_estilos(cliente):
    texto = cliente.get("/").text

    assert "htmx.org" in texto
    assert "/static/estilo.css" in texto
    assert cliente.get("/static/estilo.css").status_code == 200


# ---------------------------------------------------------------------------
# Fragmento HTML
# ---------------------------------------------------------------------------

def test_el_fragmento_no_es_una_pagina_entera(cliente, extrae):
    """HTMX lo inserta dentro del DOM: si trae <html> queda anidado."""
    extrae(consulta_semantica="un vestido")
    texto = cliente.get("/buscar/html", params={"q": "un vestido", "limite": 3}).text

    assert "<html" not in texto.lower()
    assert "tarjeta" in texto


def test_el_fragmento_ensena_el_json_del_modelo(cliente, extrae):
    """El panel de filtros es lo que hace explicable la demo."""
    extrae(publico="Menswear", colores=["Blue"], consulta_semantica="camiseta")
    texto = cliente.get("/buscar/html", params={"q": "camiseta azul de hombre"}).text

    assert "Lo que entendio el modelo" in texto
    assert "Menswear" in texto
    assert "Blue" in texto


def test_el_fragmento_avisa_si_se_relajo_algo(cliente, extrae):
    """Ampliar la busqueda en silencio es mentirle al usuario."""
    extrae(publico="Menswear", categoria="Swimwear", colores=["Metal"],
           estampado="Lace", consulta_semantica="banador")
    texto = cliente.get("/buscar/html", params={"q": "banador metalico de encaje"}).text

    assert "Se amplio la busqueda" in texto
    assert "filtro--relajado" in texto


def test_el_fragmento_no_avisa_si_no_relajo_nada(cliente, extrae):
    extrae(publico="Menswear", consulta_semantica="camiseta")
    texto = cliente.get("/buscar/html", params={"q": "camiseta de hombre"}).text

    assert "Se amplio la busqueda" not in texto


def test_el_html_se_escapa(cliente, extrae):
    """Jinja2 autoescapa: la frase del usuario no puede inyectar etiquetas."""
    extrae(consulta_semantica="<script>alert(1)</script>")
    texto = cliente.get("/buscar/html", params={"q": "<script>alert(1)</script>"}).text

    assert "<script>alert(1)</script>" not in texto


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def test_buscar_devuelve_json(cliente, extrae):
    extrae(publico="Menswear", consulta_semantica="camiseta")
    datos = cliente.get("/buscar", params={"q": "camiseta de hombre"}).json()

    assert datos["frase"] == "camiseta de hombre"
    assert datos["filtros_aplicados"]["publico"] == "Menswear"
    assert datos["resultados"]


def test_buscar_exige_la_consulta(cliente):
    assert cliente.get("/buscar").status_code == 422


def test_filtros_no_llama_al_llm(cliente, monkeypatch):
    """La ruta de depuracion tiene que funcionar con Groq caido."""
    import extractor

    def explota(frase):
        raise AssertionError("no deberia llamarse al extractor")

    monkeypatch.setattr(extractor, "extraer_filtros", explota)

    datos = cliente.get("/filtros", params={"publico": "Menswear"}).json()
    assert datos["filtros_aplicados"]["publico"] == "Menswear"


def test_filtros_rechaza_valores_fuera_del_enum(cliente):
    """Los Literal de models.py validan solos, sin codigo a mano."""
    assert cliente.get("/filtros", params={"publico": "Perros"}).status_code == 422


def test_el_limite_tiene_tope(cliente):
    """Sin tope, una peticion podria pedir el catalogo entero."""
    assert cliente.get("/buscar", params={"q": "x", "limite": 500}).status_code == 422
