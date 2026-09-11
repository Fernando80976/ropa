"""
Tests del extractor.

Casi todos mockean la llamada a Groq: se sustituye `_llamar` por una funcion
que devuelve el JSON que se quiera. Asi se pueden provocar a voluntad los
fallos que importan (valor fuera del enum, JSON roto, campo vacio), que con
la API de verdad no se sabe cuando van a pasar, y la suite corre sin red ni
API key.

El unico test que llama a Groq de verdad esta marcado como integracion:

    pytest -q                        # rapido, se salta integracion
    pytest -q -m integracion         # necesita GROQ_API_KEY
"""

import json

import pytest
import extractor
from models import SearchFilters


@pytest.fixture
def responder(monkeypatch):
    """Hace que Groq devuelva un texto fijo, o una secuencia de textos."""
    def _responder(*respuestas):
        pendientes = list(respuestas)
        llamadas = []

        def falso_llamar(frase):
            llamadas.append(frase)
            siguiente = pendientes.pop(0) if len(pendientes) > 1 else pendientes[0]
            if isinstance(siguiente, Exception):
                raise siguiente
            return siguiente

        monkeypatch.setattr(extractor, "_llamar", falso_llamar)
        return llamadas

    return _responder


def respuesta(**campos) -> str:
    """JSON completo como el que devuelve el modelo en modo strict."""
    base = {
        "publico": None,
        "categoria": None,
        "colores": [],
        "colores_excluidos": [],
        "tono": None,
        "estampado": None,
        "consulta_semantica": "una prenda",
    }
    return json.dumps({**base, **campos})


# ---------------------------------------------------------------------------
# El schema que se le manda al modelo
# ---------------------------------------------------------------------------

def test_el_esquema_sale_de_pydantic_no_escrito_a_mano():
    """Los valores validos tienen que venir de models.py, no duplicados aqui."""
    esquema = extractor.esquema_json()
    publico = esquema["properties"]["publico"]["anyOf"][0]["enum"]
    assert set(publico) == {"Ladieswear", "Menswear", "Divided", "Baby/Children", "Sport"}


def test_el_esquema_cumple_lo_que_exige_groq_en_strict():
    """Sin estas dos claves la API responde 400."""
    esquema = extractor.esquema_json()
    assert esquema["additionalProperties"] is False
    assert set(esquema["required"]) == set(esquema["properties"])


def test_el_system_prompt_no_repite_los_enums():
    """Si alguien los copia al prompt, dejan de actualizarse con models.py."""
    assert "Ladieswear" not in extractor.SISTEMA
    assert "Garment Upper body" not in extractor.SISTEMA


def test_el_system_prompt_pide_la_consulta_semantica_en_ingles():
    """
    El modelo de embeddings es solo-ingles: quien traduce es el LLM. Si esta
    instruccion desaparece del prompt, la busqueda semantica se degrada sin
    dar ningun error.
    """
    assert "INGLES" in extractor.SISTEMA


# ---------------------------------------------------------------------------
# Las seis frases de ejemplo
# ---------------------------------------------------------------------------

# Frase -> lo que el modelo devuelve -> campos que deben quedar rellenos.
# Fija el contrato de la capa, no la calidad del modelo: de eso se ocupa el
# test de integracion.
FRASES = [
    (
        "chaqueta vaquera oversize para una boda en verano, que no sea negra",
        respuesta(categoria="Garment Upper body", colores_excluidos=["Black"],
                  estampado="Denim",
                  consulta_semantica="chaqueta oversize ligera para evento"),
        {"categoria": "Garment Upper body", "colores_excluidos": ["Black"],
         "estampado": "Denim"},
    ),
    (
        "vestido rojo largo para una fiesta",
        respuesta(categoria="Garment Full body", colores=["Red"],
                  consulta_semantica="vestido largo de fiesta"),
        {"categoria": "Garment Full body", "colores": ["Red"]},
    ),
    (
        "camiseta de hombre azul oscuro lisa",
        respuesta(publico="Menswear", categoria="Garment Upper body",
                  colores=["Blue"], tono="Dark", estampado="Solid",
                  consulta_semantica="camiseta"),
        {"publico": "Menswear", "colores": ["Blue"], "tono": "Dark",
         "estampado": "Solid"},
    ),
    (
        "pijama para un nino de 5 anos",
        respuesta(publico="Baby/Children", consulta_semantica="pijama infantil"),
        {"publico": "Baby/Children"},
    ),
    (
        "ropa deportiva para correr, ni negra ni gris",
        respuesta(publico="Sport", colores_excluidos=["Black", "Grey"],
                  consulta_semantica="ropa tecnica para correr"),
        {"publico": "Sport", "colores_excluidos": ["Black", "Grey"]},
    ),
    (
        "algo comodo para estar en casa",
        respuesta(consulta_semantica="prenda comoda de estar por casa"),
        {},  # ningun filtro duro: todo es semantico
    ),
]


@pytest.mark.parametrize("frase, json_del_modelo, esperado",
                         FRASES, ids=[f[0][:30] for f in FRASES])
def test_frases_de_ejemplo(responder, frase, json_del_modelo, esperado):
    responder(json_del_modelo)
    filtros = extractor.extraer_filtros(frase)

    for campo, valor in esperado.items():
        assert getattr(filtros, campo) == valor, campo

    # Invariante de models.py: nunca vacia.
    assert filtros.consulta_semantica.strip()


def test_los_campos_no_mencionados_quedan_vacios(responder):
    """Un filtro inventado deja al usuario sin resultados."""
    responder(respuesta(consulta_semantica="prenda comoda de estar por casa"))
    filtros = extractor.extraer_filtros("algo comodo para estar en casa")

    assert filtros.publico is None
    assert filtros.categoria is None
    assert filtros.colores == []
    assert filtros.tono is None


# ---------------------------------------------------------------------------
# Degradar, no fallar
# ---------------------------------------------------------------------------

def test_un_valor_fuera_del_enum_solo_tira_ese_campo(responder):
    """
    Turquesa no es un Color valido (seria Turquoise), pero el resto de la
    frase es aprovechable: se pierde el color, no la busqueda.
    """
    responder(respuesta(publico="Menswear", colores=["Turquesa"],
                        consulta_semantica="camiseta"))
    filtros = extractor.extraer_filtros("camiseta de hombre turquesa")

    assert filtros.publico == "Menswear"
    assert filtros.colores == []
    assert filtros.consulta_semantica == "camiseta"


def test_varios_campos_invalidos_a_la_vez(responder):
    responder(respuesta(publico="Hombres", categoria="Camisetas", tono="Oscuro",
                        estampado="Denim", consulta_semantica="camiseta"))
    filtros = extractor.extraer_filtros("camiseta de hombre oscura vaquera")

    assert (filtros.publico, filtros.categoria, filtros.tono) == (None, None, None)
    assert filtros.estampado == "Denim"


def test_si_cae_consulta_semantica_se_repone_con_la_frase(responder):
    """Es obligatoria: sin ella no hay nada que buscar por semantica."""
    responder(json.dumps({"consulta_semantica": None, "publico": "Menswear"}))
    filtros = extractor.extraer_filtros("camiseta de hombre")

    assert filtros.consulta_semantica == "camiseta de hombre"
    assert filtros.publico == "Menswear"


def test_consulta_semantica_vacia_se_repone_con_la_frase(responder):
    """El prompt lo prohibe, pero el modelo puede devolverla vacia igualmente."""
    responder(respuesta(consulta_semantica="   "))
    assert extractor.extraer_filtros("un abrigo").consulta_semantica == "un abrigo"


def test_un_json_vacio_no_tira_la_consulta(responder):
    """
    Si el modelo devuelve {}, se pierden los filtros pero queda la frase: la
    busqueda degrada a puramente semantica en vez de fallar.
    """
    responder("{}")
    filtros = extractor.extraer_filtros("un abrigo de invierno")

    assert filtros == SearchFilters(consulta_semantica="un abrigo de invierno")


def test_consulta_semantica_con_tipo_absurdo_se_repone():
    """El campo obligatorio se reconstruye siempre a partir de la frase."""
    filtros = extractor.validar_degradando(
        {"consulta_semantica": {"no": "es texto"}, "publico": "Menswear"},
        "camiseta de hombre",
    )
    assert filtros.consulta_semantica == "camiseta de hombre"
    assert filtros.publico == "Menswear"


# ---------------------------------------------------------------------------
# Reintento y fallback
# ---------------------------------------------------------------------------

def test_json_roto_se_reintenta_una_vez(responder):
    llamadas = responder("{esto no es json", respuesta(publico="Menswear"))
    filtros = extractor.extraer_filtros("camiseta de hombre")

    assert len(llamadas) == 2
    assert filtros.publico == "Menswear"


def test_si_el_reintento_tambien_falla_se_cae_al_fallback(responder):
    """Sin filtros duros, pero con la frase entera como consulta semantica."""
    responder("{roto", "{roto tambien")
    filtros = extractor.extraer_filtros("camiseta de hombre azul")

    assert filtros == SearchFilters(consulta_semantica="camiseta de hombre azul")


def test_un_error_de_red_no_propaga(responder):
    """El buscador no puede devolver un 500 porque Groq este caido."""
    responder(ConnectionError("Groq no responde"))
    filtros = extractor.extraer_filtros("un abrigo")

    assert filtros.consulta_semantica == "un abrigo"


def test_extraer_filtros_nunca_lanza(responder):
    responder(Exception("lo que sea"))
    assert isinstance(extractor.extraer_filtros("x"), SearchFilters)


def test_se_loguea_la_frase_y_los_filtros(responder, caplog):
    """Los logs son lo que permite depurar el prompt con consultas reales."""
    responder(respuesta(publico="Menswear"))
    with caplog.at_level("INFO", logger="extractor"):
        extractor.extraer_filtros("camiseta de hombre")

    registro = "\n".join(caplog.messages)
    assert "camiseta de hombre" in registro
    assert "Menswear" in registro


# ---------------------------------------------------------------------------
# Integracion: llama a Groq de verdad
# ---------------------------------------------------------------------------

@pytest.mark.integracion
def test_integracion_groq_devuelve_filtros_sensatos():
    """
    Comprueba que el modelo y el schema siguen entendiendose.

    Se afirma poco a proposito: que el color pedido sale y que la parte
    semantica no viene vacia. Afirmar los seis campos convertiria cada cambio
    de version del modelo en un test rojo.
    """
    import os

    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("sin GROQ_API_KEY")

    filtros = extractor.extraer_filtros("vestido rojo largo para una boda")

    assert "Red" in filtros.colores
    assert filtros.categoria == "Garment Full body"
    assert filtros.consulta_semantica.strip()

    # Traducido al ingles, que es lo que espera el modelo de embeddings.
    assert "dress" in filtros.consulta_semantica.lower()
