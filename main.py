"""
App FastAPI.

    GET /              la pagina, con el formulario
    GET /buscar/html   fragmento HTML con los resultados, para HTMX
    GET /buscar?q=...  lo mismo en JSON
    GET /filtros?...   solo el WHERE, con los filtros puestos a mano

Las dos ultimas existen para depurar. /buscar en JSON permite ver la
respuesta entera sin el front de por medio, y /filtros dice si un resultado
raro viene del SQL o de lo que extrajo el modelo, que son las dos cosas que
se confunden cuando todo pasa por la misma ruta.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from buscador import buscar, buscar_por_filtros
from db import conectar, crear_esquema
from models import Categoria, Color, Estampado, Publico, SearchFilters, Tono

# Los logs del extractor (frase de entrada y filtros extraidos) son lo que
# permite depurar el prompt con consultas reales, asi que se encienden aqui.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Prepara la BD y calienta el modelo antes de aceptar peticiones."""
    con = conectar()
    crear_esquema(con)
    con.close()

    # Importar embeddings carga el modelo (~450 MB, unos 10 segundos). Si se
    # deja para la primera busqueda, esa peticion tarda 15 segundos y las
    # siguientes 1,3: medido. Mejor pagarlo en el arranque, donde no hay nadie
    # esperando, que en la primera consulta de quien abre la demo.
    import embeddings  # noqa: F401

    log = logging.getLogger("main")
    log.info("Modelo cargado. Listo en http://127.0.0.1:8000")

    yield


app = FastAPI(
    title="Buscador de ropa",
    description="Busqueda sobre el catalogo de H&M por lenguaje natural.",
    lifespan=ciclo_de_vida,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
plantillas = Jinja2Templates(directory="templates")


def obtener_conexion():
    """
    Una conexion por peticion.

    SQLite prohibe por defecto usar una conexion desde un hilo distinto al que
    la creo, y FastAPI reparte las peticiones entre hilos. Abrir y cerrar por
    peticion es barato en SQLite (es un fichero local, no hay handshake) y
    evita tener que razonar sobre concurrencia.
    """
    con = conectar()
    try:
        yield con
    finally:
        con.close()


@app.get("/buscar", summary="Busqueda por lenguaje natural")
def endpoint_buscar(
    q: str = Query(description="La frase del usuario, tal cual la escribe."),
    limite: int = Query(default=24, ge=1, le=100),
    con=Depends(obtener_conexion),
) -> dict:
    """
    Frase libre -> prendas.

    Devuelve tambien los filtros que extrajo el LLM y los que hubo que
    relajar. No es informacion de depuracion: es lo que le dice al usuario
    por que esta viendo estos resultados.
    """
    return buscar(con, q, limite)


@app.get("/filtros", summary="Solo filtros estructurados, sin LLM")
def endpoint_filtros(
    publico: Publico | None = None,
    categoria: Categoria | None = None,
    colores: list[Color] = Query(default=[]),
    colores_excluidos: list[Color] = Query(default=[]),
    tono: Tono | None = None,
    estampado: Estampado | None = None,
    limite: int = Query(default=24, ge=1, le=100),
    con=Depends(obtener_conexion),
) -> dict:
    """
    La capa SQL sola, sin IA ni embeddings de por medio.

    Los tipos son los Literal de models.py, asi que /docs sale con un
    desplegable por campo y los valores fuera del enum son un 422 sin
    validacion escrita a mano.
    """
    filtros = SearchFilters(
        publico=publico,
        categoria=categoria,
        colores=colores,
        colores_excluidos=colores_excluidos,
        tono=tono,
        estampado=estampado,
        # Obligatoria en el modelo, y esta ruta no la usa: aqui no hay
        # ranking semantico.
        consulta_semantica="",
    )

    resultados = buscar_por_filtros(con, filtros, limite)

    return {
        "total": len(resultados),
        "filtros_aplicados": filtros.model_dump(exclude={"consulta_semantica"}),
        "resultados": resultados,
    }


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def pagina(request: Request):
    """La pagina completa. A partir de aqui todo lo mueve HTMX."""
    return plantillas.TemplateResponse(request, "index.html")


# Campos de SearchFilters que acaban en el WHERE, en el orden en que se
# ensenan. consulta_semantica no esta: no es un filtro duro, y el panel la
# muestra aparte justo para dejar clara esa diferencia.
CAMPOS_DUROS = [
    "publico", "categoria", "colores", "colores_excluidos", "tono", "estampado",
]


@app.get("/buscar/html", response_class=HTMLResponse, include_in_schema=False)
def endpoint_buscar_html(
    request: Request,
    q: str = Query(),
    limite: int = Query(default=24, ge=1, le=100),
    con=Depends(obtener_conexion),
):
    """
    Fragmento de resultados que HTMX mete en la pagina.

    Devuelve HTML y no JSON a proposito: con HTMX el servidor manda la vista
    ya montada y el navegador no necesita ninguna plantilla ni estado propio.
    """
    respuesta = buscar(con, q, limite)

    # Los filtros que se ensenan en el panel son los PEDIDOS, no los
    # aplicados: lo interesante es lo que entendio el modelo. Que alguno se
    # haya soltado despues se marca tachandolo.
    pedidos = respuesta["filtros_pedidos"]
    campos_duros = [
        (campo, ", ".join(pedidos[campo]) if isinstance(pedidos[campo], list)
         else pedidos[campo])
        for campo in CAMPOS_DUROS
        if pedidos[campo]
    ]

    return plantillas.TemplateResponse(
        request,
        "_resultados.html",
        {
            "respuesta": respuesta,
            "campos_duros": campos_duros,
            # ensure_ascii False para que las tildes se vean como tildes.
            "filtros_json": json.dumps(pedidos, indent=2, ensure_ascii=False),
        },
    )
