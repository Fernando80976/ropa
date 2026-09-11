"""
App FastAPI.

Dos endpoints, y el segundo existe para depurar:

    GET /buscar?q=...   la busqueda de verdad: LLM + filtros + semantica
    GET /filtros?...    solo el WHERE, con los filtros puestos a mano

Cuando una consulta devuelve algo raro, /filtros dice si el problema esta en
el SQL o en lo que extrajo el modelo, que son las dos cosas que se pueden
confundir cuando todo pasa por la misma ruta.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query

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
    """Crea el esquema al arrancar, por si la BD todavia no existe."""
    con = conectar()
    crear_esquema(con)
    con.close()
    yield


app = FastAPI(
    title="Buscador de ropa",
    description="Busqueda sobre el catalogo de H&M por lenguaje natural.",
    lifespan=ciclo_de_vida,
)


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
