"""
App FastAPI.

De momento expone la busqueda estructurada tal cual: los filtros llegan como
parametros de query y se validan contra SearchFilters, que es el mismo modelo
que rellenara el LLM en la etapa 5. Probar la capa SQL sola, sin IA de por
medio, es lo que permite saber luego si un resultado raro viene del modelo o
de la consulta.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query

from db import conectar, crear_esquema
from models import Categoria, Color, Estampado, Publico, SearchFilters, Tono
from buscador import buscar_por_filtros


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


@app.get("/buscar", summary="Busqueda por filtros estructurados")
def endpoint_buscar(
    con=Depends(obtener_conexion),
    publico: Publico | None = None,
    categoria: Categoria | None = None,
    colores: list[Color] = Query(default=[]),
    colores_excluidos: list[Color] = Query(default=[]),
    tono: Tono | None = None,
    estampado: Estampado | None = None,
    limite: int = Query(default=24, ge=1, le=100),
) -> dict:
    """
    Aplica solo filtros duros. Todavia no hay busqueda semantica.

    Los tipos son los Literal de models.py, asi que /docs muestra cada campo
    como un desplegable con los valores validos y FastAPI rechaza el resto
    con un 422 sin que haya que escribir ninguna validacion.
    """
    filtros = SearchFilters(
        publico=publico,
        categoria=categoria,
        colores=colores,
        colores_excluidos=colores_excluidos,
        tono=tono,
        estampado=estampado,
        # Obligatorio en el modelo y aqui no se usa: esta etapa ignora la
        # parte semantica. Se rellena en la etapa 6.
        consulta_semantica="",
    )

    resultados = buscar_por_filtros(con, filtros, limite)

    return {
        "total": len(resultados),
        "filtros_aplicados": filtros.model_dump(exclude={"consulta_semantica"}),
        "resultados": resultados,
    }
