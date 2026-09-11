"""
Busqueda sobre el catalogo.

Tres capas, cada una util por si sola:

  buscar_por_filtros   WHERE en SQL a partir de SearchFilters. Sin IA.
  buscar_semantica     ranking por distancia vectorial. Sin IA. (etapa 4)
  buscar               fusion de las dos + extraccion LLM. (etapa 6)

La que importa es la tercera, pero las dos primeras se prueban solas y eso es
lo que permite saber cual de las tres esta fallando cuando algo va mal.
"""

import sqlite3

from models import SearchFilters

# Columnas que se devuelven al front. article_id no esta aqui porque lo anade
# el SELECT como min(article_id); es la que forma la URL de la imagen en el
# catalogo de H&M, asi que tiene que salir si o si.
COLUMNAS_RESULTADO = """
    product_code,
    prod_name,
    product_type_name,
    product_group_name,
    index_group_name,
    perceived_colour_master_name,
    perceived_colour_value_name,
    graphical_appearance_name,
    detail_desc
"""


def construir_where(filtros: SearchFilters) -> tuple[str, list]:
    """
    Traduce SearchFilters a un fragmento SQL y sus parametros.

    Devuelve ("1 = 1", []) si no hay ningun filtro, para que el que llama
    pueda concatenar sin casos especiales.

    Los valores van SIEMPRE como parametros (?), nunca interpolados: aunque
    los Literal de Pydantic ya garantizan que el valor es uno de los del enum,
    depender de eso para la seguridad seria confiar en la validacion de una
    capa de arriba para evitar inyeccion en esta.
    """
    condiciones: list[str] = []
    parametros: list = []

    if filtros.publico:
        condiciones.append("index_group_name = ?")
        parametros.append(filtros.publico)

    if filtros.categoria:
        condiciones.append("product_group_name = ?")
        parametros.append(filtros.categoria)

    if filtros.colores:
        marcadores = ", ".join("?" * len(filtros.colores))
        condiciones.append(f"perceived_colour_master_name IN ({marcadores})")
        parametros.extend(filtros.colores)

    if filtros.colores_excluidos:
        marcadores = ", ".join("?" * len(filtros.colores_excluidos))
        # NOT IN devuelve NULL (y por tanto excluye la fila) si la columna es
        # NULL. Comprobado que no hay nulos en esta columna, asi que no hace
        # falta el COALESCE que haria falta en otro dataset.
        condiciones.append(f"perceived_colour_master_name NOT IN ({marcadores})")
        parametros.extend(filtros.colores_excluidos)

    if filtros.tono:
        condiciones.append("perceived_colour_value_name = ?")
        parametros.append(filtros.tono)

    if filtros.estampado:
        condiciones.append("graphical_appearance_name = ?")
        parametros.append(filtros.estampado)

    return " AND ".join(condiciones) or "1 = 1", parametros


def buscar_por_filtros(
    con: sqlite3.Connection,
    filtros: SearchFilters,
    limite: int = 24,
) -> list[dict]:
    """
    Aplica solo los filtros duros. Ignora `consulta_semantica` a proposito.

    Una prenda puede estar en la BD hasta en 10 variantes de color; sin
    agrupar, una busqueda de 24 resultados podria devolver 3 prendas.
    """
    where, parametros = construir_where(filtros)

    # GROUP BY product_code deja una fila por producto, y min(article_id)
    # elige cual de las variantes lo representa. No es un detalle estetico:
    # SQLite documenta que, cuando hay un min() o un max() en el SELECT, las
    # columnas sueltas del mismo SELECT toman su valor de ESA fila. Sin el
    # min(), prod_name y el color vendrian de una fila arbitraria del grupo y
    # podrian no corresponderse entre si.
    consulta = f"""
        SELECT min(article_id) AS article_id, {COLUMNAS_RESULTADO},
               count(*) AS variantes
        FROM articulos
        WHERE {where}
        GROUP BY product_code
        ORDER BY product_code
        LIMIT ?
    """

    filas = con.execute(consulta, [*parametros, limite]).fetchall()
    return [dict(fila) for fila in filas]


def contar_por_filtros(con: sqlite3.Connection, filtros: SearchFilters) -> int:
    """Cuantos productos distintos pasan los filtros. Para la etapa de relajado."""
    where, parametros = construir_where(filtros)
    consulta = f"SELECT count(DISTINCT product_code) FROM articulos WHERE {where}"
    return con.execute(consulta, parametros).fetchone()[0]
