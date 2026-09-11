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

# Columnas que se devuelven al front.
# Van cualificadas con "a." porque la busqueda semantica hace JOIN contra
# vec_productos, que tambien tiene una columna product_code.
#
# article_id no esta en la lista: lo anade cada SELECT como min(article_id).
# Se devuelve porque es el identificador estable de la variante concreta; el
# dataset de Kaggle trae las fotos en una carpeta aparte que aqui no se ha
# descargado, asi que el front no las muestra.
COLUMNAS_RESULTADO = """
    a.product_code,
    a.prod_name,
    a.product_type_name,
    a.product_group_name,
    a.index_group_name,
    a.perceived_colour_master_name,
    a.perceived_colour_value_name,
    a.graphical_appearance_name,
    a.detail_desc
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
        SELECT min(a.article_id) AS article_id, {COLUMNAS_RESULTADO},
               count(*) AS variantes
        FROM articulos a
        WHERE {where}
        GROUP BY a.product_code
        ORDER BY a.product_code
        LIMIT ?
    """

    filas = con.execute(consulta, [*parametros, limite]).fetchall()
    return [dict(fila) for fila in filas]


def contar_por_filtros(con: sqlite3.Connection, filtros: SearchFilters) -> int:
    """Cuantos productos distintos pasan los filtros. Para la etapa de relajado."""
    where, parametros = construir_where(filtros)
    consulta = f"SELECT count(DISTINCT product_code) FROM articulos WHERE {where}"
    return con.execute(consulta, parametros).fetchone()[0]


# ---------------------------------------------------------------------------
# Capa semantica (etapa 4)
# ---------------------------------------------------------------------------

def buscar_semantica(
    con: sqlite3.Connection,
    texto: str,
    limite: int = 24,
) -> list[dict]:
    """
    Productos mas parecidos al texto, sin aplicar ningun filtro duro.

    Devuelve los datos del producto mas su `distancia` (menor = mas parecido).
    Solo se usa suelta para depurar: la busqueda real la hace `buscar()`,
    que primero filtra y luego ordena lo que sobrevive.
    """
    # Import diferido: cargar embeddings.py arrastra el modelo de 450 MB, y
    # buscar_por_filtros tiene que poder usarse sin pagar eso.
    from embeddings import embeder_consulta

    # MATCH + k = ... es la sintaxis de KNN de sqlite-vec: recorre el indice
    # vectorial, no la tabla entera.
    #
    # El KNN va aislado en un CTE porque vec0 solo admite consultas muy
    # simples: exige un "ORDER BY distance" a secas y no tolera JOIN ni
    # GROUP BY en la misma consulta. Se sacan primero los k vecinos y se
    # adorna despues.
    filas = con.execute(
        f"""
        WITH vecinos AS (
            SELECT product_code, distance
            FROM vec_productos
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        )
        SELECT v.distance AS distancia, min(a.article_id) AS article_id,
               {COLUMNAS_RESULTADO}, count(*) AS variantes
        FROM vecinos v
        JOIN articulos a ON a.product_code = v.product_code
        GROUP BY a.product_code
        ORDER BY v.distance
        """,
        (embeder_consulta(texto), limite),
    ).fetchall()

    return [dict(fila) for fila in filas]


# ---------------------------------------------------------------------------
# Fusion: filtros duros + ranking semantico (etapa 6)
# ---------------------------------------------------------------------------

# Orden en que se van soltando los filtros cuando no hay resultados, del menos
# al mas importante. La idea: quitar primero lo que el usuario mencionó de
# pasada y dejar para el final lo que, si se ignora, convierte el resultado en
# algo que no ha pedido.
#
#   estampado  suele ser un adorno ("de rayas"); sin el la prenda sigue valiendo
#   tono       "azul oscuro" -> "azul" es una perdida pequena
#   colores    ya es una peticion explicita, pero hay alternativas
#   categoria  cambiar de zona del cuerpo ya es otra prenda
#   publico    el ultimo: ensenar ropa de bebe a quien pidio de hombre es el
#              peor fallo posible, y un tercio del catalogo es Baby/Children
ORDEN_DE_RELAJADO = ["estampado", "tono", "colores", "categoria", "publico"]


def _sin_campo(filtros: SearchFilters, campo: str) -> SearchFilters:
    """Copia de los filtros con un campo vaciado."""
    vacio = [] if campo == "colores" else None
    return filtros.model_copy(update={campo: vacio})


def relajar(
    con: sqlite3.Connection,
    filtros: SearchFilters,
) -> tuple[SearchFilters, list[str]]:
    """
    Suelta filtros hasta que haya resultados. Devuelve los filtros y cuales cayeron.

    `colores_excluidos` no se relaja nunca: es lo unico que el usuario ha
    pedido en negativo ("que no sea negra"), y devolverle justo eso es peor
    que no devolver nada.
    """
    relajados: list[str] = []

    for campo in ORDEN_DE_RELAJADO:
        if contar_por_filtros(con, filtros) > 0:
            break

        # Si el campo ya estaba vacio, soltarlo no cambia nada: no se anuncia
        # al usuario un filtro que nunca aplicó.
        if not getattr(filtros, campo):
            continue

        filtros = _sin_campo(filtros, campo)
        relajados.append(campo)

    return filtros, relajados


def buscar(con: sqlite3.Connection, frase: str, limite: int = 24) -> dict:
    """
    Busqueda completa: frase en lenguaje natural -> resultados.

    El orden importa y es la decision de diseno del proyecto: los filtros van
    primero y el ranking semantico despues, solo sobre lo que sobrevive. Al
    reves (buscar los k mas parecidos y filtrarlos luego) los filtros se
    comerian parte de los k y una consulta muy filtrada devolveria casi nada.
    """
    from embeddings import embeder_consulta
    from extractor import extraer_filtros

    filtros_pedidos = extraer_filtros(frase)
    filtros, relajados = relajar(con, filtros_pedidos)

    where, parametros = construir_where(filtros)

    # El subconjunto que pasa los filtros se ordena por distancia calculandola
    # contra todos sus vectores, en vez de con el KNN de sqlite-vec. El KNN
    # devuelve los k mas parecidos del catalogo ENTERO, que luego habria que
    # filtrar: con filtros restrictivos casi ninguno sobrevive. Aqui la
    # distancia se calcula solo sobre los candidatos, asi que el resultado es
    # exacto. Comprobado que da las mismas distancias que el KNN.
    consulta = f"""
        WITH candidatos AS (
            SELECT min(a.article_id) AS article_id, {COLUMNAS_RESULTADO},
                   count(*) AS variantes
            FROM articulos a
            WHERE {where}
            GROUP BY a.product_code
        )
        SELECT c.*, vec_distance_L2(v.embedding, ?) AS distancia
        FROM candidatos c
        JOIN vec_productos v ON v.product_code = c.product_code
        ORDER BY distancia
        LIMIT ?
    """

    vector = embeder_consulta(filtros.consulta_semantica)
    filas = con.execute(consulta, [*parametros, vector, limite]).fetchall()

    return {
        "frase": frase,
        "total": len(filas),
        # Los pedidos y los aplicados se devuelven por separado para que el
        # front pueda ensenar que se pidio y que se solto. Es lo que hace la
        # demo explicable en vez de magica.
        "filtros_pedidos": filtros_pedidos.model_dump(),
        "filtros_aplicados": filtros.model_dump(),
        "filtros_relajados": relajados,
        "resultados": [dict(fila) for fila in filas],
    }
