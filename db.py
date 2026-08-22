"""
Conexion a SQLite y creacion del esquema.

Dos responsabilidades y nada mas: abrir la BD con la extension sqlite-vec ya
cargada, y crear las tablas si no existen. Ningun otro modulo debe abrir
conexiones por su cuenta.

El esquema son dos tablas:

  articulos      una fila por article_id (variante de color de un producto).
                 Es lo que se filtra con WHERE.
  vec_productos  una fila por product_code, con el embedding de 384 dims.
                 Es lo que se ordena por similitud.

Estan separadas a proposito: el catalogo tiene duplicados por variante de
color (45.875 prod_name unicos frente a 105.542 filas), asi que embeber por
fila seria calcular el mismo vector varias veces. Se embebe una vez por
producto y se une por product_code.
"""

import os
import sqlite3

import sqlite_vec
from dotenv import load_dotenv

load_dotenv()

# Ruta por defecto para no depender de que .env exista al importar el modulo.
RUTA_BD = os.getenv("DB_PATH", "ropa.db")

# Dimensiones del modelo de embeddings multilingue (ver embeddings.py).
# Vive aqui porque la tabla virtual necesita el tamano fijo al crearse.
DIMENSIONES = 384


def conectar(ruta: str | None = None) -> sqlite3.Connection:
    """Abre la BD con sqlite-vec cargado y filas accesibles por nombre."""
    con = sqlite3.connect(ruta or RUTA_BD)

    # La carga de extensiones se habilita solo el tiempo justo: dejarla
    # abierta permitiria cargar cualquier .dll desde SQL.
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    # sqlite3.Row permite fila["prod_name"] en vez de fila[2], que es lo que
    # hace legibles las consultas de buscador.py.
    con.row_factory = sqlite3.Row
    return con


def crear_esquema(con: sqlite3.Connection) -> None:
    """Crea tablas e indices si no existen. Es idempotente."""

    # Solo las columnas que usa el proyecto. Las otras 16 del CSV
    # (department_name, section_name, los codigos numericos...) se descartan
    # en el ingest: no son filtro ni se muestran.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS articulos (
            article_id                   INTEGER PRIMARY KEY,
            product_code                 INTEGER NOT NULL,
            prod_name                    TEXT    NOT NULL,
            product_type_name            TEXT,
            product_group_name           TEXT,    -- filtro: categoria
            index_group_name             TEXT,    -- filtro: publico
            perceived_colour_master_name TEXT,    -- filtro: colores
            perceived_colour_value_name  TEXT,    -- filtro: tono
            graphical_appearance_name    TEXT,    -- filtro: estampado
            detail_desc                  TEXT
        )
        """
    )

    # Un indice por columna filtrable. Son indices sueltos y no compuestos
    # porque el WHERE se construye dinamicamente segun que campos rellene el
    # LLM: no hay un orden de columnas fijo al que ajustar un indice compuesto.
    for columna in (
        "product_code",
        "product_group_name",
        "index_group_name",
        "perceived_colour_master_name",
        "perceived_colour_value_name",
        "graphical_appearance_name",
    ):
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_articulos_{columna} "
            f"ON articulos({columna})"
        )

    # Tabla virtual de sqlite-vec. product_code es la clave primaria entera,
    # asi que el resultado de una busqueda vectorial se une directamente
    # contra articulos.product_code sin tabla intermedia.
    con.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_productos USING vec0(
            product_code INTEGER PRIMARY KEY,
            embedding    FLOAT[{DIMENSIONES}]
        )
        """
    )

    con.commit()


if __name__ == "__main__":
    # Comprobacion a mano: python db.py
    con = conectar()
    crear_esquema(con)

    version, = con.execute("SELECT vec_version()").fetchone()
    print(f"sqlite-vec {version}")
    print(f"BD: {RUTA_BD}\n")

    print("Tablas:")
    for fila in con.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table', 'index') ORDER BY type, name"
    ):
        print(f"  {fila['type']:<5} {fila['name']}")

    print("\nColumnas de articulos:")
    for fila in con.execute("PRAGMA table_info(articulos)"):
        print(f"  {fila['name']:<30} {fila['type']}")

    con.close()
