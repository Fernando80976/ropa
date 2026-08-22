"""
Carga articles.csv en la tabla `articulos`. Script de un solo uso.

    python ingest.py                 # muestra de 5000 filas (desarrollo)
    python ingest.py --limite 20000
    python ingest.py --limite 0      # catalogo completo

Es idempotente: se puede ejecutar tantas veces como haga falta sin duplicar,
porque inserta con OR REPLACE sobre article_id, que es la clave primaria.
"""

import argparse
from typing import get_args

import pandas as pd

from db import conectar, crear_esquema
from models import Categoria

RUTA_CSV = "articles.csv"

# Filas por lote en el executemany. 1000 es suficiente para que el coste por
# llamada se diluya; subirlo mas no se nota y gasta memoria.
LOTE = 1000

# Las 10 columnas que usa el proyecto, de las 25 del CSV. Mismo orden que en
# la tabla, para que el INSERT sea posicional y no haya que nombrarlas dos veces.
COLUMNAS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "index_group_name",
    "perceived_colour_master_name",
    "perceived_colour_value_name",
    "graphical_appearance_name",
    "detail_desc",
]

# Lista blanca, no lista negra: se acepta lo que models.Categoria declara y se
# descarta todo lo demas. Asi la BD nunca contiene un valor que el LLM no pueda
# emitir, y anadir una categoria en models.py basta para que entre aqui.
# Consecuencia: ademas de las categorias que no son ropa (Furniture, Cosmetic,
# Bags...) cae tambien "Underwear/nightwear", 54 filas que el CSV usa como
# etiqueta suelta y que no tienen equivalente en el contrato.
CATEGORIAS_DE_ROPA = set(get_args(Categoria))


def cargar_csv(ruta: str, limite: int) -> pd.DataFrame:
    """Lee el CSV quedandose con las columnas y las filas necesarias."""
    # Se leen las primeras `limite` filas, sin muestreo aleatorio: comprobado
    # que el reparto por publico de las primeras 5000 se parece al del catalogo
    # completo (todos los grupos presentes), y asi no hay que leer 36 MB para
    # trabajar con una muestra.
    return pd.read_csv(
        ruta,
        usecols=COLUMNAS,
        nrows=limite if limite > 0 else None,
    )


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta lo que no es ropa y lo que no se puede embeber."""
    df = df[df["product_group_name"].isin(CATEGORIAS_DE_ROPA)]

    # detail_desc es el texto principal del embedding. Sin el, el producto
    # nunca apareceria en una busqueda semantica: mejor no tenerlo en la BD.
    return df.dropna(subset=["detail_desc"])


def insertar(con, df: pd.DataFrame) -> int:
    """Inserta el DataFrame por lotes. Devuelve el numero de filas enviadas."""
    # astype(object) + where convierte los NaN de las columnas opcionales en
    # None, que es lo que sqlite3 sabe guardar como NULL.
    filas = df[COLUMNAS].astype(object).where(df[COLUMNAS].notna(), None).values.tolist()

    marcadores = ", ".join("?" * len(COLUMNAS))
    sentencia = f"INSERT OR REPLACE INTO articulos VALUES ({marcadores})"

    for inicio in range(0, len(filas), LOTE):
        con.executemany(sentencia, filas[inicio:inicio + LOTE])

    con.commit()
    return len(filas)


def resumen(con) -> None:
    """Imprime lo que ha quedado en la BD, para verificarlo a ojo."""
    total, productos = con.execute(
        "SELECT count(*), count(DISTINCT product_code) FROM articulos"
    ).fetchone()

    print(f"\nEn la BD: {total} filas, {productos} product_code distintos")
    print(f"({total - productos} filas son variantes de color de un producto ya presente)")

    print("\nReparto por publico:")
    for fila in con.execute(
        "SELECT index_group_name, count(*) AS n FROM articulos "
        "GROUP BY index_group_name ORDER BY n DESC"
    ):
        print(f"  {fila['index_group_name']:<15} {fila['n']:>6}")

    print("\nReparto por categoria:")
    for fila in con.execute(
        "SELECT product_group_name, count(*) AS n FROM articulos "
        "GROUP BY product_group_name ORDER BY n DESC"
    ):
        print(f"  {fila['product_group_name']:<22} {fila['n']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga articles.csv en ropa.db")
    parser.add_argument(
        "--limite",
        type=int,
        default=5000,
        help="Filas del CSV a leer. 0 = todas (105.542).",
    )
    parser.add_argument("--csv", default=RUTA_CSV, help="Ruta del CSV.")
    args = parser.parse_args()

    print(f"Leyendo {args.csv} (limite: {args.limite or 'sin limite'})...")
    df = cargar_csv(args.csv, args.limite)
    leidas = len(df)

    df = limpiar(df)
    print(f"Leidas {leidas} filas, descartadas {leidas - len(df)} "
          f"(no son ropa o no tienen detail_desc)")

    con = conectar()
    crear_esquema(con)  # por si se ejecuta el ingest sin haber corrido db.py
    insertadas = insertar(con, df)
    print(f"Insertadas {insertadas} filas")

    resumen(con)
    con.close()


if __name__ == "__main__":
    main()
