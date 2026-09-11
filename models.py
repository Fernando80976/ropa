"""
Esquema de filtros del buscador de ropa.

Este modelo es el contrato entre el LLM y la base de datos: el LLM recibe una
frase en lenguaje natural y debe devolver exactamente esta estructura.

Regla de diseño: un campo solo es filtro duro si sus valores son POCOS,
ORTOGONALES entre si y NO AMBIGUOS. Todo lo demas (estilo, ocasion, corte,
tipo concreto de prenda) va a `consulta_semantica` y se resuelve por embeddings.

`consulta_semantica` se pide EN INGLES aunque el usuario escriba en espanol.
El LLM ya esta reescribiendo la frase, asi que traducirla no le cuesta nada, y
a cambio el buscador compara ingles contra ingles (las descripciones del
catalogo lo estan) en vez de cruzar dos idiomas. Eso permite usar un modelo de
embeddings solo-ingles, que ademas de ser mejor en su idioma ocupa 99 MB en
memoria frente a los 451 MB del multilingue equivalente: la diferencia entre
caber o no caber en un servidor gratuito.

Los valores de los Literal son los del dataset (en ingles) porque son los que
estan literalmente en la BD. Las descripciones estan en espanol porque acaban
en el JSON Schema que se le pasa al modelo, y ahi si importa el idioma del
usuario.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Vocabularios cerrados (extraidos de articles.csv)
# ---------------------------------------------------------------------------

# index_group_name (5 valores). Ojo: mezcla publico con categoria deportiva.
Publico = Literal[
    "Ladieswear",     # mujer adulta
    "Menswear",       # hombre adulto
    "Divided",        # linea joven / teen, sin genero claro
    "Baby/Children",  # bebe y nino
    "Sport",          # ropa deportiva (NO es un publico, ver nota abajo)
]

# product_group_name, depurado: fuera Furniture, Stationery, Cosmetic,
# Interior textile, Fun, Items, Garment and Shoe care, Bags, Unknown.
Categoria = Literal[
    "Garment Upper body",   # camisetas, camisas, jerseis, chaquetas
    "Garment Lower body",   # pantalones, faldas, shorts
    "Garment Full body",    # vestidos, monos, conjuntos
    "Accessories",
    "Underwear",
    "Shoes",
    "Swimwear",
    "Socks & Tights",
    "Nightwear",
]

# perceived_colour_master_name (20 valores), sin Unknown ni undefined.
# Preferido sobre colour_group_name, que tiene 50 valores con ruido
# del tipo "Other Turquoise" o "Light Red".
Color = Literal[
    "Black", "White", "Grey", "Beige", "Brown", "Mole",
    "Blue", "Turquoise", "Green", "Khaki green", "Yellowish Green",
    "Bluish Green", "Yellow", "Orange", "Red", "Pink", "Lilac Purple",
    "Metal",
]

# perceived_colour_value_name (8 valores), sin Undefined ni Unknown.
# Es la luminosidad, ortogonal al color: "azul oscuro" = Blue + Dark.
Tono = Literal["Dark", "Light", "Bright", "Dusty Light", "Medium", "Medium Dusty"]

# graphical_appearance_name (30 valores), depurado a los que un usuario
# nombraria de verdad. El resto (Slub, Neps, Chambray, Argyle...) es jerga
# textil interna y se queda para la parte semantica.
Estampado = Literal[
    "Solid",                # liso
    "All over pattern",     # estampado integral
    "Stripe",               # rayas
    "Check",                # cuadros
    "Dot",                  # lunares
    "Denim",                # vaquero
    "Melange",              # jaspeado
    "Lace",                 # encaje
    "Embroidery",           # bordado
    "Sequin",               # lentejuelas
    "Glittering/Metallic",  # brillante
    "Front print",          # estampado frontal
    "Transparent",
    "Mesh",
]


# ---------------------------------------------------------------------------
# El contrato
# ---------------------------------------------------------------------------

class SearchFilters(BaseModel):
    """Filtros extraidos de una consulta en lenguaje natural."""

    publico: Publico | None = Field(
        default=None,
        description=(
            "Para quien es la prenda. 'Ladieswear' mujer, 'Menswear' hombre, "
            "'Divided' linea joven o estilo urbano, 'Baby/Children' ninos y "
            "bebes, 'Sport' ropa deportiva. Deja null si la consulta no lo "
            "indica ni se puede deducir."
        ),
    )

    categoria: Categoria | None = Field(
        default=None,
        description=(
            "Zona del cuerpo o familia de producto. Camisetas, camisas, "
            "jerseis y chaquetas son 'Garment Upper body'. Pantalones, faldas "
            "y shorts, 'Garment Lower body'. Vestidos y monos, 'Garment Full "
            "body'. No intentes precisar mas de esto: el tipo exacto de prenda "
            "va en consulta_semantica."
        ),
    )

    colores: list[Color] = Field(
        default_factory=list,
        description=(
            "Colores pedidos explicitamente. Usa la familia de color, no el "
            "matiz: 'azul marino' y 'celeste' son ambos 'Blue' (el matiz lo "
            "aporta el campo tono)."
        ),
    )

    colores_excluidos: list[Color] = Field(
        default_factory=list,
        description="Colores que el usuario rechaza ('que no sea negro').",
    )

    tono: Tono | None = Field(
        default=None,
        description=(
            "Luminosidad, independiente del color. 'oscuro' es 'Dark', "
            "'claro' o 'pastel' es 'Light' o 'Dusty Light', 'vivo' o "
            "'chillon' es 'Bright'."
        ),
    )

    estampado: Estampado | None = Field(
        default=None,
        description=(
            "Acabado o patron visual. 'liso' es 'Solid', 'vaquero' o "
            "'denim' es 'Denim', 'de rayas' es 'Stripe'."
        ),
    )

    consulta_semantica: str = Field(
        description=(
            "TODO lo que no ha entrado en los campos anteriores, reescrito "
            "como una descripcion breve de la prenda EN INGLES: tipo "
            "concreto, corte, ocasion, tejido, estilo. Ejemplo: para "
            "'chaqueta vaquera oversize para una boda en verano que no sea "
            "negra', aqui iria 'oversized lightweight jacket for an event'. "
            "Escribelo siempre en ingles aunque la consulta venga en espanol. "
            "Nunca lo dejes vacio: si la consulta era puramente de filtros, "
            "repite aqui el tipo de prenda."
        ),
    )


# ---------------------------------------------------------------------------
# Notas de diseno (utiles para el README)
# ---------------------------------------------------------------------------
#
# 1. product_type_name (131 valores) NO es filtro. Meter 131 opciones en el
#    prompt encarece cada llamada y aumenta el riesgo de que el modelo elija
#    una categoria vecina pero equivocada. El tipo concreto se resuelve por
#    similitud semantica, que es justo lo que hace bien.
#
# 2. section_name (56) y department_name (250) son taxonomia comercial interna
#    de H&M ("Womens Trend", "Divided Projects"). Un usuario no habla asi.
#    Fuera.
#
# 3. garment_group_name (21) parecia candidato, pero mezcla criterios
#    ("Jersey Fancy", "Dressed", "Special Offers"). Descartado por ambiguo.
#
# 4. 'Sport' vive dentro de `publico` porque el dataset lo mete en
#    index_group_name, aunque conceptualmente sea ortogonal al genero. Es una
#    limitacion de la fuente, no del diseno: filtrar por Sport pierde el
#    genero y viceversa.
#
# 5. No hay precio en articles.csv. Vive en transactions_train.csv como precio
#    por transaccion y normalizado. Queda fuera de la v1.

#
# 6. El coste de pedir consulta_semantica en ingles es que el camino de
#    emergencia empeora: si Groq no responde, extractor.py cae al fallback y
#    manda la frase del usuario en espanol a un modelo que solo entiende
#    ingles. Se acepta porque ese camino ya era el peor (se queda sin ningun
#    filtro duro) y sigue devolviendo algo en vez de nada.
