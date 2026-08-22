"""
Existe para que pytest anada la raiz del proyecto a sys.path.

Sin este fichero, `from tests/ import db` falla: pytest solo mete en sys.path
el directorio del propio test (tests/), no la raiz, y los modulos del proyecto
viven en la raiz. No hace falta que contenga nada mas.
"""
