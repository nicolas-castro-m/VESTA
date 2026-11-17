"""Crea la base local data/vesta.sqlite (o verifica que exista) y muestra su contenido.

Uso:  python scripts/inicializar_db.py
La aplicación también la crea automáticamente al iniciar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesta.persistencia import db  # noqa: E402

if __name__ == "__main__":
    version = db.inicializar()
    print(f"Base lista: {db.ruta_actual()} (esquema v{version})")
    for tabla, n in db.resumen_base().items():
        print(f"  {tabla:<18} {n}")
