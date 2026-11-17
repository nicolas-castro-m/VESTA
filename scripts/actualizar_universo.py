"""Actualiza el universo del S&P 500: lista de empresas, beta, perfil y precio reciente.

Uso (desde la carpeta del proyecto):
    python scripts/actualizar_universo.py            # solo si tiene más de 30 días
    python scripts/actualizar_universo.py --forzar   # recalcula siempre

Puede programarse con el Programador de tareas de Windows para que corra cada mes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesta.datos import universo as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forzar", action="store_true", help="Recalcular aunque el universo esté vigente")
    args = ap.parse_args()

    actual = U.cargar_universo()
    if not args.forzar and not U.universo_vencido(actual):
        print(f"El universo está vigente (calculado el {actual['fecha_consulta'].iloc[0]}). Use --forzar para recalcular.")
        return
    u = U.actualizar(progreso=lambda f, t: print(f"[{f:5.0%}] {t}", flush=True))
    print(f"\nActivos: {len(u)} · con beta: {u['beta'].notna().sum()}")
    print(u["perfil"].value_counts().to_string())


if __name__ == "__main__":
    main()
