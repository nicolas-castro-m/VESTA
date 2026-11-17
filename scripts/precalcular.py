"""Precalcula la modelación (selección y evaluación rodante) de varios activos.

Al abrir esos activos en la aplicación, «Generar predicción» responde de inmediato.
No registra predicciones: eso ocurre solo cuando el usuario las genera.

Uso:
    python scripts/precalcular.py MSFT KO TSLA
    python scripts/precalcular.py --muestra            # los 21 activos de la fase 2
    python scripts/precalcular.py --muestra --replica  # incluye la réplica de la sección 12.10 (lenta)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesta import ajustes as A  # noqa: E402
from vesta import servicios as S  # noqa: E402
from vesta.datos import universo as U  # noqa: E402


def procesar(ticker: str, replica: bool) -> str:
    t0 = time.time()
    aj = A.cargar()
    datos = S.cargar_activo(ticker)
    inf = S.revisar_calidad(datos)
    if not inf.apta:
        return f"{ticker}: omitido ({' '.join(inf.errores)})"
    m = S.modelacion(ticker, datos.precios, aj)
    S.actualizar_seguimiento(ticker, datos.precios)
    if replica:
        S.replica_documento(ticker, datos.precios)
    return f"{ticker}: {m.seleccion.mejor.etiqueta} ({time.time() - t0:.0f} s)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--muestra", action="store_true", help="Usar la muestra estratificada de la fase 2")
    ap.add_argument("--replica", action="store_true", help="Calcular también la réplica de la sección 12.10")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    tickers = [t.upper() for t in args.tickers]
    if args.muestra:
        muestra, origen = U.cargar_muestra(U.cargar_universo())
        if muestra is None:
            sys.exit("No hay universo: ejecute primero scripts/actualizar_universo.py")
        print(f"Muestra: {origen}")
        tickers += muestra["ticker"].tolist()
    if not tickers:
        sys.exit("Indique tickers o use --muestra")
    with ProcessPoolExecutor(max_workers=args.procesos) as ex:
        futuros = {ex.submit(procesar, t, args.replica): t for t in dict.fromkeys(tickers)}
        for f in as_completed(futuros):
            try:
                print(f.result(), flush=True)
            except Exception as exc:
                print(f"{futuros[f]}: ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
