"""Compara los resultados de VESTA con los exportados desde R.

Uso:
    Rscript validacion/exportar_referencia.R MSFT KO
    python validacion/comparar_con_r.py MSFT KO
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vesta.datos import almacen  # noqa: E402
from vesta.estadistica import pruebas as P  # noqa: E402
from vesta.modelos.seleccion import auto_arima  # noqa: E402
from vesta.procesamiento.rendimientos import limpiar_precios, preparar_serie  # noqa: E402


def valores_vesta(ticker: str) -> dict:
    d = limpiar_precios(pd.read_csv(almacen._archivo(ticker)))
    d = almacen.recortar_ventana(d, 5)
    s = preparar_serie(d, ticker)
    lp, r = s.log_precio, s.rentabilidad[1:]
    sel = auto_arima(s.r_train, d=1, max_p=3, max_q=3, permitir_media=False, criterio="bic").mejor
    return {
        "rachas": P.rachas(r).p_valor, "lilliefors": P.lilliefors(r).p_valor,
        "adf_nivel": P.adf(lp).p_valor, "adf_nivel_stat": P.adf(lp).estadistico, "adf_dif": P.adf(r).p_valor,
        "kpss_nivel": P.kpss(lp).p_valor, "kpss_nivel_stat": P.kpss(lp).estadistico, "kpss_dif": P.kpss(r).p_valor,
        "ljung10": P.ljung_box(r, 10).p_valor, "ljung20": P.ljung_box(r, 20).p_valor, "arch10": P.arch_lm(r, 10).p_valor,
        "arima_p": sel.p, "arima_q": sel.q, "arima_bic": sel.bic, "arima_aic": sel.aic,
    }


def main():
    tickers = sys.argv[1:] or ["MSFT"]
    filas = []
    for tk in tickers:
        ref = pd.read_csv(RAIZ / "validacion" / "referencia_r" / f"{tk}.csv").set_index("prueba")["valor"]
        vv = valores_vesta(tk)
        for k, v_r in ref.items():
            v_p = vv.get(k, np.nan)
            filas.append({"ticker": tk, "medida": k, "R": v_r, "VESTA": v_p,
                          "diferencia": abs(v_p - v_r), "coincide": bool(np.isclose(v_p, v_r, rtol=1e-3, atol=1e-4))})
    t = pd.DataFrame(filas)
    print(t.to_string(index=False))
    print(f"\nCoinciden {t['coincide'].sum()} de {len(t)} medidas (tolerancia relativa 1e-3).")


if __name__ == "__main__":
    main()
