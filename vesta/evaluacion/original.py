"""Réplica literal de la evaluación publicada en el documento (secciones 12.6 y 12.10).

Se conserva tal cual para trazabilidad, con sus limitaciones documentadas en
docs/METODOLOGIA.md (horizontes 1/21/63 en la caminata aleatoria, métricas
en escalas distintas, pronóstico multipaso desde el final del entrenamiento y
rendimientos solapados). La evaluación recomendada está en `rodante.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROP_TRAIN = 0.80   # sección 12.6 y 12.10
from vesta.evaluacion import metricas as M
from vesta.modelos.arma import pronosticar
from vesta.modelos.seleccion import auto_arima
from vesta.procesamiento.rendimientos import preparar_horizontes

HORIZONTES_RW_DOCUMENTO = {"Diario": 1, "Mensual": 21, "Trimestral": 63}


def evaluar_rw_documento(datos: pd.DataFrame, nombre: str, proporcion: float = PROP_TRAIN):
    """`evaluar_rw` (sección 12.6): caminata aleatoria sobre el precio ajustado."""
    d = datos[["Date", "Adj.Close"]].dropna().sort_values("Date").reset_index(drop=True)
    n = len(d)
    corte = int(np.floor(n * proporcion))
    precio = d["Adj.Close"].to_numpy(float)
    filas, metr = [], []
    for nombre_h, h in HORIZONTES_RW_DOCUMENTO.items():
        origenes = np.arange(corte, n - h + 1)            # seq(corte, n - h) en base 1
        idx = origenes - 1
        pred = precio[idx]
        real = precio[idx + h]
        filas.append(pd.DataFrame({"Activo": nombre, "Horizonte": nombre_h,
                                   "Fecha_Origen": d["Date"].iloc[idx].to_numpy(),
                                   "Fecha_Pronostico": d["Date"].iloc[idx + h].to_numpy(),
                                   "Predicho": pred, "Real": real}))
        metr.append({"Activo": nombre, "Horizonte": nombre_h, "MAE": M.mae(real, pred),
                     "RMSE": M.rmse(real, pred), "MAPE": M.mape(real, pred), "Observaciones": len(real)})
    return pd.concat(filas, ignore_index=True), pd.DataFrame(metr)


def evaluar_arima_documento(datos: pd.DataFrame, nombre: str, progreso=None):
    """`ajustar_auto_arima` + métricas de la sección 12.10, para los tres horizontes."""
    series = preparar_horizontes(datos, nombre)
    modelos, predicciones, metr, candidatos = [], [], [], {}
    for i, (h, s) in enumerate(series.items()):
        def prog(frac, texto, i=i, h=h):
            if progreso:
                progreso((i + frac) / len(series), f"{h}: {texto}")

        # Literal: max.p = max.q = 5, max.order = 10, BIC, allowmean = TRUE (sección 12.10)
        sel = auto_arima(s.train, d=0, max_p=5, max_q=5, max_orden=10, permitir_media=True, criterio="bic",
                         progreso=prog)
        mod = sel.mejor
        candidatos[h] = sel.candidatos
        pr = pronosticar(mod, len(s.test), 0.95)
        modelos.append({"Activo": nombre, "Horizonte": h, "Modelo": f"ARIMA({mod.p},0,{mod.q})",
                        "p": mod.p, "d": 0, "q": mod.q, "Media": mod.con_media, "AIC": mod.aic, "BIC": mod.bic,
                        "N_train": len(s.train), "N_test": len(s.test), **mod.coeficientes()})
        pred = pd.DataFrame({"Activo": nombre, "Horizonte": h, "Fecha": s.fechas_test,
                             "Real": s.test, "Predicho": pr.media, "Inferior": pr.inferior, "Superior": pr.superior})
        predicciones.append(pred)
        metr.append({"Activo": nombre, "Horizonte": h, "MAE": M.mae(s.test, pr.media),
                     "RMSE": M.rmse(s.test, pr.media), "Sesgo": M.sesgo(s.test, pr.media),
                     "Direccion": M.direccion_documento(s.test, pr.media),
                     "Pct_reales_positivos": float(np.mean(s.test > 0) * 100),
                     "Cobertura_95": M.cobertura(s.test, pr.inferior, pr.superior),
                     "Observaciones": len(s.test)})
    return {"modelos": pd.DataFrame(modelos), "predicciones": pd.concat(predicciones, ignore_index=True),
            "metricas": pd.DataFrame(metr), "candidatos": candidatos}
