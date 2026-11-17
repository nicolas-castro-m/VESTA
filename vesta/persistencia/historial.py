"""Seguimiento de predicciones (sección 12).

Cada predicción se guarda una sola vez y nunca se modifica. Cuando ya existe
el precio observado h sesiones después del último dato usado, se registra una
realización aparte con el error y la posición frente al intervalo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vesta.persistencia import db


def registrar(prediccion: dict) -> int:
    return db.guardar_prediccion(prediccion)


def posicion_intervalo(real: float, inferior: float | None, superior: float | None) -> str | None:
    if inferior is None or superior is None:
        return None
    if real < inferior:
        return "Debajo"
    if real > superior:
        return "Encima"
    return "Dentro"


def actualizar_realizados(ticker: str, precios: pd.DataFrame) -> int:
    """Completa las predicciones vencidas usando el precio ajustado h sesiones después del origen.

    Si los datos actuales no contienen exactamente la fecha de origen (por ejemplo,
    la fuente revisó la serie), la predicción queda pendiente en lugar de adivinar.
    """
    d = precios[["Date", "Adj.Close"]].sort_values("Date").reset_index(drop=True)
    fechas = pd.to_datetime(d["Date"]).dt.normalize()
    indice = {f: i for i, f in enumerate(fechas)}
    n = 0
    for pid, f_origen, h, p0, pred, inf_, sup, precio_pred in db.pendientes(ticker):
        i = indice.get(pd.Timestamp(f_origen).normalize())
        if i is None or i + h >= len(d):
            continue
        p0_actual = float(d["Adj.Close"].iloc[i])
        p1 = float(d["Adj.Close"].iloc[i + h])
        real = float(np.log(p1 / p0_actual))
        # El precio ajustado puede reescalarse por dividendos posteriores: el error de
        # precio se mide en la escala del registro original (p0 guardado).
        precio_real_escala = p0 * np.exp(real)
        db.guardar_realizacion({
            "prediccion_id": pid,
            "fecha_real": str(fechas.iloc[i + h].date()),
            "precio_real": precio_real_escala,
            "rend_real": real,
            "error_abs_precio": abs(precio_real_escala - precio_pred),
            "error_pct_precio": abs(precio_real_escala - precio_pred) / precio_real_escala * 100,
            "error_rend": real - pred,
            "acierto_direccion": None if pred == 0 or real == 0 else int(np.sign(real) == np.sign(pred)),
            "posicion_intervalo": posicion_intervalo(real, inf_, sup),
        })
        n += 1
    return n


def leer(ticker: str | None = None) -> pd.DataFrame:
    return db.leer_predicciones(ticker)


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    r = df.dropna(subset=["rend_real"])
    if r.empty:
        return pd.DataFrame()
    return (r.groupby(["ticker", "horizonte"])
            .agg(Evaluadas=("id", "count"),
                 MAE_rend=("error_rend", lambda e: float(np.mean(np.abs(e)))),
                 RMSE_rend=("error_rend", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                 Error_pct_precio=("error_pct_precio", "mean"),
                 Acierto_direccion=("acierto_direccion", lambda a: float(np.nanmean(a) * 100) if a.notna().any() else np.nan),
                 Dentro_intervalo=("posicion_intervalo", lambda p: float((p == "Dentro").mean() * 100)))
            .reset_index())
