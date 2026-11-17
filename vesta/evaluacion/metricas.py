"""Métricas de desempeño predictivo y pruebas de comparación de pronósticos."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _par(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(real) & np.isfinite(pred)
    return real[ok], pred[ok]


def mae(real, pred) -> float:
    r, p = _par(real, pred)
    return float(np.mean(np.abs(r - p))) if len(r) else np.nan


def rmse(real, pred) -> float:
    r, p = _par(real, pred)
    return float(np.sqrt(np.mean((r - p) ** 2))) if len(r) else np.nan


def mape(real, pred) -> float:
    r, p = _par(real, pred)
    ok = r != 0
    return float(np.mean(np.abs((r[ok] - p[ok]) / r[ok])) * 100) if ok.any() else np.nan


def sesgo(real, pred) -> float:
    """Sesgo = mean(Predicho - Real), como en el documento."""
    r, p = _par(real, pred)
    return float(np.mean(p - r)) if len(r) else np.nan


def direccion_documento(real, pred) -> float:
    """Definición literal del documento: mean(sign(Predicho) == sign(Real)) * 100."""
    r, p = _par(real, pred)
    return float(np.mean(np.sign(p) == np.sign(r)) * 100) if len(r) else np.nan


def acierto_direccion(real, pred) -> dict:
    """Acierto de dirección contando solo pronósticos con signo definido.

    Incluye la prueba binomial exacta frente al 50 % (H0: acertar es azar).
    """
    r, p = _par(real, pred)
    ok = (p != 0) & (r != 0)
    n = int(ok.sum())
    if n == 0:
        return {"aciertos": 0, "n": 0, "porcentaje": np.nan, "p_binomial": np.nan}
    k = int(np.sum(np.sign(p[ok]) == np.sign(r[ok])))
    prueba = stats.binomtest(k, n, 0.5, alternative="two-sided")
    return {"aciertos": k, "n": n, "porcentaje": 100 * k / n, "p_binomial": float(prueba.pvalue)}


def cobertura(real, inferior, superior) -> float:
    real = np.asarray(real, float)
    inf_ = np.asarray(inferior, float)
    sup = np.asarray(superior, float)
    ok = np.isfinite(real) & np.isfinite(inf_) & np.isfinite(sup)
    if not ok.any():
        return np.nan
    return float(np.mean((real[ok] >= inf_[ok]) & (real[ok] <= sup[ok])) * 100)


def theil_u(real, pred_modelo, pred_referencia) -> float:
    """Cociente de RMSE frente a la referencia (U de Theil relativa). < 1: el modelo mejora."""
    r1 = rmse(real, pred_modelo)
    r2 = rmse(real, pred_referencia)
    return float(r1 / r2) if r2 and np.isfinite(r2) and r2 > 0 else np.nan


def _autocov(x: np.ndarray, lag_max: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    return np.array([np.dot(x[k:], x[:n - k]) / n for k in range(lag_max + 1)])


def diebold_mariano(e1, e2, h: int = 1, potencia: int = 2) -> dict:
    """Prueba de Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold.

    Réplica de `forecast::dm.test(e1, e2, h, power)`: e1 son los errores del
    modelo y e2 los de la referencia. Estadístico negativo => el modelo tiene
    menor pérdida. Si la varianza de largo plazo sale negativa se repite con h = 1.
    """
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = np.abs(e1[ok]) ** potencia - np.abs(e2[ok]) ** potencia
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return {"estadistico": np.nan, "p_valor": np.nan, "h_usado": h,
                "nota": "Los errores de ambos pronósticos son idénticos."}
    gamma = _autocov(d, h - 1)
    var = (gamma[0] + 2 * gamma[1:].sum()) / n
    nota = ""
    if var <= 0:
        if h == 1:
            return {"estadistico": np.nan, "p_valor": np.nan, "h_usado": h, "nota": "Varianza nula."}
        res = diebold_mariano(e1, e2, 1, potencia)
        res["nota"] = "Varianza negativa: se usó h = 1, como en dm.test."
        return res
    stat = d.mean() / np.sqrt(var)
    k = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    stat *= k
    p = 2 * stats.t.cdf(-abs(stat), df=n - 1)
    return {"estadistico": float(stat), "p_valor": float(p), "h_usado": h, "nota": nota}
