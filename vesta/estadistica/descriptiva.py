"""Medidas descriptivas (secciones 7.4 y 8) e indicadores de riesgo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vesta.config import HORIZONTES
from vesta.procesamiento.rendimientos import VARIABLE_HORIZONTE, preparar_activo


def resumen(x) -> dict:
    """Media, Mediana, Desv_Estandar, Mínimo y Máximo, como `resumen_activos`."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, Media=np.nan, Mediana=np.nan, Desv_Estandar=np.nan, Minimo=np.nan, Maximo=np.nan)
    return dict(n=len(x), Media=x.mean(), Mediana=np.median(x), Desv_Estandar=x.std(ddof=1),
                Minimo=x.min(), Maximo=x.max())


def resumen_ampliado(x) -> dict:
    """Sección 9.1: media, mediana, desviación, varianza, extremos, cuartiles, RIC, asimetría y curtosis.

    Cuartiles con el tipo 7 de R (el de `quantile` por defecto). Asimetría y
    curtosis en exceso con corrección por sesgo muestral.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    base = resumen(x)
    if len(x) < 4:
        return base
    q1, q3 = np.quantile(x, [0.25, 0.75])
    base.update(
        Varianza=float(x.var(ddof=1)),
        Q1=float(q1), Q3=float(q3), RIC=float(q3 - q1),
        Asimetria=float(stats.skew(x, bias=False)),
        Curtosis_exceso=float(stats.kurtosis(x, bias=False)),
        Pct_positivos=float(np.mean(x > 0) * 100),
    )
    return base


def resumen_horizontes(datos: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Reconstrucción de `resumen_horizontes` (la función está oculta en el HTML).

    Las columnas coinciden con la tabla publicada en la sección 8.
    """
    d = preparar_activo(datos) if "rent_sem" not in datos else datos
    filas = []
    for h in HORIZONTES:
        r = resumen(d[VARIABLE_HORIZONTE[h]])
        filas.append({"Activo": nombre, "Horizonte": h, **{k: v for k, v in r.items() if k != "n"}})
    return pd.DataFrame(filas)


def indicadores_riesgo(rent_diaria, dias_anio: int = 252) -> dict:
    """Indicadores para el usuario general a partir de rendimientos log diarios."""
    r = np.asarray(rent_diaria, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return {}
    vol_anual = r.std(ddof=1) * np.sqrt(dias_anio)
    rent_anual = r.mean() * dias_anio
    nivel = np.exp(np.cumsum(r))
    max_dd = float(np.min(nivel / np.maximum.accumulate(nivel) - 1))
    var95 = float(np.quantile(r, 0.05))
    es95 = float(r[r <= var95].mean())
    vol_30 = r[-21:].std(ddof=1) * np.sqrt(dias_anio) if len(r) >= 21 else np.nan
    return {
        "rent_anual": float(rent_anual),
        "vol_anual": float(vol_anual),
        "vol_reciente": float(vol_30),
        "max_drawdown": max_dd,
        "var95_diario": var95,
        "es95_diario": es95,
    }


def volatilidad_movil(rent_diaria: pd.Series, ventana: int = 21, dias_anio: int = 252) -> pd.Series:
    return rent_diaria.rolling(ventana).std() * np.sqrt(dias_anio)


def volatilidad_horizonte(rent_diaria, h: int) -> dict:
    """Volatilidad en la escala del horizonte.

    * Empírica: desviación estándar de los rendimientos acumulados de h días
      tomados sin solapamiento (cada h sesiones).
    * Escalada: sigma_diaria * sqrt(h), que supone rendimientos independientes.
    La diferencia entre ambas indica dependencia en los rendimientos.
    """
    r = np.asarray(rent_diaria, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 * h:
        return {"empirica": np.nan, "escalada": np.nan, "n_periodos": 0}
    n = (len(r) // h) * h
    bloques = r[len(r) - n:].reshape(-1, h).sum(axis=1)
    return {"empirica": float(np.std(bloques, ddof=1)) if len(bloques) > 2 else np.nan,
            "escalada": float(r.std(ddof=1) * np.sqrt(h)), "n_periodos": len(bloques)}


def clasificar_volatilidad(percentil: float | None) -> str:
    """Categoría relativa al universo S&P 500 (terciles). No es un umbral universal."""
    if percentil is None or not np.isfinite(percentil):
        return "Sin referencia"
    if percentil < 100 / 3:
        return "Baja frente al S&P 500"
    if percentil < 200 / 3:
        return "Media frente al S&P 500"
    return "Alta frente al S&P 500"
