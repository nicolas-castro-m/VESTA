"""Caminata aleatoria: ARIMA(0,1,0) sobre el log-precio, sin deriva (sección 12.1).

Y_t = Y_{t-1} + e_t  =>  el mejor pronóstico de Y_{t+h} es Y_t, es decir,
rendimiento acumulado esperado igual a 0 y precio pronosticado igual al último.
La varianza del error a h pasos es h·sigma², con sigma² = mean(r_t²) (estimación
de máxima verosimilitud sin media).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sigma2_rw(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(np.mean(r**2))


def pronostico_rw(r, h: int, nivel: float = 0.95) -> dict:
    """Trayectoria k = 1..h del rendimiento acumulado (0) con su banda."""
    s2 = sigma2_rw(r)
    k = np.arange(1, h + 1)
    ee = np.sqrt(s2 * k)
    z = stats.norm.ppf(0.5 + nivel / 2)
    return {"media": np.zeros(h), "ee": ee, "inferior": -z * ee, "superior": z * ee}
