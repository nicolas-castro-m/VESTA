"""Pronóstico vigente desde el último dato disponible (sección 10.2 y 10.3 del prompt).

Reconstrucción del precio (sin multiplicar el precio por un rendimiento diario):
    Y_T = ln(P_T)
    Ŷ_{T+h} = Y_T + sum_{j=1}^{h} r̂_{T+j}
    P̂_{T+h} = exp(Ŷ_{T+h}) = P_T · exp(sum r̂)

Intervalo de predicción (no de confianza de parámetros): para el rendimiento
acumulado S_h, media ± z·ee con ee² = sigma² · sum_m (sum_{i<=h-m} psi_i)².
Supone errores normales, homocedásticos y parámetros conocidos (no incorpora
la incertidumbre de estimación). Los límites del precio se obtienen aplicando
exp a los límites del log-precio, por eso la banda de precio es asimétrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from vesta.ajustes import Ajustes
from vesta.modelos.arma import ModeloARMA, ajustar_arma, reconstruir_precio, trayectoria_acumulada
from vesta.modelos.caminata import pronostico_rw
from vesta.modelos.seleccion import etiqueta
from vesta.procesamiento.rendimientos import SerieTrabajo


@dataclass
class Prediccion:
    horizonte: str
    h: int
    fecha_ultimo_dato: pd.Timestamp
    precio_actual: float
    modelo: ModeloARMA
    etiqueta: str
    fechas: list                    # T, T+1, ..., T+h (días hábiles estimados)
    rend_acum: np.ndarray           # S_k, k = 1..h
    rend_inf: np.ndarray
    rend_sup: np.ndarray
    precio: np.ndarray              # trayectoria de precio k = 1..h
    precio_inf: np.ndarray
    precio_sup: np.ndarray
    rw_inf: np.ndarray
    rw_sup: np.ndarray
    prob_subida: float
    nivel: float

    @property
    def rend_final(self) -> float:
        return float(self.rend_acum[-1])

    @property
    def precio_final(self) -> float:
        return float(self.precio[-1])

    @property
    def fecha_objetivo(self) -> pd.Timestamp:
        return self.fechas[-1]


def generar(serie: SerieTrabajo, orden: ModeloARMA, horizonte: str, h: int, ajustes: Ajustes) -> Prediccion:
    r = serie.rentabilidad[1:]
    modelo = ajustar_arma(r, orden.p, orden.q, orden.con_media, warm_start=orden.par_transformado,
                          errores_estandar=True)
    modelo.d = orden.d
    tr = trayectoria_acumulada(modelo, h, ajustes.nivel_intervalo)
    p_t = float(serie.precio[-1])
    rw = pronostico_rw(r, h, ajustes.nivel_intervalo)
    ultimo = pd.Timestamp(serie.fechas.iloc[-1])
    fechas = [ultimo] + [ultimo + pd.offsets.BDay(k) for k in range(1, h + 1)]
    prob = float(1 - stats.norm.cdf(0, loc=tr["media"][-1], scale=tr["ee"][-1])) if tr["ee"][-1] > 0 else np.nan
    return Prediccion(
        horizonte=horizonte, h=h, fecha_ultimo_dato=ultimo, precio_actual=p_t, modelo=modelo,
        etiqueta=etiqueta(modelo), fechas=fechas,
        rend_acum=tr["media"], rend_inf=tr["inferior"], rend_sup=tr["superior"],
        precio=reconstruir_precio(p_t, tr["media"]),
        precio_inf=reconstruir_precio(p_t, tr["inferior"]),
        precio_sup=reconstruir_precio(p_t, tr["superior"]),
        rw_inf=reconstruir_precio(p_t, rw["inferior"]), rw_sup=reconstruir_precio(p_t, rw["superior"]),
        prob_subida=prob, nivel=ajustes.nivel_intervalo,
    )
