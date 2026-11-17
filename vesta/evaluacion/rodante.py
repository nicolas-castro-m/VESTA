"""Evaluación predictiva fuera de muestra (sección 11.3 del prompt; ajuste pendiente de la sección 14).

Condiciones comparables para todos los modelos:
  * Misma variable objetivo: rendimiento logarítmico acumulado R_{t,h} = ln(P_{t+h} / P_t).
  * Mismos horizontes: 5, 21 y 63 días bursátiles.
  * Mismos orígenes de pronóstico: cada día del periodo de prueba (20 % final).
  * Mismo conjunto de observaciones evaluadas.

Sin fuga de información:
  * El orden (p, q) se elige solo con el entrenamiento.
  * En cada origen t los parámetros se reestiman solo con datos hasta t
    (`reestimar_cada = 1`, como REEST_CADA = 1 en el documento) y el pronóstico
    usa únicamente información disponible en t.

Caminata aleatoria sin deriva: pronóstico del acumulado igual a 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from vesta.ajustes import Ajustes
from vesta.config import HORIZONTES
from vesta.evaluacion import metricas as M
from vesta.modelos.arma import (ErrorModelo, ModeloARMA, ajustar_arma, estados_filtrados, matriz_suma_pasos,
                                psi_pesos)
from vesta.modelos.seleccion import etiqueta
from vesta.procesamiento.rendimientos import SerieTrabajo


@dataclass
class EvaluacionModelo:
    nombre: str                    # "Mejor BIC", "Mejor AIC"
    etiqueta: str
    pronosticos: pd.DataFrame      # una fila por origen y horizonte
    metricas: pd.DataFrame         # una fila por horizonte
    reestimaciones: int
    fallos_reestimacion: int


@dataclass
class ResultadoEvaluacion:
    modelos: list[EvaluacionModelo]
    origenes: dict = field(default_factory=dict)   # horizonte -> número de orígenes
    fecha_corte: pd.Timestamp | None = None


def _pronosticos_modelo(serie: SerieTrabajo, orden: ModeloARMA, ajustes: Ajustes, horizontes, progreso=None):
    r = serie.rentabilidad
    n = serie.n
    origen_ini = serie.corte - 1
    origen_fin = n - 2
    paso = max(1, ajustes.reestimar_cada)
    z = stats.norm.ppf(0.5 + ajustes.nivel_intervalo / 2)

    estados: dict[int, np.ndarray] = {}
    modelos: dict[int, ModeloARMA] = {}
    modelo = orden
    fallos = 0
    bloques = list(range(origen_ini, origen_fin + 1, paso))
    for b_i, b in enumerate(bloques):
        if progreso and (b_i % 5 == 0 or b_i == len(bloques) - 1):
            progreso((b_i + 1) / len(bloques), f"Origen {b_i + 1} de {len(bloques)}")
        if b != origen_ini:
            try:
                nuevo = ajustar_arma(r[1:b + 1], orden.p, orden.q, orden.con_media,
                                     warm_start=modelo.par_transformado)
                nuevo.d = orden.d
                modelo = nuevo
            except ErrorModelo:
                fallos += 1          # se conserva el último modelo válido (sin mirar el futuro)
        fin = min(bloques[b_i + 1] - 1, origen_fin) if b_i + 1 < len(bloques) else origen_fin
        if paso == 1:
            estados[b] = modelo.a_final if (modelo.p or modelo.q) else np.zeros(1)
            modelos[b] = modelo
        else:
            w = r[1:fin + 1] - modelo.mu
            est = estados_filtrados(w, modelo) if (modelo.p or modelo.q) else np.zeros((len(w), 1))
            for o in range(b, fin + 1):
                estados[o] = est[o - 1]
                modelos[o] = modelo

    cache_g: dict[tuple, tuple] = {}
    filas = []
    for nombre_h, h in horizontes.items():
        for o in range(origen_ini, n - h):
            mod = modelos[o]
            clave = (id(mod), h)
            if clave not in cache_g:
                g = matriz_suma_pasos(mod, h) if (mod.p or mod.q) else None
                psi = psi_pesos(mod.phi, mod.theta, h)
                cache_g[clave] = (g, float(np.sqrt(mod.sigma2 * np.sum(np.cumsum(psi) ** 2))))
            g, ee = cache_g[clave]
            pred = h * mod.mu + (float(g @ estados[o]) if g is not None else 0.0)
            filas.append((nombre_h, h, serie.fechas.iloc[o], serie.fechas.iloc[o + h], serie.precio[o],
                          serie.precio[o + h], serie.log_precio[o + h] - serie.log_precio[o], pred,
                          pred - z * ee, pred + z * ee))
    pr = pd.DataFrame(filas, columns=["Horizonte", "h", "Fecha_Origen", "Fecha_Objetivo", "Precio_Origen",
                                      "Precio_Real", "Real", "Pred_Modelo", "Inferior", "Superior"])
    pr["Pred_RW"] = 0.0
    pr["Precio_Modelo"] = pr["Precio_Origen"] * np.exp(pr["Pred_Modelo"])
    pr["Precio_RW"] = pr["Precio_Origen"]
    return pr, len(bloques), fallos


def veredicto(fila: dict | pd.Series, alfa: float) -> str:
    """Supera, iguala o queda por debajo de la caminata aleatoria (según Diebold-Mariano)."""
    if fila.get("Identico_RW"):
        return "Idéntico a la caminata aleatoria"
    p = fila.get("DM_p_valor", np.nan)
    if not np.isfinite(p):
        return "Sin prueba aplicable"
    if p < alfa:
        return "Supera a la caminata aleatoria" if fila["DM_estadistico"] < 0 else "Por debajo de la caminata aleatoria"
    return "Sin diferencia significativa"


def _metricas(pr: pd.DataFrame, mod: ModeloARMA, ajustes: Ajustes, horizontes) -> pd.DataFrame:
    identico = mod.p == 0 and mod.q == 0 and not mod.con_media
    filas = []
    for nombre_h, h in horizontes.items():
        s = pr[pr["Horizonte"] == nombre_h]
        e_m = s["Real"] - s["Pred_Modelo"]
        e_rw = s["Real"] - s["Pred_RW"]
        dm = M.diebold_mariano(e_m, e_rw, h=h) if not identico else \
            {"estadistico": np.nan, "p_valor": np.nan, "nota": "Modelo idéntico a la caminata aleatoria"}
        dirr = M.acierto_direccion(s["Real"], s["Pred_Modelo"])
        fila = {
            "Horizonte": nombre_h, "h": h, "Origenes": len(s),
            "MAE": M.mae(s["Real"], s["Pred_Modelo"]), "MAE_RW": M.mae(s["Real"], s["Pred_RW"]),
            "RMSE": M.rmse(s["Real"], s["Pred_Modelo"]), "RMSE_RW": M.rmse(s["Real"], s["Pred_RW"]),
            "Sesgo": M.sesgo(s["Real"], s["Pred_Modelo"]),
            "MAPE_precio": M.mape(s["Precio_Real"], s["Precio_Modelo"]),
            "MAPE_precio_RW": M.mape(s["Precio_Real"], s["Precio_RW"]),
            "U_Theil": M.theil_u(s["Real"], s["Pred_Modelo"], s["Pred_RW"]),
            "DM_estadistico": dm["estadistico"], "DM_p_valor": dm["p_valor"], "DM_nota": dm.get("nota", ""),
            "Direccion_pct": dirr["porcentaje"], "Direccion_n": dirr["n"], "Direccion_p_binomial": dirr["p_binomial"],
            "Pct_periodos_positivos": float(np.mean(s["Real"] > 0) * 100),
            "Cobertura_intervalo": M.cobertura(s["Real"], s["Inferior"], s["Superior"]),
            "Identico_RW": identico,
        }
        fila["Veredicto"] = veredicto(fila, ajustes.alfa)
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar(serie: SerieTrabajo, especificaciones: list[tuple[str, ModeloARMA]], ajustes: Ajustes,
            horizontes: dict[str, int] = HORIZONTES, progreso=None) -> ResultadoEvaluacion:
    resultados = []
    for i, (nombre, orden) in enumerate(especificaciones):
        prog = (lambda f, t, i=i, nm=nombre: progreso((i + f) / len(especificaciones), f"{nm}: {t}")) if progreso else None
        pr, reest, fallos = _pronosticos_modelo(serie, orden, ajustes, horizontes, prog)
        resultados.append(EvaluacionModelo(nombre=nombre, etiqueta=etiqueta(orden), pronosticos=pr,
                                           metricas=_metricas(pr, orden, ajustes, horizontes),
                                           reestimaciones=reest, fallos_reestimacion=fallos))
    origenes = {h: int((resultados[0].pronosticos["Horizonte"] == h).sum()) for h in horizontes}
    return ResultadoEvaluacion(modelos=resultados, origenes=origenes, fecha_corte=serie.fecha_corte)
