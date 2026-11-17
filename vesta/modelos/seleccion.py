"""Selección de la especificación ARIMA (secciones 12.8 a 12.10).

ARIMA(p, 1, q) sobre el log-precio Y_t = ln(P_t) sin deriva es exactamente
ARMA(p, q) sobre la rentabilidad r_t = ΔY_t sin media. Por eso todos los
modelos se estiman sobre r_t y se etiquetan con d = 1. Con d = 0 se modela la
rentabilidad directamente y se permite la media.

Búsqueda exhaustiva (`stepwise = FALSE, approximation = FALSE`) sobre
p <= max_p, q <= max_q. Igual que `auto.arima`, se descartan los modelos que no
se pueden estimar y los que tienen raíces de módulo menor que 1.01.

Se informan AIC y BIC de cada candidato y el mejor según cada criterio. El
criterio elegido en los ajustes define la especificación principal, pero la
decisión de si el modelo sirve para pronosticar se toma con la evaluación
fuera de muestra (evaluacion/rodante.py), no con AIC o BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vesta.modelos.arma import ErrorModelo, ModeloARMA, ajustar_arma, etiqueta_modelo


@dataclass
class ResultadoSeleccion:
    mejor: ModeloARMA            # según el criterio configurado
    mejor_aic: ModeloARMA
    mejor_bic: ModeloARMA
    candidatos: pd.DataFrame
    criterio: str
    d: int

    def especificaciones(self) -> list[tuple[str, ModeloARMA]]:
        """Especificaciones distintas a evaluar fuera de muestra (AIC y BIC si difieren)."""
        salida = [("Mejor BIC", self.mejor_bic)]
        if (self.mejor_aic.p, self.mejor_aic.q, self.mejor_aic.con_media) != \
           (self.mejor_bic.p, self.mejor_bic.q, self.mejor_bic.con_media):
            salida.append(("Mejor AIC", self.mejor_aic))
        if self.criterio == "aic":
            salida.reverse()
        return salida


def etiqueta(m: ModeloARMA) -> str:
    base = etiqueta_modelo(m.p, m.d, m.q)
    if m.con_media:
        base += " con deriva" if m.d == 1 else " con media"
    return base


def auto_arima(r, d: int = 1, max_p: int = 3, max_q: int = 3, max_orden: int | None = None,
               permitir_media: bool = False, criterio: str = "bic", progreso=None) -> ResultadoSeleccion:
    """r: rentabilidades logarítmicas diarias (d = 1 equivale a modelar el log-precio)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    max_orden = max_orden if max_orden is not None else max_p + max_q
    opciones_media = [True, False] if permitir_media else [False]
    combinaciones = [(p, q, m) for m in opciones_media for p in range(max_p + 1) for q in range(max_q + 1)
                     if p + q <= max_orden]
    filas, validos = [], []
    for i, (p, q, m) in enumerate(combinaciones):
        if progreso:
            progreso((i + 1) / len(combinaciones), f"ARIMA({p},{d},{q})")
        fila = {"Modelo": etiqueta_modelo(p, d, q), "p": p, "d": d, "q": q,
                ("Deriva" if d == 1 else "Media"): m}
        try:
            mod = ajustar_arma(r, p, q, con_media=m)
            mod.d = d
            raiz = mod.raiz_minima()
            ok = np.isfinite(mod.bic) and raiz >= 1.01
            fila.update(AIC=mod.aic, BIC=mod.bic, LogLik=mod.loglik, Sigma2=mod.sigma2, Raiz_min=raiz, Valido=ok,
                        Motivo="" if ok else "Raíz de módulo < 1.01")
            if ok:
                validos.append(mod)
        except (ErrorModelo, np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            fila.update(AIC=np.nan, BIC=np.nan, LogLik=np.nan, Sigma2=np.nan, Raiz_min=np.nan, Valido=False,
                        Motivo=str(exc))
        filas.append(fila)
    if not validos:
        raise ErrorModelo("Ningún modelo candidato pudo estimarse.")
    mejor_aic = min(validos, key=lambda m: m.aic)
    mejor_bic = min(validos, key=lambda m: m.bic)
    tabla = pd.DataFrame(filas)
    tabla["Delta_AIC"] = tabla["AIC"] - mejor_aic.aic
    tabla["Delta_BIC"] = tabla["BIC"] - mejor_bic.bic
    tabla = tabla.sort_values(criterio.upper()).reset_index(drop=True)
    return ResultadoSeleccion(mejor=mejor_aic if criterio == "aic" else mejor_bic, mejor_aic=mejor_aic,
                              mejor_bic=mejor_bic, candidatos=tabla, criterio=criterio, d=d)
