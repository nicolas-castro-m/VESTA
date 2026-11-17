"""Universo S&P 500: beta, perfiles de riesgo y muestreo (sección 13).

La beta se calcula igual que en el documento:
  * Ventana de cinco años que termina al cierre del último mes completo.
  * Rendimientos logarítmicos mensuales con precio ajustado (`monthlyReturn`
    de quantmod: el primer mes es parcial, desde el primer cierre disponible).
  * Pendiente de la regresión del activo sobre ^GSPC: Cov(r_i, r_m) / Var(r_m).
  * Se exigen al menos 48 meses comunes; si no, beta = NA.

El universo se recalcula periódicamente (no en cada visita) y se guarda en
SQLite (tablas activos y calculos_universo) con la fecha de cálculo.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from vesta.config import (ANIOS_BETA, ARCHIVO_MUESTRA, ARCHIVO_TICKERS_RESPALDO, DIAS_VALIDEZ_UNIVERSO,
                          INDICE_REFERENCIA, MIN_MESES_BETA, N_POR_PERFIL, PERFILES, PILOTO, SEMILLA_MUESTREO,
                          UMBRAL_AGRESIVO, UMBRAL_DEFENSIVO)
from vesta.persistencia import db
from vesta.datos.fuentes import componentes_sp500, descargar_precios
from vesta.procesamiento.rendimientos import limpiar_precios

FUENTE_BETA = "Calculado: regresión de rendimientos logarítmicos mensuales vs ^GSPC, 5 años"


def ventana_beta(hoy: date | None = None, anios: int = ANIOS_BETA) -> tuple[date, date]:
    """Inicio y fin de la ventana: del día 1 de hace cinco años al último día del mes anterior."""
    hoy = hoy or date.today()
    inicio_mes = hoy.replace(day=1)
    fin = inicio_mes - timedelta(days=1)
    inicio = inicio_mes.replace(year=inicio_mes.year - anios)
    return inicio, fin


def rend_mensual(precios: pd.DataFrame) -> pd.Series:
    """Rendimiento log mensual indexado por 'AAAA-MM', como `monthlyReturn(Ad(x), type = "log")`."""
    d = precios[["Date", "Adj.Close"]].dropna().sort_values("Date")
    s = d.set_index("Date")["Adj.Close"]
    ultimo = s.groupby(s.index.to_period("M")).last()
    primero = float(s.iloc[0])
    previos = np.r_[primero, ultimo.to_numpy()[:-1]]
    r = np.log(ultimo.to_numpy() / previos)
    return pd.Series(r, index=ultimo.index.astype(str))


def calcular_beta(r_activo: pd.Series, r_mercado: pd.Series, min_meses: int = MIN_MESES_BETA) -> tuple[float, int]:
    d = pd.concat({"a": r_activo, "m": r_mercado}, axis=1, join="inner").dropna()
    n = len(d)
    if n < min_meses:
        return np.nan, n
    cov = np.cov(d["a"], d["m"], ddof=1)
    return float(cov[0, 1] / cov[1, 1]), n


def asignar_perfil(beta: float) -> str | None:
    """`cut(beta, c(-Inf, 0.8, 1.2, Inf), right = FALSE)`."""
    if beta is None or not np.isfinite(beta):
        return None
    if beta < UMBRAL_DEFENSIVO:
        return "Defensivo"
    if beta < UMBRAL_AGRESIVO:
        return "Neutral"
    return "Agresivo"


def construir_universo(tickers: pd.DataFrame | None = None, hoy: date | None = None, lote: int = 100,
                       progreso=None) -> pd.DataFrame:
    """Calcula beta, perfil, precio y variación reciente para todo el universo."""
    if tickers is None:
        try:
            tickers = componentes_sp500()
        except Exception:
            respaldo = ARCHIVO_TICKERS_RESPALDO
            if not respaldo.exists():
                raise
            tickers = pd.read_csv(respaldo)   # columnas ticker y empresa, como en el documento
            for col in ("sector", "subindustria"):
                if col not in tickers:
                    tickers[col] = None
    inicio, fin = ventana_beta(hoy)
    mercado = descargar_precios([INDICE_REFERENCIA], inicio=inicio, fin=fin + timedelta(days=1))
    if INDICE_REFERENCIA not in mercado:
        raise RuntimeError("No se pudo descargar el índice de referencia ^GSPC.")
    r_m = rend_mensual(limpiar_precios(mercado[INDICE_REFERENCIA]))
    # Precio reciente: se descarga hasta hoy para la ficha (no entra en la beta)
    fin_precio = (hoy or date.today()) + timedelta(days=1)
    lista = tickers["ticker"].tolist()
    filas = []
    for i in range(0, len(lista), lote):
        parte = lista[i:i + lote]
        if progreso:
            progreso(i / len(lista), f"Descargando {i + 1}-{min(i + lote, len(lista))} de {len(lista)}")
        datos = descargar_precios(parte, inicio=inicio, fin=fin_precio)
        for tk in parte:
            fila = {"ticker": tk, "beta": np.nan, "n_meses": 0, "precio": np.nan, "fecha_precio": None,
                    "var_5d": np.nan, "var_21d": np.nan, "vol_anual": np.nan}
            if tk in datos:
                try:
                    d = limpiar_precios(datos[tk])
                    d_beta = d[d["Date"] <= pd.Timestamp(fin)]
                    b, n = calcular_beta(rend_mensual(d_beta), r_m)
                    p = d["Adj.Close"].to_numpy()
                    r_d = np.diff(np.log(d_beta["Adj.Close"].to_numpy()))
                    fila.update(beta=b, n_meses=n, precio=float(d["Close"].iloc[-1]),
                                vol_anual=float(np.std(r_d, ddof=1) * np.sqrt(252)) if len(r_d) > 250 else np.nan,
                                fecha_precio=str(d["Date"].iloc[-1].date()),
                                var_5d=float(p[-1] / p[-6] - 1) if len(p) > 6 else np.nan,
                                var_21d=float(p[-1] / p[-22] - 1) if len(p) > 22 else np.nan)
                except Exception:
                    pass
            filas.append(fila)
    if progreso:
        progreso(1.0, "Clasificando perfiles")
    u = tickers.merge(pd.DataFrame(filas), on="ticker", how="left")
    u["perfil"] = u["beta"].map(asignar_perfil)
    u["mercado"] = "S&P 500"
    u["fuente"] = FUENTE_BETA
    u["ventana_inicio"] = str(inicio)
    u["ventana_fin"] = str(fin)
    u["fecha_consulta"] = str(hoy or date.today())
    return u


def guardar_universo(u: pd.DataFrame) -> None:
    db.guardar_universo(u)


def cargar_universo() -> pd.DataFrame | None:
    return db.leer_universo()


def actualizar(progreso=None) -> pd.DataFrame:
    """Construye y guarda el universo registrando la actualización en la base."""
    id_ = db.iniciar_actualizacion("universo", "S&P 500: beta, perfil y precio")
    try:
        u = construir_universo(progreso=progreso)
        guardar_universo(u)
        db.cerrar_actualizacion(id_, "completada", f"{u['beta'].notna().sum()} de {len(u)} activos con beta")
        return u
    except Exception as exc:
        db.cerrar_actualizacion(id_, "fallida", str(exc))
        raise


def universo_vencido(u: pd.DataFrame | None, dias: int = DIAS_VALIDEZ_UNIVERSO) -> bool:
    if u is None or u.empty or "fecha_consulta" not in u:
        return True
    fecha = pd.to_datetime(u["fecha_consulta"].iloc[0]).date()
    return (date.today() - fecha).days > dias


def depurar(u: pd.DataFrame) -> pd.DataFrame:
    """Bloque 5: se excluyen activos sin beta y tickers repetidos."""
    return u[u["beta"].notna()].drop_duplicates("ticker").reset_index(drop=True)


def muestreo_sistematico(marco: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Bloque 7: muestreo sistemático con arranque aleatorio dentro del perfil ordenado por beta."""
    marco = marco.sort_values("beta").reset_index(drop=True)
    N = len(marco)
    if N <= n:
        return marco
    k = N / n
    u = rng.uniform(0, k)
    posiciones = np.ceil(u + np.arange(n) * k).astype(int)
    return marco.iloc[posiciones - 1]


def muestra_estratificada(u: pd.DataFrame, n_por_perfil: int = N_POR_PERFIL,
                          semilla: int = SEMILLA_MUESTREO) -> pd.DataFrame:
    """Muestra sistemática estratificada por perfil (7 por perfil, n = 21).

    El generador de números aleatorios de numpy no es el de R, así que con la
    misma semilla la muestra no coincide con `set.seed(2026)` en R. Si existe
    data/muestra_fase2.csv (exportado desde R), la aplicación usa ese archivo.
    """
    rng = np.random.default_rng(semilla)
    partes = [muestreo_sistematico(u[u["perfil"] == p], n_por_perfil, rng) for p in PERFILES]
    return pd.concat(partes, ignore_index=True)


def cargar_muestra(u: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Muestra académica de la fase 2 (distinta del universo exploratorio)."""
    if ARCHIVO_MUESTRA.exists():
        return pd.read_csv(ARCHIVO_MUESTRA), "Archivo data/muestra_fase2.csv (muestra oficial del proyecto)"
    if u is None:
        return None, ""
    return muestra_estratificada(depurar(u)), "Generada en Python con semilla 2026 (no idéntica a la de R)"


def piloto() -> pd.DataFrame:
    """Activos del estudio piloto (fase 1, sección 6.2)."""
    return pd.DataFrame({"ticker": list(PILOTO), "empresa": list(PILOTO.values())})


def percentil_volatilidad(vol_anual: float, u: pd.DataFrame | None) -> float | None:
    """Percentil de la volatilidad de un activo dentro del universo S&P 500 (0 a 100)."""
    if u is None or "vol_anual" not in u or not np.isfinite(vol_anual):
        return None
    ref = pd.to_numeric(u["vol_anual"], errors="coerce").dropna()
    if len(ref) < 30:
        return None
    return float((ref < vol_anual).mean() * 100)
