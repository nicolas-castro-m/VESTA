"""Preparación de series y rendimientos (secciones 6.5, 7 y 12.8).

Replica `preparar_activo`, `preparar_caminata`, `preparar_serie` y
`preparar_horizontes` del documento metodológico.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vesta.config import HORIZONTES

PROP_TRAIN = 0.80

COLUMNAS_PRECIO = ["Open", "High", "Low", "Close", "Adj.Close", "Volume"]


def limpiar_precios(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, ordena por fecha y elimina filas incompletas.

    Igual que `descargar_activo`: los días sin cotización (NA) se eliminan.
    """
    d = df.copy()
    d.columns = [str(c).strip().replace("Adj Close", "Adj.Close") for c in d.columns]
    if "Date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
    if "Adj.Close" not in d.columns and "Close" in d.columns:
        d["Adj.Close"] = d["Close"]
    columnas = ["Date"] + [c for c in COLUMNAS_PRECIO if c in d.columns]
    d = d[columnas]
    for c in columnas[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    d = d[d["Adj.Close"] > 0].reset_index(drop=True)
    return d


def preparar_activo(datos: pd.DataFrame) -> pd.DataFrame:
    """`preparar_activo`: rendimiento log diario y acumulados a 5, 21 y 63 días.

    r_t = ln(P_t / P_{t-1}) con P el precio ajustado; los acumulados usan
    ln(P_t / P_{t-h}), por eso las primeras h observaciones quedan en NA.
    """
    d = datos.copy()
    p = d["Adj.Close"].to_numpy(dtype=float)
    d["rentabilidad"] = np.log(p / np.r_[np.nan, p[:-1]])
    d["rent_porcent"] = d["rentabilidad"] * 100
    for nombre, h in (("rent_sem", 5), ("rent_mes", 21), ("rent_trim", 63)):
        rezago = np.r_[np.full(h, np.nan), p[:-h]] if len(p) > h else np.full(len(p), np.nan)
        d[nombre] = np.log(p / rezago)
    return d


VARIABLE_HORIZONTE = {"Semanal": "rent_sem", "Mensual": "rent_mes", "Trimestral": "rent_trim"}


@dataclass
class SerieTrabajo:
    """Equivalente a la lista que devuelve `preparar_serie` (sección 12.8)."""

    nombre: str
    fechas: pd.Series
    precio: np.ndarray
    log_precio: np.ndarray
    rentabilidad: np.ndarray       # r_t, primer valor NA
    n: int
    corte: int                     # último índice (base 1) del entrenamiento
    fecha_corte: pd.Timestamp
    y_train: np.ndarray = field(repr=False)   # log-precios 1..corte
    r_train: np.ndarray = field(repr=False)   # rentabilidades 2..corte


def preparar_serie(datos: pd.DataFrame, nombre: str, proporcion: float = PROP_TRAIN) -> SerieTrabajo:
    d = datos[["Date", "Adj.Close"]].dropna()
    d = d[d["Adj.Close"] > 0].sort_values("Date").reset_index(drop=True)
    precio = d["Adj.Close"].to_numpy(dtype=float)
    lp = np.log(precio)
    r = np.r_[np.nan, np.diff(lp)]
    n = len(d)
    corte = int(np.floor(n * proporcion))
    return SerieTrabajo(
        nombre=nombre,
        fechas=d["Date"],
        precio=precio,
        log_precio=lp,
        rentabilidad=r,
        n=n,
        corte=corte,
        fecha_corte=d["Date"].iloc[corte - 1],
        y_train=lp[:corte],
        r_train=r[1:corte],
    )


@dataclass
class SerieHorizonte:
    """Equivalente a cada elemento de `preparar_horizontes` (sección 12.10)."""

    activo: str
    horizonte: str
    fechas: pd.Series
    valores: np.ndarray
    n: int
    corte: int
    fecha_corte: pd.Timestamp
    train: np.ndarray = field(repr=False)
    test: np.ndarray = field(repr=False)
    fechas_train: pd.Series = field(repr=False)
    fechas_test: pd.Series = field(repr=False)


def preparar_horizontes(datos: pd.DataFrame, nombre: str, proporcion: float = PROP_TRAIN) -> dict[str, SerieHorizonte]:
    d = preparar_activo(datos)
    salida = {}
    for h, variable in VARIABLE_HORIZONTE.items():
        s = d[["Date", variable]]
        s = s[np.isfinite(s[variable])].reset_index(drop=True)
        n = len(s)
        corte = int(np.floor(n * proporcion))
        v = s[variable].to_numpy(dtype=float)
        salida[h] = SerieHorizonte(
            activo=nombre,
            horizonte=h,
            fechas=s["Date"],
            valores=v,
            n=n,
            corte=corte,
            fecha_corte=s["Date"].iloc[corte - 1],
            train=v[:corte],
            test=v[corte:],
            fechas_train=s["Date"].iloc[:corte].reset_index(drop=True),
            fechas_test=s["Date"].iloc[corte:].reset_index(drop=True),
        )
    return salida


def rendimientos_no_solapados(precio: pd.Series, h: int) -> pd.Series:
    """Rendimientos log tomando una observación cada h días (como en Cochran)."""
    muestra = precio.iloc[::h]
    return np.log(muestra / muestra.shift(1)).dropna()


def alinear_precios(precios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Precios ajustados en las fechas comunes a todos los activos (inner join)."""
    series = {k: v.set_index("Date")["Adj.Close"] for k, v in precios.items()}
    return pd.concat(series, axis=1, join="inner").sort_index()


def variacion_reciente(datos: pd.DataFrame, dias: int = 5) -> float | None:
    p = datos["Adj.Close"].to_numpy(dtype=float)
    if len(p) <= dias:
        return None
    return float(p[-1] / p[-1 - dias] - 1)


__all__ = [
    "HORIZONTES",
    "limpiar_precios",
    "preparar_activo",
    "preparar_serie",
    "preparar_horizontes",
    "rendimientos_no_solapados",
    "alinear_precios",
    "variacion_reciente",
    "SerieTrabajo",
    "SerieHorizonte",
    "VARIABLE_HORIZONTE",
]
