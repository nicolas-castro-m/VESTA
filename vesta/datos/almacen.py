"""Caché local de precios: un CSV por activo en data/precios.

Evita descargar en cada visita. Si la descarga falla y existe una copia, se
usa la copia y se informa su fecha.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from vesta.config import DIR_PRECIOS, HORAS_VALIDEZ_PRECIOS

ANIOS_HISTORIA = 5
from vesta.datos.fuentes import ErrorDatos, descargar_precios
from vesta.procesamiento.rendimientos import limpiar_precios


@dataclass
class DatosActivo:
    ticker: str
    precios: pd.DataFrame          # serie limpia (sin duplicados, faltantes ni precios no positivos)
    origen: str                    # "descarga", "caché" o "caché (descarga fallida)"
    actualizado: pd.Timestamp      # cuándo se obtuvo de la fuente
    aviso: str = ""
    crudo: pd.DataFrame | None = None   # tal como llegó de la fuente, para el control de calidad


def _archivo(ticker: str) -> Path:
    nombre = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    return DIR_PRECIOS / f"{nombre}.csv"


def _vigente(ruta: Path, horas: float) -> bool:
    return ruta.exists() and (time.time() - ruta.stat().st_mtime) < horas * 3600


def guardar(ticker: str, df: pd.DataFrame) -> None:
    """Guarda la descarga sin limpiar: la limpieza se hace al leer y queda auditada."""
    DIR_PRECIOS.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if "Date" not in d.columns:
        d = d.reset_index()
    d.to_csv(_archivo(ticker), index=False)


def _leer(ruta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    crudo = pd.read_csv(ruta)
    return limpiar_precios(crudo), crudo


def cargar(ticker: str, forzar: bool = False, horas_validez: float = HORAS_VALIDEZ_PRECIOS,
           anios: float = ANIOS_HISTORIA) -> DatosActivo:
    ruta = _archivo(ticker)
    if not forzar and _vigente(ruta, horas_validez):
        df, crudo = _leer(ruta)
        return DatosActivo(ticker, df, "caché", pd.Timestamp(ruta.stat().st_mtime, unit="s"), crudo=crudo)
    try:
        descargado = descargar_precios([ticker], anios=anios).get(ticker)
        if descargado is None or descargado.empty:
            raise ErrorDatos(f"Yahoo Finance no devolvió datos para {ticker}.")
        guardar(ticker, descargado)
        df, crudo = _leer(ruta)
        return DatosActivo(ticker, df, "descarga", pd.Timestamp.now(), crudo=crudo)
    except Exception as exc:
        if ruta.exists():
            df, crudo = _leer(ruta)
            fecha = pd.Timestamp(ruta.stat().st_mtime, unit="s")
            return DatosActivo(ticker, df, "caché (descarga fallida)", fecha, crudo=crudo,
                               aviso=f"No se pudo actualizar desde Yahoo Finance ({exc}). Se usan los datos guardados "
                                     f"el {fecha:%Y-%m-%d %H:%M}.")
        raise ErrorDatos(f"No hay datos para {ticker}: {exc}") from exc


def recortar_ventana(df: pd.DataFrame, anios: float = ANIOS_HISTORIA, fecha_fin: date | None = None) -> pd.DataFrame:
    """Ventana de `anios` años que termina en `fecha_fin` (o en el último dato)."""
    d = df.copy()
    fin = pd.Timestamp(fecha_fin) if fecha_fin else d["Date"].max()
    inicio = fin - pd.Timedelta(days=round(365.25 * anios))
    return d[(d["Date"] > inicio) & (d["Date"] <= fin)].reset_index(drop=True)


def tickers_en_cache() -> list[str]:
    if not DIR_PRECIOS.exists():
        return []
    return sorted(p.stem for p in DIR_PRECIOS.glob("*.csv"))
