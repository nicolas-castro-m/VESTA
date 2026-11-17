"""Caché en disco de resultados intermedios costosos (data/cache).

No es la fuente de verdad: lo que debe conservarse (predicciones, evaluaciones)
va a SQLite. La clave incluye ticker, última fecha de datos, tipo de cálculo,
la firma de los ajustes y la versión del cálculo, así que un cambio en
cualquiera de ellos produce un cálculo nuevo.
"""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Any, Callable

from vesta.config import DIR_CACHE as DIR_RESULTADOS

VERSION_CALCULO = "2.0"


def _ruta(ticker: str, ultima_fecha: str, tipo: str, firma: str) -> Path:
    tk = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    clave = hashlib.sha1(f"{firma}|{VERSION_CALCULO}".encode()).hexdigest()[:10]
    return DIR_RESULTADOS / f"{tk}__{ultima_fecha}__{tipo}__{clave}.pkl"


def obtener(ticker: str, ultima_fecha: str, tipo: str, calcular: Callable[[], Any], firma: str = "",
            forzar: bool = False) -> Any:
    ruta = _ruta(ticker, ultima_fecha, tipo, firma)
    if ruta.exists() and not forzar:
        try:
            with ruta.open("rb") as f:
                return pickle.load(f)
        except Exception:
            ruta.unlink(missing_ok=True)
    valor = calcular()
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with ruta.open("wb") as f:
        pickle.dump(valor, f)
    _limpiar_antiguos(ticker, tipo, ruta)
    return valor


def existe(ticker: str, ultima_fecha: str, tipo: str, firma: str = "") -> bool:
    return _ruta(ticker, ultima_fecha, tipo, firma).exists()


def _limpiar_antiguos(ticker: str, tipo: str, actual: Path, conservar: int = 3) -> None:
    tk = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    archivos = sorted(DIR_RESULTADOS.glob(f"{tk}__*__{tipo}__*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in archivos[conservar:]:
        if p != actual:
            p.unlink(missing_ok=True)
