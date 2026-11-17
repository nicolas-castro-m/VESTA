"""Obtención de datos externos: Yahoo Finance (precios) y Wikipedia (componentes del S&P 500)."""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

URL_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CABECERAS = {"User-Agent": "Mozilla/5.0 (VESTA academic project; +https://github.com)"}


class ErrorDatos(Exception):
    pass


def simbolo_yahoo(ticker: str) -> str:
    """Yahoo escribe BRK-B y BF-B con guion (sección 13.3, bloque 2)."""
    return ticker.strip().upper().replace(".", "-")


def descargar_precios(tickers: list[str] | str, inicio: date | None = None, fin: date | None = None,
                      anios: float = 5) -> dict[str, pd.DataFrame]:
    """Descarga precios diarios (Open, High, Low, Close, Adj Close, Volume).

    Equivale a `getSymbols(..., src = "yahoo", periodicity = "daily")`. Devuelve
    un diccionario ticker -> DataFrame con columna Date; omite los que fallan.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ErrorDatos("Falta el paquete yfinance. Ejecute: pip install -r requirements.txt") from exc

    if isinstance(tickers, str):
        tickers = [tickers]
    simbolos = [simbolo_yahoo(t) if not t.startswith("^") else t for t in tickers]
    fin = fin or date.today() + timedelta(days=1)
    inicio = inicio or (date.today() - timedelta(days=round(365.25 * anios)))
    datos = yf.download(simbolos, start=inicio.isoformat(), end=fin.isoformat(), interval="1d",
                        auto_adjust=False, actions=False, group_by="ticker", threads=True, progress=False)
    salida: dict[str, pd.DataFrame] = {}
    if datos is None or datos.empty:
        return salida
    for original, simbolo in zip(tickers, simbolos):
        try:
            df = datos[simbolo] if isinstance(datos.columns, pd.MultiIndex) else datos
        except KeyError:
            continue
        df = df.dropna(how="all")
        if df.empty or "Close" not in df:
            continue
        df = df.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
        salida[original] = df
    return salida


def componentes_sp500() -> pd.DataFrame:
    """Lista vigente del S&P 500: ticker, empresa, sector y subindustria (GICS).

    Se conserva una sola clase de acción por empresa (mismo CIK), como en la
    sección 13.1 del documento.
    """
    import requests

    try:
        resp = requests.get(URL_SP500, headers=CABECERAS, timeout=30)
        resp.raise_for_status()
        tablas = pd.read_html(io.StringIO(resp.text))
    except Exception as exc:
        raise ErrorDatos(f"No se pudo obtener la lista del S&P 500: {exc}") from exc
    tabla = next((t for t in tablas if "Symbol" in t.columns), None)
    if tabla is None:
        raise ErrorDatos("La página de Wikipedia cambió de formato: no se encontró la columna Symbol.")
    df = pd.DataFrame({
        "ticker": tabla["Symbol"].astype(str).str.strip(),
        "empresa": tabla["Security"].astype(str).str.strip(),
        "sector": tabla.get("GICS Sector", pd.Series([None] * len(tabla))),
        "subindustria": tabla.get("GICS Sub-Industry", pd.Series([None] * len(tabla))),
        "cik": tabla.get("CIK", pd.Series(range(len(tabla)))),
    })
    df = df.drop_duplicates("cik", keep="first").drop(columns="cik").reset_index(drop=True)
    return df
