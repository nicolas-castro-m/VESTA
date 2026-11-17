"""Base de datos local (SQLite).

Guarda el universo, las betas con su fecha de cálculo, el historial de
actualizaciones, las predicciones, sus realizaciones, las evaluaciones y la
configuración. Los precios descargados viven aparte (data/precios) y los
resultados intermedios reutilizables en data/cache.

Para migrar a otro motor (PostgreSQL, etc.) basta con reimplementar este
módulo: el resto de la aplicación solo usa sus funciones.

Las predicciones son inmutables: un disparador impide modificarlas. Lo que
ocurre después (precio realizado, error) se guarda en `realizaciones`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vesta.config import ARCHIVO_DB

VERSION_ESQUEMA = 1

ESQUEMA = """
CREATE TABLE IF NOT EXISTS esquema (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS activos (
    ticker TEXT PRIMARY KEY,
    empresa TEXT, sector TEXT, subindustria TEXT, mercado TEXT
);

CREATE TABLE IF NOT EXISTS calculos_universo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL REFERENCES activos(ticker),
    fecha_calculo TEXT NOT NULL,
    ventana_inicio TEXT, ventana_fin TEXT,
    beta REAL, n_meses INTEGER, perfil TEXT, fuente TEXT,
    precio REAL, fecha_precio TEXT, var_5d REAL, var_21d REAL, vol_anual REAL
);
CREATE INDEX IF NOT EXISTS ix_calc_ticker ON calculos_universo (ticker, fecha_calculo);

CREATE TABLE IF NOT EXISTS actualizaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo TEXT NOT NULL, inicio TEXT NOT NULL, fin TEXT, estado TEXT, detalle TEXT
);

CREATE TABLE IF NOT EXISTS configuracion (
    clave TEXT PRIMARY KEY, valor TEXT NOT NULL, actualizado TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecutada_en TEXT NOT NULL,
    ticker TEXT NOT NULL, empresa TEXT,
    horizonte TEXT NOT NULL, h INTEGER NOT NULL,
    fecha_ultimo_dato TEXT NOT NULL, fecha_objetivo_estimada TEXT,
    precio_inicial REAL NOT NULL,
    rend_pronosticado REAL NOT NULL, rend_inferior REAL, rend_superior REAL,
    precio_pronosticado REAL NOT NULL, precio_inferior REAL, precio_superior REAL,
    nivel_intervalo REAL,
    modelo TEXT NOT NULL, orden TEXT, criterio TEXT,
    metricas TEXT,          -- JSON: desempeño fuera de muestra al momento de predecir
    ajustes TEXT,           -- JSON: parámetros usados
    datos TEXT,             -- JSON: periodo, observaciones y fuente
    version TEXT
);

CREATE TRIGGER IF NOT EXISTS predicciones_inmutables
BEFORE UPDATE ON predicciones
BEGIN SELECT RAISE(ABORT, 'Las predicciones históricas no se modifican'); END;

CREATE TABLE IF NOT EXISTS realizaciones (
    prediccion_id INTEGER PRIMARY KEY REFERENCES predicciones(id),
    evaluada_en TEXT NOT NULL,
    fecha_real TEXT NOT NULL, precio_real REAL NOT NULL, rend_real REAL NOT NULL,
    error_abs_precio REAL, error_pct_precio REAL, error_rend REAL,
    acierto_direccion INTEGER, posicion_intervalo TEXT
);

CREATE TABLE IF NOT EXISTS evaluaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecutada_en TEXT NOT NULL, ticker TEXT NOT NULL, horizonte TEXT NOT NULL,
    modelo TEXT NOT NULL, metricas TEXT NOT NULL, ajustes TEXT, datos TEXT
);
"""

_RUTA = ARCHIVO_DB


def usar_ruta(ruta: Path) -> None:
    """Cambia la base de datos (útil en pruebas)."""
    global _RUTA
    _RUTA = Path(ruta)


def ruta_actual() -> Path:
    return _RUTA


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=conv, ensure_ascii=False)


@contextmanager
def conexion():
    _RUTA.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_RUTA)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def inicializar() -> int:
    with conexion() as con:
        con.executescript(ESQUEMA)
        fila = con.execute("SELECT version FROM esquema").fetchone()
        if fila is None:
            con.execute("INSERT INTO esquema (version) VALUES (?)", (VERSION_ESQUEMA,))
            return VERSION_ESQUEMA
        return fila[0]


def ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Configuración ────────────────────────────────────────────────────────
def leer_configuracion() -> dict:
    inicializar()
    with conexion() as con:
        return {k: json.loads(v) for k, v in con.execute("SELECT clave, valor FROM configuracion")}


def guardar_configuracion(valores: dict) -> None:
    inicializar()
    with conexion() as con:
        con.executemany("INSERT OR REPLACE INTO configuracion (clave, valor, actualizado) VALUES (?, ?, ?)",
                        [(k, _json(v), ahora()) for k, v in valores.items()])


def borrar_configuracion() -> None:
    inicializar()
    with conexion() as con:
        con.execute("DELETE FROM configuracion")


# ── Actualizaciones ──────────────────────────────────────────────────────
def iniciar_actualizacion(tipo: str, detalle: str = "") -> int:
    inicializar()
    with conexion() as con:
        cur = con.execute("INSERT INTO actualizaciones (tipo, inicio, estado, detalle) VALUES (?, ?, 'en curso', ?)",
                          (tipo, ahora(), detalle))
        return cur.lastrowid


def cerrar_actualizacion(id_: int, estado: str, detalle: str = "") -> None:
    with conexion() as con:
        con.execute("UPDATE actualizaciones SET fin = ?, estado = ?, detalle = ? WHERE id = ?",
                    (ahora(), estado, detalle, id_))


def leer_actualizaciones(limite: int = 50) -> pd.DataFrame:
    inicializar()
    with conexion() as con:
        return pd.read_sql_query("SELECT * FROM actualizaciones ORDER BY id DESC LIMIT ?", con, params=(limite,))


# ── Universo ─────────────────────────────────────────────────────────────
def guardar_universo(u: pd.DataFrame) -> None:
    inicializar()
    with conexion() as con:
        con.executemany(
            "INSERT OR REPLACE INTO activos (ticker, empresa, sector, subindustria, mercado) VALUES (?, ?, ?, ?, ?)",
            [(r.ticker, r.empresa, _txt(r.sector), _txt(getattr(r, "subindustria", None)), r.mercado)
             for r in u.itertuples()])
        con.executemany(
            """INSERT INTO calculos_universo (ticker, fecha_calculo, ventana_inicio, ventana_fin, beta, n_meses,
               perfil, fuente, precio, fecha_precio, var_5d, var_21d, vol_anual)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(r.ticker, r.fecha_consulta, r.ventana_inicio, r.ventana_fin, _num(r.beta), _int(r.n_meses),
              _txt(r.perfil), r.fuente, _num(r.precio), _txt(r.fecha_precio), _num(r.var_5d), _num(r.var_21d),
              _num(getattr(r, "vol_anual", None))) for r in u.itertuples()])


def leer_universo() -> pd.DataFrame | None:
    """Último cálculo de cada activo."""
    inicializar()
    with conexion() as con:
        u = pd.read_sql_query(
            """SELECT a.ticker, a.empresa, a.sector, a.subindustria, a.mercado, c.beta, c.n_meses, c.perfil,
                      c.precio, c.fecha_precio, c.var_5d, c.var_21d, c.vol_anual, c.fuente,
                      c.ventana_inicio, c.ventana_fin, c.fecha_calculo AS fecha_consulta
               FROM activos a JOIN calculos_universo c ON c.ticker = a.ticker
               WHERE c.id = (SELECT MAX(id) FROM calculos_universo c2 WHERE c2.ticker = a.ticker)""", con)
    return None if u.empty else u


def historial_beta(ticker: str) -> pd.DataFrame:
    inicializar()
    with conexion() as con:
        return pd.read_sql_query("SELECT fecha_calculo, beta, perfil, n_meses FROM calculos_universo "
                                 "WHERE ticker = ? ORDER BY id", con, params=(ticker,))


# ── Predicciones ─────────────────────────────────────────────────────────
def guardar_prediccion(p: dict) -> int:
    inicializar()
    columnas = ["ejecutada_en", "ticker", "empresa", "horizonte", "h", "fecha_ultimo_dato", "fecha_objetivo_estimada",
                "precio_inicial", "rend_pronosticado", "rend_inferior", "rend_superior", "precio_pronosticado",
                "precio_inferior", "precio_superior", "nivel_intervalo", "modelo", "orden", "criterio",
                "metricas", "ajustes", "datos", "version"]
    valores = [(_json(p[c]) if c in ("metricas", "ajustes", "datos") else _limpio(p.get(c))) for c in columnas]
    with conexion() as con:
        cur = con.execute(f"INSERT INTO predicciones ({', '.join(columnas)}) VALUES ({', '.join('?' * len(columnas))})",
                          valores)
        return cur.lastrowid


def leer_predicciones(ticker: str | None = None) -> pd.DataFrame:
    inicializar()
    consulta = """SELECT p.*, r.evaluada_en, r.fecha_real, r.precio_real, r.rend_real, r.error_abs_precio,
                         r.error_pct_precio, r.error_rend, r.acierto_direccion, r.posicion_intervalo
                  FROM predicciones p LEFT JOIN realizaciones r ON r.prediccion_id = p.id"""
    with conexion() as con:
        if ticker:
            return pd.read_sql_query(consulta + " WHERE p.ticker = ? ORDER BY p.id DESC", con, params=(ticker,))
        return pd.read_sql_query(consulta + " ORDER BY p.id DESC", con)


def pendientes(ticker: str) -> list[tuple]:
    inicializar()
    with conexion() as con:
        return con.execute(
            """SELECT p.id, p.fecha_ultimo_dato, p.h, p.precio_inicial, p.rend_pronosticado, p.rend_inferior,
                      p.rend_superior, p.precio_pronosticado
               FROM predicciones p LEFT JOIN realizaciones r ON r.prediccion_id = p.id
               WHERE p.ticker = ? AND r.prediccion_id IS NULL""", (ticker,)).fetchall()


def guardar_realizacion(r: dict) -> None:
    with conexion() as con:
        con.execute(
            """INSERT OR IGNORE INTO realizaciones (prediccion_id, evaluada_en, fecha_real, precio_real, rend_real,
               error_abs_precio, error_pct_precio, error_rend, acierto_direccion, posicion_intervalo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["prediccion_id"], ahora(), r["fecha_real"], r["precio_real"], r["rend_real"], r["error_abs_precio"],
             r["error_pct_precio"], r["error_rend"], r["acierto_direccion"], r["posicion_intervalo"]))


# ── Evaluaciones ─────────────────────────────────────────────────────────
def guardar_evaluacion(ticker: str, horizonte: str, modelo: str, metricas: dict, ajustes: dict, datos: dict) -> None:
    inicializar()
    with conexion() as con:
        con.execute("INSERT INTO evaluaciones (ejecutada_en, ticker, horizonte, modelo, metricas, ajustes, datos) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ahora(), ticker, horizonte, modelo, _json(metricas), _json(ajustes), _json(datos)))


def leer_evaluaciones(ticker: str | None = None) -> pd.DataFrame:
    inicializar()
    with conexion() as con:
        if ticker:
            return pd.read_sql_query("SELECT * FROM evaluaciones WHERE ticker = ? ORDER BY id DESC", con, params=(ticker,))
        return pd.read_sql_query("SELECT * FROM evaluaciones ORDER BY id DESC", con)


def resumen_base() -> dict:
    inicializar()
    with conexion() as con:
        cuenta = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return {t: cuenta(t) for t in ("activos", "calculos_universo", "actualizaciones", "predicciones",
                                       "realizaciones", "evaluaciones")}


# ── utilidades ───────────────────────────────────────────────────────────
def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _int(x):
    v = _num(x)
    return None if v is None else int(v)


def _txt(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _limpio(x):
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if hasattr(x, "isoformat") and not isinstance(x, str):
        return x.isoformat()
    return x
