"""Pruebas de integración del flujo de predicción, persistencia y reglas metodológicas.

Usan series SIMULADAS (identificadas como tales) para no depender de internet.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesta import servicios as S  # noqa: E402
from vesta.ajustes import Ajustes  # noqa: E402
from vesta.config import HORIZONTES  # noqa: E402
from vesta.datos import calidad  # noqa: E402
from vesta.datos.almacen import DatosActivo  # noqa: E402
from vesta.modelos import arma as A  # noqa: E402
from vesta.persistencia import cache, db, historial  # noqa: E402
from vesta.procesamiento.rendimientos import preparar_activo, preparar_serie  # noqa: E402


def serie_simulada(n=1255, phi=0.2, semilla=11, hasta=None):
    """Serie de precios SIMULADA (AR(1) en rendimientos). No representa ningún activo real."""
    rng = np.random.default_rng(semilla)
    r = np.zeros(n)
    e = rng.normal(0, 0.015, n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    fechas = pd.bdate_range(end=hasta or pd.Timestamp.now().normalize(), periods=n)
    p = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"Date": fechas, "Open": p, "High": p, "Low": p, "Close": p, "Adj.Close": p,
                         "Volume": 1e6})


@pytest.fixture()
def entorno(tmp_path, monkeypatch):
    db.usar_ruta(tmp_path / "prueba.sqlite")
    monkeypatch.setattr(cache, "DIR_RESULTADOS", tmp_path / "cache")
    yield tmp_path
    db.usar_ruta(tmp_path / "otra.sqlite")


def test_horizontes_son_5_21_63():
    assert HORIZONTES == {"Semanal": 5, "Mensual": 21, "Trimestral": 63}
    d = preparar_activo(serie_simulada(200))
    p = d["Adj.Close"].to_numpy()
    assert np.isclose(d["rent_mes"].iloc[100], np.log(p[100] / p[79]))
    assert np.isclose(d["rent_trim"].iloc[150], np.log(p[150] / p[87]))


def test_reconstruccion_del_precio():
    # P(T+h) = exp(ln P_T + sum r) = P_T * exp(sum r); nunca P_T * (1 + r_diaria)
    r = np.array([0.01, -0.02, 0.015])
    assert np.isclose(A.reconstruir_precio(50.0, r.sum()), 50 * np.exp(0.005))
    assert np.allclose(A.reconstruir_precio(50.0, np.cumsum(r)), 50 * np.exp(np.cumsum(r)))


def test_trayectoria_coincide_con_acumulado():
    d = serie_simulada(800)
    m = A.ajustar_arma(np.diff(np.log(d["Adj.Close"])), 1, 0, con_media=False)
    tr = A.trayectoria_acumulada(m, 21)
    media, ee, inf_, sup = A.pronostico_acumulado(m, 21)
    assert np.isclose(tr["media"][-1], media) and np.isclose(tr["ee"][-1], ee)
    assert np.all(np.diff(tr["ee"]) > 0)


def test_calidad_bloquea_series_cortas():
    d = serie_simulada(300)
    inf = calidad.revisar(d, d)
    assert not inf.apta and "se necesitan" in inf.errores[0]


def test_flujo_completo_de_prediccion(entorno):
    d = serie_simulada()
    datos = DatosActivo("SIMULADO", d, "prueba", pd.Timestamp.now(), crudo=d)
    aj = Ajustes(reestimar_cada=21, max_p=2, max_q=2)
    res = S.generar_prediccion("SIMULADO", "Serie simulada", datos, "Mensual", aj)
    pr = res.prediccion
    # precio = P_T * exp(acumulado) y el intervalo contiene al pronóstico
    assert np.isclose(pr.precio_final, pr.precio_actual * np.exp(pr.rend_final))
    assert pr.precio_inf[-1] < pr.precio_final < pr.precio_sup[-1]
    assert len(pr.fechas) == 22 and pr.h == 21
    # misma variable, horizonte y orígenes para modelo y caminata
    ev = res.modelacion.principal().pronosticos
    mens = ev[ev["Horizonte"] == "Mensual"]
    assert (mens["Pred_RW"] == 0).all() and mens["Fecha_Origen"].is_unique
    assert set(res.metricas) >= {"MAE", "RMSE", "U_Theil", "DM_p_valor", "Direccion_pct", "Veredicto"}
    # registro guardado con métricas, ajustes y datos para reproducir
    reg = db.leer_predicciones("SIMULADO").iloc[0]
    assert reg["modelo"] == pr.etiqueta and json.loads(reg["ajustes"])["reestimar_cada"] == 21
    assert "coeficientes" in json.loads(reg["datos"])


def test_seleccion_solo_con_entrenamiento(entorno):
    """El orden elegido no cambia si se alteran los datos de prueba (no hay fuga)."""
    d = serie_simulada()
    s = preparar_serie(d, "X")
    d2 = d.copy()
    d2.loc[s.corte + 5:, "Adj.Close"] *= np.exp(np.random.default_rng(3).normal(0, 0.05, len(d2) - s.corte - 5).cumsum())
    from vesta.modelos.seleccion import auto_arima
    s2 = preparar_serie(d2, "X")
    a = auto_arima(s.r_train, max_p=2, max_q=2).mejor
    b = auto_arima(s2.r_train, max_p=2, max_q=2).mejor
    assert (a.p, a.q) == (b.p, b.q) and np.isclose(a.bic, b.bic)


def test_predicciones_inmutables_y_realizacion(entorno):
    fechas = pd.bdate_range("2026-01-01", periods=40)
    precios = pd.DataFrame({"Date": fechas, "Adj.Close": np.linspace(100, 139, 40)})
    pid = historial.registrar({
        "ejecutada_en": "2026-01-10T10:00:00", "ticker": "X", "empresa": "X", "horizonte": "Semanal", "h": 5,
        "fecha_ultimo_dato": str(fechas[10].date()), "fecha_objetivo_estimada": str(fechas[15].date()),
        "precio_inicial": 110.0, "rend_pronosticado": 0.01, "rend_inferior": -0.05, "rend_superior": 0.07,
        "precio_pronosticado": 111.1, "precio_inferior": 104.6, "precio_superior": 118.0, "nivel_intervalo": 0.95,
        "modelo": "AR(1)", "orden": "(1,1,0)", "criterio": "BIC", "metricas": {}, "ajustes": {}, "datos": {},
        "version": "test"})
    with pytest.raises(sqlite3.DatabaseError):
        with db.conexion() as con:
            con.execute("UPDATE predicciones SET precio_pronosticado = 1 WHERE id = ?", (pid,))
    assert historial.actualizar_realizados("X", precios) == 1
    assert historial.actualizar_realizados("X", precios) == 0          # no se duplica
    fila = historial.leer("X").iloc[0]
    assert fila["precio_pronosticado"] == 111.1                       # el registro original no cambia
    assert np.isclose(fila["rend_real"], np.log(115 / 110)) and fila["acierto_direccion"] == 1
    assert fila["posicion_intervalo"] == "Dentro"


def test_ajustes_validados():
    assert Ajustes().validar() == []
    assert Ajustes(alfa=0.7).validar() and Ajustes(criterio="hqic").validar()
