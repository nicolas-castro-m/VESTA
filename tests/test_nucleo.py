"""Pruebas del núcleo estadístico y de modelación. Ejecutar: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vesta.datos.universo import asignar_perfil, calcular_beta, muestra_estratificada, rend_mensual  # noqa: E402
from vesta.estadistica import pruebas as P  # noqa: E402
from vesta.evaluacion import metricas as M  # noqa: E402
from vesta.ajustes import Ajustes  # noqa: E402
from vesta.evaluacion import rodante  # noqa: E402
from vesta.modelos import arma as A  # noqa: E402
from vesta.modelos.seleccion import auto_arima  # noqa: E402
from vesta.procesamiento.rendimientos import preparar_activo, preparar_serie  # noqa: E402

RNG = np.random.default_rng(2026)


def serie_ar1(n=1255, phi=0.25, mu=0.0004, sd=0.015, semilla=7):
    rng = np.random.default_rng(semilla)
    r = np.zeros(n)
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + e[t]
    return pd.DataFrame({"Date": pd.bdate_range("2021-09-07", periods=n), "Adj.Close": 100 * np.exp(np.cumsum(r))})


# ── Rendimientos ─────────────────────────────────────────────────────────
def test_rendimientos_log_y_acumulados():
    d = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=70), "Adj.Close": np.linspace(100, 170, 70)})
    a = preparar_activo(d)
    assert np.isnan(a["rentabilidad"].iloc[0])
    assert np.isclose(a["rentabilidad"].iloc[1], np.log(d["Adj.Close"].iloc[1] / 100))
    assert a["rent_sem"].isna().sum() == 5 and a["rent_mes"].isna().sum() == 21 and a["rent_trim"].isna().sum() == 63
    # el acumulado es la suma de los diarios
    assert np.isclose(a["rent_sem"].iloc[10], a["rentabilidad"].iloc[6:11].sum())


# ── Pruebas ──────────────────────────────────────────────────────────────
def test_rachas_alternante_rechaza():
    x = np.tile([1.0, -1.0], 100)
    r = P.rachas(x)
    assert r.estadistico > 10 and r.p_valor < 1e-6


def test_lilliefors_estadistico_igual_ks():
    x = RNG.standard_t(4, 800)
    z = (x - x.mean()) / x.std(ddof=1)
    assert np.isclose(P.lilliefors(x).estadistico, stats.kstest(z, "norm").statistic)
    assert P.lilliefors(x).p_valor < 0.05


def test_lilliefors_normal_no_rechaza():
    assert P.lilliefors(np.random.default_rng(1).normal(size=500)).p_valor > 0.05


def test_kruskal_igual_scipy():
    g = {"a": RNG.normal(0, 1, 50), "b": RNG.normal(0.5, 1, 60), "c": RNG.normal(0, 2, 40)}
    assert np.isclose(P.kruskal_wallis(g).estadistico, stats.kruskal(*g.values()).statistic)


def test_cochran_ejemplo_clasico():
    # Ejemplo de libro (Siegel & Castellan): Q conocido por cálculo directo
    M_ = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0]])
    k = 3
    C = M_.sum(0); R = M_.sum(1); N = M_.sum()
    q_esperado = (k - 1) * (k * (C**2).sum() - N**2) / (k * N - (R**2).sum())
    r, exitos, bloques = P.cochran_q(M_)
    assert np.isclose(r.estadistico, q_esperado) and bloques == 6 and list(exitos) == [5, 3, 1]


def test_adf_y_kpss_distinguen_raiz_unitaria():
    caminata = np.cumsum(RNG.normal(size=1000))
    ruido = RNG.normal(size=1000)
    assert P.adf(caminata).p_valor > 0.1 and P.adf(ruido).p_valor == 0.01
    assert P.kpss(caminata).p_valor == 0.01 and P.kpss(ruido).p_valor == 0.1


def test_ljung_box_formula():
    x = RNG.normal(size=300)
    r = P.acf(x, 10)
    n = len(x)
    q = n * (n + 2) * sum(r[k] ** 2 / (n - k - 1) for k in range(10))
    assert np.isclose(P.ljung_box(x, 10).estadistico, q)


def test_pacf_coincide_con_regresion():
    x = serie_ar1()["Adj.Close"].pipe(np.log).diff().dropna().to_numpy()
    X = np.column_stack([np.ones(len(x) - 2), x[1:-1], x[:-2]])
    b = np.linalg.lstsq(X, x[2:], rcond=None)[0]
    assert abs(P.pacf(x, 5)[1] - b[2]) < 0.01


def test_arch_detecta_garch():
    n, w, a, b = 2000, 1e-5, 0.15, 0.8
    rng = np.random.default_rng(3)
    s2, x = np.empty(n), np.empty(n)
    s2[0] = w / (1 - a - b)
    for t in range(n):
        if t:
            s2[t] = w + a * x[t - 1] ** 2 + b * s2[t - 1]
        x[t] = np.sqrt(s2[t]) * rng.normal()
    assert P.arch_lm(x, 10).p_valor < 0.01
    assert P.arch_lm(rng.normal(size=2000), 10).p_valor > 0.01


# ── ARMA ─────────────────────────────────────────────────────────────────
def test_arma_recupera_parametros_y_atajo_exacto():
    rng = np.random.default_rng(1)
    n = 1400
    e = rng.normal(0, 0.02, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + e[t] + 0.4 * e[t - 1]
    y = y[200:]
    m = A.ajustar_arma(y, 2, 1, con_media=False)
    assert np.allclose(m.phi, [0.5, -0.3], atol=0.08) and abs(m.theta[0] - 0.4) < 0.08
    rapido = A.filtro_kalman(y, m.phi, m.theta)
    tol, A.TOL_CONVERGENCIA = A.TOL_CONVERGENCIA, -1
    try:
        completo = A.filtro_kalman(y, m.phi, m.theta)
    finally:
        A.TOL_CONVERGENCIA = tol
    assert np.max(np.abs(rapido.v - completo.v)) < 1e-8


def test_d1_equivale_a_rentabilidad_sin_media():
    lp = np.log(serie_ar1(600)["Adj.Close"].to_numpy())
    m1 = A.ajustar_arma(lp, 1, 0, d=1)
    m0 = A.ajustar_arma(np.diff(lp), 1, 0, con_media=False)
    assert np.isclose(m1.bic, m0.bic, atol=1e-6)


def test_partrans_inversa():
    phi = np.array([0.5, -0.2, 0.1])
    assert np.allclose(A.partrans(A.invpartrans(phi)), phi)


def test_pronostico_acumulado_es_suma():
    m = A.ajustar_arma(np.diff(np.log(serie_ar1(800)["Adj.Close"].to_numpy())), 1, 0, con_media=True)
    media, ee, inf_, sup = A.pronostico_acumulado(m, 5)
    assert np.isclose(media, A.pronosticar(m, 5).media.sum()) and inf_ < media < sup


def test_rodante_sin_mirar_futuro_y_detecta_ar():
    d = serie_ar1()
    s = preparar_serie(d, "SIM")
    sel = auto_arima(s.r_train, d=1, max_p=2, max_q=2)
    assert sel.mejor.p >= 1
    aj = Ajustes(reestimar_cada=21)
    res = rodante.evaluar(s, [("Mejor BIC", sel.mejor)], aj)
    s2 = preparar_serie(d.iloc[:1150], "SIM")
    s2.corte, s2.r_train = s.corte, s.r_train
    res2 = rodante.evaluar(s2, [("Mejor BIC", sel.mejor)], aj)
    a = res.modelos[0].pronosticos.query("Horizonte == 'Semanal'").set_index("Fecha_Origen")["Pred_Modelo"]
    b = res2.modelos[0].pronosticos.query("Horizonte == 'Semanal'").set_index("Fecha_Origen")["Pred_Modelo"]
    assert np.allclose(a.loc[b.index], b)
    assert res.modelos[0].metricas.set_index("Horizonte").loc["Semanal", "U_Theil"] < 1


# ── Métricas ─────────────────────────────────────────────────────────────
def test_diebold_mariano_signo():
    real = RNG.normal(size=300)
    bueno = real + RNG.normal(0, 0.3, 300)
    malo = real + RNG.normal(0, 1.0, 300)
    dm = M.diebold_mariano(real - bueno, real - malo, h=1)
    assert dm["estadistico"] < 0 and dm["p_valor"] < 0.01


def test_direccion_ignora_ceros():
    d = M.acierto_direccion([0.1, -0.2, 0.3], [0.0, 0.0, 0.0])
    assert d["n"] == 0 and np.isnan(d["porcentaje"])


# ── Universo ─────────────────────────────────────────────────────────────
def test_beta_y_perfiles():
    fechas = pd.bdate_range("2020-01-01", periods=1300)
    rm = RNG.normal(0, 0.01, 1300)
    ra = 1.5 * rm + RNG.normal(0, 0.001, 1300)
    m = pd.DataFrame({"Date": fechas, "Adj.Close": 100 * np.exp(np.cumsum(rm))})
    a = pd.DataFrame({"Date": fechas, "Adj.Close": 100 * np.exp(np.cumsum(ra))})
    b, n = calcular_beta(rend_mensual(a), rend_mensual(m))
    assert abs(b - 1.5) < 0.05 and n >= 48
    assert [asignar_perfil(x) for x in (0.79, 0.8, 1.19, 1.2)] == ["Defensivo", "Neutral", "Neutral", "Agresivo"]


def test_muestreo_siete_por_perfil():
    u = pd.DataFrame({"ticker": [f"T{i}" for i in range(90)], "beta": np.linspace(0.1, 2.0, 90)})
    u["perfil"] = u["beta"].map(asignar_perfil)
    mu = muestra_estratificada(u)
    assert mu.groupby("perfil").size().tolist() == [7, 7, 7] and mu["ticker"].is_unique


