import numpy as np
import pandas as pd
import streamlit as st

from vesta import servicios as S
from vesta.config import HORIZONTES, VENTANAS_HISTORIA
from vesta.datos import universo as U
from vesta.estadistica.descriptiva import clasificar_volatilidad
from vesta.ui import graficos as G
from vesta.ui.componentes import (ajustes, aviso, cabecera, con_signo, ficha, info_activo, mostrar_pruebas,
                                  mostrar_tabla, num, pct, pie, selector_horizonte, selector_ventana, tarjeta, tarjetas,
                                  universo_cacheado)

ticker = (st.session_state.get("ticker") or "").strip().upper()
aj = ajustes()
if not ticker:
    aviso("Elija un activo en la barra lateral o en el Explorador.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def datos_activo(tk: str, forzar: int = 0):
    return S.cargar_activo(tk, forzar=bool(forzar))


@st.cache_data(show_spinner=False)
def analisis_horizonte(tk: str, fecha: str, horizonte: str, construccion: str, firma: str, _precios, _aj):
    return S.analisis_horizonte(_precios, horizonte, construccion, _aj)


with st.spinner(f"Consultando datos de {ticker}…"):
    try:
        datos = datos_activo(ticker, st.session_state.get(f"forzar_{ticker}", 0))
    except Exception as exc:
        cabecera("Análisis del activo", "Sin datos", ticker)
        st.error(f"No fue posible obtener datos de {ticker}: {exc}", icon=":material/error:")
        st.stop()

precios = datos.precios
fecha = S.ultima_fecha(precios)
info = info_activo(ticker)
cabecera("Análisis del activo", info.get("empresa") or ticker, ticker, datos)
if datos.aviso:
    st.warning(datos.aviso, icon=":material/cloud_off:")
calidad = S.revisar_calidad(datos)
for a in calidad.avisos:
    st.caption(f"Datos: {a}")

# ── Resumen general ──────────────────────────────────────────────────────
ultimo = float(precios["Close"].iloc[-1])
var_d = float(precios["Adj.Close"].iloc[-1] / precios["Adj.Close"].iloc[-2] - 1) if len(precios) > 1 else np.nan
texto_var, clase_var = con_signo(var_d)
ficha([
    ("Empresa", info.get("empresa") or "—"), ("Ticker", ticker), ("Sector", info.get("sector") or "—"),
    ("Mercado", info.get("mercado") or "—"), ("Último cierre (USD)", num(ultimo)),
    ("Variación última sesión", f'<span class="{clase_var}">{texto_var}</span>'),
    ("Beta", num(info.get("beta"))), ("Perfil", info.get("perfil") or "—"),
    ("Datos hasta", f"{pd.Timestamp(precios['Date'].max()):%Y-%m-%d}"),
])

# ── Comportamiento histórico (ventana independiente del horizonte) ──────
c_v, c_m = st.columns([2, 1])
with c_v:
    ventana = selector_ventana(VENTANAS_HISTORIA)
with c_m:
    modo = st.segmented_control("Vista", ["Precio", "Rentabilidad acumulada"], default="Precio", key="modo_precio")
vh = S.vista_historica(precios, VENTANAS_HISTORIA[ventana])
if vh["sesiones"] < 0.9 * 252 * VENTANAS_HISTORIA[ventana]:
    st.caption(f"Hay {vh['sesiones']} sesiones disponibles, menos de las que cubre la ventana elegida.")
g1, g2 = st.columns([3, 2])
g1.plotly_chart(G.precio_ventana(vh["datos"], "acumulada" if modo == "Rentabilidad acumulada" else "precio"),
                use_container_width=True)
g2.plotly_chart(G.volatilidad_movil(vh["datos"]), use_container_width=True)

# ── Horizonte de inversión ───────────────────────────────────────────────
st.markdown("## Horizonte de inversión")
hz = selector_horizonte()
h = HORIZONTES[hz]
construccion = st.session_state.get("construccion", "solapada")
ah = analisis_horizonte(ticker, fecha, hz, construccion, aj.firma(), precios, aj)
st.plotly_chart(G.serie_horizonte(ah["serie"], hz, h, construccion), use_container_width=True)

# ── Indicadores principales ──────────────────────────────────────────────
u = universo_cacheado()
percentil = U.percentil_volatilidad(ah["vol_diaria_anual"], u)
vol_h = ah["volatilidad"]["empirica"]
rent_txt, rent_clase = con_signo(vh["rent_acumulada"], 1, "")
lista = [
    tarjeta("Rentabilidad acumulada", rent_txt, "%",
            f"{'Ganancia' if vh['rent_acumulada'] > 0 else 'Pérdida'} entre {vh['inicio']:%d %b %Y} y {vh['fin']:%d %b %Y} "
            f"(ventana de {ventana}), con precio ajustado por dividendos.", clase=rent_clase),
    tarjeta(f"Volatilidad · {hz.lower()}", pct(vol_h, 2), "%",
            f"Desviación estándar de los rendimientos acumulados de {h} días (periodos sin solapar, "
            f"n = {ah['volatilidad']['n_periodos']}). Mide cuánto varían los rendimientos; no su dirección.",
            referencia=(f"Volatilidad anual {pct(ah['vol_diaria_anual'], 1)} % · percentil {percentil:.0f} del S&P 500 · "
                        f"{clasificar_volatilidad(percentil).lower()}" if percentil is not None else
                        f"Volatilidad anual {pct(ah['vol_diaria_anual'], 1)} % · sin universo para comparar")),
]
tarjetas("Comportamiento del activo (descriptivo)", lista)

clave_pred = (ticker, hz, aj.firma())
pred = st.session_state.get("predicciones", {}).get(clave_pred)
if pred is None:
    st.markdown('<div class="vq-grupo">Desempeño predictivo (fuera de muestra)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vq-nota">MAE, RMSE y acierto direccional se calculan al generar una predicción para '
                f'{ticker} en el horizonte {hz.lower()}.</div>', unsafe_allow_html=True)
    if st.button("Ir a Predicciones", icon=":material/timeline:"):
        st.switch_page("paginas/predicciones.py")
else:
    m = pred.metricas
    tarjetas("Desempeño predictivo (fuera de muestra)", [
        tarjeta("MAE", num(m["MAE"] * 100, 2), "p.p.", "Error absoluto medio del rendimiento acumulado. Ideal: 0.",
                referencia=f"Caminata aleatoria: {num(m['MAE_RW'] * 100, 2)} p.p."),
        tarjeta("RMSE", num(m["RMSE"] * 100, 2), "p.p.", "Raíz del error cuadrático medio; penaliza más los errores grandes. Ideal: 0.",
                referencia=f"U de Theil {num(m['U_Theil'], 3)} (menor que 1: mejora a la caminata)"),
        tarjeta("Acierto direccional", num(m["Direccion_pct"], 1) if np.isfinite(m["Direccion_pct"]) else "—", "%",
                "Veces que el signo pronosticado coincidió con el real (0 a 100 %).",
                referencia=(f"Referencia 50 % · binomial p = {num(m['Direccion_p_binomial'], 3)}"
                            if np.isfinite(m["Direccion_pct"]) else "El modelo no pronostica dirección (equivale a la caminata)")),
    ])

# ── Más información estadística ──────────────────────────────────────────
with st.expander("Más información estadística", icon=":material/functions:"):
    st.session_state["_construccion_w"] = construccion
    st.radio("Construcción de la serie del horizonte", list(S.CONSTRUCCIONES), key="_construccion_w", horizontal=True,
             format_func=lambda c: S.CONSTRUCCIONES[c],
             on_change=lambda: st.session_state.update(construccion=st.session_state["_construccion_w"]))
    if construccion == "solapada":
        st.caption("Con la construcción solapada, observaciones consecutivas comparten h − 1 días: eso induce "
                   "autocorrelación artificial (afecta a rachas, Ljung-Box y ACF). Compare con la versión no solapada.")
    variable = f"Rend. log acumulado {h} d ({'solapado' if construccion == 'solapada' else 'no solapado'})"
    t1, t2, t3 = st.tabs(["9.1 Estadística descriptiva", "9.2 Pruebas estadísticas", "9.3 Dependencia temporal"])
    with t1:
        dsc = ah["descriptivos"]
        orden = ["n", "Media", "Mediana", "Desv_Estandar", "Varianza", "Minimo", "Q1", "Q3", "Maximo", "RIC",
                 "Asimetria", "Curtosis_exceso", "Pct_positivos"]
        mostrar_tabla(pd.DataFrame([{"Variable": variable, **{k: dsc.get(k) for k in orden}}]),
                      {k: "%.6f" for k in orden[1:11]} | {"Asimetria": "%.3f", "Curtosis_exceso": "%.3f",
                                                          "Pct_positivos": "%.1f"})
        st.caption("Cuartiles tipo 7 (como `quantile` en R). Curtosis en exceso: 0 para una normal.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(G.histograma(ah["serie"]["valor"], f"Distribución · {variable}"), use_container_width=True)
        c2.plotly_chart(G.qq(ah["serie"]["valor"], f"Q-Q frente a la normal · {variable}"), use_container_width=True)
    with t2:
        st.markdown(f"**Sobre la variable del horizonte** · {variable} · α = {aj.alfa}")
        mostrar_pruebas(ah["pruebas"], aj.alfa, variable)
        st.markdown("**Estacionariedad** · no depende del horizonte")
        est = ah["estacionariedad"]
        for r, etq in zip(est, ["ADF · log-precio", "KPSS · log-precio", "ADF · rentabilidad diaria",
                                 "KPSS · rentabilidad diaria"]):
            r.prueba = etq
        mostrar_pruebas(est, aj.alfa, "Log-precio y su primera diferencia")
        st.caption("No rechazar H0 no demuestra que sea verdadera: solo indica que los datos no aportan evidencia "
                   "suficiente en su contra. ADF y KPSS replican tseries; sus p-valores se interpolan en tablas y "
                   "quedan acotados (ADF 0.01–0.99, KPSS 0.01–0.10). α se cambia en Configuración.")
    with t3:
        c1, c2 = st.columns(2)
        c1.plotly_chart(G.correlograma(ah["acf"], ah["banda"], f"ACF · {variable}"), use_container_width=True)
        c2.plotly_chart(G.correlograma(ah["pacf"], ah["banda"], f"PACF · {variable}"), use_container_width=True)
        fuera_acf = [i + 1 for i, v in enumerate(ah["acf"]) if abs(v) > ah["banda"]]
        fuera_pacf = [i + 1 for i, v in enumerate(ah["pacf"]) if abs(v) > ah["banda"]]
        mostrar_tabla(pd.DataFrame([{"Rezagos evaluados": len(ah["acf"]), "Banda 95 % (±)": ah["banda"], "n": ah["n"],
                                     "Fuera de banda · ACF": ", ".join(map(str, fuera_acf)) or "ninguno",
                                     "Fuera de banda · PACF": ", ".join(map(str, fuera_pacf)) or "ninguno"}]),
                      {"Banda 95 % (±)": "%.4f"})
        st.caption("Banda ±1.96/√n bajo ruido blanco. Con muchos rezagos, alrededor del 5 % queda fuera por azar.")
    st.caption("La comparación entre activos (Kruskal-Wallis, Q de Cochran) se hace dentro de la muestra académica, "
               "en Comparación de activos.")

with st.sidebar:
    if st.button("Actualizar precios", icon=":material/refresh:", use_container_width=True,
                 help="Fuerza una nueva descarga desde Yahoo Finance."):
        st.session_state[f"forzar_{ticker}"] = st.session_state.get(f"forzar_{ticker}", 0) + 1
        st.rerun()
pie()
