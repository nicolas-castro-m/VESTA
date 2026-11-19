from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from vesta import servicios as S
from vesta.config import HORIZONTES
from vesta.estadistica import pruebas as P
from vesta.ui import graficos as G
from vesta.ui.componentes import (ajustes, aviso, cabecera, con_signo, info_activo, mostrar_pruebas, mostrar_tabla,
                                  num, pct, pie, selector_horizonte, tarjeta, tarjetas)

ticker = (st.session_state.get("ticker") or "").strip().upper()
aj = ajustes()
if not ticker:
    aviso("Elija un activo en la barra lateral.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def datos_activo(tk: str, forzar: int = 0):
    return S.cargar_activo(tk, forzar=bool(forzar))


try:
    datos = datos_activo(ticker, st.session_state.get(f"forzar_{ticker}", 0))
except Exception as exc:
    cabecera("Predicciones", "Sin datos", ticker)
    st.error(f"No fue posible obtener datos de {ticker}: {exc}", icon=":material/error:")
    st.stop()

info = info_activo(ticker)
cabecera("Predicciones", info.get("empresa") or ticker, ticker, datos)
hz = selector_horizonte()
h = HORIZONTES[hz]


def _cambiar_criterio():
    elegido = st.session_state.get("_criterio_w")
    if elegido:  # el control segmentado permite deseleccionar; en ese caso se conserva el criterio actual
        st.session_state["ajustes"] = replace(ajustes(), criterio=elegido.lower())


st.session_state["_criterio_w"] = aj.criterio.upper()
st.segmented_control(
    "Criterio de selección del ARIMA", ["BIC", "AIC"], key="_criterio_w", on_change=_cambiar_criterio,
    help="BIC (el del documento) castiga más los parámetros y en acciones líquidas suele elegir la caminata "
         "aleatoria, ARIMA(0,1,0). AIC castiga menos y elige modelos con más rezagos. En ambos casos la "
         "evaluación fuera de muestra dice si el modelo realmente mejora a la caminata aleatoria.")
aj = ajustes()
st.session_state.setdefault("predicciones", {})
clave = (ticker, hz, aj.firma())

# ── Calidad y botón ──────────────────────────────────────────────────────
calidad = S.revisar_calidad(datos)
c1, c2 = st.columns([3, 1])
with c1:
    estado = "apta para modelar" if calidad.apta else "no apta para modelar"
    st.markdown(f'<div class="vq-nota">Serie de {calidad.observaciones} sesiones ({calidad.inicio:%Y-%m-%d} a '
                f'{calidad.fin:%Y-%m-%d}) · <b>{estado}</b> · modelo con {aj.anios_historia} años de historia, '
                f'{aj.prop_train:.0%} para entrenamiento, ARIMA(p,{aj.d},q) con p ≤ {aj.max_p}, q ≤ {aj.max_q}, '
                f'criterio {aj.criterio.upper()}, reajuste cada {aj.reestimar_cada} origen(es).</div>',
                unsafe_allow_html=True)
    for e in calidad.errores:
        st.error(e, icon=":material/block:")
    for a in calidad.avisos:
        st.caption(f"Datos: {a}")
    if datos.aviso:
        st.warning(datos.aviso, icon=":material/cloud_off:")
with c2:
    pulsado = st.button("Generar predicción", type="primary", icon=":material/play_arrow:", use_container_width=True,
                        disabled=not calidad.apta)

if pulsado:
    primera = not S.modelacion_disponible(ticker, datos.precios, aj)
    with st.status(f"Generando la predicción {hz.lower()} de {ticker}…", expanded=True) as status:
        if primera:
            st.write("Primera ejecución con estos ajustes: se reestiman los modelos en cada origen de la prueba. "
                     "Puede tardar alrededor de un minuto; después queda guardado.")
        barra = st.progress(0.0, text="Preparando")
        try:
            res = S.generar_prediccion(ticker, info.get("empresa") or ticker, datos, hz, aj,
                                       progreso=lambda f, t: barra.progress(min(max(f, 0.0), 1.0), text=t))
        except Exception as exc:
            status.update(label="No se pudo generar la predicción", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label=f"Predicción {hz.lower()} generada y guardada (registro #{res.id_registro})",
                      state="complete", expanded=False)
    st.session_state["predicciones"][clave] = res
    st.toast("Predicción guardada en el historial", icon=":material/check_circle:")

res = st.session_state["predicciones"].get(clave)
if res is None:
    aviso(f"Pulse «Generar predicción» para ajustar los modelos de {ticker} y pronosticar el horizonte {hz.lower()} "
          f"({h} días bursátiles). Cambiar de horizonte no ejecuta predicciones por sí solo.")
    pie()
    st.stop()

pr = res.prediccion
m = res.metricas
mod = res.modelacion

# ── Resultados ───────────────────────────────────────────────────────────
rend_txt, rend_clase = con_signo(pr.rend_final, 2, "")
tarjetas(f"Resultado · {ticker} · {hz} ({h} días bursátiles)", [
    tarjeta("Último precio observado", num(pr.precio_actual), "USD",
            f"Precio ajustado al {pr.fecha_ultimo_dato:%d %b %Y} (último dato, no tiempo real)."),
    tarjeta("Rentabilidad estimada", rend_txt, "%", f"Rendimiento logarítmico acumulado esperado a {h} días.",
            clase=rend_clase,
            referencia=f"Intervalo {pr.nivel:.0%}: {pct(pr.rend_inf[-1], 2, True)} % a {pct(pr.rend_sup[-1], 2, True)} %"),
    tarjeta("Precio proyectado", num(pr.precio_final), "USD", "P(T) · exp(suma de rendimientos pronosticados).",
            referencia=f"Intervalo {pr.nivel:.0%}: {num(pr.precio_inf[-1])} a {num(pr.precio_sup[-1])} USD"),
    tarjeta("Modelo seleccionado", pr.etiqueta, "", f"Especificación por {aj.criterio.upper()} con el entrenamiento.",
            referencia=f"Fin estimado del horizonte: {pr.fecha_objetivo:%d %b %Y} (días hábiles, sin festivos)"),
])
st.plotly_chart(G.trayectoria(datos.precios, pr), use_container_width=True)

veredicto = m["Veredicto"]
if pr.modelo.p == 0 and pr.modelo.q == 0:
    st.info(f"Con el criterio {aj.criterio.upper()}, ningún término AR o MA de {ticker} mejoró el ajuste lo "
            "suficiente para justificar su costo: el modelo elegido es la caminata aleatoria, ARIMA(0,1,0). "
            "No es un error; es la conclusión de los datos (eficiencia débil del mercado)."
            + (" Puede probar el criterio AIC arriba para ver una especificación más rica y si mejora fuera de muestra."
               if aj.criterio == "bic" else ""), icon=":material/info:")
tarjetas("Desempeño fuera de muestra del modelo seleccionado", [
    tarjeta("MAE", num(m["MAE"] * 100, 2), "p.p.", "Error absoluto medio del rendimiento acumulado. Rango 0 a ∞; ideal 0. "
            "Su magnitud depende de la escala y el horizonte.", referencia=f"Caminata aleatoria: {num(m['MAE_RW'] * 100, 2)} p.p."),
    tarjeta("RMSE", num(m["RMSE"] * 100, 2), "p.p.", "Raíz del error cuadrático medio; penaliza más los errores grandes. "
            "Rango 0 a ∞; ideal 0.", referencia=f"U de Theil = {num(m['U_Theil'], 3)} · menor que 1 mejora a la caminata"),
    tarjeta("Acierto direccional", num(m["Direccion_pct"], 1) if np.isfinite(m["Direccion_pct"]) else "—", "%",
            "Porcentaje de orígenes con el signo correcto (0 a 100 %). Superar el 50 % no basta sin significancia.",
            referencia=(f"Binomial frente al 50 %: p = {num(m['Direccion_p_binomial'], 3)} · periodos positivos "
                        f"{num(m['Pct_periodos_positivos'], 1)} %" if np.isfinite(m["Direccion_pct"]) else
                        "Pronóstico nulo: sin dirección (igual que la caminata aleatoria)")),
    tarjeta("Frente a la caminata aleatoria", veredicto, "",
            f"Diebold-Mariano sobre {m['Origenes']} orígenes rodantes, mismas fechas y misma variable.",
            referencia=(f"DM = {num(m['DM_estadistico'], 3)} · p = {num(m['DM_p_valor'], 4)}"
                        if np.isfinite(m["DM_p_valor"]) else "Prueba no aplicable")),
])

# ── Diagnóstico del modelo ───────────────────────────────────────────────
with st.expander("Diagnóstico del modelo", icon=":material/biotech:"):
    b1, b2, b3, b4 = st.tabs(["11.1 Información técnica", "11.2 Diagnóstico estadístico", "11.3 Desempeño predictivo",
                              "11.4 Incertidumbre"])
    with b1:
        mo = pr.modelo
        info_tec = pd.DataFrame([
            ("Modelo", pr.etiqueta), ("Orden ARIMA(p,d,q)", f"({mo.p}, {mo.d}, {mo.q})"),
            ("Transformaciones", "Y_t = ln(Adj Close); r_t = ΔY_t (ARIMA con d = 1 sobre Y_t ≡ ARMA sobre r_t)"),
            ("Variable modelada", "Rentabilidad logarítmica diaria r_t; el horizonte se obtiene sumando 1..h pasos"),
            ("Estimación", "CSS-ML: CSS para valores iniciales y máxima verosimilitud exacta (filtro de Kalman)"),
            ("Observaciones", f"{mod.n} sesiones"),
            ("Periodo histórico", f"{mod.inicio:%Y-%m-%d} a {mod.fin:%Y-%m-%d}"),
            ("Entrenamiento / prueba", f"{mod.corte} / {mod.n - mod.corte} sesiones · corte {mod.fecha_corte:%Y-%m-%d}"),
            ("Horizonte de pronóstico", f"{hz}: {h} días bursátiles"),
            ("Criterio de especificación", f"{aj.criterio.upper()} (se informan AIC y BIC)"),
            ("Evaluación", f"Rodante, reajuste cada {aj.reestimar_cada} origen(es), {m['Origenes']} orígenes"),
            ("σ² diaria", f"{mo.sigma2:.3e}"),
            ("Ejecución", res.ejecutada_en), ("Modelación calculada", mod.ejecutada_en),
        ], columns=["Campo", "Valor"])
        st.dataframe(info_tec, hide_index=True, use_container_width=True)
        coef = mo.coeficientes()
        if coef:
            tabla = pd.DataFrame({"Parámetro": list(coef), "Estimación": list(coef.values())})
            if mo.errores_estandar is not None and len(mo.errores_estandar) == len(coef):
                tabla["Error estándar"] = mo.errores_estandar
                tabla["z"] = tabla["Estimación"] / tabla["Error estándar"]
            mostrar_tabla(tabla, {"Estimación": "%.6f", "Error estándar": "%.6f", "z": "%.2f"})
        else:
            st.caption("Sin parámetros AR/MA: el pronóstico del rendimiento acumulado es 0 (caminata aleatoria).")
        st.markdown("**Criterios de información de todos los candidatos** (entrenamiento)")
        mostrar_tabla(mod.seleccion.candidatos, {"AIC": "%.2f", "BIC": "%.2f", "Delta_AIC": "%.2f", "Delta_BIC": "%.2f",
                                                 "LogLik": "%.2f", "Sigma2": "%.3e", "Raiz_min": "%.3f"}, alto=300)
        st.caption(f"Mejor AIC: {mod.seleccion.mejor_aic.etiqueta} · mejor BIC: {mod.seleccion.mejor_bic.etiqueta}. "
                   "AIC y BIC miden ajuste penalizado dentro de muestra; no son métricas de pronóstico. Se descartan "
                   "los modelos con raíces de módulo < 1.01, como auto.arima.")
    with b2:
        st.markdown("**Estacionariedad** (log-precio y su primera diferencia, periodo del modelo)")
        mostrar_pruebas(res.estacionariedad, aj.alfa, "Log-precio / rentabilidad diaria")
        st.markdown("**Residuos del modelo final**")
        mostrar_pruebas(res.diagnostico_residuos, aj.alfa, "Residuos estandarizados")
        st.caption(f"Ljung-Box con grados de libertad corregidos por los {pr.modelo.p + pr.modelo.q} parámetros AR/MA. "
                   "Que una prueba no rechace no hace confiable al modelo; la evidencia de pronóstico está en 11.3.")
        st.plotly_chart(G.residuos(res.residuos), use_container_width=True)
        banda = P.banda_confianza(len(res.residuos))
        c1, c2 = st.columns(2)
        c1.plotly_chart(G.correlograma(P.acf(res.residuos, aj.lag_max_acf), banda, "ACF de los residuos"),
                        use_container_width=True)
        c2.plotly_chart(G.correlograma(P.pacf(res.residuos, aj.lag_max_acf), banda, "PACF de los residuos"),
                        use_container_width=True)
    with b3:
        cols = ["Modelo", "Origenes", "MAE", "MAE_RW", "RMSE", "RMSE_RW", "U_Theil", "MAPE_precio", "MAPE_precio_RW",
                "Direccion_pct", "Direccion_p_binomial", "Pct_periodos_positivos", "DM_estadistico", "DM_p_valor",
                "Cobertura_intervalo", "Veredicto"]
        mostrar_tabla(res.metricas_todos[cols], {c: "%.5f" for c in ["MAE", "MAE_RW", "RMSE", "RMSE_RW"]}
                      | {"U_Theil": "%.4f", "MAPE_precio": "%.2f", "MAPE_precio_RW": "%.2f", "Direccion_pct": "%.1f",
                         "Direccion_p_binomial": "%.4f", "Pct_periodos_positivos": "%.1f", "DM_estadistico": "%.3f",
                         "DM_p_valor": "%.4f", "Cobertura_intervalo": "%.1f"})
        st.caption("Misma variable (rendimiento log acumulado), mismo horizonte, mismos orígenes y mismas observaciones "
                   "para todos los modelos. MAE y RMSE en rendimiento; MAPE sobre el precio reconstruido (el MAPE de un "
                   "rendimiento cercano a cero no está bien definido). Diebold-Mariano con corrección de Harvey-Leybourne-"
                   "Newbold y h − 1 rezagos; estadístico negativo favorece al modelo. Si el BIC y el AIC eligen modelos "
                   "distintos, ambos se evalúan.")
        principal = mod.principal()
        st.plotly_chart(G.evaluacion_rodante(principal.pronosticos, hz, principal.etiqueta), use_container_width=True)
        if not m["Identico_RW"]:
            st.plotly_chart(G.ventaja_acumulada(principal.pronosticos, hz), use_container_width=True)
    with b4:
        ancho = pr.precio_sup[-1] - pr.precio_inf[-1]
        st.markdown(
            f"El **intervalo de predicción del {pr.nivel:.0%}** para el precio al final del horizonte va de "
            f"**{num(pr.precio_inf[-1])}** a **{num(pr.precio_sup[-1])} USD** (amplitud {num(ancho)} USD, "
            f"{ancho / pr.precio_actual:.1%} del precio actual). Describe dónde se espera el precio futuro si el modelo "
            "y sus supuestos se cumplen; no es un intervalo de confianza de los parámetros.")
        st.markdown(
            "- **Construcción:** rendimiento acumulado ± z·ee, con ee² = σ²·Σ(Σψ)² a partir de los pesos ψ del ARMA; "
            "los límites del precio son exp de los límites del log-precio, por eso la banda es asimétrica.\n"
            "- **Supuestos:** errores normales, independientes y de varianza constante; parámetros tratados como "
            "conocidos (no incorpora la incertidumbre de estimación).\n"
            "- **Comprobación empírica:** en la evaluación rodante el valor real cayó dentro del intervalo en el "
            f"**{num(m['Cobertura_intervalo'], 1)} %** de los orígenes (nominal {pr.nivel:.0%}).")
        arch = [r for r in res.diagnostico_residuos if r.clave == "arch"]
        if arch and arch[0].p_valor < aj.alfa:
            st.warning("Los residuos muestran efectos ARCH: la varianza cambia en el tiempo y el intervalo, que la "
                       "supone constante, puede ser demasiado estrecho en periodos turbulentos.", icon=":material/warning:")
        st.caption("Un intervalo estrecho no implica más confiabilidad: su amplitud depende del activo, del horizonte, de "
                   "la escala del precio y del procedimiento. Las evidencias se presentan por separado; VESTA no "
                   "calcula ningún índice compuesto de confiabilidad.")

with st.expander("Réplica de la evaluación publicada en el documento (secciones 12.6 y 12.10)",
                 icon=":material/history_edu:"):
    st.markdown("Se reproduce literalmente para trazabilidad: rejilla p, q ≤ 5 sobre rendimientos solapados, un pronóstico "
                "multipaso desde el corte, y caminata aleatoria sobre precios con horizontes 1/21/63. Sus métricas no "
                "son comparables entre sí; ver Configuración → Metodología.")
    if S.replica_disponible(ticker, datos.precios):
        rep = S.replica_documento(ticker, datos.precios)
    else:
        rep = None
        st.caption("Ajusta 72 modelos por horizonte sobre series solapadas: puede tardar dos o tres minutos.")
        if st.button("Calcular réplica", icon=":material/play_arrow:"):
            barra = st.progress(0.0, text="Preparando")
            rep = S.replica_documento(ticker, datos.precios, progreso=lambda f, t: barra.progress(min(f, 1.0), text=t))
            barra.empty()
    if rep is not None:
        mostrar_tabla(rep["modelos"].drop(columns=["Activo"]), {"AIC": "%.2f", "BIC": "%.2f"})
        mostrar_tabla(rep["metricas"].drop(columns=["Activo"]), {"MAE": "%.5f", "RMSE": "%.5f", "Sesgo": "%.5f",
                                                                  "Direccion": "%.2f", "Pct_reales_positivos": "%.1f",
                                                                  "Cobertura_95": "%.1f"})
        st.caption("Compare 'Direccion' con 'Pct_reales_positivos': un pronóstico multipaso converge a la media y el "
                   "acierto de dirección termina reflejando solo el signo de esa media.")
        st.plotly_chart(G.replica_pronostico(rep["predicciones"], hz), use_container_width=True)
        mostrar_tabla(rep["rw_metricas"].drop(columns=["Activo"]), {"MAE": "%.3f", "RMSE": "%.3f", "MAPE": "%.3f"})

for a in res.avisos:
    st.caption(f"Aviso de datos al predecir: {a}")
pie()
