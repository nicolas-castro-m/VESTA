import json

import streamlit as st

from vesta import servicios as S
from vesta.persistencia import historial as H
from vesta.ui import graficos as G
from vesta.ui.componentes import aviso, cabecera, mostrar_tabla, pie

cabecera("Seguimiento", "Historial de predicciones")
st.markdown('<div class="vq-nota">Cada predicción se guarda tal como se emitió y nunca se modifica. Cuando pasan los '
            'días del horizonte, se agrega su resultado real: es la prueba fuera de muestra más honesta.</div>',
            unsafe_allow_html=True)

h = H.leer()
if h.empty:
    aviso("Aún no hay predicciones. Abra Predicciones y pulse «Generar predicción».")
    st.stop()

c1, c2, c3 = st.columns([3, 2, 1])
tickers = sorted(h["ticker"].unique())
filtro = c1.multiselect("Activos", tickers, default=tickers)
horizontes = c2.multiselect("Horizontes", ["Semanal", "Mensual", "Trimestral"], default=["Semanal", "Mensual", "Trimestral"])
if c3.button("Evaluar vencidas", icon=":material/sync:", use_container_width=True,
             help="Descarga precios recientes y evalúa las predicciones cuyo horizonte ya terminó."):
    total = 0
    with st.spinner("Consultando precios…"):
        for tk in filtro:
            try:
                total += S.actualizar_seguimiento(tk, S.cargar_activo(tk).precios)
            except Exception as exc:
                st.warning(f"{tk}: {exc}")
    st.toast(f"{total} predicciones evaluadas", icon=":material/check_circle:")
    h = H.leer()

h = h[h["ticker"].isin(filtro) & h["horizonte"].isin(horizontes)]
st.markdown(f'<div class="vq-nota">{len(h)} predicciones · {int(h["rend_real"].notna().sum())} evaluadas · '
            f'{int(h["rend_real"].isna().sum())} pendientes</div>', unsafe_allow_html=True)

res = H.resumen(h)
if not res.empty:
    st.markdown("## Evaluación de las predicciones realizadas")
    mostrar_tabla(res, {"MAE_rend": "%.5f", "RMSE_rend": "%.5f", "Error_pct_precio": "%.2f", "Acierto_direccion": "%.1f",
                        "Dentro_intervalo": "%.1f"})
    st.plotly_chart(G.seguimiento(h), use_container_width=True)
    st.caption("Con pocas predicciones evaluadas estos porcentajes varían mucho; no son concluyentes hasta acumular "
               "decenas de observaciones por horizonte.")

st.markdown("## Registro")
columnas = ["id", "ejecutada_en", "ticker", "horizonte", "modelo", "fecha_ultimo_dato", "fecha_objetivo_estimada",
            "precio_inicial", "rend_pronosticado", "precio_pronosticado", "precio_inferior", "precio_superior",
            "fecha_real", "precio_real", "rend_real", "error_abs_precio", "error_pct_precio", "acierto_direccion",
            "posicion_intervalo"]
evento = mostrar_tabla(h[columnas], {c: "%.2f" for c in ["precio_inicial", "precio_pronosticado", "precio_inferior",
                                                          "precio_superior", "precio_real", "error_abs_precio",
                                                          "error_pct_precio"]}
                       | {"rend_pronosticado": "%+.5f", "rend_real": "%+.5f"}, alto=380,
                       on_select="rerun", selection_mode="single-row", key="tabla_historial")
filas = evento.selection.rows if evento and evento.selection else []
if filas:
    fila = h.iloc[filas[0]]
    with st.expander(f"Reproducibilidad del registro #{fila['id']}", expanded=True, icon=":material/fact_check:"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Métricas al momento de predecir**")
        c1.json(json.loads(fila["metricas"] or "{}"))
        c2.markdown("**Ajustes usados**")
        c2.json(json.loads(fila["ajustes"] or "{}"))
        c3.markdown("**Datos y parámetros**")
        c3.json(json.loads(fila["datos"] or "{}"))
st.download_button("Descargar historial (CSV)", h.to_csv(index=False).encode("utf-8"), "historial_vesta.csv",
                   "text/csv", icon=":material/download:")
pie()
