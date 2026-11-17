import pandas as pd
import streamlit as st

from vesta.config import HORIZONTES, PERFILES
from vesta.datos import almacen
from vesta.persistencia import db
from vesta.ui.componentes import aviso, cabecera, pie, tag_perfil, tarjeta, tarjetas, universo_cacheado

cabecera("Inicio", "VESTA | Quantitative Intelligence")

st.markdown(
    "**Volatilidad, estimación de series de tiempo y análisis.** Laboratorio cuantitativo para estudiar acciones "
    "del S&P 500 con herramientas estadísticas aplicadas a sus rendimientos, y para evaluar si los modelos de series "
    "de tiempo pronostican mejor que una referencia simple en inversiones de corto plazo.")

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.markdown("## Objetivo")
    st.markdown(
        "Describir el comportamiento de cada activo, contrastar los supuestos estadísticos habituales "
        "(aleatoriedad, normalidad, estacionariedad, dependencia) y comparar modelos ARIMA con la caminata aleatoria "
        "bajo las mismas condiciones. La pregunta no es cuánto subirá una acción, sino si su historia contiene "
        "información que mejore el pronóstico más simple: que el precio futuro sea el actual.")
    st.markdown("## Horizontes de inversión")
    st.dataframe(pd.DataFrame({
        "Horizonte": list(HORIZONTES), "Días bursátiles": list(HORIZONTES.values()),
        "Variable analizada": [f"ln(P(t+{h}) / P(t)) · rendimiento logarítmico acumulado" for h in HORIZONTES.values()],
    }), hide_index=True, use_container_width=True)
    st.caption("Se cuentan días bursátiles porque los datos son diarios y el mercado no cotiza fines de semana ni "
               "festivos. La ventana de historia que se consulta es independiente del horizonte.")
    b1, b2, _ = st.columns([1, 1, 1])
    if b1.button("Abrir el explorador", type="primary", icon=":material/table_rows:", use_container_width=True):
        st.switch_page("paginas/explorador.py")
    if b2.button(f"Analizar {st.session_state.get('ticker', 'MSFT')}", icon=":material/query_stats:",
                 use_container_width=True):
        st.switch_page("paginas/analisis.py")

with c2:
    st.markdown("## Universo exploratorio · S&P 500")
    u = universo_cacheado()
    if u is None:
        aviso("El universo aún no se ha construido. Hágalo desde el Explorador: se descarga una vez y se reutiliza "
              "durante 30 días.")
    else:
        conteo = u["perfil"].value_counts()
        tarjetas("Clasificación por beta", [
            tarjeta(p, str(int(conteo.get(p, 0))), "activos",
                    {"Defensivo": "β < 0.8", "Neutral": "0.8 ≤ β < 1.2", "Agresivo": "β ≥ 1.2"}[p]) for p in PERFILES])
        st.markdown(f'<div class="vq-nota">{len(u)} empresas · {int(u["beta"].notna().sum())} con beta · ventana '
                    f'{u["ventana_inicio"].iloc[0]} a {u["ventana_fin"].iloc[0]} · calculado el '
                    f'{u["fecha_consulta"].iloc[0]}</div>', unsafe_allow_html=True)
        st.caption("La beta mide la sensibilidad frente al mercado. No es una medida completa del riesgo ni una "
                   "garantía de seguridad.")
    st.markdown("## Muestra académica")
    st.markdown('<div class="vq-nota">Distinta del universo exploratorio: el piloto (Dior, Grupo Bimbo, Disney) y la '
                'muestra sistemática estratificada de 21 acciones (7 por perfil) con la que se valida la metodología. '
                'Las comparaciones entre activos se hacen solo dentro de ella.</div>', unsafe_allow_html=True)
    st.markdown("## Última actualización")
    act = db.leer_actualizaciones(5)
    if act.empty:
        st.markdown('<div class="vq-nota">Sin actualizaciones registradas.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(act[["tipo", "inicio", "estado", "detalle"]], hide_index=True, use_container_width=True)
    base = db.resumen_base()
    st.markdown(f'<div class="vq-nota">Precios guardados localmente: {len(almacen.tickers_en_cache())} activos · '
                f'predicciones registradas: {base["predicciones"]} · evaluadas: {base["realizaciones"]}. '
                'El último dato disponible no es una cotización en tiempo real.</div>', unsafe_allow_html=True)
pie()
