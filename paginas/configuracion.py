import shutil
from dataclasses import fields

import streamlit as st

from vesta import ajustes as A
from vesta.config import DIR_CACHE, RAIZ, VERSION
from vesta.datos import almacen
from vesta.persistencia import db
from vesta.ui.componentes import ajustes, cabecera, pie, universo_cacheado

cabecera("Sistema", "Configuración")
t_analisis, t_datos, t_metodo, t_acerca = st.tabs(["Análisis", "Datos y base local", "Metodología", "Acerca de"])

with t_analisis:
    actual = ajustes()
    defecto = A.Ajustes()
    st.markdown('<div class="vq-nota">Los valores por defecto replican el documento metodológico. Cada predicción guarda '
                'los ajustes con que se generó.</div>', unsafe_allow_html=True)
    with st.form("form_ajustes"):
        c1, c2, c3 = st.columns(3)
        alfa = c1.number_input("Nivel de significancia α", 0.001, 0.2, actual.alfa, 0.005, format="%.3f")
        nivel = c1.number_input("Nivel del intervalo de predicción", 0.80, 0.99, actual.nivel_intervalo, 0.01, format="%.2f")
        anios = c1.selectbox("Historia para modelar (años)", [3, 4, 5], index=[3, 4, 5].index(actual.anios_historia))
        max_p = c2.number_input("p máximo", 0, 5, actual.max_p)
        max_q = c2.number_input("q máximo", 0, 5, actual.max_q)
        criterio = c2.selectbox("Criterio de especificación", ["bic", "aic"], index=["bic", "aic"].index(actual.criterio),
                                format_func=str.upper)
        prop = c3.number_input("Proporción de entrenamiento", 0.5, 0.95, actual.prop_train, 0.05, format="%.2f")
        reest = c3.number_input("Reestimar cada (orígenes)", 1, 63, actual.reestimar_cada,
                                help="1 = reajuste en cada origen (documento). Valores mayores aceleran el cálculo.")
        deriva = c3.checkbox("Incluir deriva en ARIMA(p,1,q)", actual.incluir_deriva,
                             help="El documento usa allowdrift = FALSE.")
        c4, c5, c6 = st.columns(3)
        lb = c4.number_input("Rezagos Ljung-Box", 1, 40, actual.lags_ljung_box)
        arch = c5.number_input("Rezagos ARCH-LM", 1, 40, actual.lags_arch)
        acf = c6.number_input("Rezagos ACF/PACF", 5, 60, actual.lag_max_acf)
        guardar = st.form_submit_button("Guardar ajustes", type="primary")
    if guardar:
        nuevo = A.Ajustes(alfa=alfa, anios_historia=anios, max_p=int(max_p), max_q=int(max_q), d=1,
                          incluir_deriva=deriva, prop_train=prop, criterio=criterio, reestimar_cada=int(reest),
                          nivel_intervalo=nivel, lags_ljung_box=int(lb), lags_arch=int(arch), lag_max_acf=int(acf))
        errores = nuevo.validar()
        if errores:
            for e in errores:
                st.error(e)
        else:
            A.guardar(nuevo)
            st.session_state["ajustes"] = nuevo
            st.toast("Ajustes guardados", icon=":material/check_circle:")
            st.rerun()
    cambiados = [f.name for f in fields(A.Ajustes) if getattr(actual, f.name) != getattr(defecto, f.name)]
    st.caption(f"Firma de los ajustes: {actual.firma()} · "
               + (f"distintos del documento: {', '.join(cambiados)}" if cambiados else "iguales a los del documento"))
    if cambiados and st.button("Restablecer valores del documento", icon=":material/restart_alt:"):
        db.borrar_configuracion()
        st.session_state["ajustes"] = A.cargar()
        st.rerun()

with t_datos:
    u = universo_cacheado()
    base = db.resumen_base()
    st.markdown(f"**Base local:** `{db.ruta_actual().relative_to(RAIZ) if db.ruta_actual().is_relative_to(RAIZ) else db.ruta_actual()}` "
                f"· {base['activos']} activos · {base['calculos_universo']} cálculos de beta · {base['predicciones']} "
                f"predicciones · {base['realizaciones']} evaluadas · {base['evaluaciones']} evaluaciones guardadas")
    st.markdown(f"**Universo:** " + (f"calculado el {u['fecha_consulta'].iloc[0]}" if u is not None else "sin construir")
                + " · se actualiza desde el Explorador o con `python scripts/actualizar_universo.py`.")
    st.markdown(f"**Precios en caché:** {len(almacen.tickers_en_cache())} activos en `data/precios` (se renuevan si "
                "tienen más de 12 horas).")
    st.markdown("**Historial de actualizaciones**")
    st.dataframe(db.leer_actualizaciones(50), hide_index=True, use_container_width=True)
    if st.button("Vaciar caché de cálculos", icon=":material/delete_sweep:",
                 help="Borra data/cache (resultados intermedios). No toca la base ni el historial de predicciones."):
        shutil.rmtree(DIR_CACHE, ignore_errors=True)
        st.toast("Caché de cálculos vaciada", icon=":material/check_circle:")

with t_metodo:
    ruta = RAIZ / "docs" / "METODOLOGIA.md"
    st.markdown(ruta.read_text(encoding="utf-8") if ruta.exists() else "No se encontró docs/METODOLOGIA.md.")

with t_acerca:
    st.markdown(f"**VESTA | Quantitative Intelligence** · versión {VERSION}")
    st.markdown("Proyecto académico *Modelación estadística para inversiones de corto plazo (semanal, mensual y "
                "trimestral)*. Alejandra Beltrán y Johan Castro.")
    st.markdown("Datos: Yahoo Finance (vía yfinance) y Wikipedia (lista del S&P 500). Licencias de dependencias y "
                "condiciones de las fuentes en `THIRD_PARTY_NOTICES.md`.")
    st.caption("Trabajo académico. No es una recomendación de inversión. Los pronósticos estadísticos de rendimientos "
               "a corto plazo tienen gran incertidumbre y no garantizan resultados futuros.")
pie()
