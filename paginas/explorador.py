import pandas as pd
import streamlit as st

from vesta.config import COLOR_PERFIL, PERFILES
from vesta.datos import universo as U
from vesta.ui import graficos as G
from vesta.ui.componentes import aviso, cabecera, ir_a_activo, mostrar_tabla, pie, tag_perfil, universo_cacheado

cabecera("Mercado", "Explorador de activos")
u = universo_cacheado()


def construir():
    with st.status("Actualizando el universo del S&P 500…", expanded=True) as estado:
        barra = st.progress(0.0, text="Obteniendo la lista de componentes")
        try:
            nuevo = U.actualizar(progreso=lambda f, t: barra.progress(min(f, 1.0), text=t))
        except Exception as exc:
            estado.update(label="No se pudo actualizar el universo", state="error")
            st.error(f"{exc}. Si no hay conexión con Wikipedia, coloque data/tickers_sp500.csv con columnas ticker y empresa.")
            return
        estado.update(label=f"Universo actualizado: {nuevo['beta'].notna().sum()} activos con beta", state="complete")
    universo_cacheado.clear()
    st.toast("Universo actualizado", icon=":material/check_circle:")
    st.rerun()


if u is None:
    aviso("El universo todavía no existe. Se descargan cinco años de precios de unas 500 empresas, se calcula la beta "
          "de cada una y se guarda en la base local. Tarda varios minutos y no se repite en cada visita.")
    if st.button("Construir universo", type="primary", icon=":material/download:"):
        construir()
    st.stop()

pest_universo, pest_muestra = st.tabs(["Universo exploratorio · S&P 500", "Muestra académica"])

with pest_universo:
    f1, f2, f3 = st.columns([3, 3, 2])
    busqueda = f1.text_input("Buscar por ticker, empresa o sector", placeholder="AAPL, Coca-Cola, Energy…")
    perfiles = f2.segmented_control("Perfil", PERFILES, default=PERFILES, selection_mode="multi") or PERFILES
    vencido = U.universo_vencido(u)
    with f3:
        st.markdown(f'<div class="vq-nota">Betas calculadas el <b>{u["fecha_consulta"].iloc[0]}</b>'
                    f'{" · conviene actualizar" if vencido else ""}</div>', unsafe_allow_html=True)
        if st.button("Actualizar universo", icon=":material/refresh:", use_container_width=True):
            construir()

    vista = U.depurar(u)
    vista = vista[vista["perfil"].isin(perfiles)]
    if busqueda:
        b = busqueda.strip().lower()
        vista = vista[vista["ticker"].str.lower().str.contains(b, regex=False)
                      | vista["empresa"].str.lower().str.contains(b, regex=False)
                      | vista["sector"].fillna("").str.lower().str.contains(b, regex=False)]
    st.markdown(" ".join(f'{tag_perfil(p)}<span class="vq-nota">{int((vista["perfil"] == p).sum())}</span>&nbsp;&nbsp;'
                         for p in PERFILES), unsafe_allow_html=True)
    if vista.empty:
        aviso("Ningún activo coincide con la búsqueda.")
    else:
        tabla = vista.assign(var_5d=vista["var_5d"] * 100, var_21d=vista["var_21d"] * 100,
                             vol_anual=vista["vol_anual"] * 100)
        tabla = tabla[["ticker", "empresa", "precio", "var_5d", "var_21d", "beta", "perfil", "sector", "vol_anual",
                       "fecha_precio"]].sort_values("ticker")
        columnas = {
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "empresa": st.column_config.TextColumn("Empresa"),
            "precio": st.column_config.NumberColumn("Precio (USD)", format="%.2f"),
            "var_5d": st.column_config.NumberColumn("Var. 5 d (%)", format="%+.2f"),
            "var_21d": st.column_config.NumberColumn("Var. 21 d (%)", format="%+.2f"),
            "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
            "perfil": st.column_config.TextColumn("Perfil", width="small"),
            "sector": st.column_config.TextColumn("Sector"),
            "vol_anual": st.column_config.NumberColumn("Volatilidad anual (%)", format="%.1f"),
            "fecha_precio": st.column_config.TextColumn("Fecha del precio", width="small"),
        }
        evento = st.dataframe(tabla, hide_index=True, use_container_width=True, column_config=columnas, height=460,
                              on_select="rerun", selection_mode="single-row", key="tabla_universo")
        st.caption("Ordene con un clic en el encabezado. Seleccione una fila para abrir el análisis del activo. "
                   "Precio de cierre del último día disponible (no es tiempo real).")
        filas = evento.selection.rows if evento and evento.selection else []
        if filas:
            ir_a_activo(tabla.iloc[filas[0]]["ticker"])
    with st.expander("Distribución de la beta", icon=":material/bar_chart:"):
        st.plotly_chart(G.distribucion_beta(U.depurar(u)), use_container_width=True)

with pest_muestra:
    st.markdown('<div class="vq-nota">La muestra académica valida la metodología y es la base de las comparaciones '
                'entre activos (Kruskal-Wallis, Q de Cochran). No sustituye al universo exploratorio.</div>',
                unsafe_allow_html=True)
    st.markdown("## Piloto · fase 1")
    st.dataframe(U.piloto(), hide_index=True, use_container_width=True)
    st.markdown("## Muestra sistemática estratificada · fase 2")
    muestra, origen = U.cargar_muestra(u)
    if muestra is not None:
        st.caption(origen)
        mostrar_tabla(muestra[[c for c in ["perfil", "ticker", "empresa", "beta"] if c in muestra]], {"beta": "%.3f"})
        pob = U.depurar(u)
        resumen = pd.concat([pob.assign(Conjunto="Población"), muestra.assign(Conjunto="Muestra")])
        resumen = (resumen.groupby(["perfil", "Conjunto"])["beta"]
                   .agg(n="count", Min="min", Mediana="median", Media="mean", Max="max").reset_index())
        mostrar_tabla(resumen, {c: "%.3f" for c in ["Min", "Mediana", "Media", "Max"]})
pie()
