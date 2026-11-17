import pandas as pd
import streamlit as st

from vesta import servicios as S
from vesta.config import HORIZONTES
from vesta.datos import universo as U
from vesta.estadistica.pruebas import CATALOGO
from vesta.procesamiento.rendimientos import VARIABLE_HORIZONTE
from vesta.ui import graficos as G
from vesta.ui.componentes import ajustes, aviso, cabecera, mostrar_tabla, pie, selector_horizonte, universo_cacheado

cabecera("Muestra académica", "Comparación de activos")
st.markdown('<div class="vq-nota">Kruskal-Wallis (sección 10) y Q de Cochran (sección 11) comparan los activos '
            '<b>dentro de la muestra académica</b>. No son pruebas sobre una sola acción.</div>', unsafe_allow_html=True)
aj = ajustes()
u = universo_cacheado()
muestra, origen = U.cargar_muestra(u)
opciones = {"Piloto · fase 1 (Dior, Grupo Bimbo, Disney)": U.piloto()["ticker"].tolist()}
if muestra is not None:
    opciones["Muestra estratificada · fase 2 (21 acciones)"] = muestra["ticker"].tolist()
c1, c2 = st.columns([2, 3])
eleccion = c1.radio("Conjunto", list(opciones), key="conjunto_comparacion")
with c2:
    hz = selector_horizonte()
tickers = opciones[eleccion]
if muestra is None:
    st.caption("La muestra de la fase 2 aparece cuando existe el universo (Explorador) o data/muestra_fase2.csv.")
elif "fase 2" in eleccion:
    st.caption(origen)


@st.cache_data(ttl=3600, show_spinner=False)
def cargar(lista: tuple[str, ...]):
    precios, fallidos = {}, []
    for tk in lista:
        try:
            precios[tk] = S.cargar_activo(tk).precios
        except Exception as exc:
            fallidos.append(f"{tk} ({exc})")
    return precios, fallidos


with st.spinner(f"Cargando {len(tickers)} activos…"):
    precios, fallidos = cargar(tuple(tickers))
if fallidos:
    st.warning("Sin datos para: " + "; ".join(fallidos), icon=":material/cloud_off:")
if len(precios) < 2:
    aviso("Se necesitan al menos dos activos con datos para comparar.")
    st.stop()

res = S.comparar_muestra(precios)
h = HORIZONTES[hz]
st.plotly_chart(G.cajas(res["preparados"], VARIABLE_HORIZONTE[hz],
                        f"Rendimiento log acumulado a {h} días ({hz.lower()}, solapado) por activo"),
                use_container_width=True)


def tabla(resultados: dict, clave: str, extra=None) -> pd.DataFrame:
    filas = []
    for horizonte, r in resultados.items():
        fila = {"Horizonte": horizonte, "Estadístico": r.estadistico, "gl": r.gl, "n / bloques": r.n,
                "p-valor": r.p_valor, "α": aj.alfa, "Decisión": r.decision(aj.alfa), "Interpretación": r.interpretacion(aj.alfa)}
        if extra:
            fila.update(extra(horizonte))
        filas.append(fila)
    return pd.DataFrame(filas)


c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("## Kruskal-Wallis")
    st.markdown(f'<div class="vq-nota">H0: todos los activos comparten la misma distribución de rendimientos. '
                f'H1: {CATALOGO["kruskal"][1].lower()}.</div>', unsafe_allow_html=True)
    mostrar_tabla(tabla(res["kruskal"], "kruskal"), {"Estadístico": "%.3f", "p-valor": "%.4f", "α": "%.2f"})
    st.caption("Usa todos los rendimientos solapados de cada activo, como el documento. La dependencia entre "
               "observaciones solapadas hace que los p-valores sean optimistas.")
with c2:
    st.markdown("## Q de Cochran")
    st.markdown(f'<div class="vq-nota">H0: la proporción de periodos con ganancia es igual entre activos. '
                f'H1: {CATALOGO["cochran"][1].lower()}.</div>', unsafe_allow_html=True)
    tq = tabla(res["cochran"], "cochran")
    mostrar_tabla(tq, {"Estadístico": "%.3f", "p-valor": "%.4f", "α": "%.2f"})
    pocos = tq[tq["n / bloques"] < 30]["Horizonte"].tolist()
    st.caption(f"Periodos no solapados sobre {res['fechas_comunes']} fechas comunes a todos los activos.")
    if pocos:
        st.warning(f"Pocos bloques en {', '.join(pocos)}: la aproximación chi-cuadrado de la Q de Cochran pierde "
                   "precisión con pocos bloques y la prueba tiene poca potencia (limitación señalada en el documento).",
                   icon=":material/warning:")
with st.expander("Éxitos por activo (periodos con rendimiento positivo)", icon=":material/table:"):
    mostrar_tabla(pd.DataFrame(res["exitos"]).T.reset_index(names="Horizonte"))
pie()
