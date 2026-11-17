"""VESTA | Quantitative Intelligence.

Ejecutar:  streamlit run app.py      (detener: Ctrl + C en la terminal)
"""

import streamlit as st

from vesta.config import RAIZ
from vesta.persistencia import db
from vesta.ui.componentes import barra_lateral
from vesta.ui.tema import CSS

st.set_page_config(page_title="VESTA | Quantitative Intelligence", page_icon=str(RAIZ / "assets" / "icono.svg"),
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
db.inicializar()

paginas = [
    st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("paginas/explorador.py", title="Explorador de activos", icon=":material/table_rows:"),
    st.Page("paginas/analisis.py", title="Análisis del activo", icon=":material/query_stats:"),
    st.Page("paginas/comparacion.py", title="Comparación de activos", icon=":material/compare_arrows:"),
    st.Page("paginas/predicciones.py", title="Predicciones", icon=":material/timeline:"),
    st.Page("paginas/historial.py", title="Historial", icon=":material/history:"),
    st.Page("paginas/configuracion.py", title="Configuración", icon=":material/tune:"),
]

navegacion = st.navigation(paginas, position="sidebar")
barra_lateral()
navegacion.run()
