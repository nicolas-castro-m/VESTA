"""Componentes reutilizables del sistema de diseño."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from vesta import ajustes as A
from vesta.config import AVISO, COLOR_PERFIL, COLORES as C, HORIZONTES, PERFILES, RAIZ
from vesta.datos import universo as U
from vesta.estadistica.pruebas import ResultadoPrueba

TICKER_DEFECTO = "MSFT"


# ── Formato ──────────────────────────────────────────────────────────────
def pct(x, dec: int = 2, signo: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{dec}f}" if signo else f"{x * 100:.{dec}f}"


def num(x, dec: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{x:,.{dec}f}" if np.isfinite(x) else "—"


def pval(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def con_signo(x, dec: int = 2, sufijo: str = " %") -> tuple[str, str]:
    """Texto con flecha y signo + clase de color. El color nunca va solo (CVD)."""
    if x is None or not np.isfinite(x):
        return "—", ""
    flecha = "▲" if x > 0 else ("▼" if x < 0 else "■")
    clase = "vq-pos" if x > 0 else ("vq-neg" if x < 0 else "")
    return f"{flecha} {x * 100:+.{dec}f}{sufijo}", clase


# ── Estado global ────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def universo_cacheado() -> pd.DataFrame | None:
    try:
        return U.cargar_universo()
    except Exception:
        return None


def ajustes() -> A.Ajustes:
    if "ajustes" not in st.session_state:
        st.session_state["ajustes"] = A.cargar()
    return st.session_state["ajustes"]


def estado_inicial() -> None:
    """Estado compartido entre páginas.

    "ticker", "horizonte" y "ventana" son claves propias, no de widgets: desde
    Streamlit 1.5x el valor de un widget se borra al cambiar de página (salvo
    persist_state="session"), así que los widgets solo copian su valor aquí.
    """
    st.session_state.setdefault("ticker", TICKER_DEFECTO)
    st.session_state.setdefault("horizonte", "Mensual")
    st.session_state.setdefault("ventana", "5 años")
    pendiente = st.session_state.pop("ticker_pendiente", None)
    if pendiente:
        st.session_state["ticker"] = pendiente


def ir_a_activo(ticker: str, pagina: str = "paginas/analisis.py") -> None:
    st.session_state["ticker"] = ticker
    st.switch_page(pagina)


def info_activo(ticker: str) -> dict:
    u = universo_cacheado()
    if u is not None:
        fila = u[u["ticker"] == ticker]
        if not fila.empty:
            return fila.iloc[0].to_dict()
    return {"ticker": ticker, "empresa": ticker, "sector": None, "mercado": None, "beta": np.nan, "perfil": None,
            "fecha_consulta": None}


def barra_lateral() -> None:
    """Logotipo y activo seleccionado (se conserva entre páginas)."""
    estado_inicial()
    try:
        st.logo(str(RAIZ / "assets" / "logo.svg"), size="large", icon_image=str(RAIZ / "assets" / "icono.svg"))
    except Exception:
        pass
    u = universo_cacheado()

    def copiar_seleccion():
        st.session_state["ticker"] = st.session_state["_ticker_w"]

    def usar_libre():
        valor = (st.session_state.get("_ticker_libre") or "").strip().upper()
        if valor:
            st.session_state["ticker"] = valor
            st.session_state["_ticker_libre"] = ""

    with st.sidebar:
        if u is not None and not u.empty:
            orden = u.sort_values("ticker")
            etiquetas = {r.ticker: f"{r.ticker} · {r.empresa}" for r in orden.itertuples()}
            lista = list(etiquetas)
            if st.session_state["ticker"] not in etiquetas:
                lista = [st.session_state["ticker"]] + lista
            st.session_state["_ticker_w"] = st.session_state["ticker"]
            st.selectbox("Activo", lista, key="_ticker_w", on_change=copiar_seleccion,
                         format_func=lambda t: etiquetas.get(t, t))
            st.text_input("Otro ticker (Yahoo Finance)", key="_ticker_libre", on_change=usar_libre,
                          placeholder="GRBMF, DIO.DE…", help="Escriba un ticker fuera del S&P 500 y pulse Enter.")
        else:
            st.session_state["_ticker_libre"] = st.session_state["ticker"]
            st.text_input("Ticker (Yahoo Finance)", key="_ticker_libre", on_change=usar_libre,
                          help="Construya el universo en el Explorador para buscar por nombre.")
        st.caption(AVISO)


def selector_horizonte() -> str:
    """Selector principal del horizonte. Su valor persiste aunque la página cambie."""
    opciones = list(HORIZONTES)

    def copiar():
        st.session_state["horizonte"] = st.session_state["_hz"]

    st.session_state["_hz"] = st.session_state.get("horizonte", "Mensual")
    st.segmented_control("Horizonte de inversión", opciones, key="_hz", on_change=copiar,
                         format_func=lambda h: f"{h} · {HORIZONTES[h]} d", selection_mode="single")
    if st.session_state.get("_hz") is None:          # el control permite deseleccionar
        st.session_state["_hz"] = st.session_state["horizonte"]
    return st.session_state["horizonte"]


def selector_ventana(opciones: dict) -> str:
    def copiar():
        st.session_state["ventana"] = st.session_state["_vent"] or st.session_state["ventana"]

    st.session_state["_vent"] = st.session_state.get("ventana", "5 años")
    st.segmented_control("Ventana histórica", list(opciones), key="_vent", on_change=copiar, selection_mode="single",
                         help="Periodo de historia que se muestra. Es independiente del horizonte de inversión.")
    return st.session_state["ventana"]


# ── Cabecera, ficha y tarjetas ───────────────────────────────────────────
def cabecera(seccion: str, titulo: str, ticker: str | None = None, datos=None) -> None:
    estado = []
    if datos is not None:
        ultimo = pd.Timestamp(datos.precios["Date"].max())
        habiles = int(np.busday_count(ultimo.date(), pd.Timestamp.now().date()))
        color = C["positivo"] if habiles <= 2 else (C["serie_ref"] if habiles <= 5 else C["negativo"])
        texto = "al día" if habiles <= 2 else f"{habiles} días hábiles de antigüedad"
        estado.append(f'<span class="vq-punto" style="background:{color}"></span>Último dato {ultimo:%d %b %Y} · {texto}')
        estado.append(f"Fuente: Yahoo Finance · {html.escape(datos.origen)} {datos.actualizado:%Y-%m-%d %H:%M}")
    activo = ""
    if ticker:
        info = info_activo(ticker)
        perfil = info.get("perfil")
        tag = (f' <span class="vq-tag" style="border-color:{COLOR_PERFIL[perfil]}">{perfil}</span>'
               if isinstance(perfil, str) else "")
        activo = f' · <span class="tk">{html.escape(ticker)}</span>{tag}'
    st.markdown(f'<div class="vq-cab"><div><div class="sec">{html.escape(seccion)}</div>'
                f'<div class="tit">{html.escape(titulo)}{activo}</div></div>'
                f'<div class="estado">{"<br>".join(estado)}</div></div>', unsafe_allow_html=True)


def ficha(pares: list[tuple[str, str]]) -> None:
    celdas = "".join(f'<div><div class="k">{html.escape(k)}</div><div class="v">{v}</div></div>' for k, v in pares)
    st.markdown(f'<div class="vq-ficha">{celdas}</div>', unsafe_allow_html=True)


def tarjeta(nombre: str, valor: str, unidad: str = "", descripcion: str = "", referencia: str = "", clase: str = "") -> str:
    ref = f'<div class="ref">{referencia}</div>' if referencia else ""
    return (f'<div class="vq-tarjeta"><div class="n">{html.escape(nombre)}</div>'
            f'<div class="val {clase}">{valor}<span class="u">{html.escape(unidad)}</span></div>'
            f'<div class="d">{descripcion}</div>{ref}</div>')


def tarjetas(grupo: str, lista: list[str]) -> None:
    st.markdown(f'<div class="vq-grupo">{html.escape(grupo)}</div><div class="vq-tarjetas">{"".join(lista)}</div>',
                unsafe_allow_html=True)


# ── Tablas de pruebas ────────────────────────────────────────────────────
def tabla_pruebas(resultados: list[ResultadoPrueba], alfa: float, variable: str = "") -> pd.DataFrame:
    filas = []
    for r in resultados:
        criticos = ", ".join(f"{k}: {v:.3f}" for k, v in r.valores_criticos.items()) if r.valores_criticos else "—"
        filas.append({
            "Prueba": r.prueba, "Variable": variable, "Qué evalúa": r.que_evalua, "H0": r.h0, "H1": r.h1,
            "Estadístico": r.estadistico, "p-valor": r.p_valor, "Rezagos": r.rezagos, "n": r.n,
            "Valores críticos": criticos, "α": alfa, "Decisión": r.decision(alfa), "Interpretación": r.interpretacion(alfa),
        })
    return pd.DataFrame(filas)


def mostrar_tabla(df: pd.DataFrame, formatos: dict | None = None, alto: int | None = None, **kw):
    config = {}
    for col, fmt in (formatos or {}).items():
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(format=fmt)
    if alto is not None:
        kw["height"] = alto
    return st.dataframe(df, hide_index=True, use_container_width=True, column_config=config, **kw)


def mostrar_pruebas(resultados: list[ResultadoPrueba], alfa: float, variable: str) -> None:
    mostrar_tabla(tabla_pruebas(resultados, alfa, variable), {"Estadístico": "%.4f", "p-valor": "%.4f", "α": "%.2f"})


def tag_perfil(perfil: str) -> str:
    return f'<span class="vq-tag" style="border-color:{COLOR_PERFIL.get(perfil, C["borde"])}">{perfil}</span>'


def pie() -> None:
    st.markdown(f'<div class="vq-pie">VESTA | Quantitative Intelligence · Modelación estadística para inversiones de '
                f'corto plazo · {AVISO}</div>', unsafe_allow_html=True)


def aviso(mensaje: str) -> None:
    st.info(mensaje, icon=":material/info:")


__all__ = ["PERFILES"]
