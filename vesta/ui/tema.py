"""Sistema de diseño de VESTA | Quantitative Intelligence: CSS, plantilla de Plotly y tema de Streamlit."""

from __future__ import annotations

from vesta.config import COLORES as C

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --vq-fondo: {C['fondo']}; --vq-sup: {C['superficie']}; --vq-panel: {C['panel']}; --vq-borde: {C['borde']};
  --vq-texto: {C['texto']}; --vq-tenue: {C['tenue']}; --vq-acento: {C['acento']}; --vq-acento2: {C['acento_2']};
  --vq-pos: {C['positivo']}; --vq-neg: {C['negativo']}; --vq-neutro: {C['neutral']};
  --vq-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
}}
html, body, [class*="css"], .stMarkdown, p, label, li {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.01em; font-weight: 600; color: var(--vq-texto); }}
h2 {{ font-size: 1.15rem; margin-top: 1.2rem; }}
h3 {{ font-size: 1.0rem; color: var(--vq-texto); }}
hr {{ border-color: var(--vq-borde); margin: 1rem 0; }}
code, .vq-mono {{ font-family: var(--vq-mono); font-variant-numeric: tabular-nums; }}

/* Barra lateral */
[data-testid="stSidebar"] {{ background: var(--vq-sup); border-right: 1px solid var(--vq-borde); }}
[data-testid="stSidebarNav"] a {{ border-radius: 4px; transition: background .15s ease; }}
[data-testid="stSidebarNav"] a:hover {{ background: var(--vq-panel); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--vq-panel); border-left: 2px solid var(--vq-acento); }}

/* Botones */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: 4px; border: 1px solid var(--vq-borde); background: var(--vq-panel); color: var(--vq-texto);
  transition: border-color .15s ease, background .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--vq-acento2); }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: var(--vq-acento); border-color: var(--vq-acento); color: #fff; font-weight: 500;
}}
.stButton > button[kind="primary"]:hover {{ background: #a02d47; }}

/* Tablas y paneles desplegables */
[data-testid="stDataFrame"] {{ border: 1px solid var(--vq-borde); border-radius: 4px; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--vq-borde); border-radius: 4px; background: var(--vq-sup); }}
[data-testid="stExpander"] summary {{ font-weight: 500; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .2rem; border-bottom: 1px solid var(--vq-borde); }}
.stTabs [data-baseweb="tab"] {{ color: var(--vq-tenue); padding: .45rem .8rem; }}
.stTabs [aria-selected="true"] {{ color: var(--vq-texto); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--vq-acento); }}

/* Cabecera de sección */
.vq-cab {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: .6rem 1.5rem;
  border-bottom: 1px solid var(--vq-borde); padding-bottom: .7rem; margin-bottom: 1.1rem; }}
.vq-cab .sec {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--vq-tenue); }}
.vq-cab .tit {{ font-size: 1.45rem; font-weight: 600; color: var(--vq-texto); line-height: 1.2; }}
.vq-cab .tk {{ font-family: var(--vq-mono); font-weight: 500; }}
.vq-cab .estado {{ font-size: .78rem; color: var(--vq-tenue); text-align: right; font-family: var(--vq-mono); }}
.vq-punto {{ display: inline-block; width: .5rem; height: .5rem; border-radius: 50%; margin-right: .35rem; vertical-align: middle; }}

/* Ficha del activo */
.vq-ficha {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 1px;
  background: var(--vq-borde); border: 1px solid var(--vq-borde); border-radius: 4px; overflow: hidden; margin-bottom: 1rem; }}
.vq-ficha > div {{ background: var(--vq-sup); padding: .55rem .8rem; }}
.vq-ficha .k {{ font-size: .7rem; color: var(--vq-tenue); text-transform: uppercase; letter-spacing: .06em; }}
.vq-ficha .v {{ font-family: var(--vq-mono); font-variant-numeric: tabular-nums; font-size: 1rem; color: var(--vq-texto); margin-top: .1rem; }}

/* Tarjetas de indicadores */
.vq-grupo {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--vq-tenue); margin: .4rem 0 .4rem; }}
.vq-tarjetas {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .6rem; margin-bottom: .8rem; }}
.vq-tarjeta {{ background: var(--vq-panel); border: 1px solid var(--vq-borde); border-radius: 4px; padding: .75rem .9rem;
  transition: border-color .15s ease; animation: vqEntrada .25s ease-out; }}
.vq-tarjeta:hover {{ border-color: #454a55; }}
.vq-tarjeta .n {{ font-size: .74rem; color: var(--vq-tenue); }}
.vq-tarjeta .val {{ font-family: var(--vq-mono); font-variant-numeric: tabular-nums; font-size: 1.35rem; color: var(--vq-texto); margin: .15rem 0 .05rem; }}
.vq-tarjeta .u {{ font-size: .75rem; color: var(--vq-tenue); margin-left: .25rem; font-family: 'Inter', sans-serif; }}
.vq-tarjeta .d {{ font-size: .76rem; color: var(--vq-tenue); line-height: 1.35; }}
.vq-tarjeta .ref {{ font-size: .72rem; color: var(--vq-neutro); margin-top: .35rem; border-top: 1px solid var(--vq-borde); padding-top: .3rem; }}
.vq-pos {{ color: var(--vq-pos) !important; }}
.vq-neg {{ color: var(--vq-neg) !important; }}
@keyframes vqEntrada {{ from {{ opacity: 0; transform: translateY(2px); }} to {{ opacity: 1; transform: none; }} }}

/* Etiquetas */
.vq-tag {{ display: inline-block; font-size: .72rem; padding: .08rem .5rem; border-radius: 3px; border: 1px solid var(--vq-borde);
  color: var(--vq-texto); font-family: var(--vq-mono); margin-right: .3rem; }}
.vq-nota {{ font-size: .8rem; color: var(--vq-tenue); }}
.vq-pie {{ color: var(--vq-tenue); font-size: .72rem; border-top: 1px solid var(--vq-borde); margin-top: 2.2rem; padding-top: .6rem; }}
.vq-logo-sub {{ font-size: .62rem; letter-spacing: .22em; color: var(--vq-tenue); text-transform: uppercase; }}
@media (max-width: 640px) {{ .vq-cab .estado {{ text-align: left; }} .vq-tarjeta .val {{ font-size: 1.15rem; }} }}
</style>
"""


def plantilla_plotly() -> dict:
    return dict(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["superficie"],
        font=dict(family="Inter, system-ui, sans-serif", color=C["texto"], size=12),
        colorway=[C["serie_obs"], C["serie_mod"], C["serie_ref"]],
        xaxis=dict(gridcolor=C["grid"], linecolor=C["eje"], zerolinecolor=C["eje"], tickcolor=C["eje"], showgrid=False,
                   tickfont=dict(family="JetBrains Mono, monospace", size=11, color=C["tenue"])),
        yaxis=dict(gridcolor=C["grid"], linecolor=C["eje"], zerolinecolor=C["eje"], tickcolor=C["eje"],
                   tickfont=dict(family="JetBrains Mono, monospace", size=11, color=C["tenue"])),
        hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["borde"],
                        font=dict(family="JetBrains Mono, monospace", color=C["texto"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=C["tenue"], size=11),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(font=dict(size=13, color=C["texto"]), x=0, xanchor="left"),
        transition=dict(duration=250, easing="cubic-in-out"),
    ))


def tema_streamlit_toml() -> str:
    return f"""[theme]
base = "dark"
primaryColor = "{C['acento']}"
backgroundColor = "{C['fondo']}"
secondaryBackgroundColor = "{C['superficie']}"
textColor = "{C['texto']}"
font = "sans serif"
"""
