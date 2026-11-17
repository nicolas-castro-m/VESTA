"""Gráficos interactivos de VESTA (Plotly).

Reglas: una escala por gráfico, líneas finas, cuadrícula sutil, tooltip en
todas las marcas y colores de serie del trío validado (observado, modelo,
referencia). Positivo y negativo siempre llevan signo en el tooltip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from vesta.config import COLOR_PERFIL, COLORES as C, UMBRAL_AGRESIVO, UMBRAL_DEFENSIVO
from vesta.ui.tema import plantilla_plotly

PLANTILLA = go.layout.Template(plantilla_plotly())


def _fig(titulo: str = "", alto: int = 320, **kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=PLANTILLA, title=titulo, height=alto, **kw)
    return fig


def precio_ventana(d: pd.DataFrame, modo: str = "precio") -> go.Figure:
    """Precio ajustado o rentabilidad acumulada desde el inicio de la ventana."""
    if modo == "acumulada":
        y = d["Adj.Close"] / d["Adj.Close"].iloc[0] - 1
        fig = _fig("Rentabilidad acumulada desde el inicio de la ventana", hovermode="x unified")
        fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
        fig.add_trace(go.Scatter(x=d["Date"], y=y, mode="lines", line=dict(color=C["serie_obs"], width=1.8),
                                 hovertemplate="%{y:+.2%}<extra></extra>"))
        fig.update_yaxes(tickformat="+.0%")
    else:
        fig = _fig("Precio ajustado (Adj Close, USD)", hovermode="x unified")
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Adj.Close"], mode="lines", line=dict(color=C["serie_obs"], width=1.8),
                                 hovertemplate="%{y:,.2f} USD<extra></extra>"))
        fig.update_yaxes(tickformat=",.0f")
    fig.update_layout(showlegend=False)
    return fig


def volatilidad_movil(d: pd.DataFrame, ventana: int = 21) -> go.Figure:
    r = np.log(d.set_index("Date")["Adj.Close"]).diff()
    vol = r.rolling(ventana).std() * np.sqrt(252)
    fig = _fig(f"Volatilidad histórica anualizada (ventana móvil de {ventana} sesiones)", alto=260, hovermode="x unified")
    fig.add_trace(go.Scatter(x=vol.index, y=vol, mode="lines", line=dict(color=C["serie_obs"], width=1.5),
                             hovertemplate="%{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    return fig


def serie_horizonte(s: pd.DataFrame, horizonte: str, h: int, construccion: str) -> go.Figure:
    titulo = f"Rendimiento logarítmico acumulado a {h} días ({horizonte.lower()}) · " + \
             ("calculado cada día" if construccion == "solapada" else "un periodo cada h sesiones")
    fig = _fig(titulo, alto=300, hovermode="x unified")
    fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
    if construccion == "no_solapada":
        colores = np.where(s["valor"] >= 0, C["positivo"], C["negativo"])
        fig.add_trace(go.Bar(x=s["Date"], y=s["valor"], marker=dict(color=colores, cornerradius=2),
                             hovertemplate="%{x|%d %b %Y}: %{y:+.2%}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=s["Date"], y=s["valor"], mode="lines", line=dict(color=C["serie_obs"], width=1.3),
                                 hovertemplate="%{y:+.2%}<extra></extra>"))
    fig.update_yaxes(tickformat="+.0%")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


def histograma(x, titulo: str, bins: int = 40) -> go.Figure:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.quantile(x, [0.01, 0.99])
    central = x[(x >= lo) & (x <= hi)]
    fig = _fig(titulo, alto=290)
    fig.add_trace(go.Histogram(x=central, nbinsx=bins, histnorm="probability density", name="Observado",
                               marker=dict(color=C["serie_obs"], line=dict(color=C["superficie"], width=1)),
                               hovertemplate="%{x:+.2%} · densidad %{y:.2f}<extra></extra>"))
    malla = np.linspace(central.min(), central.max(), 200)
    fig.add_trace(go.Scatter(x=malla, y=stats.norm.pdf(malla, x.mean(), x.std(ddof=1)), mode="lines",
                             name="Normal de referencia", line=dict(color=C["serie_mod"], width=1.8), hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color=C["eje"], dash="dot", width=1))
    fig.update_xaxes(tickformat="+.0%")
    fig.add_annotation(text=f"Intervalo central 98 % · {len(x) - len(central)} extremos fuera de la vista",
                       xref="paper", yref="paper", x=0, y=-0.2, showarrow=False, font=dict(color=C["tenue"], size=10))
    fig.update_layout(margin=dict(b=42))
    return fig


def qq(x, titulo: str) -> go.Figure:
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    n = len(x)
    teo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    t1, t3 = stats.norm.ppf([0.25, 0.75])
    pend = (q3 - q1) / (t3 - t1)
    fig = _fig(titulo, alto=290)
    fig.add_trace(go.Scatter(x=teo, y=x, mode="markers", marker=dict(color=C["serie_obs"], size=4, opacity=0.6),
                             hovertemplate="teórico %{x:.2f} · observado %{y:+.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[teo[0], teo[-1]], y=[q1 + pend * (teo[0] - t1), q1 + pend * (teo[-1] - t1)],
                             mode="lines", line=dict(color=C["serie_mod"], width=1.8), hoverinfo="skip"))
    fig.update_xaxes(title="Cuantiles teóricos (normal)")
    fig.update_yaxes(title="Cuantiles observados", tickformat="+.0%")
    fig.update_layout(showlegend=False)
    return fig


def correlograma(valores: np.ndarray, banda: float, titulo: str, alto: int = 250) -> go.Figure:
    lags = np.arange(1, len(valores) + 1)
    fuera = np.abs(valores) > banda
    fig = _fig(titulo, alto=alto)
    fig.add_hrect(y0=-banda, y1=banda, fillcolor=C["panel"], opacity=0.9, line_width=0)
    fig.add_hline(y=banda, line=dict(color=C["neutral"], dash="dot", width=1))
    fig.add_hline(y=-banda, line=dict(color=C["neutral"], dash="dot", width=1))
    fig.add_trace(go.Bar(x=lags, y=valores, width=0.3,
                         marker=dict(color=np.where(fuera, C["serie_mod"], C["serie_obs"]), cornerradius=2),
                         customdata=np.where(fuera, "fuera de la banda", "dentro de la banda"),
                         hovertemplate="rezago %{x}: %{y:.4f} (%{customdata})<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
    fig.update_xaxes(title="Rezago", dtick=5)
    fig.update_layout(showlegend=False)
    return fig


def trayectoria(d: pd.DataFrame, pred, dias_historia: int = 126) -> go.Figure:
    """Histórico, punto de inicio, trayectoria estimada y bandas (modelo y caminata aleatoria)."""
    hist = d.tail(dias_historia)
    f = pred.fechas
    p0 = pred.precio_actual
    fig = _fig(f"Trayectoria proyectada · {pred.horizonte.lower()} ({pred.h} días) · intervalo de predicción "
               f"del {pred.nivel:.0%}", alto=400, hovermode="x unified")
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Adj.Close"], mode="lines", name="Histórico",
                             line=dict(color=C["serie_obs"], width=1.8), hovertemplate="%{y:,.2f}<extra>Histórico</extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.r_[p0, pred.rw_sup], mode="lines", name="Banda caminata aleatoria",
                             line=dict(color=C["serie_ref"], width=1, dash="dot"),
                             hovertemplate="%{y:,.2f}<extra>Límite caminata</extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.r_[p0, pred.rw_inf], mode="lines", showlegend=False,
                             line=dict(color=C["serie_ref"], width=1, dash="dot"),
                             hovertemplate="%{y:,.2f}<extra>Límite caminata</extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.r_[p0, pred.precio_sup], mode="lines", line=dict(width=0), showlegend=False,
                             hovertemplate="%{y:,.2f}<extra>Límite superior</extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.r_[p0, pred.precio_inf], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=C["banda"], name=f"Intervalo {pred.nivel:.0%} del modelo",
                             hovertemplate="%{y:,.2f}<extra>Límite inferior</extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.r_[p0, pred.precio], mode="lines", name=f"Trayectoria {pred.etiqueta}",
                             line=dict(color=C["serie_mod"], width=2), hovertemplate="%{y:,.2f}<extra>Pronóstico</extra>"))
    fig.add_trace(go.Scatter(x=[f[0]], y=[p0], mode="markers", name="Inicio del pronóstico",
                             marker=dict(size=9, color=C["texto"], line=dict(color=C["fondo"], width=2)),
                             hovertemplate="Último dato %{x|%d %b %Y}: %{y:,.2f}<extra></extra>"))
    fig.add_vline(x=f[0], line=dict(color=C["eje"], dash="dot", width=1))
    fig.update_yaxes(tickformat=",.0f", title="USD")
    return fig


def evaluacion_rodante(pr: pd.DataFrame, horizonte: str, etiqueta: str) -> go.Figure:
    s = pr[pr["Horizonte"] == horizonte]
    fig = _fig(f"Fuera de muestra · rendimiento acumulado {horizonte.lower()}: real frente a pronósticos",
               alto=330, hovermode="x unified")
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=s["Superior"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=s["Inferior"], fill="tonexty", fillcolor=C["banda"],
                             line=dict(width=0), name="Intervalo del modelo", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=s["Real"], name="Real", line=dict(color=C["serie_obs"], width=1.6),
                             hovertemplate="%{y:+.2%}<extra>Real</extra>"))
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=s["Pred_Modelo"], name=etiqueta,
                             line=dict(color=C["serie_mod"], width=1.6), hovertemplate="%{y:+.2%}<extra>Modelo</extra>"))
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=s["Pred_RW"], name="Caminata aleatoria",
                             line=dict(color=C["serie_ref"], width=1.6, dash="dash"),
                             hovertemplate="%{y:+.2%}<extra>Caminata</extra>"))
    fig.update_xaxes(title="Fecha de origen del pronóstico")
    fig.update_yaxes(tickformat="+.0%")
    return fig


def ventaja_acumulada(pr: pd.DataFrame, horizonte: str) -> go.Figure:
    s = pr[pr["Horizonte"] == horizonte]
    d = ((s["Real"] - s["Pred_Modelo"]) ** 2 - (s["Real"] - s["Pred_RW"]) ** 2).cumsum()
    fig = _fig("Diferencia acumulada de pérdida: error² del modelo − error² de la caminata aleatoria", alto=250,
               hovermode="x unified")
    fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
    fig.add_trace(go.Scatter(x=s["Fecha_Origen"], y=d, mode="lines", line=dict(color=C["serie_mod"], width=1.6),
                             hovertemplate="%{y:.5f}<extra></extra>"))
    fig.add_annotation(text="Bajo cero: el modelo acumula menos error", xref="paper", yref="paper", x=0, y=1,
                       yanchor="bottom", showarrow=False, font=dict(color=C["tenue"], size=10))
    fig.update_layout(showlegend=False)
    return fig


def residuos(res: np.ndarray) -> go.Figure:
    fig = _fig("Residuos estandarizados del modelo", alto=250, hovermode="x unified")
    fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
    for v in (-2, 2):
        fig.add_hline(y=v, line=dict(color=C["neutral"], dash="dot", width=1))
    fig.add_trace(go.Scatter(x=np.arange(1, len(res) + 1), y=res, mode="lines", line=dict(color=C["serie_obs"], width=1),
                             hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="Observación")
    fig.update_layout(showlegend=False)
    return fig


def replica_pronostico(pred: pd.DataFrame, horizonte: str) -> go.Figure:
    s = pred[pred["Horizonte"] == horizonte]
    fig = _fig(f"Réplica sección 12.10 · {horizonte}: un pronóstico multipaso desde el corte", alto=300, hovermode="x unified")
    fig.add_trace(go.Scatter(x=s["Fecha"], y=s["Superior"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s["Fecha"], y=s["Inferior"], fill="tonexty", fillcolor=C["banda"], line=dict(width=0),
                             name="Intervalo 95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s["Fecha"], y=s["Real"], name="Real", line=dict(color=C["serie_obs"], width=1.5),
                             hovertemplate="%{y:+.2%}<extra>Real</extra>"))
    fig.add_trace(go.Scatter(x=s["Fecha"], y=s["Predicho"], name="Predicho", line=dict(color=C["serie_mod"], width=1.5),
                             hovertemplate="%{y:+.2%}<extra>Predicho</extra>"))
    fig.update_yaxes(tickformat="+.0%")
    return fig


def distribucion_beta(u: pd.DataFrame) -> go.Figure:
    fig = _fig("Distribución de la beta en el universo S&P 500", alto=250)
    for perfil, color in COLOR_PERFIL.items():
        s = u[u["perfil"] == perfil]["beta"]
        fig.add_trace(go.Histogram(x=s, name=f"{perfil} ({len(s)})", xbins=dict(size=0.1),
                                   marker=dict(color=color, line=dict(color=C["superficie"], width=1)),
                                   hovertemplate="β %{x}: %{y} activos<extra></extra>"))
    for v in (UMBRAL_DEFENSIVO, UMBRAL_AGRESIVO):
        fig.add_vline(x=v, line=dict(color=C["tenue"], dash="dot", width=1), annotation_text=f"β = {v}",
                      annotation_font_color=C["tenue"], annotation_font_size=10)
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title="Beta (rendimientos log mensuales, 5 años, frente a ^GSPC)")
    return fig


def cajas(preparados: dict[str, pd.DataFrame], columna: str, titulo: str) -> go.Figure:
    fig = _fig(titulo, alto=330)
    for k, d in preparados.items():
        fig.add_trace(go.Box(y=d[columna].dropna(), name=k, boxpoints=False, marker_color=C["serie_obs"],
                             line=dict(width=1.3), fillcolor="rgba(79,134,198,0.18)",
                             hovertemplate=f"{k}<br>%{{y:+.2%}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C["eje"], width=1))
    fig.update_yaxes(tickformat="+.0%")
    fig.update_layout(showlegend=False)
    return fig


def seguimiento(h: pd.DataFrame) -> go.Figure:
    r = h.dropna(subset=["rend_real"])
    fig = _fig("Rendimiento pronosticado frente al realizado", alto=320)
    if r.empty:
        return fig
    lim = float(np.nanmax(np.abs(np.r_[r["rend_real"], r["rend_pronosticado"]]))) * 1.15
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(color=C["eje"], dash="dot", width=1),
                             hoverinfo="skip", showlegend=False))
    for color, hz in zip([C["serie_obs"], C["serie_mod"], C["serie_ref"]], ["Semanal", "Mensual", "Trimestral"]):
        s = r[r["horizonte"] == hz]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s["rend_pronosticado"], y=s["rend_real"], mode="markers", name=hz,
                                 marker=dict(size=9, color=color, line=dict(color=C["fondo"], width=2)),
                                 customdata=np.c_[s["ticker"], s["fecha_ultimo_dato"]],
                                 hovertemplate="%{customdata[0]} · %{customdata[1]}<br>pronosticado %{x:+.2%} · real %{y:+.2%}<extra></extra>"))
    fig.update_xaxes(title="Pronosticado", tickformat="+.0%")
    fig.update_yaxes(title="Realizado", tickformat="+.0%")
    return fig
