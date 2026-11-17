"""Orquestación de los cálculos para la interfaz.

La interfaz solo llama a estas funciones. Aquí se combinan datos, calidad,
procesamiento, estadística, modelación, evaluación y persistencia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from vesta import ajustes as A
from vesta.config import HORIZONTES, VERSION
from vesta.datos import almacen, calidad
from vesta.estadistica import descriptiva as D
from vesta.estadistica import pruebas as P
from vesta.evaluacion import rodante
from vesta.evaluacion.original import evaluar_arima_documento, evaluar_rw_documento
from vesta.modelos import prediccion as PR
from vesta.modelos.seleccion import ResultadoSeleccion, auto_arima, etiqueta
from vesta.persistencia import cache, db, historial
from vesta.procesamiento.rendimientos import (VARIABLE_HORIZONTE, alinear_precios, preparar_activo, preparar_serie,
                                              rendimientos_no_solapados)

CONSTRUCCIONES = {
    "solapada": "Acumulado de h días calculado cada día (como el documento)",
    "no_solapada": "Acumulado de h días tomando un periodo cada h sesiones",
}


# ─────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────
def cargar_activo(ticker: str, forzar: bool = False, anios: float = 5) -> almacen.DatosActivo:
    datos = almacen.cargar(ticker, forzar=forzar)
    datos.precios = almacen.recortar_ventana(datos.precios, anios)
    return datos


def ultima_fecha(precios: pd.DataFrame) -> str:
    return str(pd.Timestamp(precios["Date"].max()).date())


def revisar_calidad(datos: almacen.DatosActivo) -> calidad.InformeCalidad:
    return calidad.revisar(datos.crudo, datos.precios)


# ─────────────────────────────────────────────────────────────────────────
# Análisis descriptivo
# ─────────────────────────────────────────────────────────────────────────
def vista_historica(precios: pd.DataFrame, anios: int) -> dict:
    """Comportamiento en la ventana elegida (independiente del horizonte de inversión)."""
    d = almacen.recortar_ventana(precios, anios)
    p = d["Adj.Close"].to_numpy(float)
    r = np.diff(np.log(p))
    return {
        "datos": d,
        "rent_acumulada": float(p[-1] / p[0] - 1) if len(p) > 1 else np.nan,
        "vol_anual": float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) > 20 else np.nan,
        "volumen_promedio": float(d["Volume"].mean()) if "Volume" in d and (d["Volume"] > 0).any() else np.nan,
        "inicio": d["Date"].min(), "fin": d["Date"].max(), "sesiones": len(d),
        "max_drawdown": float(np.min(p / np.maximum.accumulate(p) - 1)) if len(p) else np.nan,
    }


def serie_horizonte(precios: pd.DataFrame, horizonte: str, construccion: str = "solapada") -> pd.DataFrame:
    """Rendimiento logarítmico acumulado del horizonte, identificado por su fecha final."""
    h = HORIZONTES[horizonte]
    if construccion == "no_solapada":
        s = rendimientos_no_solapados(precios.set_index("Date")["Adj.Close"], h)
        return pd.DataFrame({"Date": s.index, "valor": s.to_numpy()})
    d = preparar_activo(precios)
    s = d[["Date", VARIABLE_HORIZONTE[horizonte]]].dropna()
    return pd.DataFrame({"Date": s["Date"].to_numpy(), "valor": s.iloc[:, 1].to_numpy()})


def analisis_horizonte(precios: pd.DataFrame, horizonte: str, construccion: str, aj: A.Ajustes) -> dict:
    h = HORIZONTES[horizonte]
    s = serie_horizonte(precios, horizonte, construccion)
    x = s["valor"].to_numpy()
    r = np.diff(np.log(precios["Adj.Close"].to_numpy(float)))
    lags = max(1, min(aj.lags_ljung_box, len(x) // 5))
    lag_acf = max(1, min(aj.lag_max_acf, len(x) // 4))
    pruebas = [P.rachas(x), P.lilliefors(x)]
    if len(x) > lags + 5:
        pruebas += [P.ljung_box(x, lags), P.arch_lm(x, max(1, min(aj.lags_arch, len(x) // 5)))]
    lp = np.log(precios["Adj.Close"].to_numpy(float))
    return {
        "serie": s, "n": len(x), "h": h, "construccion": construccion,
        "descriptivos": D.resumen_ampliado(x),
        "pruebas": pruebas,
        "estacionariedad": [P.adf(lp), P.kpss(lp), P.adf(r), P.kpss(r)],
        "acf": P.acf(x, lag_acf), "pacf": P.pacf(x, lag_acf), "banda": P.banda_confianza(len(x)),
        "volatilidad": D.volatilidad_horizonte(r, h),
        "vol_diaria_anual": float(np.std(r, ddof=1) * np.sqrt(252)),
    }


def comparar_muestra(precios: dict[str, pd.DataFrame]) -> dict:
    """Kruskal-Wallis (sección 10) y Q de Cochran (sección 11) dentro de la muestra académica."""
    preparados = {k: preparar_activo(v) for k, v in precios.items()}
    kw = {}
    for h, var in VARIABLE_HORIZONTE.items():
        kw[h] = P.kruskal_wallis({k: d[var].dropna().to_numpy() for k, d in preparados.items()})
    alineados = alinear_precios(precios)
    cochran, exitos = {}, {}
    for h, k in HORIZONTES.items():
        muestra = alineados.iloc[::k]
        rend = np.log(muestra / muestra.shift(1)).dropna()
        exito = (rend > 0).astype(int)
        res, e, bloques = P.cochran_q(exito.to_numpy())
        cochran[h] = res
        exitos[h] = dict(zip(exito.columns, map(int, e)))
    return {"kruskal": kw, "cochran": cochran, "exitos": exitos, "fechas_comunes": len(alineados),
            "preparados": preparados}


# ─────────────────────────────────────────────────────────────────────────
# Modelación, evaluación y predicción
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ResultadoModelacion:
    seleccion: ResultadoSeleccion
    evaluacion: rodante.ResultadoEvaluacion
    n: int
    corte: int
    inicio: pd.Timestamp
    fecha_corte: pd.Timestamp
    fin: pd.Timestamp
    ajustes: dict
    ejecutada_en: str

    def principal(self) -> rodante.EvaluacionModelo:
        return self.evaluacion.modelos[0]


def _precios_modelo(precios: pd.DataFrame, aj: A.Ajustes) -> pd.DataFrame:
    return almacen.recortar_ventana(precios, aj.anios_historia)


def modelacion(ticker: str, precios: pd.DataFrame, aj: A.Ajustes, progreso=None, forzar: bool = False) -> ResultadoModelacion:
    """Pasos 6 a 8: serie, selección de la especificación (solo entrenamiento) y evaluación rodante."""
    p = _precios_modelo(precios, aj)

    def calcular():
        serie = preparar_serie(p, ticker, aj.prop_train)
        sel = auto_arima(serie.r_train, d=aj.d, max_p=aj.max_p, max_q=aj.max_q,
                         permitir_media=(aj.incluir_deriva if aj.d == 1 else True), criterio=aj.criterio,
                         progreso=(lambda f, t: progreso(0.3 * f, f"Especificación · {t}")) if progreso else None)
        ev = rodante.evaluar(serie, sel.especificaciones(), aj,
                             progreso=(lambda f, t: progreso(0.3 + 0.65 * f, f"Evaluación rodante · {t}")) if progreso else None)
        return ResultadoModelacion(sel, ev, serie.n, serie.corte, serie.fechas.iloc[0], serie.fecha_corte,
                                   serie.fechas.iloc[-1], aj.como_dict(), db.ahora())

    return cache.obtener(ticker, ultima_fecha(p), "modelacion", calcular, aj.firma(), forzar)


def modelacion_disponible(ticker: str, precios: pd.DataFrame, aj: A.Ajustes) -> bool:
    return cache.existe(ticker, ultima_fecha(_precios_modelo(precios, aj)), "modelacion", aj.firma())


@dataclass
class ResultadoPrediccion:
    ticker: str
    empresa: str
    horizonte: str
    prediccion: PR.Prediccion
    modelacion: ResultadoModelacion
    metricas: dict                      # fila de la evaluación del modelo principal para el horizonte
    metricas_todos: pd.DataFrame        # todos los modelos evaluados, ese horizonte
    diagnostico_residuos: list
    residuos: np.ndarray
    estacionariedad: list
    calidad: calidad.InformeCalidad
    ejecutada_en: str
    id_registro: int | None = None
    avisos: list = field(default_factory=list)


def generar_prediccion(ticker: str, empresa: str, datos: almacen.DatosActivo, horizonte: str, aj: A.Ajustes,
                       progreso=None, registrar: bool = True) -> ResultadoPrediccion:
    """Flujo de la sección 10: calidad → serie → modelos → selección → pronóstico → diagnóstico → registro."""
    if progreso:
        progreso(0.0, "Verificando la calidad de los datos")
    inf = revisar_calidad(datos)
    if not inf.apta:
        raise ValueError("La serie no cumple las condiciones mínimas: " + " ".join(inf.errores))
    p = _precios_modelo(datos.precios, aj)
    mod = modelacion(ticker, datos.precios, aj, progreso=(lambda f, t: progreso(0.05 + 0.85 * f, t)) if progreso else None)
    if progreso:
        progreso(0.92, "Pronóstico desde el último dato")
    serie = preparar_serie(p, ticker, aj.prop_train)
    h = HORIZONTES[horizonte]
    pred = PR.generar(serie, mod.seleccion.mejor, horizonte, h, aj)
    res = pred.modelo.residuos
    k = pred.modelo.p + pred.modelo.q
    diag = [P.ljung_box(res, aj.lags_ljung_box, fitdf=k), P.ljung_box(res, 2 * aj.lags_ljung_box, fitdf=k),
            P.arch_lm(res, aj.lags_arch), P.lilliefors(res)]
    principal = mod.principal()
    fila = principal.metricas.set_index("Horizonte").loc[horizonte].to_dict()
    todos = pd.concat([m.metricas.assign(Modelo=f"{m.nombre}: {m.etiqueta}") for m in mod.evaluacion.modelos])
    todos = todos[todos["Horizonte"] == horizonte]
    ahora = datetime.now().isoformat(timespec="seconds")
    resultado = ResultadoPrediccion(
        ticker=ticker, empresa=empresa, horizonte=horizonte, prediccion=pred, modelacion=mod, metricas=fila,
        metricas_todos=todos, diagnostico_residuos=diag, residuos=res,
        estacionariedad=P.evaluar_caminata(serie.log_precio, (aj.lags_ljung_box, 2 * aj.lags_ljung_box), aj.lags_arch)[:4],
        calidad=inf, ejecutada_en=ahora, avisos=list(inf.avisos))
    if registrar:
        resultado.id_registro = _registrar(resultado, aj, datos)
    if progreso:
        progreso(1.0, "Listo")
    return resultado


def _registrar(res: ResultadoPrediccion, aj: A.Ajustes, datos: almacen.DatosActivo) -> int:
    pr = res.prediccion
    metricas = {k: res.metricas.get(k) for k in ("Origenes", "MAE", "RMSE", "MAE_RW", "RMSE_RW", "U_Theil",
                                                    "DM_estadistico", "DM_p_valor", "Direccion_pct",
                                                    "Direccion_p_binomial", "Cobertura_intervalo", "Veredicto",
                                                    "MAPE_precio")}
    m = res.modelacion
    id_ = historial.registrar({
        "ejecutada_en": res.ejecutada_en, "ticker": res.ticker, "empresa": res.empresa,
        "horizonte": res.horizonte, "h": pr.h, "fecha_ultimo_dato": str(pr.fecha_ultimo_dato.date()),
        "fecha_objetivo_estimada": str(pr.fecha_objetivo.date()), "precio_inicial": pr.precio_actual,
        "rend_pronosticado": pr.rend_final, "rend_inferior": float(pr.rend_inf[-1]),
        "rend_superior": float(pr.rend_sup[-1]), "precio_pronosticado": pr.precio_final,
        "precio_inferior": float(pr.precio_inf[-1]), "precio_superior": float(pr.precio_sup[-1]),
        "nivel_intervalo": pr.nivel, "modelo": pr.etiqueta, "orden": f"({pr.modelo.p},{pr.modelo.d},{pr.modelo.q})",
        "criterio": aj.criterio.upper(), "metricas": metricas, "ajustes": aj.como_dict(),
        "datos": {"fuente": "Yahoo Finance (Adj Close)", "origen": datos.origen,
                  "obtenidos": str(datos.actualizado), "inicio": str(m.inicio.date()), "fin": str(m.fin.date()),
                  "observaciones": m.n, "corte_entrenamiento": str(m.fecha_corte.date()),
                  "coeficientes": pr.modelo.coeficientes(), "sigma2": pr.modelo.sigma2},
        "version": VERSION,
    })
    db.guardar_evaluacion(res.ticker, res.horizonte, pr.etiqueta, metricas, aj.como_dict(),
                          {"inicio": str(m.inicio.date()), "fin": str(m.fin.date()), "n": m.n})
    return id_


def actualizar_seguimiento(ticker: str, precios: pd.DataFrame) -> int:
    return historial.actualizar_realizados(ticker, precios)


def replica_documento(ticker: str, precios: pd.DataFrame, progreso=None) -> dict:
    p = almacen.recortar_ventana(precios, 5)

    def calcular():
        arima = evaluar_arima_documento(p, ticker, progreso=progreso)
        rw_pred, rw_metr = evaluar_rw_documento(p, ticker)
        arima["rw_predicciones"] = rw_pred
        arima["rw_metricas"] = rw_metr
        return arima

    return cache.obtener(ticker, ultima_fecha(p), "replica_documento", calcular, "documento")


def replica_disponible(ticker: str, precios: pd.DataFrame) -> bool:
    return cache.existe(ticker, ultima_fecha(almacen.recortar_ventana(precios, 5)), "replica_documento", "documento")
