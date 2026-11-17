"""Pruebas estadísticas replicadas de los paquetes de R usados en VESTA.

Cada función reproduce el algoritmo de la función de R indicada (mismo
estadístico, misma regla para el p-valor), no una aproximación genérica:

    rachas            tseries::runs.test
    lilliefors        nortest::lillie.test
    kruskal_wallis    stats::kruskal.test
    cochran_q         función `cochran_q` del documento (sección 11)
    adf               tseries::adf.test   (con tendencia, tabla interpolada)
    kpss              tseries::kpss.test  (null = "Level", lshort = TRUE)
    ljung_box         stats::Box.test(type = "Ljung-Box")
    arch_lm           FinTS::ArchTest(demean = FALSE)
    t_media_cero      stats::t.test(mu = 0)
    acf / pacf        stats::acf / stats::pacf
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

from dataclasses import field

ALFA = 0.05

# Qué evalúa cada prueba, su hipótesis alternativa y cómo leer cada decisión.
# Las lecturas evitan presentar un "no se rechaza" como prueba de que H0 es cierta.
CATALOGO = {
    "rachas": ("Aleatoriedad del orden de los valores por encima y por debajo de la mediana",
               "La secuencia no es aleatoria (hay agrupamiento o alternancia)",
               "El orden de los valores muestra un patrón no aleatorio.",
               "No hay evidencia suficiente de un patrón no aleatorio en el orden de los valores."),
    "lilliefors": ("Normalidad con media y desviación estimadas de la muestra",
                   "Los datos no provienen de una distribución normal",
                   "La distribución se aparta de la normal (típicamente por colas pesadas).",
                   "No hay evidencia suficiente para descartar la normalidad con este tamaño de muestra."),
    "jarque_bera": ("Normalidad a partir de la asimetría y la curtosis",
                    "Asimetría o curtosis distintas de las de una normal",
                    "La asimetría o la curtosis son incompatibles con una normal.",
                    "No hay evidencia suficiente contra la normalidad en asimetría y curtosis."),
    "adf": ("Presencia de raíz unitaria (no estacionariedad)",
            "La serie es estacionaria (alrededor de una tendencia determinística)",
            "Hay evidencia de estacionariedad: se rechaza la raíz unitaria.",
            "No hay evidencia suficiente para rechazar la raíz unitaria."),
    "kpss": ("Estacionariedad en nivel",
             "La serie tiene raíz unitaria (no es estacionaria)",
             "Hay evidencia contra la estacionariedad en nivel.",
             "No hay evidencia suficiente contra la estacionariedad en nivel."),
    "ljung_box": ("Autocorrelación conjunta hasta el rezago indicado",
                  "Al menos una autocorrelación hasta ese rezago es distinta de cero",
                  "Hay autocorrelación: el pasado contiene información lineal sobre el presente.",
                  "No hay evidencia suficiente de autocorrelación conjunta hasta ese rezago."),
    "arch": ("Heterocedasticidad condicional (efectos ARCH)",
             "La varianza depende de los cuadrados pasados (volatilidad agrupada)",
             "Hay evidencia de efectos ARCH: la volatilidad se agrupa en el tiempo.",
             "No hay evidencia suficiente de efectos ARCH con esos rezagos."),
    "t_media": ("Si el rendimiento medio es cero",
                "El rendimiento medio es distinto de cero",
                "El rendimiento medio es distinto de cero.",
                "No hay evidencia suficiente de que el rendimiento medio sea distinto de cero."),
    "kruskal": ("Si los activos comparten la misma distribución de rendimientos",
                "Al menos un activo tiene una distribución distinta (desplazada)",
                "Al menos un activo tiene una distribución de rendimientos distinta.",
                "No hay evidencia suficiente de diferencias entre las distribuciones."),
    "cochran": ("Si la proporción de periodos con ganancia es igual entre activos",
                "Al menos un activo tiene una proporción de éxitos distinta",
                "La proporción de periodos con ganancia difiere entre activos.",
                "No hay evidencia suficiente de diferencias en la proporción de periodos con ganancia."),
}


@dataclass
class ResultadoPrueba:
    prueba: str
    estadistico: float
    p_valor: float
    h0: str
    gl: float | None = None
    n: int | None = None
    detalle: str = ""
    clave: str = ""
    rezagos: int | None = None
    valores_criticos: dict = field(default_factory=dict)

    @property
    def que_evalua(self) -> str:
        return CATALOGO.get(self.clave, ("", "", "", ""))[0]

    @property
    def h1(self) -> str:
        return CATALOGO.get(self.clave, ("", "", "", ""))[1]

    def decision(self, alfa: float = ALFA) -> str:
        if self.p_valor is None or not np.isfinite(self.p_valor):
            return "No calculable"
        return "Se rechaza H0" if self.p_valor < alfa else "No se rechaza H0"

    def interpretacion(self, alfa: float = ALFA) -> str:
        if self.p_valor is None or not np.isfinite(self.p_valor):
            return "No calculable con estos datos."
        _, _, si_rechaza, si_no = CATALOGO.get(self.clave, ("", "", "Se rechaza H0.", "No se rechaza H0."))
        return si_rechaza if self.p_valor < alfa else si_no

    def como_dict(self, alfa: float = ALFA) -> dict:
        d = asdict(self)
        d.update(decision=self.decision(alfa), interpretacion=self.interpretacion(alfa),
                 que_evalua=self.que_evalua, h1=self.h1)
        return d


def _limpiar(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _interpolar(x: np.ndarray, y: np.ndarray, v: float) -> float:
    """`approx(x, y, v, rule = 2)` de R: lineal y constante fuera del rango."""
    orden = np.argsort(x)
    return float(np.interp(v, x[orden], y[orden]))


# ─────────────────────────────────────────────────────────────────────────
# Aleatoriedad y normalidad (sección 9)
# ─────────────────────────────────────────────────────────────────────────
def rachas(x, binaria: bool = False) -> ResultadoPrueba:
    """Prueba de rachas de Wald-Wolfowitz como `tseries::runs.test`.

    Si `binaria` es False, la serie se dicotomiza como en el documento:
    `as.factor(serie > median(serie))`.
    """
    x = _limpiar(x)
    if not binaria:
        x = x > np.median(x)
    x = np.asarray(x, dtype=bool)
    niveles = np.unique(x)
    if len(niveles) != 2 or len(x) < 2:
        return ResultadoPrueba("Rachas", np.nan, np.nan, "La serie es aleatoria", n=len(x), clave="rachas")
    s = np.where(x == niveles[0], -1, 1)
    n = len(s)
    R = 1 + int(np.sum(s[1:] != s[:-1]))
    n1 = int(np.sum(s == -1))
    n2 = int(np.sum(s == 1))
    m = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (R - m) / np.sqrt(var)
    p = 2 * stats.norm.cdf(-abs(z))
    return ResultadoPrueba("Rachas (Wald-Wolfowitz)", float(z), float(p), "La serie es aleatoria", n=n,
                           detalle=f"Rachas observadas {R}, esperadas {m:.1f}", clave="rachas",
                           valores_criticos={"5 % (|z|)": 1.96, "1 % (|z|)": 2.576})


def lilliefors(x) -> ResultadoPrueba:
    """Lilliefors (Kolmogorov-Smirnov con parámetros estimados), `nortest::lillie.test`.

    p-valor con la aproximación de Dallal y Wilkinson (1986) y, si supera 0.1,
    la fórmula de Stephens, exactamente como en nortest.
    """
    x = np.sort(_limpiar(x))
    n = len(x)
    if n < 5:
        return ResultadoPrueba("Lilliefors", np.nan, np.nan, "La serie es normal", n=n, clave="lilliefors")
    p = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    K = max(np.max(i / n - p), np.max(p - (i - 1) / n))
    if n <= 100:
        Kd, nd = K, n
    else:
        Kd, nd = K * (n / 100) ** 0.49, 100
    pv = np.exp(-7.01256 * Kd**2 * (nd + 2.78019) + 2.99587 * Kd * np.sqrt(nd + 2.78019)
                - 0.122119 + 0.974598 / np.sqrt(nd) + 1.67997 / nd)
    if pv > 0.1:
        KK = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * K
        if KK <= 0.302:
            pv = 1.0
        elif KK <= 0.5:
            pv = 2.76773 - 19.828315 * KK + 80.709644 * KK**2 - 138.55152 * KK**3 + 81.218052 * KK**4
        elif KK <= 0.9:
            pv = -4.901232 + 40.662806 * KK - 97.490286 * KK**2 + 94.029866 * KK**3 - 32.355711 * KK**4
        elif KK <= 1.31:
            pv = 6.198765 - 19.558097 * KK + 23.186922 * KK**2 - 12.234627 * KK**3 + 2.423045 * KK**4
        else:
            pv = 0.0
    return ResultadoPrueba("Lilliefors", float(K), float(pv), "La serie es normal", n=n, clave="lilliefors")


def jarque_bera(x) -> ResultadoPrueba:
    """Complemento (no está en el documento): Jarque-Bera como `tseries::jarque.bera.test`."""
    x = _limpiar(x)
    n = len(x)
    m = x - x.mean()
    m2 = np.mean(m**2)
    sk = np.mean(m**3) / m2**1.5
    ku = np.mean(m**4) / m2**2
    jb = n / 6 * (sk**2 + (ku - 3) ** 2 / 4)
    return ResultadoPrueba("Jarque-Bera", float(jb), float(stats.chi2.sf(jb, 2)), "La serie es normal", gl=2, n=n, clave="jarque_bera")


# ─────────────────────────────────────────────────────────────────────────
# Comparación entre activos (secciones 10 y 11)
# ─────────────────────────────────────────────────────────────────────────
def kruskal_wallis(grupos: dict[str, np.ndarray]) -> ResultadoPrueba:
    """`kruskal.test(valor ~ Activo)`: H con corrección por empates."""
    muestras = [_limpiar(v) for v in grupos.values()]
    muestras = [m for m in muestras if len(m) > 0]
    if len(muestras) < 2:
        return ResultadoPrueba("Kruskal-Wallis", np.nan, np.nan, "Las distribuciones son iguales", clave="kruskal")
    H, p = stats.kruskal(*muestras)
    return ResultadoPrueba("Kruskal-Wallis", float(H), float(p), "Las distribuciones son iguales",
                           gl=len(muestras) - 1, n=int(sum(len(m) for m in muestras)), clave="kruskal")


def cochran_q(M) -> tuple[ResultadoPrueba, np.ndarray, int]:
    """Q de Cochran como la función `cochran_q` del documento.

    M: matriz 0/1, filas = bloques (periodos), columnas = activos.
    Devuelve (resultado, éxitos por columna, número de bloques).
    """
    M = np.asarray(M, dtype=float)
    k = M.shape[1]
    Cj = M.sum(axis=0)
    Ri = M.sum(axis=1)
    N = M.sum()
    den = k * N - np.sum(Ri**2)
    if den == 0:
        Q = p = np.nan
    else:
        Q = (k - 1) * (k * np.sum(Cj**2) - N**2) / den
        p = stats.chi2.sf(Q, k - 1)
    r = ResultadoPrueba("Q de Cochran", float(Q), float(p), "La proporción de éxitos es igual entre activos",
                        gl=k - 1, n=M.shape[0], clave="cochran")
    return r, Cj, M.shape[0]


# ─────────────────────────────────────────────────────────────────────────
# Estacionariedad y dependencia (sección 12.5)
# ─────────────────────────────────────────────────────────────────────────
def _ols(y: np.ndarray, X: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    gl = len(y) - X.shape[1]
    s2 = resid @ resid / gl
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), resid


def _embed(x: np.ndarray, dim: int) -> np.ndarray:
    """`embed(x, dim)` de R: columna j = x desplazada j-1 hacia atrás."""
    n = len(x) - dim + 1
    return np.column_stack([x[dim - 1 - j: dim - 1 - j + n] for j in range(dim)])


_ADF_TABLA = -np.array([
    [4.38, 4.15, 4.04, 3.99, 3.98, 3.96],
    [3.95, 3.80, 3.73, 3.69, 3.68, 3.66],
    [3.60, 3.50, 3.45, 3.43, 3.42, 3.41],
    [3.24, 3.18, 3.15, 3.13, 3.13, 3.12],
    [1.14, 1.19, 1.22, 1.23, 1.24, 1.25],
    [0.80, 0.87, 0.90, 0.92, 0.93, 0.94],
    [0.50, 0.58, 0.62, 0.64, 0.65, 0.66],
    [0.15, 0.24, 0.28, 0.31, 0.32, 0.33],
]).T
_ADF_T = np.array([25, 50, 100, 250, 500, 1e5])
_ADF_P = np.array([0.01, 0.025, 0.05, 0.1, 0.9, 0.95, 0.975, 0.99])


def adf(x, k: int | None = None) -> ResultadoPrueba:
    """Dickey-Fuller aumentada como `tseries::adf.test(alternative = "stationary")`.

    Regresión con constante y tendencia, k = trunc((n-1)^(1/3)) rezagos y
    p-valor interpolado en la tabla de Banerjee et al. (acotado a [0.01, 0.99]).
    """
    x = _limpiar(x)
    if k is None:
        k = int(np.trunc((len(x) - 1) ** (1 / 3)))
    k = k + 1
    y = np.diff(x)
    n = len(y)
    z = _embed(y, k)
    yt = z[:, 0]
    xt1 = x[k - 1: n]
    tt = np.arange(k, n + 1, dtype=float)
    cols = [np.ones_like(yt), xt1, tt]
    if k > 1:
        cols += [z[:, j] for j in range(1, k)]
    X = np.column_stack(cols)
    beta, se, _ = _ols(yt, X)
    stat = beta[1] / se[1]
    tabla_interp = np.array([np.interp(n, _ADF_T, _ADF_TABLA[:, j]) for j in range(_ADF_TABLA.shape[1])])
    p = _interpolar(tabla_interp, _ADF_P, stat)
    criticos = {"1 %": float(tabla_interp[0]), "5 %": float(tabla_interp[2]), "10 %": float(tabla_interp[3])}
    return ResultadoPrueba("ADF", float(stat), p, "La serie tiene raíz unitaria (no estacionaria)",
                           n=len(x), detalle="Regresión con constante y tendencia. p-valor interpolado y acotado entre 0.01 y 0.99",
                           clave="adf", rezagos=k - 1, valores_criticos=criticos)


def _pp_sum(e: np.ndarray, l: int) -> float:
    n = len(e)
    total = 0.0
    for i in range(1, l + 1):
        total += (1 - i / (l + 1)) * float(np.dot(e[i:], e[:-i]))
    return 2 * total / n


def kpss(x, lshort: bool = True) -> ResultadoPrueba:
    """KPSS de nivel como `tseries::kpss.test(null = "Level")`. p-valor acotado [0.01, 0.1]."""
    x = _limpiar(x)
    n = len(x)
    e = x - x.mean()
    s = np.cumsum(e)
    eta = np.sum(s**2) / n**2
    s2 = np.sum(e**2) / n
    l = int(np.trunc((4 if lshort else 12) * (n / 100) ** 0.25))
    s2 = s2 + _pp_sum(e, l)
    stat = eta / s2
    p = _interpolar(np.array([0.347, 0.463, 0.574, 0.739]), np.array([0.1, 0.05, 0.025, 0.01]), stat)
    return ResultadoPrueba("KPSS", float(stat), p, "La serie es estacionaria en nivel", n=n,
                           detalle="Ventana de Bartlett. p-valor interpolado y acotado entre 0.01 y 0.10",
                           clave="kpss", rezagos=l,
                           valores_criticos={"10 %": 0.347, "5 %": 0.463, "2.5 %": 0.574, "1 %": 0.739})


def acf(x, lag_max: int = 30) -> np.ndarray:
    """Autocorrelaciones 1..lag_max como `stats::acf` (media muestral, denominador n)."""
    x = _limpiar(x)
    x = x - x.mean()
    n = len(x)
    den = np.dot(x, x)
    lag_max = min(lag_max, n - 1)
    return np.array([np.dot(x[k:], x[:n - k]) / den for k in range(1, lag_max + 1)])


def pacf(x, lag_max: int = 30) -> np.ndarray:
    """Autocorrelaciones parciales vía Durbin-Levinson, como `stats::pacf`."""
    r = acf(x, lag_max)
    m = len(r)
    salida = np.zeros(m)
    phi = np.zeros(m)
    v = 1.0
    for k in range(m):
        if k == 0:
            a = r[0]
        else:
            a = (r[k] - np.dot(phi[:k], r[k - 1::-1][:k])) / v
            phi[:k] = phi[:k] - a * phi[:k][::-1]
        phi[k] = a
        salida[k] = a
        v *= 1 - a**2
    return salida


def banda_confianza(n: int, nivel: float = 0.95) -> float:
    return float(stats.norm.ppf(0.5 + nivel / 2) / np.sqrt(n))


def rezagos_significativos(x, lag_max: int = 20) -> dict:
    """`lags_signif` del documento (sección 12.8)."""
    x = _limpiar(x)
    lim = banda_confianza(len(x))
    a = acf(x, lag_max)
    p = pacf(x, lag_max)
    return {"banda": lim,
            "acf": [int(i + 1) for i in np.where(np.abs(a) > lim)[0]],
            "pacf": [int(i + 1) for i in np.where(np.abs(p) > lim)[0]]}


def ljung_box(x, lag: int = 10, fitdf: int = 0) -> ResultadoPrueba:
    """`Box.test(x, lag, type = "Ljung-Box", fitdf)`."""
    x = _limpiar(x)
    n = len(x)
    r = acf(x, lag)
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lag + 1)))
    gl = lag - fitdf
    return ResultadoPrueba(f"Ljung-Box ({lag})", float(q), float(stats.chi2.sf(q, gl)),
                           "No hay autocorrelación hasta el rezago indicado", gl=gl, n=n, clave="ljung_box",
                           rezagos=lag, valores_criticos={"5 %": float(stats.chi2.ppf(0.95, gl))})


def arch_lm(x, lags: int = 10, demean: bool = False) -> ResultadoPrueba:
    """`FinTS::ArchTest(x, lags, demean = FALSE)`: n·R² de la regresión de x² en sus rezagos."""
    x = _limpiar(x)
    if demean:
        x = x - x.mean()
    mat = _embed(x**2, lags + 1)
    y = mat[:, 0]
    X = np.column_stack([np.ones(len(y)), mat[:, 1:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    stat = r2 * len(resid)
    return ResultadoPrueba(f"ARCH-LM ({lags})", float(stat), float(stats.chi2.sf(stat, lags)),
                           "No hay efectos ARCH (varianza condicional constante)", gl=lags, n=len(x), clave="arch",
                           rezagos=lags, valores_criticos={"5 %": float(stats.chi2.ppf(0.95, lags))})


def t_media_cero(x) -> ResultadoPrueba:
    x = _limpiar(x)
    t, p = stats.ttest_1samp(x, 0.0)
    return ResultadoPrueba("t de Student (media = 0)", float(t), float(p), "La media es cero", gl=len(x) - 1, n=len(x),
                           clave="t_media")


# ─────────────────────────────────────────────────────────────────────────
# Baterías usadas por la interfaz
# ─────────────────────────────────────────────────────────────────────────
def evaluar_caminata(log_precio: np.ndarray, lags_lb: tuple = (10, 20), lags_arch: int = 10) -> list[ResultadoPrueba]:
    """`evaluar_caminata` (sección 12.5): pruebas sobre niveles y primeras diferencias."""
    niveles = _limpiar(log_precio)
    dif = np.diff(niveles)
    resultados = []
    for nombre, prueba in (("ADF", adf), ("KPSS", kpss)):
        r1 = prueba(niveles)
        r1.prueba = f"{nombre} · log-precio (nivel)"
        r2 = prueba(dif)
        r2.prueba = f"{nombre} · primeras diferencias"
        resultados += [r1, r2]
    resultados += [ljung_box(dif, l) for l in lags_lb]
    r = t_media_cero(dif)
    r.prueba = "Media de las diferencias = 0"
    resultados += [r, arch_lm(dif, lags_arch)]
    return resultados


def diagnostico_horizonte(serie) -> dict:
    """`diagnostico_horizonte`: rachas y Lilliefors (p-valores) para una serie."""
    s = _limpiar(serie)
    if len(s) < 5:
        return {"n": len(s), "p_aleatoriedad": np.nan, "p_normalidad": np.nan}
    return {"n": len(s), "p_aleatoriedad": rachas(s).p_valor, "p_normalidad": lilliefors(s).p_valor}
