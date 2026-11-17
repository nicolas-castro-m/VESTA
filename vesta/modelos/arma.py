"""Modelos ARMA/ARIMA con verosimilitud exacta, siguiendo a `stats::arima` de R.

Convención de R:
    (y_t - mu) = sum_i phi_i (y_{t-i} - mu) + e_t + sum_j theta_j e_{t-j}

Estimación por el método "CSS-ML" (el que usa `auto.arima` con
`approximation = FALSE`):
  1. Mínimos cuadrados condicionales (CSS) para obtener valores iniciales.
     Si la parte AR resultante no es estacionaria, el modelo se descarta,
     igual que R ("non-stationary AR part from CSS").
  2. Máxima verosimilitud exacta con el filtro de Kalman (representación de
     Gardner et al., 1980) partiendo de los valores CSS, con la parte AR
     reparametrizada por autocorrelaciones parciales (`transform.pars`).

Para acelerar, cuando la covarianza del filtro converge a su estado
estacionario los residuos restantes se calculan con un filtro lineal; el
resultado es el mismo que el del filtro completo dentro de la tolerancia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize, signal

TOL_CONVERGENCIA = 1e-9


# ─────────────────────────────────────────────────────────────────────────
# Utilidades de raíces y reparametrización (partrans / invpartrans de R)
# ─────────────────────────────────────────────────────────────────────────
def raices_min_modulo(coef: np.ndarray, signo: float) -> float:
    """Módulo mínimo de las raíces de 1 + signo*(c1 z + ... + cp z^p)."""
    coef = np.asarray(coef, dtype=float)
    if len(coef) == 0 or np.all(coef == 0):
        return np.inf
    ultimo = np.max(np.nonzero(coef)[0]) + 1
    poli = np.r_[1.0, signo * coef[:ultimo]]
    return float(np.min(np.abs(np.roots(poli[::-1]))))


def ar_estacionario(phi: np.ndarray) -> bool:
    return raices_min_modulo(phi, -1.0) > 1.0


def partrans(raw: np.ndarray) -> np.ndarray:
    p = len(raw)
    new = np.tanh(raw).astype(float)
    work = new.copy()
    for j in range(1, p):
        a = new[j]
        for k in range(j):
            work[k] -= a * new[j - k - 1]
        new[:j] = work[:j]
    return new


def invpartrans(phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    new = np.asarray(phi, dtype=float).copy()
    work = new.copy()
    for j in range(p - 1, 0, -1):
        a = new[j]
        for k in range(j):
            work[k] = (new[k] + a * new[j - k - 1]) / (1 - a * a)
        new[:j] = work[:j]
    return np.arctanh(np.clip(new, -0.999999, 0.999999))


def ma_invertir(theta: np.ndarray) -> np.ndarray:
    """`maInvert` de R: refleja las raíces MA que caen dentro del círculo unidad."""
    q = len(theta)
    if q == 0 or np.all(theta == 0):
        return theta
    q0 = np.max(np.nonzero(theta)[0]) + 1
    raices = np.roots(np.r_[1.0, theta[:q0]][::-1])
    if np.all(np.abs(raices) >= 1):
        return theta
    raices = np.where(np.abs(raices) < 1, 1 / raices, raices)
    x = np.array([1.0 + 0j])
    for r in raices:
        x = np.r_[x, 0] - np.r_[0, x] / r
    nuevo = np.real(x[1:])
    salida = np.zeros(q)
    salida[:q0] = nuevo
    return salida


def psi_pesos(phi: np.ndarray, theta: np.ndarray, m: int) -> np.ndarray:
    """Pesos psi_0..psi_{m-1} de la representación MA(infinito)."""
    psi = np.zeros(m)
    psi[0] = 1.0
    p, q = len(phi), len(theta)
    for j in range(1, m):
        v = theta[j - 1] if j <= q else 0.0
        for i in range(1, min(j, p) + 1):
            v += phi[i - 1] * psi[j - i]
        psi[j] = v
    return psi


# ─────────────────────────────────────────────────────────────────────────
# Espacio de estados y filtro de Kalman
# ─────────────────────────────────────────────────────────────────────────
def espacio_estados(phi: np.ndarray, theta: np.ndarray):
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    T = np.zeros((r, r))
    T[:p, 0] = phi
    if r > 1:
        T[:-1, 1:] = np.eye(r - 1)
    Rv = np.zeros(r)
    Rv[0] = 1.0
    Rv[1:q + 1] = theta
    RR = np.outer(Rv, Rv)
    if p == 0:
        # Sin parte AR la covarianza estacionaria es suma finita
        P0 = np.zeros((r, r))
        Tk = np.eye(r)
        for _ in range(r):
            P0 += Tk @ RR @ Tk.T
            Tk = T @ Tk
    else:
        P0 = linalg.solve_discrete_lyapunov(T, RR)
    return T, Rv, RR, P0


@dataclass
class SalidaFiltro:
    v: np.ndarray            # innovaciones
    F: np.ndarray            # varianzas (unidades de sigma^2)
    a_final: np.ndarray      # estado predicho para n+1
    P_final: np.ndarray
    estados: np.ndarray | None = None   # a_{t+1|t} para cada t (si se pide)


def filtro_kalman(w: np.ndarray, phi: np.ndarray, theta: np.ndarray, guardar_estados: bool = False) -> SalidaFiltro:
    """Filtro de Kalman para w = y - mu con sigma^2 = 1."""
    n = len(w)
    T, Rv, RR, P = espacio_estados(phi, theta)
    r = T.shape[0]
    a = np.zeros(r)
    v = np.empty(n)
    F = np.empty(n)
    estados = np.empty((n, r)) if guardar_estados else None
    K_estacionario = T @ Rv
    t = 0
    convergido = False
    while t < n:
        f = P[0, 0]
        vt = w[t] - a[0]
        v[t] = vt
        F[t] = f
        K = (T @ P[:, 0]) / f
        a = T @ a + K * vt
        P = T @ P @ T.T + RR - np.outer(K, K) * f
        if guardar_estados:
            estados[t] = a
        t += 1
        if np.max(np.abs(P - RR)) < TOL_CONVERGENCIA:
            convergido = True
            break
    if convergido and t < n:
        if guardar_estados:
            # La recursión en estado estacionario es a <- T a + (T R) v
            for s in range(t, n):
                vt = w[s] - a[0]
                v[s] = vt
                F[s] = 1.0
                a = T @ a + K_estacionario * vt
                estados[s] = a
        else:
            v_rest, a = _residuos_estacionarios(w, v, t, phi, theta, T, Rv, a)
            v[t:] = v_rest
            F[t:] = 1.0
        P = RR.copy()
    return SalidaFiltro(v=v, F=F, a_final=a, P_final=P, estados=estados)


def _residuos_estacionarios(w, v, t, phi, theta, T, Rv, a):
    """Residuos e_s, s >= t, con la recursión ARMA (equivale al filtro convergido)."""
    p, q = len(phi), len(theta)
    b = np.r_[1.0, -phi]
    den = np.r_[1.0, theta]
    x_prev = w[:t][::-1][:p] if p else np.array([])
    y_prev = v[:t][::-1][:q] if q else np.array([])
    x_prev = np.r_[x_prev, np.zeros(max(0, p - len(x_prev)))]
    y_prev = np.r_[y_prev, np.zeros(max(0, q - len(y_prev)))]
    if p == 0 and q == 0:
        resto = w[t:].copy()
    else:
        zi = signal.lfiltic(b, den, y_prev, x_prev)
        resto, _ = signal.lfilter(b, den, w[t:], zi=zi)
    # Estado predicho a_{n+1|n} a partir de los últimos valores y residuos
    e = np.r_[v[:t], resto]
    return resto, _estado_desde_historia(w, e, len(w), phi, theta, T.shape[0])


def _estado_desde_historia(w, e, s, phi, theta, r):
    """Estado a_{s|s-1} a partir de valores y residuos pasados (forma de Gardner)."""
    p, q = len(phi), len(theta)
    a = np.zeros(r)
    for i in range(r):
        val = 0.0
        for j in range(i, p):
            idx = s - 1 - (j - i)
            if idx >= 0:
                val += phi[j] * w[idx]
        for j in range(i, q):
            idx = s - 1 - (j - i)
            if idx >= 0:
                val += theta[j] * e[idx]
        a[i] = val
    return a


# ─────────────────────────────────────────────────────────────────────────
# Estimación
# ─────────────────────────────────────────────────────────────────────────
def _css_residuos(w: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    p = len(phi)
    n = len(w)
    u = w[p:].copy()
    for i in range(p):
        u -= phi[i] * w[p - 1 - i: n - 1 - i]
    if len(theta):
        u = signal.lfilter([1.0], np.r_[1.0, theta], u)
    return u


def _desempacar(par, p, q, con_media):
    phi = par[:p]
    theta = par[p:p + q]
    mu = par[p + q] if con_media else 0.0
    return phi, theta, mu


def _objetivo_css(par, y, p, q, con_media):
    phi, theta, mu = _desempacar(par, p, q, con_media)
    with np.errstate(all="ignore"):
        e = _css_residuos(y - mu, phi, theta)
        s2 = np.mean(e**2)
    if not np.isfinite(s2) or s2 <= 0:
        return 1e10
    return 0.5 * np.log(s2)


def _nll_lote(Xt: np.ndarray, y: np.ndarray, p: int, q: int, con_media: bool) -> np.ndarray:
    """Objetivo ML de R (0.5·log s2 + 0.5·mean(log F)) para B vectores de parámetros a la vez.

    Xt: matriz (B, k) con la parte AR en la escala transformada (partrans).
    El filtro de Kalman avanza en paralelo para todo el lote; la covarianza
    no depende de los datos, así que cuando todos los miembros convergen el
    resto de residuos se obtiene con la recursión ARMA (filtro lineal).
    """
    B = Xt.shape[0]
    n = len(y)
    r = max(p, q + 1)
    PHI = np.array([partrans(x[:p]) for x in Xt]) if p else np.zeros((B, 0))
    THETA = Xt[:, p:p + q]
    MU = Xt[:, p + q] if con_media else np.zeros(B)
    salida = np.full(B, 1e10)
    T = np.zeros((B, r, r))
    T[:, :p, 0] = PHI
    if r > 1:
        T[:, :-1, 1:] = np.eye(r - 1)
    Rv = np.zeros((B, r))
    Rv[:, 0] = 1.0
    Rv[:, 1:q + 1] = THETA
    RR = Rv[:, :, None] * Rv[:, None, :]
    P = np.empty((B, r, r))
    validos = np.ones(B, dtype=bool)
    for b in range(B):
        try:
            P[b] = espacio_estados(PHI[b], THETA[b])[3]
        except (np.linalg.LinAlgError, ValueError):
            validos[b] = False
            P[b] = RR[b]
    W = y[None, :] - MU[:, None]
    a = np.zeros((B, r))
    V = np.empty((B, n))
    F = np.empty((B, n))
    Tt = np.transpose(T, (0, 2, 1))
    t = 0
    with np.errstate(all="ignore"):
        while t < n:
            f = P[:, 0, 0]
            v = W[:, t] - a[:, 0]
            V[:, t] = v
            F[:, t] = f
            TP = T @ P
            K = TP[:, :, 0] / f[:, None]
            a = np.einsum("bij,bj->bi", T, a) + K * v[:, None]
            P = TP @ Tt + RR - K[:, :, None] * K[:, None, :] * f[:, None, None]
            t += 1
            if np.max(np.abs(P - RR)) < TOL_CONVERGENCIA:
                break
        if t < n:
            b_poli = np.c_[np.ones(B), -PHI]
            a_poli = np.c_[np.ones(B), THETA]
            for b in range(B):
                x_prev = W[b, :t][::-1][:p]
                y_prev = V[b, :t][::-1][:q]
                x_prev = np.r_[x_prev, np.zeros(p - len(x_prev))]
                y_prev = np.r_[y_prev, np.zeros(q - len(y_prev))]
                zi = signal.lfiltic(b_poli[b], a_poli[b], y_prev, x_prev)
                V[b, t:] = signal.lfilter(b_poli[b], a_poli[b], W[b, t:], zi=zi)[0]
            F[:, t:] = 1.0
        ok = validos & np.all(F > 0, axis=1) & np.all(np.isfinite(V), axis=1)
        s2 = np.sum(V**2 / F, axis=1) / n
        obj = 0.5 * np.log(s2) + 0.5 * np.sum(np.log(np.where(F > 0, F, 1.0)), axis=1) / n
    salida[ok & np.isfinite(obj)] = obj[ok & np.isfinite(obj)]
    return salida


@dataclass
class ModeloARMA:
    p: int
    q: int
    d: int
    con_media: bool
    phi: np.ndarray
    theta: np.ndarray
    mu: float
    sigma2: float
    loglik: float
    n: int
    aic: float
    bic: float
    aicc: float
    convergio: bool
    residuos: np.ndarray = field(repr=False)
    a_final: np.ndarray = field(repr=False)
    P_final: np.ndarray = field(repr=False)
    errores_estandar: np.ndarray | None = field(default=None, repr=False)
    par_transformado: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_par(self) -> int:
        return self.p + self.q + int(self.con_media) + 1

    @property
    def etiqueta(self) -> str:
        return etiqueta_modelo(self.p, self.d, self.q)

    def coeficientes(self) -> dict:
        c = {f"ar{i + 1}": float(v) for i, v in enumerate(self.phi)}
        c.update({f"ma{j + 1}": float(v) for j, v in enumerate(self.theta)})
        if self.con_media:
            c["media"] = float(self.mu)
        return c

    def raiz_minima(self) -> float:
        return min(raices_min_modulo(self.phi, -1.0), raices_min_modulo(self.theta, 1.0))


def etiqueta_modelo(p: int, d: int, q: int) -> str:
    """`etiqueta_modelo` del documento (sección 12.8)."""
    if d == 0:
        if p == 0 and q == 0:
            return "Ruido blanco"
        if q == 0:
            return f"AR({p})"
        if p == 0:
            return f"MA({q})"
        return f"ARMA({p},{q})"
    if p == 0 and q == 0 and d == 1:
        return "ARIMA(0,1,0) [RW]"
    return f"ARIMA({p},{d},{q})"


class ErrorModelo(Exception):
    pass


def ajustar_arma(y, p: int, q: int, con_media: bool = True, d: int = 0,
                 errores_estandar: bool = False, warm_start=None) -> ModeloARMA:
    """Ajusta ARIMA(p, d, q) por CSS-ML. d en {0, 1}: con d = 1 se modela diff(y) sin media."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if d == 1:
        y = np.diff(y)
        con_media = False
    n = len(y)
    if n < max(30, 3 * (p + q) + 5):
        raise ErrorModelo("No hay suficientes observaciones para ajustar el modelo.")
    k = p + q + int(con_media)

    mu0 = float(np.mean(y)) if con_media else 0.0
    if p == 0 and q == 0:
        mu = mu0
        w = y - mu
        s2 = float(np.mean(w**2))
        loglik = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
        sal = SalidaFiltro(v=w, F=np.ones(n), a_final=np.zeros(1), P_final=np.ones((1, 1)))
        return _construir(p, q, d, con_media, np.array([]), np.array([]), mu, s2, loglik, n, sal, True,
                          np.array([np.sqrt(s2 / n)]) if con_media and errores_estandar else None)

    # Escala de parámetros como `parscale` de R: 1 para AR/MA y 10·ee para la media.
    escala = np.ones(k)
    if con_media:
        escala[-1] = 10 * np.std(y, ddof=1) / np.sqrt(n)

    # 1) CSS
    x0 = np.zeros(k)
    if con_media:
        x0[-1] = mu0
    res_css = optimize.minimize(lambda xs: _objetivo_css(xs * escala, y, p, q, con_media), x0 / escala,
                                method="BFGS", options={"maxiter": 100, "gtol": 1e-7})
    init = res_css.x * escala
    if p and not ar_estacionario(init[:p]):
        raise ErrorModelo("Parte AR no estacionaria tras CSS.")
    if q:
        init[p:p + q] = ma_invertir(init[p:p + q])

    # 2) ML exacta, gradiente por diferencias centrales con ndeps = 1e-3 (como optim)
    init_t = init.copy()
    if p:
        init_t[:p] = invpartrans(init[:p])
    if warm_start is not None and len(warm_start) == k:
        init_t = np.asarray(warm_start, dtype=float).copy()
    paso = 1e-3
    eye = np.eye(k) * paso

    def valor_y_gradiente(xs):
        lote = np.vstack([xs, xs + eye, xs - eye]) * escala
        vals = _nll_lote(lote, y, p, q, con_media)
        g = (vals[1:k + 1] - vals[k + 1:]) / (2 * paso)
        if not np.all(np.isfinite(g)) or vals[0] >= 1e10:
            g = np.zeros(k)
        return vals[0], g

    vigilante = _Vigilante(valor_y_gradiente, MAX_EVALUACIONES_ML)
    try:
        res = optimize.minimize(vigilante, init_t / escala, jac=True, method="BFGS",
                                options={"maxiter": 100, "gtol": 1e-5})
        x_final, ok = res.x * escala, bool(res.success)
    except _LimiteEvaluaciones:
        x_final, ok = vigilante.mejor_x * escala, False
    par = x_final.copy()
    if p:
        par[:p] = partrans(x_final[:p])
    par_transformado = x_final.copy()
    phi, theta, mu = _desempacar(par, p, q, con_media)
    sal = filtro_kalman(y - mu, phi, theta)
    s2 = float(np.sum(sal.v**2 / sal.F) / n)
    loglik = -0.5 * (n * np.log(2 * np.pi * s2) + np.sum(np.log(sal.F)) + n)
    ee = _errores_estandar(par, y, p, q, con_media) if errores_estandar else None
    modelo = _construir(p, q, d, con_media, phi, theta, mu, s2, loglik, n, sal, ok, ee)
    modelo.par_transformado = par_transformado
    return modelo


MAX_EVALUACIONES_ML = 250


class _LimiteEvaluaciones(Exception):
    pass


class _Vigilante:
    """Envuelve la función objetivo, guarda el mejor punto y corta tras N evaluaciones."""

    def __init__(self, f, maximo):
        self.f, self.maximo, self.n = f, maximo, 0
        self.mejor_x, self.mejor_f = None, np.inf

    def __call__(self, x, *args):
        self.n += 1
        if self.n > self.maximo:
            raise _LimiteEvaluaciones
        salida = self.f(x, *args)
        v = salida[0] if isinstance(salida, tuple) else salida
        if v < self.mejor_f:
            self.mejor_f, self.mejor_x = v, np.array(x, dtype=float).copy()
        return salida


def _construir(p, q, d, con_media, phi, theta, mu, s2, loglik, n, sal, ok, ee):
    npar = p + q + int(con_media) + 1
    aic = -2 * loglik + 2 * npar
    bic = aic + npar * (np.log(n) - 2)
    aicc = aic + 2 * npar * (npar + 1) / (n - npar - 1) if n - npar - 1 > 0 else np.inf
    return ModeloARMA(p=p, q=q, d=d, con_media=con_media, phi=np.asarray(phi, float), theta=np.asarray(theta, float),
                      mu=float(mu), sigma2=s2, loglik=float(loglik), n=n, aic=float(aic), bic=float(bic),
                      aicc=float(aicc), convergio=ok, residuos=sal.v / np.sqrt(sal.F), a_final=sal.a_final,
                      P_final=sal.P_final, errores_estandar=ee)


def _errores_estandar(par, y, p, q, con_media):
    """Errores estándar a partir del hessiano numérico de -loglik (escala natural)."""
    n = len(y)

    def nll(theta_vec):
        ph, th, mu = _desempacar(theta_vec, p, q, con_media)
        sal = filtro_kalman(y - mu, ph, th)
        s2 = np.sum(sal.v**2 / sal.F) / n
        return 0.5 * (n * np.log(2 * np.pi * s2) + np.sum(np.log(sal.F)) + n)

    k = len(par)
    h = 1e-4 * np.maximum(np.abs(par), 1e-2)
    H = np.zeros((k, k))
    f0 = nll(par)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (nll(par + ei + ej) - nll(par + ei) - nll(par + ej) + f0) / (h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Pronóstico
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Pronostico:
    media: np.ndarray
    error_estandar: np.ndarray
    inferior: np.ndarray
    superior: np.ndarray
    nivel: float


def pronosticar(modelo: ModeloARMA, h: int, nivel: float = 0.95) -> Pronostico:
    """Pronóstico de 1..h pasos como `forecast(modelo, h, level)` (Kalman)."""
    from scipy import stats

    T, Rv, RR, _ = espacio_estados(modelo.phi, modelo.theta)
    a = modelo.a_final.copy()
    P = modelo.P_final.copy()
    if a.shape[0] != T.shape[0]:
        a = np.zeros(T.shape[0])
        P = RR.copy()
    medias = np.empty(h)
    var = np.empty(h)
    for j in range(h):
        medias[j] = a[0]
        var[j] = P[0, 0]
        a = T @ a
        P = T @ P @ T.T + RR
    medias = medias + modelo.mu
    ee = np.sqrt(var * modelo.sigma2)
    z = stats.norm.ppf(0.5 + nivel / 2)
    return Pronostico(medias, ee, medias - z * ee, medias + z * ee, nivel)


def pronostico_acumulado(modelo: ModeloARMA, h: int, nivel: float = 0.95) -> tuple[float, float, float, float]:
    """Pronóstico del rendimiento acumulado de h días (suma de rendimientos log diarios).

    Devuelve (media, error estándar, inferior, superior). La varianza usa los
    pesos psi: Var = sigma^2 * sum_m (sum_{i<=h-m} psi_i)^2.
    """
    from scipy import stats

    pr = pronosticar(modelo, h, nivel)
    psi = psi_pesos(modelo.phi, modelo.theta, h)
    acum = np.cumsum(psi)
    var = modelo.sigma2 * np.sum(acum[::-1] ** 2)
    media = float(np.sum(pr.media))
    ee = float(np.sqrt(var))
    z = stats.norm.ppf(0.5 + nivel / 2)
    return media, ee, media - z * ee, media + z * ee


def trayectoria_acumulada(modelo: ModeloARMA, h: int, nivel: float = 0.95) -> dict:
    """Rendimiento acumulado pronosticado para k = 1..h con su banda de predicción.

    S_k = sum_{j=1}^{k} r_{T+j}. Media: suma de los pronósticos diarios.
    Varianza: sigma^2 * sum_{m=1}^{k} (sum_{i=0}^{k-m} psi_i)^2, que supone
    errores normales homocedásticos y parámetros conocidos.
    """
    from scipy import stats

    pr = pronosticar(modelo, h, nivel)
    psi = psi_pesos(modelo.phi, modelo.theta, h)
    acum_psi = np.cumsum(psi)
    var = np.array([modelo.sigma2 * np.sum(acum_psi[:k][::-1] ** 2) for k in range(1, h + 1)])
    media = np.cumsum(pr.media)
    ee = np.sqrt(var)
    z = stats.norm.ppf(0.5 + nivel / 2)
    return {"media": media, "ee": ee, "inferior": media - z * ee, "superior": media + z * ee}


def reconstruir_precio(precio_actual: float, rend_acumulado):
    """P(T+h) = exp(ln P_T + sum r) = P_T · exp(sum r) (sección 10.2)."""
    return np.exp(np.log(precio_actual) + np.asarray(rend_acumulado, dtype=float))


def estados_filtrados(w: np.ndarray, modelo: ModeloARMA) -> np.ndarray:
    """Estados a_{t+1|t} para cada t de la serie w = y - mu, con parámetros fijos."""
    return filtro_kalman(w, modelo.phi, modelo.theta, guardar_estados=True).estados


def matriz_suma_pasos(modelo: ModeloARMA, h: int) -> np.ndarray:
    """Vector g tal que sum_{j=1}^{h} (y_{t+j|t} - mu) = g · a_{t+1|t}."""
    T, *_ = espacio_estados(modelo.phi, modelo.theta)
    r = T.shape[0]
    acumulada = np.zeros((r, r))
    Tk = np.eye(r)
    for _ in range(h):
        acumulada += Tk
        Tk = T @ Tk
    return acumulada[0]
