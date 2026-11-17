"""Parámetros de análisis configurables.

Orden de prioridad: valores por defecto (documento metodológico) <
config/ajustes.toml < lo guardado desde la página Configuración (SQLite).
Cada ejecución guarda los ajustes usados, para poder reproducirla.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields

from vesta.config import ARCHIVO_AJUSTES


@dataclass(frozen=True)
class Ajustes:
    # Significancia de las pruebas (sección 9)
    alfa: float = 0.05
    # Datos
    anios_historia: int = 5
    # Modelación (sección 12.8): rejilla p, q en 0..3, d = 1, 80 % entrenamiento
    max_p: int = 3
    max_q: int = 3
    d: int = 1
    incluir_deriva: bool = False          # allowdrift = FALSE en la sección 12.9
    prop_train: float = 0.80
    criterio: str = "bic"                 # "aic" o "bic": criterio de especificación
    reestimar_cada: int = 1               # 1 = reajuste en cada origen (REEST_CADA = 1)
    nivel_intervalo: float = 0.95
    lags_ljung_box: int = 10
    lags_arch: int = 10
    lag_max_acf: int = 30

    def validar(self) -> list[str]:
        errores = []
        if not 0 < self.alfa < 0.5:
            errores.append("α debe estar entre 0 y 0.5.")
        if not 0.5 <= self.prop_train <= 0.95:
            errores.append("La proporción de entrenamiento debe estar entre 0.5 y 0.95.")
        if not (0 <= self.max_p <= 5 and 0 <= self.max_q <= 5):
            errores.append("p y q máximos deben estar entre 0 y 5.")
        if self.d not in (0, 1):
            errores.append("d debe ser 0 o 1.")
        if self.criterio not in ("aic", "bic"):
            errores.append("El criterio debe ser AIC o BIC.")
        if self.reestimar_cada < 1:
            errores.append("La reestimación debe hacerse cada 1 o más orígenes.")
        if not 0.5 < self.nivel_intervalo < 1:
            errores.append("El nivel del intervalo debe estar entre 0.5 y 1.")
        if self.anios_historia not in (3, 4, 5):
            errores.append("La historia para modelar debe ser de 3, 4 o 5 años.")
        return errores

    def firma(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def como_dict(self) -> dict:
        return asdict(self)


def _filtrar(d: dict) -> dict:
    validos = {f.name: f.type for f in fields(Ajustes)}
    return {k: v for k, v in d.items() if k in validos}


def cargar() -> Ajustes:
    datos = {}
    if ARCHIVO_AJUSTES.exists():
        with ARCHIVO_AJUSTES.open("rb") as f:
            datos.update(_filtrar(tomllib.load(f).get("analisis", {})))
    try:
        from vesta.persistencia import db
        datos.update(_filtrar(db.leer_configuracion()))
    except Exception:
        pass
    try:
        a = Ajustes(**datos)
        return a if not a.validar() else Ajustes()
    except TypeError:
        return Ajustes()


def guardar(a: Ajustes) -> None:
    from vesta.persistencia import db
    db.guardar_configuracion(a.como_dict())
