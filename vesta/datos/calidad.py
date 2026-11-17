"""Control de calidad de la serie de precios (sección 13 del prompt).

No se imputan valores: las filas incompletas se eliminan (como en el documento)
y se informa cuántas. Si la serie no cumple los mínimos, no se predice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vesta.config import DIAS_HABILES_DESACTUALIZADO, MIN_OBSERVACIONES


@dataclass
class InformeCalidad:
    observaciones: int
    inicio: pd.Timestamp | None
    fin: pd.Timestamp | None
    duplicados: int = 0
    filas_incompletas: int = 0
    precios_no_positivos: int = 0
    huecos: list = field(default_factory=list)      # (desde, hasta, días hábiles)
    dias_habiles_desde_ultimo: int = 0
    tiene_volumen: bool = True
    errores: list = field(default_factory=list)      # impiden predecir
    avisos: list = field(default_factory=list)

    @property
    def apta(self) -> bool:
        return not self.errores


def revisar(crudo: pd.DataFrame, limpio: pd.DataFrame, minimo: int = MIN_OBSERVACIONES,
            hoy: pd.Timestamp | None = None) -> InformeCalidad:
    hoy = pd.Timestamp(hoy or pd.Timestamp.now().normalize())
    inf = InformeCalidad(observaciones=len(limpio),
                         inicio=limpio["Date"].min() if len(limpio) else None,
                         fin=limpio["Date"].max() if len(limpio) else None)
    if crudo is not None and len(crudo):
        fechas = pd.to_datetime(crudo["Date"] if "Date" in crudo else crudo.index)
        inf.duplicados = int(fechas.duplicated().sum())
        cols = [c for c in ("Close", "Adj Close", "Adj.Close") if c in crudo]
        inf.filas_incompletas = int(crudo[cols].isna().any(axis=1).sum()) if cols else 0
        col = "Adj Close" if "Adj Close" in crudo else ("Adj.Close" if "Adj.Close" in crudo else None)
        if col:
            inf.precios_no_positivos = int((pd.to_numeric(crudo[col], errors="coerce") <= 0).sum())
    if len(limpio) < minimo:
        inf.errores.append(f"Hay {len(limpio)} sesiones válidas; se necesitan al menos {minimo} para modelar.")
    if len(limpio):
        f = pd.to_datetime(limpio["Date"]).sort_values().to_numpy()
        habiles = np.busday_count(f[:-1].astype("datetime64[D]"), f[1:].astype("datetime64[D]"))
        for i in np.where(habiles > 5)[0]:
            inf.huecos.append((pd.Timestamp(f[i]).date(), pd.Timestamp(f[i + 1]).date(), int(habiles[i])))
        if len(inf.huecos) > 0:
            inf.avisos.append(f"{len(inf.huecos)} interrupciones de más de 5 días hábiles sin cotización.")
        inf.dias_habiles_desde_ultimo = int(np.busday_count(pd.Timestamp(inf.fin).date(), hoy.date()))
        if inf.dias_habiles_desde_ultimo > DIAS_HABILES_DESACTUALIZADO:
            inf.avisos.append(f"El último dato tiene {inf.dias_habiles_desde_ultimo} días hábiles de antigüedad.")
        inf.tiene_volumen = "Volume" in limpio and bool((limpio["Volume"] > 0).any())
    if inf.duplicados:
        inf.avisos.append(f"Se eliminaron {inf.duplicados} fechas duplicadas.")
    if inf.filas_incompletas:
        inf.avisos.append(f"Se eliminaron {inf.filas_incompletas} filas con precios faltantes (sin imputar).")
    if inf.precios_no_positivos:
        inf.avisos.append(f"Se eliminaron {inf.precios_no_positivos} filas con precio no positivo (logaritmo no definido).")
    return inf
