"""Valores por defecto, rutas e identidad visual.

Los parámetros metodológicos editables por el usuario viven en `vesta/ajustes.py`
(se cargan de config/ajustes.toml y de la página Configuración). Aquí quedan los
valores por defecto que replican el documento metodológico VESTA.
"""

from pathlib import Path

VERSION = "2.0.0"
NOMBRE = "VESTA | Quantitative Intelligence"

# ── Rutas ────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_PRECIOS = DIR_DATOS / "precios"          # datos descargados (separados de los resultados)
DIR_CACHE = DIR_DATOS / "cache"              # resultados de cálculo reutilizables
ARCHIVO_DB = DIR_DATOS / "vesta.sqlite"      # universo, predicciones, evaluaciones, configuración
ARCHIVO_AJUSTES = RAIZ / "config" / "ajustes.toml"
ARCHIVO_MUESTRA = DIR_DATOS / "muestra_fase2.csv"
ARCHIVO_TICKERS_RESPALDO = DIR_DATOS / "tickers_sp500.csv"

# ── Datos (sección 6.1) ──────────────────────────────────────────────────
INDICE_REFERENCIA = "^GSPC"
HORAS_VALIDEZ_PRECIOS = 12
DIAS_HABILES_DESACTUALIZADO = 5     # último dato más viejo que esto => aviso
MIN_OBSERVACIONES = 504             # dos años bursátiles para permitir una predicción

# ── Horizontes de inversión (sección 6.4) ────────────────────────────────
HORIZONTES = {"Semanal": 5, "Mensual": 21, "Trimestral": 63}
VENTANAS_HISTORIA = {"1 año": 1, "3 años": 3, "5 años": 5}

# ── Beta y perfiles (sección 13) ─────────────────────────────────────────
ANIOS_BETA = 5
MIN_MESES_BETA = 48
UMBRAL_DEFENSIVO = 0.8
UMBRAL_AGRESIVO = 1.2
PERFILES = ["Defensivo", "Neutral", "Agresivo"]
DIAS_VALIDEZ_UNIVERSO = 30
N_POR_PERFIL = 7
SEMILLA_MUESTREO = 2026
# Piloto de la fase 1 (sección 6.2)
PILOTO = {"DIO.DE": "Christian Dior", "GRBMF": "Grupo Bimbo", "DIS": "Disney"}

AVISO = "Trabajo académico. No es una recomendación de inversión."

# ── Identidad visual: VESTA | Quantitative Intelligence ──────────────────
COLORES = {
    "fondo": "#101114",
    "superficie": "#181A20",
    "panel": "#20232B",
    "borde": "#30343D",
    "texto": "#F1F1F3",
    "tenue": "#A0A4AE",
    "acento": "#8F263D",          # rojo vino: botones principales, selección, énfasis
    "acento_2": "#C34B62",
    "positivo": "#43A982",        # siempre con signo y flecha (no distinguible en deuteranopía)
    "negativo": "#D45B65",
    "neutral": "#8498B0",         # información neutral de interfaz
    # Series de datos: trío validado (CVD y visión normal, todos los pares) sobre #181A20
    "serie_obs": "#4F86C6",       # histórico / observado
    "serie_mod": "#C34B62",       # pronóstico del modelo
    "serie_ref": "#B9862C",       # caminata aleatoria / referencia
    "banda": "rgba(195, 75, 98, 0.16)",
    "grid": "#262932",
    "eje": "#3A3E48",
}
COLOR_PERFIL = {"Defensivo": "#4F86C6", "Neutral": "#B9862C", "Agresivo": "#C34B62"}
