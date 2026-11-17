# Dependencias, recursos y fuentes de datos

## Paquetes de Python

| Paquete | Uso en VESTA | Licencia |
|---|---|---|
| streamlit | Interfaz web | Apache 2.0 |
| plotly | Gráficos interactivos | MIT |
| pandas | Tablas y series de tiempo | BSD 3-Clause |
| numpy | Cálculo numérico | BSD 3-Clause |
| scipy | Distribuciones, optimización, filtros lineales | BSD 3-Clause |
| yfinance | Descarga de precios de Yahoo Finance | Apache 2.0 |
| requests | Descarga de la lista del S&P 500 | Apache 2.0 |
| lxml | Lectura de tablas HTML | BSD 3-Clause |
| pytest | Pruebas automáticas | MIT |

Las versiones mínimas están en `requirements.txt`. Verifique las licencias vigentes de cada paquete antes de redistribuir.

## Tipografías

- **Inter** (Rasmus Andersson): SIL Open Font License 1.1.
- **JetBrains Mono** (JetBrains): SIL Open Font License 1.1.

Se cargan desde Google Fonts. Sin conexión, la interfaz usa las fuentes del sistema.

## Fuentes de datos

- **Yahoo Finance**, a través de `yfinance`, que es un proyecto independiente no afiliado a Yahoo. Los datos están sujetos a los términos de servicio de Yahoo, orientados al uso personal y no comercial. VESTA no redistribuye datos: cada usuario los descarga localmente, y `.gitignore` los excluye del repositorio.
- **Wikipedia**, *List of S&P 500 companies*: contenido bajo licencia CC BY-SA 4.0. Se usa solo para obtener la lista de componentes, tickers y sectores GICS.

## Licencia del proyecto

El repositorio todavía no declara una licencia propia. Los autores deben elegirla antes de publicar (por ejemplo MIT o Apache 2.0) y agregar el archivo `LICENSE`.
