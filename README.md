# VESTA | Quantitative Intelligence

**Volatilidad, estimación de series de tiempo y análisis.** Laboratorio cuantitativo para analizar acciones del S&P 500 y evaluar modelos predictivos en inversiones de corto plazo: **semanal (5 días bursátiles), mensual (21) y trimestral (63)**.

Implementa la metodología del proyecto académico *Modelación estadística para inversiones de corto plazo* (Beltrán y Castro, 2026): rendimientos logarítmicos, clasificación por beta, pruebas de rachas y Lilliefors, ADF, KPSS, Ljung-Box, ARCH-LM, Kruskal-Wallis y Q de Cochran, modelos ARIMA sobre el log-precio y comparación fuera de muestra con la caminata aleatoria.

> **Advertencia.** Trabajo académico. No es una recomendación de inversión. Los pronósticos de rendimientos a corto plazo tienen gran incertidumbre: en la mayoría de las acciones grandes, ningún modelo supera a la caminata aleatoria de forma significativa. VESTA lo muestra explícitamente en lugar de ocultarlo.

## Funcionalidades

| Sección | Usuario general | Usuario técnico |
|---|---|---|
| **Inicio** | Qué es VESTA, horizontes, estado del universo y de los datos | — |
| **Explorador de activos** | Tabla tipo terminal con ticker, precio, variación, beta, perfil y sector; búsqueda y filtros | Distribución de betas; universo exploratorio separado de la muestra académica |
| **Análisis del activo** | Ficha, precio o rentabilidad acumulada en ventanas de 1, 3 y 5 años, volatilidad e indicadores principales | Panel *Más información estadística*: descriptivos completos, pruebas con H0/H1/decisión, ACF/PACF |
| **Comparación de activos** | Cajas de rendimientos por horizonte | Kruskal-Wallis y Q de Cochran dentro de la muestra académica |
| **Predicciones** | Botón *Generar predicción*: precio proyectado, intervalo y trayectoria | *Diagnóstico del modelo*: información técnica, residuos, desempeño frente a la caminata aleatoria e incertidumbre |
| **Historial** | Predicciones pasadas frente a lo ocurrido | Registro inmutable con métricas, ajustes y datos para reproducir cada ejecución |
| **Configuración** | — | α, rejilla ARIMA, criterio, entrenamiento, reajuste, intervalo; base local; metodología |

## Instalación y ejecución en Windows

Requisito: **Python 3.11 o superior** ([python.org](https://www.python.org/downloads/); marque *Add Python to PATH*).

**Forma rápida:** doble clic en `iniciar_vesta.bat`. La primera vez crea el entorno virtual e instala las dependencias; después solo abre la aplicación.

**Paso a paso** (PowerShell o CMD, en la carpeta del proyecto):

```bat
:: 1. Crear el entorno virtual
py -3 -m venv .venv
:: 2. Activarlo
.venv\Scripts\activate
:: 3. Instalar dependencias
pip install -r requirements.txt
:: 4. Inicializar la base de datos (opcional: la app lo hace al iniciar)
python scripts\inicializar_db.py
:: 5. Ejecutar
streamlit run app.py
```

6. **Acceso:** el navegador se abre en `http://localhost:8501`. Si no se abre solo, escriba esa dirección.
7. **Detener:** en la terminal, `Ctrl + C`. Para salir del entorno virtual, `deactivate`.

**Configuración opcional:** copie `config/ajustes.example.toml` como `config/ajustes.toml` para cambiar los parámetros por defecto. También pueden cambiarse desde la página Configuración. VESTA no necesita credenciales ni variables de entorno.

## Primer uso

1. **Explorador → Construir universo.** Descarga cinco años de precios de unas 500 empresas, calcula la beta de cada una y la guarda en la base local con su fecha. Tarda varios minutos y se reutiliza durante 30 días.
2. **Análisis del activo.** Ficha, comportamiento histórico y estadística descriptiva. Se calcula al elegir el activo.
3. **Predicciones → Generar predicción.** La primera vez por activo tarda alrededor de un minuto, porque los modelos se reestiman en cada origen de la prueba. Después el resultado queda en caché y cambiar de horizonte es instantáneo, aunque nunca se ejecuta sin pulsar el botón.

Tareas sin abrir la aplicación:

```bat
python scripts\actualizar_universo.py            :: betas y perfiles (si tienen más de 30 días)
python scripts\precalcular.py --muestra          :: modela los 21 activos de la muestra, en paralelo
python scripts\precalcular.py MSFT KO --replica  :: incluye la réplica literal de la sección 12.10
```

## Arquitectura

```
app.py                      Entrada, navegación lateral y tema
paginas/                    Interfaz: una página por sección
vesta/
  config.py                 Rutas, valores por defecto e identidad visual
  ajustes.py                Parámetros de análisis configurables y su firma
  servicios.py              Orquestación entre interfaz y cálculo
  datos/                    Yahoo Finance y Wikipedia, caché de precios, calidad, universo, beta y muestreo
  procesamiento/            Limpieza y rendimientos (diarios y acumulados)
  estadistica/              Descriptivos y pruebas replicadas de R
  modelos/                  ARMA con verosimilitud exacta, selección AIC/BIC, caminata aleatoria, predicción
  evaluacion/               Métricas, evaluación rodante y réplica literal del documento
  persistencia/             SQLite (universo, predicciones, evaluaciones, configuración) y caché de cálculo
  ui/                       Sistema de diseño: tema, componentes y gráficos
scripts/                    Inicializar la base, actualizar el universo, precalcular
config/                     ajustes.example.toml
validacion/                 Comparación con R
tests/                      Pruebas unitarias y de integración
docs/                       METODOLOGIA.md y capturas
```

**Datos y resultados separados.** Los precios descargados viven en `data/precios`. Lo que se conserva (universo, betas con fecha, actualizaciones, predicciones, realizaciones, evaluaciones y configuración) va a `data/vesta.sqlite`. Los resultados intermedios reutilizables van a `data/cache`. Para migrar a otro motor basta con reimplementar `vesta/persistencia/db.py`.

**Por qué Python y Streamlit.** Un solo lenguaje para datos, cálculo e interfaz, y ejecución en Windows con un comando. Como la metodología se desarrolló en R, cada prueba reproduce el algoritmo exacto de su función de R (tseries, nortest, FinTS, forecast) en lugar de usar equivalentes genéricos. Ver `docs/METODOLOGIA.md`.

## Metodología estadística (resumen)

- Rendimiento logarítmico con precio ajustado: r_t = ln(P_t / P_{t−1}); acumulado del horizonte: ln(P_{t+h} / P_t).
- Beta: pendiente de la regresión de rendimientos log mensuales del activo sobre ^GSPC, 5 años, mínimo 48 meses. Perfiles: β < 0.8 defensivo, 0.8 ≤ β < 1.2 neutral, β ≥ 1.2 agresivo.
- Modelos: ARIMA(p,1,q) sobre el log-precio con p, q ∈ {0,…,3}, sin deriva, estimados por CSS-ML. Se informan AIC y BIC; el criterio configurado define la especificación principal.
- Evaluación: orígenes rodantes sobre el 20 % final, reestimación en cada origen, misma variable y mismas fechas para el ARIMA y la caminata aleatoria. Métricas: MAE, RMSE, MAPE sobre precio, acierto direccional con prueba binomial, U de Theil, Diebold-Mariano y cobertura del intervalo.
- Predicción: P̂(T+h) = P_T · exp(Σ r̂), con intervalo de predicción normal basado en los pesos ψ.

Decisiones, inconsistencias encontradas en el documento y su tratamiento: `docs/METODOLOGIA.md` (también visible en Configuración → Metodología).

## Calidad y reproducibilidad

```bat
pytest -q
```

Las pruebas cubren las funciones estadísticas y el flujo completo de predicción con series simuladas. También verifican la reconstrucción del precio, los horizontes de 5, 21 y 63 días, la ausencia de fuga de información y la inmutabilidad del historial. Cada predicción guarda los ajustes, el periodo de datos, los coeficientes y la versión.

Verificación frente a R (requiere R con `tseries`, `nortest`, `FinTS` y `forecast`):

```bat
Rscript validacion\exportar_referencia.R MSFT KO TSLA
python validacion\comparar_con_r.py MSFT KO TSLA
```

## Capturas

Guárdelas en `docs/capturas/` tras ejecutar la aplicación con datos reales.

## Publicar en GitHub

El repositorio no se publica automáticamente. Para el primer commit:

```bat
git init
git add .
git status
git commit -m "VESTA | Quantitative Intelligence: primera versión"
git branch -M main
git remote add origin https://github.com/<usuario>/vesta.git
git push -u origin main
```

`.gitignore` excluye el entorno virtual, los datos descargados, la caché, la base local (contiene el historial personal) y la configuración local. Antes de publicar, elija una licencia (ver `THIRD_PARTY_NOTICES.md`).

## Licencias y fuentes

Dependencias, tipografías y condiciones de uso de Yahoo Finance y Wikipedia: `THIRD_PARTY_NOTICES.md`.

## Autores

Alejandra Beltrán y Johan Castro.
#   V E S T A  
 