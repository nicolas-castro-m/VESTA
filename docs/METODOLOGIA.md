## Fuente metodológica

VESTA implementa el documento *VESTA: Volatilidad, estimación de series de tiempo* (Beltrán y Castro, 14 de noviembre de 2026), proyecto *Modelación estadística para inversiones de corto plazo (semanal, mensual y trimestral)*. Cada cálculo de la aplicación remite a una sección de ese documento. Lo que se agregó o se corrigió está marcado aquí, con su motivo y su efecto.

### Correspondencia entre el documento y el código

| Sección del documento | Qué se hace | Módulo |
|---|---|---|
| 6.1 Descarga | Precios diarios de Yahoo Finance, filas con NA eliminadas | `vesta/datos/fuentes.py`, `almacen.py` |
| 6.5 Rendimientos | r_t = ln(P_t / P_{t−1}) con precio ajustado; acumulados ln(P_t / P_{t−h}) con h = 5, 21, 63 | `vesta/procesamiento/rendimientos.py` |
| 7 y 8 Exploratorio | Media, mediana, desviación, mínimo, máximo; histogramas del 98 % central; Q-Q | `vesta/estadistica/descriptiva.py`, `vesta/ui/graficos.py` |
| 9.1 Rachas | `tseries::runs.test` sobre la serie dicotomizada en la mediana | `pruebas.rachas` |
| 9.2 Normalidad | `nortest::lillie.test` con la aproximación de Dallal-Wilkinson y Stephens | `pruebas.lilliefors` |
| 9.3 Por horizonte | `diagnostico_horizonte` | `pruebas.diagnostico_horizonte` |
| 10 Kruskal-Wallis | `kruskal.test` con corrección por empates | `pruebas.kruskal_wallis` |
| 11 Q de Cochran | Función `cochran_q` del documento, periodos no solapados | `pruebas.cochran_q`, `servicios.comparar_activos` |
| 12.5 Caminata aleatoria | ADF y KPSS (tseries), Ljung-Box 10 y 20, t de media cero, ARCH-LM (FinTS) | `pruebas.evaluar_caminata` |
| 12.6 Pronóstico RW | `evaluar_rw` (réplica literal) | `vesta/evaluacion/original.py` |
| 12.8 Identificación | División 80/20, ACF, PACF, ACF de cuadrados, `lags_signif` | `pruebas.acf/pacf/rezagos_significativos` |
| 12.8 a 12.10 ARIMA | ARIMA(p,1,q) sobre el log-precio, p, q ∈ {0,…,3}, sin deriva, CSS-ML; AIC y BIC | `vesta/modelos/arma.py`, `seleccion.py` |
| 12.1 Caminata aleatoria | ARIMA(0,1,0) sin deriva como referencia | `vesta/modelos/caminata.py` |
| 13 Beta y muestreo | Beta mensual a 5 años frente a ^GSPC, perfiles 0.8 / 1.2, muestreo sistemático estratificado | `vesta/datos/universo.py` |
| 14 Ajustes pendientes | Evaluación rodante con reajuste en cada origen sobre la misma variable; U de Theil, Diebold-Mariano, dirección frente al 50 % | `vesta/evaluacion/rodante.py` |
| — Predicción vigente | Reconstrucción P̂ = P_T·exp(Σ r̂) e intervalo de predicción | `vesta/modelos/prediccion.py` |

### Equivalencia con R

La metodología se desarrolló en R; VESTA está en Python. No se usaron equivalentes genéricos: cada prueba reproduce el algoritmo de la función de R correspondiente.

- **`tseries::adf.test`**: regresión con constante y tendencia, k = trunc((n−1)^(1/3)) rezagos y el p-valor interpolado en la misma tabla de Banerjee et al. (por eso queda entre 0.01 y 0.99). `statsmodels.adfuller` usaría otros rezagos y los p-valores de MacKinnon, y daría números distintos.
- **`tseries::kpss.test`**: nivel, truncamiento trunc(4·(n/100)^0.25) y tabla interpolada (entre 0.01 y 0.10).
- **`nortest::lillie.test`**: mismas constantes de Dallal-Wilkinson y de Stephens.
- **`FinTS::ArchTest`**: n·R² de la regresión de x² sobre 10 rezagos, sin centrar.
- **`forecast::auto.arima`** (stepwise = FALSE, approximation = FALSE): CSS para los valores iniciales; descarte del modelo si la parte AR sale no estacionaria; máxima verosimilitud exacta con filtro de Kalman y reparametrización de la parte AR (`partrans`); gradiente por diferencias centrales con ndeps = 1e-3 y `parscale` = 10·ee para la media; BIC = AIC + k·(ln n − 2); descarte de modelos con raíces de módulo menor que 1.01. ARIMA(p,1,q) sobre Y_t = ln P_t sin deriva se estima como ARMA(p,q) sobre r_t sin media: es el mismo modelo con la misma verosimilitud.
- **`forecast::dm.test`**: varianza de largo plazo con h − 1 autocovarianzas, corrección de Harvey-Leybourne-Newbold y distribución t con n − 1 grados de libertad.

**Lo que no es idéntico y por qué.** El optimizador BFGS de scipy no es el `optim` de R. Ambos llegan al mismo óptimo, pero en modelos con superficies de verosimilitud muy planas (sobreparametrizados, con raíces que se cancelan) los coeficientes pueden diferir en el cuarto decimal. Esos modelos casi nunca ganan por BIC. Para comprobarlo con sus propios datos: `validacion/exportar_referencia.R` genera los resultados de R y `validacion/comparar_con_r.py` los compara con VESTA.

## Parámetros de modelación y su origen

| Parámetro | Valor por defecto | Origen | Dónde se cambia |
|---|---|---|---|
| Rejilla p, q | 0 a 3 | `P_MAX = 3`, `Q_MAX = 3` (sección 12.8) y prompt §10.1 | Configuración / `config/ajustes.toml` |
| Diferenciación d | 1 sobre el log-precio | Sección 12.9 (`d = 1`) | Fijo (equivale a ARMA sobre r_t) |
| Deriva | No | `allowdrift = FALSE` (sección 12.9) | Configuración |
| Entrenamiento | 80 % inicial | `PROP_TRAIN = 0.80` | Configuración |
| Reajuste | En cada origen | `REEST_CADA = 1` | Configuración |
| Criterio de especificación | BIC (se informa también AIC) | `ic = "bic"` | Configuración |
| Nivel de significancia | 0.05 | `ALFA = 0.05` | Configuración |
| Intervalo de predicción | 95 % | `level = 95` (sección 12.10) | Configuración |

**Especificación frente a desempeño.** AIC y BIC miden el ajuste penalizado dentro de la muestra de entrenamiento. No dicen si un modelo pronostica mejor. Por eso:

1. La especificación principal se elige con el criterio configurado, usando solo el entrenamiento.
2. Si AIC y BIC eligen órdenes distintos, **ambos** se evalúan fuera de muestra.
3. La conclusión sobre su utilidad sale de la comparación con la caminata aleatoria, no de AIC o BIC.

El modelo de la predicción vigente no se elige mirando el periodo de prueba, para no sobreajustar a él.

**Reajuste en cada origen.** En cada fecha de la prueba se reestiman los **parámetros** del orden elegido, solo con los datos disponibles hasta esa fecha (con arranque en los parámetros anteriores para acelerar). El **orden** se fija con el entrenamiento: volver a buscar en la rejilla en cada origen multiplicaría el cálculo por 16 sin cambiar la conclusión, y mantenerlo fijo evita usar información de la prueba para especificar el modelo. `reestimar_cada` permite espaciar los reajustes.

## Decisiones tomadas donde el documento no era explícito

1. **Función de rachas oculta.** `prueba_aleatoriedad` no aparece en el HTML (fragmento con `echo = FALSE`). Su salida dice `data: serie_binaria`. Se asumió la misma dicotomización que `diagnostico_horizonte`: `serie > median(serie)`. Si la función original usaba otro criterio (por ejemplo el signo), cambiarlo en `pruebas.rachas`.
2. **`resumen_horizontes` oculta.** Se reconstruyó a partir de las columnas de la tabla publicada (Media, Mediana, Desv_Estandar, Mínimo, Máximo).
3. **Ventana de datos.** La descarga del piloto usa 4.5 años, pero el título de la sección 6.1 y la beta usan 5. VESTA usa 5 años en todo (`ANIOS_HISTORIA` en `config.py`).
4. **Primer mes de la beta.** `monthlyReturn` de quantmod calcula el primer mes desde el primer cierre disponible (mes parcial). Se replicó igual.
5. **Muestra de la fase 2.** El generador aleatorio de numpy no es el de R: con la semilla 2026 la muestra no coincide con la de `set.seed(2026)`. Si existe `data/muestra_fase2.csv` exportado desde R, la aplicación usa ese archivo.
6. **Fecha objetivo de las predicciones.** Se muestra como h días hábiles después del último cierre, sin descontar festivos (es aproximada). El historial, en cambio, compara siempre con el precio h sesiones después del origen.
7. **Volatilidad del horizonte.** Es la desviación estándar de los rendimientos acumulados de h días tomados sin solapamiento. También se muestra σ_diaria·√h, que supone independencia. Las etiquetas "baja, media, alta" no usan umbrales universales: indican en qué tercio del universo S&P 500 cae la volatilidad anual del activo, en la misma ventana de cálculo de la beta.
8. **Calidad de datos.** No se imputa nada. Se eliminan fechas duplicadas, filas con precios faltantes y precios no positivos, y se informa cuántas. Se avisa si hay interrupciones de más de 5 días hábiles o si el último dato tiene más de 5 días hábiles. No se generan predicciones con menos de 504 sesiones válidas (dos años bursátiles).
9. **Construcción solapada o no solapada.** Los análisis por horizonte usan por defecto la construcción del documento (acumulado calculado cada día) y permiten cambiar a periodos no solapados.
10. **Historial inmutable.** Un disparador de SQLite impide modificar una predicción guardada. El resultado real se agrega en una tabla aparte. Como el precio ajustado puede reescalarse por dividendos posteriores, el rendimiento realizado se calcula con la serie vigente y el precio realizado se expresa en la escala del registro original.
11. **Intervalo de predicción.** Es normal, a partir de los pesos ψ del ARMA, y trata los parámetros como conocidos. Los límites del precio son la exponencial de los límites del log-precio (banda asimétrica). La evaluación rodante informa la cobertura empírica. Con MSFT, la cobertura del intervalo del 95 % fue de 82 % a 90 % según el horizonte: las colas pesadas y la volatilidad agrupada hacen que el intervalo normal sea optimista. La aplicación lo advierte cuando los residuos muestran efectos ARCH.

## Inconsistencias encontradas y cómo se trataron

No se reemplazó nada sin dejar rastro. La evaluación publicada se replica tal cual (pestaña *Evaluación* → *Réplica de la evaluación publicada*), y la evaluación corregida se presenta aparte como la recomendada.

### 1. Los horizontes de la caminata aleatoria no son los del proyecto
`evaluar_rw` y `hz` usan **Diario (1), Mensual (21) y Trimestral (63)**, mientras el proyecto define **Semanal (5), Mensual (21) y Trimestral (63)**.
*Efecto:* la caminata aleatoria y el ARIMA nunca se evalúan en el mismo horizonte semanal.
*Tratamiento:* la réplica conserva 1/21/63; la evaluación rodante usa 5/21/63 para ambos modelos.

### 2. Métricas en escalas distintas
La caminata aleatoria se mide sobre **precios** (MAE en dólares, MAPE) y el ARIMA sobre **rendimientos**.
*Efecto:* las dos tablas no se pueden comparar, así que la sección "Comparación con caminata aleatoria" no puede concluir cuál es mejor.
*Tratamiento:* en la evaluación rodante ambos modelos pronostican la misma variable (rendimiento acumulado ln(P_{t+h}/P_t)). Además se informa el MAPE en precio de ambos.

### 3. El ARIMA no usa orígenes rodantes
`forecast(modelo, h = length(prueba))` produce **un solo pronóstico de ~220 pasos** desde el final del entrenamiento. Un ARMA estacionario converge a su media en pocos pasos, así que casi toda la prueba se compara contra una constante. `REEST_CADA = 1` está definido pero no se usa.
*Efecto:* el "acierto de dirección" deja de medir la capacidad predictiva y pasa a medir cuántos rendimientos reales tuvieron el signo de esa media. El 20.7 % de Dior trimestral es ese artefacto: con una media pronosticada positiva y un periodo de prueba bajista, casi todos los signos fallan.
*Tratamiento:* la evaluación rodante pronostica desde cada día del periodo de prueba con la información disponible ese día, y reestima los parámetros en cada origen (`reestimar_cada = 1`, configurable). La réplica muestra junto a la dirección el porcentaje de rendimientos reales positivos, para que el artefacto sea visible.

### 4. Rendimientos solapados en la modelación por horizonte
`rent_mes = ln(P_t / P_{t−21})` calculado cada día hace que dos observaciones consecutivas compartan 20 de sus 21 días. Esa suma móvil es, por construcción, un MA(20) con raíces sobre el círculo unidad.
*Efecto:* auto.arima "descubre" una estructura MA que es aritmética, no del mercado. Con datos reales de GOOGL, los mejores modelos por BIC para la serie semanal fueron MA(4) con raíz de módulo 1.000, la firma exacta de sumar 5 días. auto.arima los descarta y elige un modelo de segunda fila sin interpretación económica. Lo mismo infla las pruebas de rachas por horizonte (casi siempre rechazan la aleatoriedad) y vuelve optimistas los p-valores de Kruskal-Wallis. La Q de Cochran del documento, en cambio, sí usa periodos no solapados.
*Tratamiento:* la evaluación rodante modela los **rendimientos diarios** (sin solapamiento) y obtiene el acumulado sumando los pronósticos de 1 a h pasos, con su varianza calculada a partir de los pesos ψ. En el diagnóstico por horizonte se muestra además la versión no solapada.

### 5. `allowmean` cambia entre secciones
La sección 12.9 usa `allowmean = FALSE` (y `allowdrift = FALSE`), y la 12.10 `allowmean = TRUE`.
*Tratamiento:* la réplica respeta cada sección. La modelación principal sigue la 12.9: ARIMA(p,1,q) sin deriva. La deriva puede activarse en Configuración.

### 6. Rejilla de órdenes: 0 a 3 o 0 a 5
La sección 12.8 define `P_MAX = 3` y `Q_MAX = 3`, pero las llamadas a `auto.arima` de las secciones 12.9 y 12.10 usan `max.p = 5, max.q = 5`.
*Efecto:* con 5, la rejilla pasa de 16 a 36 especificaciones por variante. Con datos reales, las especificaciones grandes casi siempre se descartan por raíces que se cancelan o pierden por BIC.
*Tratamiento:* se sigue la definición explícita de parámetros (0 a 3), que además es la que fija el prompt del proyecto. Es configurable hasta 5. La réplica literal de la sección 12.10 conserva el 5.

### 7. ARIMA(p,1,q) sobre el log-precio y ARMA(p,q) sobre la rentabilidad
Sin deriva, son el mismo modelo sobre los mismos datos. Las dos selecciones de la sección 12.9 siempre coinciden. Se verificó numéricamente (BIC idéntico) y se explica en la pestaña *Modelo ARIMA*.

## Cómo leer los resultados

- **Veredicto**: "Supera" o "Por debajo de" la caminata aleatoria solo cuando Diebold-Mariano es significativo al nivel α. Si no lo es, "Sin diferencia significativa". Si el modelo elegido es ARIMA(0,1,0), "Idéntico".
- **U de Theil < 1**: el ARIMA tuvo menor error cuadrático que la caminata aleatoria. Solo es una mejora real si Diebold-Mariano la declara significativa.
- **p-valores**: no son puntuaciones de calidad. No rechazar H0 no demuestra que sea verdadera, y que un modelo pase las pruebas de diagnóstico no lo hace confiable. VESTA no calcula índices compuestos de confiabilidad: presenta cada evidencia por separado.
- **Diebold-Mariano**: un estadístico negativo favorece al ARIMA. Con p ≥ 0.05 no se puede afirmar que alguno sea mejor.
- **Dirección**: se cuenta solo cuando el pronóstico tiene signo, y se prueba frente al 50 % con una binomial exacta. Acertar el 54 % en 200 pronósticos no es distinguible del azar.
- **Si el BIC elige ruido blanco**, el ARIMA coincide con la caminata aleatoria (U = 1). Es un resultado frecuente en acciones de gran capitalización y coherente con la hipótesis de mercados eficientes en su forma débil, no un fallo de la aplicación.
- **Beta**: mide la sensibilidad sistemática frente al S&P 500. No mide el riesgo total (una acción con beta baja puede ser muy volátil por razones propias) ni garantiza seguridad.

*Trabajo académico. No es una recomendación de inversión.*
