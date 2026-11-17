# Exporta resultados de referencia calculados con los paquetes de R originales,
# para compararlos con VESTA (validacion/comparar_con_r.py).
#
# Uso, desde la carpeta del proyecto, después de abrir la app al menos una vez
# para que existan los CSV en data/precios:
#   Rscript validacion/exportar_referencia.R MSFT KO TSLA
#
# Paquetes: tseries, nortest, FinTS, forecast

suppressPackageStartupMessages({
  library(tseries); library(nortest); library(FinTS); library(forecast)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) args <- c("MSFT")
dir.create("validacion/referencia_r", showWarnings = FALSE, recursive = TRUE)

for (tk in args) {
  d <- read.csv(file.path("data/precios", paste0(gsub("[^A-Za-z0-9_-]", "_", tk), ".csv")))
  d <- d[order(as.Date(d$Date)), ]
  # misma ventana de 5 años que la aplicación
  fin <- max(as.Date(d$Date)); d <- d[as.Date(d$Date) > fin - round(365.25 * 5), ]
  p  <- d$Adj.Close
  lp <- log(p)
  r  <- diff(lp)
  n  <- length(lp); corte <- floor(n * 0.8)
  r_train <- r[1:(corte - 1)]

  pr <- list(
    rachas      = runs.test(as.factor(r > median(r)))$p.value,
    lilliefors  = lillie.test(r)$p.value,
    adf_nivel   = adf.test(lp)$p.value,  adf_nivel_stat = unname(adf.test(lp)$statistic),
    adf_dif     = adf.test(r)$p.value,
    kpss_nivel  = kpss.test(lp, null = "Level")$p.value, kpss_nivel_stat = unname(kpss.test(lp, null = "Level")$statistic),
    kpss_dif    = kpss.test(r, null = "Level")$p.value,
    ljung10     = Box.test(r, lag = 10, type = "Ljung-Box")$p.value,
    ljung20     = Box.test(r, lag = 20, type = "Ljung-Box")$p.value,
    arch10      = ArchTest(r, lags = 10)$p.value
  )
  # Mismos parámetros que VESTA por defecto: ARIMA(p,1,q) sobre el log-precio, p, q <= 3, BIC, sin deriva
  y_train <- lp[1:corte]
  fit <- auto.arima(y_train, d = 1, seasonal = FALSE, max.p = 3, max.q = 3, max.order = 6,
                    ic = "bic", stepwise = FALSE, approximation = FALSE, allowdrift = FALSE)
  o <- arimaorder(fit)
  pr$arima_p <- o[1]; pr$arima_q <- o[3]; pr$arima_bic <- fit$bic; pr$arima_aic <- fit$aic

  write.csv(data.frame(ticker = tk, prueba = names(pr), valor = unlist(pr)),
            file.path("validacion/referencia_r", paste0(tk, ".csv")), row.names = FALSE)
  cat(tk, ": ARIMA(", o[1], ",0,", o[3], ") BIC =", round(fit$bic, 2), "\n")
}
