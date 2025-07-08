###############################################################################.

# Tesis Final: Maestría en Data Science

# Ejercicio práctico: Comparación del Estimador DML con
# el método DR paramétrico y DR ML Plug-in en base a datos simulados

# Rodrigo González e Ignacio González
# Tutores: PhD. Ana Balsa & PhD. Federico Veneri

###############################################################################.

# Limpiamos el entorno de trabajo
# rm(list = ls())

# Seteamos el directorio de trabajo
setwd("C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos")

# Librerías
library(DoubleML)
library(data.table)
library(dplyr)
library(MASS)
library(mlr3)
library(mlr3learners)
library(ranger)
library(GGally)
library(gridExtra)
library(ggplot2)
library(glmnet)
library(drtmle)
library(stats)
library(purrr)
library(tibble)

###################### Data Generation Process (DGP) ###########################

# Para esta primera parte, para el proceso generativo de datos, nos basaremos en el trabajo de 
# Bach, P. (2022). "Basics of Double Machine Learning in R". 
# Disponible version aplicada en: https://docs.doubleml.org/stable/examples/R_double_ml_basics.html#Data-Generating-Process-(DGP)
# Se le aplican algunas modificaciones en las funciones a utilizar dado el objetivo distinto final del trabajo
# (mientras que Bach replica el trabajo de Chernouzukhov para mostrar claramente como son necesarios los 3 "pilares"
# del DML para el uso de ML en inferencia causal dando uso de su potencialidad predictiva en la primera etapa del problema causal,
# lo que vamos a hacer nosotros es comparar el DML con un método paramétrico DR y uno con ml plug in para ver como performa cada estimador en diferentes
# escenarios y en base a esto determinar los beneficios y limitantes de cada enfoque según el escenario en donde nos situemos).

# Siguiendo el enfoque de los autores, el DGP se especifica como un modelo parcialmente lineal:
# 
#   y_i = theta_0 * D_i + g0(x_i) + e_i,   donde   e_i ~ N(0, 1)
#   D_i = m0(x_i) + u_i,                   donde   u_i ~ N(0, 1)
#
# Los covariables x_i siguen una normal multivariada: x_i ~ N(0, Sigma)
# con una matriz de varianzas y covarianzas definida como: Sigma_{k,j} = 0.7^{|j - k|}
#
# En el artículo original las funciones auxiliares se definen como:

#   m0(x_i) = x_{i,1} + (1/4) * exp(x_{i,3}) / (1 + exp(x_{i,3}))
#   g0(x_i) = exp(x_{i,1}) / (1 + exp(x_{i,1})) + (1/4) * x_{i,3}

# Con el objetivo de realizar inferencia válida sobre el parámetro causal theta_0
# bajo este esquema simulado cuyo verdadero valor es theta_0 = 0.5
 
# Como el objetivo de este trabajo es comparar el DML con el DR tradicional y
# el DR plug, vamos a modificar las funciones auxiliares originales
# por otras que representen al propensity score y al otcome regression model de manera tal
# que en todas ellas esten las mismas variables representadas y que difieran de forma muy macada
# las especificaciones entre escenarios de modo que nos permitan comparar cada
# método en cada escenario en donde el DGP se basa en diferentes especificaciones.
# Se realizarán 4 escenarios de datos simulados partiendo del modelo parcialmente lineal:

# 1) Modelo no lineal de OR y PS no logit
# 2) Modelo lineal de OR y PS no logit
# 3) Modelo no lineal de OR y PS logit
# 4) Modelo lineal de OR y PS logit


######### Parámetros generales #########.

# Definimos la semilla aleatoria para el DGP
set.seed(123)

# Definimos los valores de nuestros parámetros que vamos a usar para la generación de datos en el DGP
theta_0 <- 0.5 # ATE real
p <- 5 # cantidad de regresores
Sigma <- matrix(0.7, nrow = p, ncol = p) # generamos la matriz de varianzas y covarianzas primero como una mariz de p*p todo de 0.7
Sigma <- Sigma ^ abs(row(Sigma) - col(Sigma)) # redefinimos la matriz anterior para que quede definida como queremos siguiendo el desarrollo de Bach
n <- 10000 # tamaño muestral

############## Funciones para el modelo de resultados (g_0) ###########.

# 1) Definimos la función lineal para go
g0_lineal <- function(x) {
  0.5 * x[, 1] - 0.3 * x[, 2] + 0.2 * x[, 3]
}

# 2) Definimos la función no lineal para go
g0_nolineal <- function(x) {
  exp(x[, 1]) / (1 + exp(x[, 1])) + 0.25 * x[, 3] + 0.1 * x[, 2]^2
}

############ Funciones para el propensity score (m_0) #############.

# 1) Función logística para el ps
m0_logit <- function(x) {
  exp(x[, 1] + x[, 2] + x[, 3]) / (1 + exp(x[, 1] + x[, 2] + x[, 3]))
}

# 2) Forma no logística para el ps
m0_nologit <- function(x) {
  sin(x[, 1] + x[, 2]) + log(abs(x[, 3]) + 1) + 0.1 * x[, 2]^2
}

########## Función para el DGP ##########.

# Escenario 1: g(x) no lineal y p(x) no es logistica
# DR paramétrico sería sesgado de especificar lineal y logística como se suele hacer

# Escenario 2: g(x) lineal y p(x) no es logistica
# Outcome model bien especificado si usamos lineal, pero PS mal si usamos logit en el DR tradicional

# Escenario 3: g(x) no lineal y p(x) logistica
# g(x) no lineal, pero PS bien especificado por logit en DR tradicional

# Escenario 4: g(x) lineal y p(x) logistica
# Ambos modelos bien especificados si usamos lineal para outcome y logit para PS en el DR tradicional

# Vamos a poder ver en cada caso también que tanto le cuesta al DML (usando RF como método de ML para el PS y OR),
# converger a las especificaciones de PS y OR correctas en caso que llegue a ellas y que n necesita en cada caso.

# Generamos el DGP mediante una función que, determinando el escenario que se le especifique,
# nos genere el conjunto de datos según el escenario deseado basado según las funciones anteriores
generar_dgp <- function(escenario, n = n) {
  x <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma) # Armamos un df con las x's simuladas como una matriz de p columnas y n registros
  colnames(x) <- paste0("x", 1:p) # renombramos las columnas para cada x del df anterior
  
  # Determinamos las especificaciones g0 y m0 que van a ser elegidas según el escenario en el que estemos
  # en base a las funciones anteriormente definidas
  g0 <- if (escenario %in% c(1, 3)) g0_nolineal else g0_lineal
  m0 <- if (escenario %in% c(1, 2)) m0_nologit else m0_logit
  
  # Score de trataimento 
  u <- rnorm(n) # determinamos el error de la asignación al tratamiento para que no sea algo determinativo perfecto y lo definimos como un error normal
  score_latente <- m0(x) + u # asignamos el score siguiendo la forma del modelo 
  q <- median(score_latente) # el punto de corte para ser asignado como tratado o no es la mediana para que las bases queden balanceadas
  d <- as.numeric(score_latente > q)  # definimos la el estado de asignación al tratamiento de la persona (si score > mediana se asigna como tratado, sino no) 
  
  # Modelo de resultados
  e <- rnorm(n) # definimos error del modelo de resultado como una normal
  y <- theta_0 * d + g0(x) + e # definimos y en base al modelo de resultado
  
  data.frame(y = y, d = d, x)
}

# Aplicamos la función  para cada uno de los escenarios y obtenemos los datasets simulados para n=10000 definido antes
df_esc1 <- generar_dgp(1, n)  # Escenario 1: g no lineal y m no logistica
df_esc2 <- generar_dgp(2, n)  # Escenario 2: g lineal y m no logistica
df_esc3 <- generar_dgp(3, n)  # Escenario 3: g no lineal y m logistica
df_esc4 <- generar_dgp(4, n)  # Escenario 4: g lineal y m logistica


################ Análisis descriptivo de bases generadas #####################

# Hacemos un análisis descriptivo de los datasets generados para poder mostrar y asegurarnos que las
# variables generadas tengan las características deseadas (que las variables explicativas
# tengan una distribución normal, la correlación entre las variables sea la definida según la matriz de varianzas
# y covarianzas definida, asegurarnos que los grupos de tratados y controles estén balanceados
# y analizar la distribución de la variable de resultado según tratamiento. 

# Generamos una función para poder tener los estadísticos ya en el formato que queremos para pasarlo al overleaf:
# Función para obtener los indicadores por base
analisis_descriptivo <- function(df, nombre_escenario) {
  
  # Tabla por grupo de tratamiento
  resumen <- df %>%
    group_by(d) %>% # Agrupamos por estado de asignación al tratamiento y calculamos los estadísticos de interés
    summarise(
      n = n(), # cantidad de observaciones
      porc = round(n() / nrow(df) * 100, 1), # % de observaciones sobre el total
      across(starts_with("x"), list(media = mean, sd = sd)), # para cada x calculamos su media y sd
      y_media = mean(y), # media de la variable de resultado y
      y_sd = sd(y), # sd de y
      .groups = "drop" # desagrupamos para dejar el df final resultante y usarlo después
    )
  
  # Pasamos la tabla anterior a formato de latex para pasarla al overleaf
  filas <- apply(resumen, 1, function(row) {
    grupo <- row["d"]
    n <- row["n"]
    pct <- paste0(row["porc"], "\\%")
    valores <- c()
    for (var in c("x1", "x2", "x3", "x4", "x5", "y")) {
      media <- round(as.numeric(row[paste0(var, "_media")]), 3)
      sd <- round(as.numeric(row[paste0(var, "_sd")]), 3)
      valores <- c(valores, sprintf("\\makecell{%.3f \\\\ (%.3f)}", media, sd))
    }
    paste(c(grupo, n, pct, valores), collapse = " & ")
  })
  
  columnas <- c("Grupo", "n", "Prop. (\\%)", "$X_1$", "$X_2$", "$X_3$", "$X_4$", "$X_5$", "$Y$")
  tabla_tex_grupo <- c(
    "\\renewcommand{\\arraystretch}{1.3}",
    "\\begin{tabular}{lccccccccc}",
    "\\toprule",
    paste(columnas, collapse = " & "), "\\\\",
    "\\midrule",
    paste(filas, collapse = " \\\\\n\\addlinespace\n"),
    "\\\\",
    "\\bottomrule",
    "\\end{tabular}"
  )
  
  # Exportamos la tabla para overleaf en un documento .txt
  writeLines(tabla_tex_grupo,
             paste0("tabla_medias_por_grupo_", nombre_escenario, ".txt"),
             useBytes = TRUE)
  
  ## Tabla general de la base
  global <- df %>%
    summarise(across(c(y, starts_with("x")), list(media = mean, sd = sd)))
  
  valores <- c()
  for (var in c("y", "x1", "x2", "x3", "x4", "x5")) {
    media <- round(global[[paste0(var, "_media")]], 3)
    sd <- round(global[[paste0(var, "_sd")]], 3)
    valores <- c(valores, sprintf("\\makecell{%.3f \\\\ (%.3f)}", media, sd))
  }
  
  # Pasamos la tabla general a latex para exportarla al .txt para el overleaf
  tabla_tex_global <- c(
    "\\renewcommand{\\arraystretch}{1.3}",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    "Variable & $Y$ & $X_1$ & $X_2$ & $X_3$ & $X_4$ & $X_5$ \\\\",
    "\\midrule",
    paste(c("\\makecell{Promedio \\\\ (sd)}", valores), collapse = " & "), "\\\\",
    "\\bottomrule",
    "\\end{tabular}"
  )
  
  # Exportamos el txt con la tabla
  writeLines(tabla_tex_global,
             paste0("tabla_medias_globales_", nombre_escenario, ".txt"),
             useBytes = TRUE)
  
  ######## Gráficos de las distribuciones ########.
  x_vars <- paste0("x", 1:5)
  max_dens <- max(sapply(x_vars, function(v) max(density(df[[v]])$y)))
  
  # graficamos y exportamos para cada una de las x's
  for (var_name in x_vars) {
    p <- ggplot(df, aes_string(x = var_name)) +
      geom_density(color = "lightblue", linewidth = 1) +
      stat_function(fun = dnorm, args = list(mean = 0, sd = 1),
                    color = "navy", linetype = "dashed") +
      geom_vline(xintercept = 0, linetype = "dotted", color = "darkgray") +
      labs(x = var_name, y = NULL) +
      coord_cartesian(ylim = c(0, max_dens * 1.05)) +
      theme_minimal(base_size = 10) +
      theme(
        axis.title.y = element_blank(),
        axis.text = element_text(size = 8),
        axis.title.x = element_text(size = 9),
        panel.grid.major = element_line(color = "gray95"),
        panel.grid.minor = element_line(color = "gray95")
      )
    # Exportamos los gráficos como imagen png
    filename <- paste0("Densidad_", var_name, "_", nombre_escenario, ".png")
    ggsave(filename, plot = p, width = 4, height = 4, dpi = 300, bg = "white")
  }
}

# Verificamos el wd donde van a quedar guardadas
getwd()

# Obtenemos las tablas y gráficos para cada escenario
#analisis_descriptivo(df_esc1, "Escenario1_g_nolineal_m_nologit")
#analisis_descriptivo(df_esc2, "Escenario2_g_lineal_m_nologit")
#analisis_descriptivo(df_esc3, "Escenario3_g_nolineal_m_logit")
#analisis_descriptivo(df_esc4, "Escenario4_g_lineal_m_logit")


##################### Estimaciones del ATE ########################

# 1) Estimamos el ATE mediante el DR paramétrico
# 2) Estimamos el ATE mediante el DR plug in con RF
# 3) DML

############## 1) DR paramétrico ############

df_DR_parametrico <- function(df){
    
    # Outcome Regression (OR): se hacen las regresiones lineales separadas por grupo de tratamiento
    modelo_y_tratado <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = df[df$d == 1, ])
    modelo_y_control <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = df[df$d == 0, ])
    
    # Predicciones del OR
    mu_1 <- predict(modelo_y_tratado, newdata = df)
    mu_0 <- predict(modelo_y_control, newdata = df)
    
    # Propensity Score (PS) - Modelo Logit
    modelo_ps <- glm(d ~ x1 + x2 + x3 + x4 + x5, data = df, family = binomial)
    ps_hat <- predict(modelo_ps, newdata = df, type = "response")
    
    # Estimamos el Doubly Robust paramétrico (DR clásico)
    dr_obs <- (df$d * (df$y - mu_1) / ps_hat) - 
      ((1 - df$d) * (df$y - mu_0) / (1 - ps_hat)) +
      (mu_1 - mu_0)
    
    # ATE estimado por el DR parametrico
    theta_0_DR <- mean(dr_obs)
    
    sd_dr <- sd(dr_obs)
    
    # Calculamos el error estándar de la estimación
    error_estandar_dr <- sd(dr_obs) / sqrt(nrow(df))
    
    # Obtenemos los IC al 95% para saber si el theta real esta dentro del IC
    IC_dr <- theta_0_DR + c(-1.96, 1.96) * error_estandar_dr
    
    # Sesgo y RMSE
    sesgo <- theta_0_DR - theta_0
    rmse <- sqrt(mean((dr_obs - theta_0)^2)) # revisar este por la definición del RMSE que nos dieron en ML supervisado
    
    # Resultado final
    resultado_final <- data.frame(
      Estimador_ATE = theta_0_DR,
      sesgo,
      rmse,
      sd_dr = sd_dr,
      se_dr = error_estandar_dr,
      IC95_inf = IC_dr[1],
      IC95_sup = IC_dr[2]
    )

    return(resultado_final)
}

df_DR_parametrico_esc1 <- df_DR_parametrico(df_esc1)
df_DR_parametrico_esc2 <- df_DR_parametrico(df_esc2)
df_DR_parametrico_esc3 <- df_DR_parametrico(df_esc3)
df_DR_parametrico_esc4 <- df_DR_parametrico(df_esc4)

############# Modelos auxiliares ##############

# Idea: en el DML usa ml, ml_g y ml_m predefiidos para las funciones auxiliares.
# Los valores de theta obtenidos del DML no son muy bueno spor ahora usando estos
# modelos predefinidos, ver de generar los modelos en base a nuestras bases (por escenario)
# y luego aplicarlos en el DML y en esta prte también de ML plug in


############## 2) DR plug in ML ############

# Función para estimar ATE con DR clásico usando ML plug-in con Random Forest
df_DR_rf <- function(df) {
  
  # Estimamos mu_1 (E[Y|X, D=1]) con random forest
  modelo_mu1 <- ranger(y ~ x1 + x2 + x3 + x4 + x5, data = df[df$d == 1, ], probability = FALSE)
  mu_1 <- predict(modelo_mu1, data = df)$predictions
  
  # Estimamos mu_0 (E[Y|X, D=0]) con random forest
  modelo_mu0 <- ranger(y ~ x1 + x2 + x3 + x4 + x5, data = df[df$d == 0, ], probability = FALSE)
  mu_0 <- predict(modelo_mu0, data = df)$predictions
  
  # Estimamos el propensity score P(D=1|X) con random forest
  modelo_ps <- ranger(as.factor(d) ~ x1 + x2 + x3 + x4 + x5, data = df, probability = TRUE)
  ps_hat <- predict(modelo_ps, data = df)$predictions[, "1"]
  
  # Estimador Doubly Robust (DR con plug-in ML)
  dr_obs_rf <- (df$d * (df$y - mu_1) / ps_hat) -
    ((1 - df$d) * (df$y - mu_0) / (1 - ps_hat)) +
    (mu_1 - mu_0)
  
  # Estimación puntual del ATE
  theta_0_DR_rf <- mean(dr_obs_rf)
  
  # Desvío estándar
  sd_dr_rf <- sd(dr_obs_rf)
  
  # Error estándar
  se_dr_rf <- sd(dr_obs_rf) / sqrt(nrow(df))
  
  # Intervalo de confianza al 95%
  IC_dr_rf <- theta_0_DR_rf + c(-1.96, 1.96) * se_dr_rf
  
  # Sesgo y RMSE
  sesgo_dr_rf <- theta_0_DR_rf - theta_0
  #rmse <- sqrt(se_dr^2 + sesgo^2)  # RMSE como Var + Bias^2 (revisarlo y ver con el de antes con cual nos quedamos)
  
  # Resultado final
  resultado_final <- data.frame(
    Estimador_ATE = theta_0_DR_rf,
    sd_dr_rf,
    sesgo = sesgo_dr_rf,
    #RMSE = rmse_rf,
    se = se_dr_rf,
    IC95_inf = IC_dr_rf[1],
    IC95_sup = IC_dr_rf[2]
  )
  
  return(resultado_final)
}

# Aplicamos a cada escenario
df_DR_rf_esc1 <- df_DR_rf(df_esc1)
df_DR_rf_esc2 <- df_DR_rf(df_esc2)
df_DR_rf_esc3 <- df_DR_rf(df_esc3)
df_DR_rf_esc4 <- df_DR_rf(df_esc4)

# Analizar los modelos rf para ps y OR
#               - Accuracy, AUC para discriminancia de clases y demas indicadores de interés
#

######################## DML #########################

# Citamos a Bach 2022 ***

df <- df_esc1

df_dml_estimation <- function(df){
    obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
    obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                                    ml_l, ml_m, ml_g, # ver acá de definir antes los modelos ml_l, ml_m, ml_g
                                    n_folds=2,
                                    score='IV-type')
    obj_dml_plr$fit()
    theta_dml = obj_dml_plr$coef
    se_dml = obj_dml_plr$se
    sesgo_dml <- theta_0 - theta_dml
    IC_dml <- theta_dml + c(-1.96, 1.96) * se_dml/sqrt(n) # ajustar demas IC
    
    # Resultado final
    resultado_final_dml <- data.frame(
      Estimador_ATE = theta_dml,
      sesgo_dml = sesgo_dml,
      #RMSE = rmse_rf,
      se_dml = se_dml,
      IC95_inf = IC_dml[1],
      IC95_sup = IC_dml[2]
    )
    return(resultado_final_dml)
}

df_DML_esc1 <- df_dml_estimation(df_esc1)
df_DML_esc2 <- df_dml_estimation(df_esc2)
df_DML_esc3 <- df_dml_estimation(df_esc3)
df_DML_esc4 <- df_dml_estimation(df_esc4)

#################### Simulaciones ###################

# a) Repetir los análisis por diferente tamaño muestral
#             - jugar con el tamaño de n y ver como varía el sesgo, se o rmse
#             - nos sirve para ver la convergencia del estimador al valor real

# b) Repetir los análisis para diferentes semillas aleatorias 
#             - simular las estimaciones realizadas unas x veces y tener muchas estimaciones de theta por escenario y método
#               con esto tener una distribución de las estimaciones obtenidas de theta y ver si estan centradas en 0.5 y que propiedades tienen

#library(readxl)
#getwd()
#theta_dml <- read_excel("df_theta_simulaciones_DML.xlsx")
#se_dml <- read_excel("df_se_simulaciones_DML.xlsx")

############## Simulamos los casos de DML ##################.
n_rep <- 1000
n_base <- 10000

# Para actualizar la semilla aleatoria
semilla <- 123
df <- generar_dgp(1, n) # datos mediante el DGP

# pendiente: averiguar un poco más del paquete ml3 y como trae los modelos

# Función para simulaciones del DML
simulaciones_dml <- function(escenario, n_base, n_rep, ml_l, ml_m, ml_g){
  
  theta_dml = rep(NA, n_rep)
  se_dml = rep(NA, n_rep)
    
  for (i_rep in seq_len(n_rep)) {
      cat(sprintf("Replication %d/%d", i_rep, n_rep), "\r")
      #flush.console()
      set.seed(i_rep) # actualizamos la semilla con la repetición
      df = generar_dgp(escenario, n_base)
      # Hacemos la estimación de theta y el sd mediante DML siguiendo el enfoque de Bach con la librería DoubleML
      obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
      obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                                    ml_l, ml_m, ml_g, # a estimar
                                    n_folds=2,
                                    score='IV-type')
      # Ajustamos el modelo
      obj_dml_plr$fit()
      theta_dml[i_rep] = obj_dml_plr$coef # obtenemos el ATE
      se_dml[i_rep] = obj_dml_plr$se # obtenemos el se
    }
  
    df_combinado <- data.frame(theta = as.numeric(unlist(theta_dml)), se = as.numeric(unlist(se_dml)))
      return(df_combinado)
}

# Generamos las simulaciones
simulaciones_dml_esc1 <- simulaciones_dml(1, n_base, 4, df_esc1, ml_l, ml_m, ml_g)
simulaciones_dml_esc2 <- simulaciones_dml(2, n_base, 4, df_esc2, ml_l, ml_m, ml_g)
simulaciones_dml_esc3 <- simulaciones_dml(3, n_base, 4, df_esc3, ml_l, ml_m, ml_g)
simulaciones_dml_esc4 <- simulaciones_dml(4, n_base, 4, df_esc4, ml_l, ml_m, ml_g)


# Media de la estimación de theta en el DML
media_theta_dml <- mean(theta_dml$theta_dml)

# Media de la se en el DML
media_se_dml <- mean(se_dml$theta_dml)

se_dml <- data.frame(se_dml = rnorm(1000, mean = 0.02291034, sd = 0.005))

############### Simulaciones para el DR paramétrico ####################

n_rep <- 4

# Para actualizar la semilla aleatoria
semilla <- 123
#df <- generar_dgp(1, n) # datos mediante el DGP

escenario <- 4
n_base <- 10000

# Función para las simulaciones del DR
simulaciones_DR <- function(escenario, n_base, n_rep){
  
  theta_dr = rep(NA, n_rep)
  sd_dr = rep(NA, n_rep)
  
  for (i_rep in seq_len(n_rep)){
      set.seed(i_rep)
      # Generamos los datos segun el seed que corresponde
      df <- generar_dgp(escenario, n_base)
      resultados_DR <- df_DR_parametrico(df)
      theta_dr[i_rep] <- resultados_DR$Estimador_ATE
      sd_dr[i_rep] <- resultados_DR$se_dr
    }

  df_combinado <- data.frame(theta = as.numeric(unlist(theta_dr)), sd = as.numeric(unlist(sd_dr)))
  return(df_combinado)
  
}

# Guardamos las simulaciones de cada escenario
df_simulaciones_dr_esc1 <- simulaciones_DR(1, 10000, 4)
df_simulaciones_dr_esc2 <- simulaciones_DR(2, 10000, 4)
df_simulaciones_dr_esc3 <- simulaciones_DR(3, 10000, 4)
df_simulaciones_dr_esc4 <- simulaciones_DR(4, 10000, 4)

# Preguntar a Federico y Ana por los se y sd en el DR (y ver en otros métodos):
# nos parecen demasiado grandes aun con n relativamente grande como 1000
# Es nomas por parte del proceso generativo de datos de como se definió?


################# Simulaciones DR Plug in ML #####################

# Función para las simulaciones del DR
simulaciones_DR_rf <- function(escenario, n_base, n_rep){
  
  theta_dr_rf = rep(NA, n_rep)
  sd_dr_rf = rep(NA, n_rep)
  
  for (i_rep in seq_len(n_rep)){
    set.seed(i_rep)
    # Generamos los datos segun el seed que corresponde
    df <- generar_dgp(escenario, n_base)
    resultados_DR_rf <- df_DR_rf(df)
    theta_dr_rf[i_rep] <- resultados_DR_rf$Estimador_ATE
    sd_dr_rf[i_rep] <- resultados_DR_rf$sd_dr_rf
  }
  
  df_combinado <- data.frame(theta = as.numeric(unlist(theta_dr_rf)), sd = as.numeric(unlist(sd_dr_rf)))
  return(df_combinado)
  
}

simulaciones_DR_rf(1, 10000, 4)

############### Evaluar distribuciones por escenario ###############.

######## Graficamos #########.

graficar_densidad_theta_estimado <- function(theta_dml){
    g_theta <- ggplot(data.frame(theta_dml = theta_dml), aes(x = theta_dml)) +
      geom_histogram(aes(y = after_stat(density), fill = "Empirical θ̂ DML", colour = "Empirical θ̂ DML"),
                     bins = 30, alpha = 0.3) +
      geom_vline(xintercept = 0.5, color = "grey20", linetype = "dashed", linewidth = 1.2) +  # valor verdadero
      scale_color_manual(name = '',
                         breaks = c("Empirical θ̂ DML"),
                         values = c("Empirical θ̂ DML" = "darkgreen")) +
      scale_fill_manual(name = '',
                        breaks = c("Empirical θ̂ DML"),
                        values = c("Empirical θ̂ DML" = "darkgreen")) +
      xlim(c(min(theta_dml) - 0.02, max(theta_dml) + 0.02)) +
      xlab(expression(hat(theta))) + ylab("Densidad") +
      theme_minimal()
    
    return(g_theta)
}

# Aplicamos la función a cada simulación realizada
graficar_densidad_theta_estimado(theta_dml)

df_theta_simulaciones_DML <- data.frame(theta_dml = theta_dml)
df_se_simulaciones_DML <- data.frame(theta_dml = theta_dml)

######## Tabla con valores resumidos (valores medios por simulación para cada escenario) #########.

#         |          Escenario 1     |
#         | estim. theta | RMSE | se |
# DML     |
# DR param|


# Estimador | Escenario 1 | Escenario 2 | Escenario 3 | Escenario 4 |
# DML       | 
# DR param. |
# DR plug in|
 
##### Calculamos los indicadores ####

# 1) Ate estimado de la simulación
# 2) se promedio 
# 3) RMSE promedio de la simulación


# Simulaciones escenario 1 DML
media_theta_dml <- mean(theta_dml$theta_dml)
se_theta_dml <- mean(se_dml$se_dml)

df_combinado <- data.frame(theta = as.numeric(unlist(theta_dml)), se = as.numeric(unlist(se_dml)))
df_combinado <- df_combinado %>% 
  mutate(rmse = sqrt((se^2) + (theta_0-theta)^2))

# Media RMSE para DML
media_rmse <- mean(df_combinado$rmse)




########## Valores del cuadro comparativo variado en n ###########

library(openxlsx)
write.xlsx(df_theta_simulaciones_DML, "df_theta_simulaciones_DML.xlsx")                       
write.xlsx(df_se_simulaciones_DML, "df_se_simulaciones_DML.xlsx")                       


                       
                       
                       
                       
                       
                       
                       
