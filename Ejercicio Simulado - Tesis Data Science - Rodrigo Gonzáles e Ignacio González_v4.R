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
library(mlr3tuning)
library(ranger)
library(GGally)
library(gridExtra)
library(ggplot2)
library(glmnet)
library(drtmle)
library(stats)
library(purrr)
library(tibble)
library(paradox)
library(pROC)
library(future)
library(future.apply)
library(progressr)

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
  u <- rnorm(n, mean = 0, sd = 0.3) # determinamos el error de la asignación al tratamiento para que no sea algo determinativo perfecto y lo definimos como un error normal
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
  theta_hat <- mean(dr_obs)
  
  # Desvío estándar y error estándar
  sd <- sd(dr_obs)
  se <- sd / sqrt(nrow(df))
  
  # Intervalo de confianza al 95%
  IC <- theta_hat + c(-1.96, 1.96) * se
  
  # Resultado final unificado
  resultado_final <- data.frame(
    theta = theta_hat,
    sd = sd,
    se = se,
    IC95_inf = IC[1],
    IC95_sup = IC[2]
  )
  
  return(resultado_final)
}

# Aplicanos la función de DR por escenario
df_DR_parametrico_esc1 <- df_DR_parametrico(df_esc1)
df_DR_parametrico_esc2 <- df_DR_parametrico(df_esc2)
df_DR_parametrico_esc3 <- df_DR_parametrico(df_esc3)
df_DR_parametrico_esc4 <- df_DR_parametrico(df_esc4)


############## 2) DML ############

############# 2.1) Estimación Modelos auxiliares ############

# Estimamos los modelos con cross-fitting (para sample splitting) y estimados 
# estos modelos en bases distintas de las que luego se usan para estimar el ATE final
# siguiendo la especificación de DML en Chernouzhukov 2018.

####### a) Modelo de resultados E[Y|X] #######.

# Debemos definir los modelos que vamos a usar para las funciones auxiliares
# Modelo para E[Y|X] usando Random Forest (regresión)

### i. OR Escenario no lineal: ###
# Definimos los parámetros de nuestro modelo de regresión para el OR

df <- df_esc1

# Función para 
entrenar_rf <- function(df, 
                        target = "y",
                        folds = 2,
                        resolution = 5,
                        evals = 25,
                        seed = 999999999,
                        num.trees = 500) {
  
  # Excluir la variable d automáticamente
  df_model <- df[, setdiff(colnames(df), "d"), drop = FALSE]
  
  # Definir learner base
  learner_rf <- lrn("regr.ranger",
                    num.trees = num.trees,
                    seed = seed)
  
  # Crear el Task sin "d"
  task_rf <- TaskRegr$new(id = "rf_tuning_task",
                          backend = df_model,
                          target = target)
  
  # Métrica y resampling
  measure <- msr("regr.mse")
  resampling <- rsmp("cv", folds = folds)
  
  # Espacio de hiperparámetros
  search_space <- ps(
    max.depth = p_int(3, 10),
    min.node.size = p_int(5, 20)
  )
  
  # Instancia de tuning
  instance <- TuningInstanceBatchSingleCrit$new(
    task = task_rf,
    learner = learner_rf$clone(deep = TRUE),
    resampling = resampling,
    measure = measure,
    search_space = search_space,
    terminator = trm("evals", n_evals = evals)
  )
  
  # Ejecutar tuning
  tuner <- tnr("grid_search", resolution = resolution)
  tuner$optimize(instance)
  
  # Aplicar parámetros óptimos y entrenar final
  learner_rf$param_set$values <- instance$result_learner_param_vals
  learner_rf$train(task_rf)
  
  return(list(
    modelo = learner_rf,
    mejores_parametros = instance$result_learner_param_vals,
    resumen_tuning = as.data.table(instance$archive)
  ))
}


# Modelo escenario no lineal
modelo_rf_esc1 <- entrenar_rf(df_esc1)

# Modelo escenario lineal
modelo_rf_esc2 <- entrenar_rf(df_esc2)

#modelo <- modelo_rf_esc1
#df <- df_esc1

# Hacemos las predicciones aplicando el modelo y monitoreamos los resultados de las
# predicciones de los modelos


evaluar_modelo <- function(modelo, df, target = "y") {
  # Excluir la variable d automáticamente
  df_model <- df[, setdiff(colnames(df), "d"), drop = FALSE]
  
  task_nuevo <- TaskRegr$new(id = "pred_rf", backend = df_model, target = target)
  pred <- modelo$predict(task_nuevo)
  
  y_pred <- pred$response
  
  return(predicciones = y_pred)
}

# Entrenamos y obtenemos el modelo sobre df_esc1
modelo_rf_esc_no_lineal <- entrenar_rf(df_esc1)

# Aplicamos el modelo y obtenemos las predicciones en la base 3 (también no linear)
predicciones_rf_esc_no_lineal <- evaluar_modelo(modelo_rf_esc_no_lineal$modelo, df_esc3)

# Validamos las predicciones en otra base distinta a la que usamos para el train pero que también tiene OR no lineal
y_real <- df_esc3$y
y_pred <- predicciones_rf_esc_no_lineal

mse <- mean((y_real - y_pred)^2)
rmse <- sqrt(mse)

var_y <- var(df_esc3$y)
mse_modelo <- mean((df_esc3$y - y_pred)^2)

mse_modelo / var_y  # Relación entre MSE y Varianza


# Escenario 1: g(x) no lineal y p(x) no es logistica
# Escenario 2: g(x) lineal y p(x) no es logistica
# Escenario 3: g(x) no lineal y p(x) logistica
# Escenario 4: g(x) lineal y p(x) logistica


##### ii. OR Escenario lineal (escenario 2 y 4) ####.

# Estimamos el modelo tuneado para un caso lineal
modelo_rf_esc_lineal <- entrenar_rf(df_esc2)

# Validamos los resultados usando la otra base lineal (df_esc4)
y_real_lineal <- df_esc4$y
y_pred_lineal <- evaluar_modelo(modelo_rf_esc_lineal$modelo, df_esc4)

# Obtenemos el mse y rmse
mse_lineal <- mean((y_real_lineal - y_pred_lineal)^2)
rmse_lineal <- sqrt(mse_lineal)



####### b) Modelo de Propensity Score E[D|X] #######

# Evaluamos de nievo cada escenario y testeamos los modelos auxiliares

##### i. Modelo PS Escenario logit (escenarios 3 y 4) #####

# Función para entrenar modelo PS con RF
entrenar_ps <- function(df,
                                 folds = 3,
                                 evals = 50,
                                 seed = 99999999) {
  
  # Preprocesamiento de la base
  df_model <- df[, setdiff(colnames(df), "y"), drop = FALSE]
  df_model$d <- factor(df_model$d, levels = c(0, 1))
  
  # Definir learner base
  learner_ps <- lrn("classif.ranger",
                    predict_type = "prob",
                    seed = seed)
  
  # Task de clasificación
  task_ps <- TaskClassif$new(id = "ps_tuning_task",
                             backend = df_model,
                             target = "d")
  
  # Métrica de evaluación priorizando AUC
  measure <- msr("classif.auc")
  resampling <- rsmp("cv", folds = folds)
  
  # Espacio de búsqueda extendido con num.trees
  search_space <- ps(
    max.depth = p_int(2, 15),
    min.node.size = p_int(1, 30),
    mtry = p_int(1, ncol(df_model) - 1),
    sample.fraction = p_dbl(0.5, 1),
    num.trees = p_int(100, 1000)
  )
  
  # Instancia de tuning
  instance <- TuningInstanceBatchSingleCrit$new(
    task = task_ps,
    learner = learner_ps$clone(deep = TRUE),
    resampling = resampling,
    measure = measure,
    search_space = search_space,
    terminator = trm("evals", n_evals = evals)
  )
  
  # Tuner random search
  tuner <- tnr("random_search", batch_size = 5)
  tuner$optimize(instance)
  
  # Entrenar modelo final con los mejores hiperparámetros
  learner_ps$param_set$values <- instance$result_learner_param_vals
  learner_ps$train(task_ps)
  
  return(list(
    modelo = learner_ps,
    mejores_parametros = instance$result_learner_param_vals,
    resumen_tuning = as.data.table(instance$archive)
  ))
}



# Estimamos el modelo entrenado para el caso logistico
modelo_ps_logit <- entrenar_ps(df_esc3)

evaluar_modelo_ps <- function(modelo, df) {
  # Eliminar la variable y
  df_model <- df[, setdiff(colnames(df), "y"), drop = FALSE]
  
  # Asegurar que d sea un factor con niveles 0 y 1
  df_model$d <- factor(df_model$d, levels = c(0, 1))
  
  # Crear Task de clasificación
  task_nuevo <- TaskClassif$new(id = "pred_ps", backend = df_model, target = "d")
  
  # Obtenemos las predicciones del modelo
  pred <- modelo$predict(task_nuevo)
  
  # Devolvemos la probabilidad de d == 1
  prob_pred <- pred$prob[, "1"]
  
  return(prob_pred)
}


# Evaluación sobre otra base logit (df_esc4)
prob_pred_logit <- evaluar_modelo_ps(modelo_ps_logit$modelo, df_esc4)

# Calculamos el AUC
roc_logit <- roc(df_esc4$d, prob_pred_logit)
auc(roc_logit)

# Corte para la clasificación de d en base a la predicción del  modelo
cutoff <- 0.5

# Clasificación la mediana como cutoff
d_predicho <- ifelse(prob_pred_logit >= cutoff, 1, 0)

# Valores reales
d_real <- df_esc4$d

# Calculamos la accuracy
accuracy_logit <- mean(d_predicho == d_real)
accuracy_logit

##### Modelo PS Escenario no logit (escenarios 1 y 2) #####
modelo_ps_nologit <- entrenar_ps(df_esc1)

# Evaluación sobre otra base no logit (df_esc2)
prob_pred_nologit <- evaluar_modelo_ps(modelo_ps_nologit$modelo, df_esc2)

roc_nologit <- roc(df_esc2$d, prob_pred_nologit)
auc(roc_nologit)

d_predicho_no_logit <- ifelse(prob_pred_nologit >= cutoff, 1, 0)

# Valores reales
d_real_no_logit <- df_esc2$d

# Calculamos la accuracy
accuracy_no_logit <- mean(d_predicho_no_logit == d_real_no_logit)
accuracy_no_logit

##########################################################################.

# Ajustamos ml_l y ml_g con los hiperparámetros de los modelos tuneados 
# para pasárselos a la estimación de DML

modelo_ps_logit
modelo_ps_nologit

modelo_rf_esc_lineal
modelo_rf_esc_no_lineal


##### Escenario de OR no lineal ####.
ml_l_nonlineal <- lrn("regr.ranger")
ml_l_nonlineal$param_set$values <- modelo_rf_esc_no_lineal$mejores_parametros

##### Escenario de OR lineal ####.
ml_l_lineal <- lrn("regr.ranger")
ml_l_lineal$param_set$values <- modelo_rf_esc_lineal$mejores_parametros

#### PS no logit ####.
ml_m_nologit <- lrn("classif.ranger", predict_type = "prob")
ml_m_nologit$param_set$values <- modelo_ps_nologit$mejores_parametros

#### PS logit ####.
ml_m_logit <- lrn("classif.ranger", predict_type = "prob")
ml_m_logit$param_set$values <- modelo_ps_logit$mejores_parametros

############# 3.1) Estimación ATE con DML ############

# Citamos a Bach 2022 ***
# El código siguiente lo obyenemos de ajustar el material de Bach (2022) disponible en su sitio web DML
# https://docs.doubleml.org/stable/examples/R_double_ml_basics.html#Data-Generating-Process-(DGP)

df_dml_estimation <- function(df, escenario, n_folds = 2, score = "IV-type") {
  obj_dml_data <- double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
  
  # Definimos los ml_l, y ml_m segun el escenario en que estemos
  if (escenario %in% c(1, 3)) {
    ml_l <- ml_l_nonlineal$clone()
    ml_g <- ml_l_nonlineal$clone()
  } else {
    ml_l <- ml_l_lineal$clone()
    ml_g <- ml_l_lineal$clone()
  }
  
  # Selección del modelo PS (ml_m)
  if (escenario %in% c(1, 2)) {
    ml_m <- ml_m_nologit$clone()
  } else {
    ml_m <- ml_m_logit$clone()
  }
  
  # DML
  obj_dml_plr <- DoubleMLPLR$new(
    data = obj_dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    ml_g = ml_g,
    n_folds = n_folds,
    score = score
  )
  
  # Ajustamos el modelo
  obj_dml_plr$fit()
  
  # Obtenemos del objeto ajustado ates las estimaciones
  theta_hat <- obj_dml_plr$coef
  se <- obj_dml_plr$se
  sd <- se * sqrt(nrow(df))
  IC <- theta_hat + c(-1.96, 1.96) * se
  
  # Guardamos los resultados en un df final
  resultado <- data.frame(
    theta = theta_hat,
    sd = sd,
    se = se,
    IC95_inf = IC[1],
    IC95_sup = IC[2]
  )
  
  return(resultado)
}

# Aplicamos el DML a cada escenario
df_DML_esc1 <- df_dml_estimation(df_esc1, 1)
df_DML_esc2 <- df_dml_estimation(df_esc2, 2)
df_DML_esc3 <- df_dml_estimation(df_esc3, 3)
df_DML_esc4 <- df_dml_estimation(df_esc4, 4)

############## 3) DR plug in ML ############

#### 3.1) Estimación de modelos auxiliares (OR y PS) ###
# Reutilizamos los hiperparámetros óptimos ya encontrados en la sección anterior de DML.
# Trabajamos sore los mismos escenarios, por lo que los hiperparámetros de los modelos nos sirven también para este caso

# Estimamos los modelos de RF para PS y OR sin sample splitting pero ya tuneados 
# para cada caso correspondiente para luego introducirlos manualmente al estimador DR
# Son los ml_l y ml_m del DML.

# Modelo de resultados
ml_l_lineal$param_set$values
ml_l_nonlineal$param_set$values

# Modelo Propensity Score
ml_m_logit$param_set$values
ml_m_nologit$param_set$values


#### 3.2) Función para el DR plug in con RF ###

# Definimos la función para estimar ATE con DR clásico usando ML plug-in con Random Forest
# y hacer mpas facil las aplicaciones posteriores sobre cada escenario

df_DR_rf <- function(df, escenario) {
  
  # Definimos los ml_l, y ml_m segun el escenario en que estemos
  if (escenario %in% c(1, 3)) {
    ml_l <- ml_l_nonlineal$clone()
    } else {
    ml_l <- ml_l_lineal$clone()
  }
  
  # Selección del modelo PS (ml_m)
  if (escenario %in% c(1, 2)) {
    ml_m <- ml_m_nologit$clone()
  } else {
    ml_m <- ml_m_logit$clone()
  }
  
  # Estimamos mu_1 (E[Y|X, D=1]) con random forest
  # usamos la función do.call para poder pasarle a la función ranger los hiperparámetros obtenidos antes del tuno de los modelos
  modelo_mu1 <- do.call(ranger, 
      c(list(formula = y ~ x1 + x2 + x3 + x4 + x5,
             data = df[df$d == 1, ]),
             ml_l$param_set$values))
  
  mu_1 <- predict(modelo_mu1, data = df)$predictions
  
  # Estimamos mu_0 (E(Y|X's) con random forest
  modelo_mu0 <- do.call(ranger, 
                c(list(formula = y ~ x1 + x2 + x3 + x4 + x5,
                 data = df[df$d == 0, ]),
                 ml_l$param_set$values))
  
  mu_0 <- predict(modelo_mu0, data = df)$predictions
  
  # Estimamos el propensity score P(D=1|X) con random forest
  modelo_ps <- do.call(
    ranger, c(list(formula = as.factor(d) ~ x1 + x2 + x3 + x4 + x5,
        data = df, probability = TRUE), ml_m$param_set$values))
  
  ps_hat <- predict(modelo_ps, data = df)$predictions[, "1"]
  
  # Estimador Doubly Robust (DR con plug-in ML)
  dr_obs_rf <- (df$d * (df$y - mu_1) / ps_hat) -
    ((1 - df$d) * (df$y - mu_0) / (1 - ps_hat)) +
    (mu_1 - mu_0)
  
  # Estimación puntual del ATE
  theta_hat <- mean(dr_obs_rf)
  
  # Desvío estándar y error estándar
  sd <- sd(dr_obs_rf)
  se <- sd / sqrt(nrow(df))
  
  # Intervalo de confianza al 95%
  IC <- theta_hat + c(-1.96, 1.96) * se
  
  # df con resultado final unificado
  resultado_final <- data.frame(
    theta = theta_hat,
    sd = sd,
    se = se,
    IC95_inf = IC[1],
    IC95_sup = IC[2]
  )
  
  return(resultado_final)
}

# Aplicamos la función por escenario
df_DR_rf_esc1 <- df_DR_rf(df_esc1, 1)
df_DR_rf_esc2 <- df_DR_rf(df_esc2)
df_DR_rf_esc3 <- df_DR_rf(df_esc3)
df_DR_rf_esc4 <- df_DR_rf(df_esc4)



################## Simulaciones ##########################

# Vamos a repetir los procesos de DGP y posterior estimación para cada método
# variando la seed aleatoria en cada una de las simulaciones para así tener n estimaciones
# diferentes para cada escenario y método y poder hacer comparaciones más robustas
# A partir de estas tendremos la distribución empírica de los theta_hat estimados y
# veremos si se centran en el verdadero valor 0.5 ais como es su dispersión.
# Vamos a poder comparar cada método en cada escenario de forma rápida también con
# el RMSE promedio y el se promedio.

########### DML ############.
# Simulamos el DML por escenario siguiendo el ejemplo de Bach
simulaciones_dml <- function(escenario, n_base, n_rep){
  theta_dml = rep(NA, n_rep)
  se_dml = rep(NA, n_rep)
  
  # Definimos los learners de nuevo en base al escenario
  if (escenario %in% c(1, 3)) {
    ml_l <- ml_l_nonlineal$clone()
    ml_g <- ml_l_nonlineal$clone()
  } else {
    ml_l <- ml_l_lineal$clone()
    ml_g <- ml_l_lineal$clone()
  }
  
  # Selección del modelo PS (ml_m)
  if (escenario %in% c(1, 2)) {
    ml_m <- ml_m_nologit$clone()
  } else {
    ml_m <- ml_m_logit$clone()
  }
  
  for (i_rep in seq_len(n_rep)) {
    cat(sprintf("Replication %d/%d\n", i_rep, n_rep))
    set.seed(i_rep)
    df = generar_dgp(escenario, n_base)
    obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
    obj_dml_plr = DoubleMLPLR$new(obj_dml_data, ml_l, ml_m, ml_g, n_folds = 2, score = 'IV-type')
    obj_dml_plr$fit()
    theta_dml[i_rep] = obj_dml_plr$coef
    se_dml[i_rep] = obj_dml_plr$se
  }
  
  df_combinado <- data.frame(theta = theta_dml, se = se_dml)
  return(df_combinado)
}

# Ejecutamos las simulaciones en cada escenario
simulaciones_dml_esc1 <- simulaciones_dml(1, 10000, 1000)
simulaciones_dml_esc2 <- simulaciones_dml(2, 10000, 1000)
simulaciones_dml_esc3 <- simulaciones_dml(3, 10000, 1000)
simulaciones_dml_esc4 <- simulaciones_dml(4, 10000, 1000)

getwd()
# Exportamos las simulaciones
#write.xlsx(simulaciones_dml_esc1, "simulaciones_dml_esc1_1000casos.xlsx")
#write.xlsx(simulaciones_dml_esc2, "simulaciones_dml_esc2_1000casos.xlsx")
#write.xlsx(simulaciones_dml_esc3, "simulaciones_dml_esc3_1000casos.xlsx")
#write.xlsx(simulaciones_dml_esc4, "simulaciones_dml_esc4_1000casos.xlsx")

simulaciones_dml_esc4

######### Simulaciones DR paramétrico ############

# Simulamos el DR para n bases aleatorias distintas
simulaciones_DR <- function(escenario, n_base, n_rep){
  theta_dr = rep(NA, n_rep)
  sd_dr = rep(NA, n_rep)
  
  for (i_rep in seq_len(n_rep)){
    set.seed(i_rep)
    df <- generar_dgp(escenario, n_base)
    resultados_DR <- df_DR_parametrico(df)
    theta_dr[i_rep] <- resultados_DR$theta
    sd_dr[i_rep] <- resultados_DR$sd
  }
  
  df_combinado <- data.frame(theta = theta_dr, sd = sd_dr)
  return(df_combinado)
}

simulaciones_dr_esc1 <- simulaciones_DR(1, 10000, 1000)
simulaciones_dr_esc2 <- simulaciones_DR(2, 10000, 1000)
simulaciones_dr_esc3 <- simulaciones_DR(3, 10000, 1000)
simulaciones_dr_esc4 <- simulaciones_DR(4, 10000, 1000)


########## DR plug in RF #########

# Repetimos el proceso pero ahora para el DR plug in RF
simulaciones_DR_rf <- function(escenario, n_base, n_rep){
  theta_rf = rep(NA, n_rep)
  sd_rf = rep(NA, n_rep)
  
  for (i_rep in seq_len(n_rep)){
    set.seed(i_rep)
    df <- generar_dgp(escenario, n_base)
    resultados_rf <- df_DR_rf(df)
    theta_rf[i_rep] <- resultados_rf$theta
    sd_rf[i_rep] <- resultados_rf$sd
  }
  
  df_combinado <- data.frame(theta = theta_rf, sd = sd_rf)
  return(df_combinado)
}

simulaciones_rf_esc1 <- simulaciones_DR_rf(1, 10000, 1000)
simulaciones_rf_esc2 <- simulaciones_DR_rf(2, 10000, 1000)
simulaciones_rf_esc3 <- simulaciones_DR_rf(3, 10000, 1000)
simulaciones_rf_esc4 <- simulaciones_DR_rf(4, 10000, 1000)


############### Descriptivos de las simulaciones ###################

#### Simulaciones DML ####.

# Función para calcular resumen de simulaciones
resumen_simulaciones <- function(df_sim, theta_0) {
  df_sim %>%
    mutate(rmse = sqrt((se^2) + (theta - theta_0)^2)) %>%
    summarise(
      theta_hat = mean(theta),
      se_promedio = mean(se),
      rmse_promedio = mean(rmse),
      sesgo_promedio = mean(theta - theta_0)
    )
}

# Aplicamos a los escenarios
resumen_dml_esc1 <- resumen_simulaciones(simulaciones_dml_esc1, theta_0)
resumen_dml_esc2 <- resumen_simulaciones(simulaciones_dml_esc2, theta_0)
resumen_dml_esc3 <- resumen_simulaciones(simulaciones_dml_esc3, theta_0)
resumen_dml_esc4 <- resumen_simulaciones(simulaciones_dml_esc4, theta_0)



###### Distribuciones de theta estimado ####.
# A partir de los df con las n simulaciones vamos a graficar la distribución
# de las estimaciones de theta

# Distribución de estimaciones (código de BACH textual para este gráfico dee la fuente anteriormente citada, DML)
graficar_densidad_theta_estimado <- function(theta_vals) {
  ggplot(data.frame(theta = theta_vals), aes(x = theta)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "darkgreen", alpha = 0.3) +
    geom_vline(xintercept = theta_0, color = "grey20", linetype = "dashed", linewidth = 1.2) +
    xlim(min(theta_vals) - 0.02, max(theta_vals) + 0.02) +
    xlab(expression(hat(theta))) +
    ylab("Densidad") +
    theme_minimal()
}

# 
graficar_densidad_theta_estimado(simulaciones_dml_esc1$theta)
graficar_densidad_theta_estimado(simulaciones_dml_esc2$theta)
graficar_densidad_theta_estimado(simulaciones_dml_esc3$theta)
graficar_densidad_theta_estimado(simulaciones_dml_esc4$theta)

               
                       
                       
                       
                       
                       
                       
