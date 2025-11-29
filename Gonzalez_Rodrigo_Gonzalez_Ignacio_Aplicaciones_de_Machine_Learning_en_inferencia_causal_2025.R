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
library(MASS)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(ranger)
library(GGally)
library(gridExtra)
library(glmnet)
library(drtmle)
library(stats)
library(purrr)
library(tibble)
library(paradox)
library(future)
library(future.apply)
library(progressr)
library(openxlsx)
library(gbm)
library(dplyr)
library(caret)
library(doParallel)
library(ggplot2)
library(pROC)
library(randomForest)


###################### Data Generation Process (DGP) ###########################

# Siguiendo el enfoque de los autores, el DGP se especifica como un modelo parcialmente lineal:
# 
#   y_i = theta_0 * D_i + g0(x_i) + e_i,   donde   e_i ~ N(0, sd)
#   D_i = m0(x_i) + u_i,                   donde   u_i ~ N(0, sd)
#
# Los covariables x_i siguen una normal multivariada: x_i ~ N(0, Sigma)
# con una matriz de varianzas y covarianzas definida como: Sigma_{k,j} = 0.7^{|j - k|}
# De este modo m0(x_i) reprsenta la función del propesity score y g0 la función del OR.

# Se realizarán 4 escenarios de datos simulados partiendo del modelo parcialmente lineal:

# 1) Modelo no lineal de OR y PS logit con modelo latente no lineal
# 2) Modelo lineal de OR y PS logit con modelo latente no lineal
# 3) Modelo no lineal de OR y PS logit con modelo latente lineal
# 4) Modelo lineal de OR y PS logit con modelo latente lineal

######### Parámetros generales #########.

# Definimos la semilla aleatoria para el DGP
set.seed(123)

# Definimos los valores de nuestros parámetros que vamos a usar para la generación de datos en el DGP
theta_0 <- 0.5 # ATE real
p <- 3 # cantidad de regresores
Sigma <- matrix(0.7, nrow = p, ncol = p) # generamos la matriz de varianzas y covarianzas primero como una mariz de p*p todo de 0.7
Sigma <- Sigma ^ abs(row(Sigma) - col(Sigma)) # redefinimos la matriz anterior para que quede definida como queremos siguiendo el desarrollo de Bach
n <- 10000 # tamaño muestral


############## Funciones para el modelo de resultados (g_0) ###########

# 1) Definimos la función lineal para go

g0_lineal <- function(x) {
  x[, 1] + x[, 2] + x[, 3]
}

# 2) Definimos la función no lineal para go
g0_nolineal <- function(x) {
  (0.5*x[, 1]^2+x[, 3]+x[, 2])
}


############ Funciones para el propensity score (m_0) #############

# 1) Función logística lineal para el ps
m0_logit <- function(x, u, media_x, sd_x) {
  plogis(((x[, 1] + x[, 2] + x[, 3]+u)-media_x)/sd_x)
}

# 2) Forma logística no lineal para el ps
m0_nologit <- function(x, u, media_x, sd_x) {
  plogis(((sin(x[, 1] + x[, 2] + x[, 3]+u))-media_x)/sd_x)
}

################# Visualizamos las funciones a utilizar #######################.

######## Graficamos los PS #########.

# Secuencia en x1
grafico_ps <- function(min_escala_x, max_escala_x){
    x1_vals <- seq(min_escala_x, max_escala_x, length.out = 400)
    Xgrid   <- cbind(x1 = x1_vals, x2 = 0, x3 = 0)
    u_val   <- 0
    
    # Índices
    z_lin <- rowSums(Xgrid) + u_val
    z_nl  <- sin(rowSums(Xgrid) + u_val)
    
    # Centrado y escalado con cada índice
    media_lin <- mean(z_lin); sd_lin <- sd(z_lin)
    media_nl  <- mean(z_nl);  sd_nl  <- sd(z_nl)
    
    # Evaluamos las funciones del DGP
    y_logit_lin <- m0_logit(Xgrid, u_val, media_lin, sd_lin)   # logit(lineal)
    y_logit_sin <- m0_nologit(Xgrid, u_val, media_nl,  sd_nl)    # logit(sen(x))
    
    # Data frame alargado para el plot
    df_plot <- data.frame(
      x1 = rep(x1_vals, 2),
      p  = c(y_logit_sin, y_logit_lin),
      modelo = factor(rep(c("Logit (no lineal)", "Logit (lineal)"),
                          each = length(x1_vals)))
    )
    
    # Gráfico
    grafico <- ggplot(df_plot, aes(x = x1, y = p, color = modelo, linetype = modelo)) +
      geom_line(linewidth = 1.2) +
      labs(
        #title = "Comparación Logit(lineal) vs Logit(sen(x))",
        subtitle = expression(m[0](x)~con~x[2]==0~","~x[3]==0~","~u==0),
        x = "x1",
        y = "Probabilidad de tratamiento",
        color = NULL,
        linetype = NULL
      ) +
      scale_y_continuous(limits = c(0, 1)) +
      scale_color_manual(values = c("Logit (no lineal)" = "navy",
                                    "Logit (lineal)" = "orange")) +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = c(0.02, 0.98),
        legend.justification = c("left", "top")
      )
    
    return(grafico)
}

grafico_ps(-10, 10) # aca son muy diferentes
 
# grafico_ps(-2, 2) # aca no hay diferencia entre las funciones
# grafico_ps(0, 5) # aca son muy diferentes

# Gráfico a exportar
grafico_ps_exp <- grafico_ps(-7,12) # este es el rango que terminamos teniendo de x1+x2+x3

# Obs: es fundamental para la mala especificación vs buena especificación definir bien el rango de x que se usa:
# por ejemplo, si tenemos el rango de -2 a 2 la especificación queda igual para ps en ambos casos
# si x de 0 a 5 es totalmente diferente

ggsave("grafico_ps.png", grafico_ps_exp, width = 6, height = 5, dpi = 300, bg = "white")

#################### Graficamos las OR #########################

grafico_or <- function(min_escala_x, max_escala_x, x2 = 0, x3 = 0,
                       normalizar = FALSE) {
  x1_vals <- seq(min_escala_x, max_escala_x, length.out = 400)
  Xgrid   <- cbind(x1 = x1_vals, x2 = x2, x3 = x3)
  
  # Evaluaciones OR
  y_lin <- g0_lineal(Xgrid)
  y_nl  <- g0_nolineal(Xgrid)
  
  df_plot <- data.frame(
    x1 = rep(x1_vals, 2),
    y  = c(y_lin, y_nl),
    modelo = factor(rep(c("OR (lineal)", "OR (no lineal)"),
                        each = length(x1_vals)),
                    levels = c("OR (lineal)", "OR (no lineal)"))
  )
  
  ggplot(df_plot, aes(x = x1, y = y, color = modelo, linetype = modelo)) +
    geom_line(linewidth = 1.2) +
    labs(
      #title = "Comparación OR lineal vs OR no lineal",
      subtitle = bquote(g[0](x) ~ " con " ~ x[2] == .(x2) ~ "," ~ x[3] == .(x3)),
      x = "x1",
      y = "g0(x)",
      color = NULL,
      linetype = NULL
    ) +
    scale_color_manual(values = c("OR (lineal)" = "lightgreen",
                                  "OR (no lineal)" = "red")) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = c(0.02, 0.98),
      legend.justification = c("left", "top")
    )
}

grafico_or_exp <- grafico_or(-8, 10)

ggsave("grafico_gs.png", grafico_or_exp, width = 6, height = 5, dpi = 300, bg = "white")


################### Función para el DGP ########################

# DGP
generar_dgp <- function(escenario, n = n) {
  x <- MASS::mvrnorm(n, mu = rep(0.7, p), Sigma = Sigma) # Armamos un df con las x's simuladas como una matriz de p columnas y n registros
  colnames(x) <- paste0("x", 1:p) # renombramos las columnas para cada x del df anterior
  
  # obtenemos media y sd para normalizar
  media_x <- mean(x[,1] + x[,2] + x[,3])
  sd_x <- sd(x[,1] + x[,2] + x[,3])
  
  # Determinamos las especificaciones g0 y m0 que van a ser elegidas según el escenario en el que estemos
  # en base a las funciones anteriormente definidas
  g0 <- if (escenario %in% c(1, 3)) g0_nolineal else g0_lineal
  m0 <- if (escenario %in% c(1, 2)) m0_nologit else m0_logit
  
  # Score de trataimento 
  u <- rnorm(mean = 0, sd = 0.4, n=n) # determinamos el error de la asignación al tratamiento para que no sea algo determinístico perfecto y lo definimos como un error normal con media 0 y sd = 0.4
  
  score_latente <- m0(x,u, media_x, sd_x) # asignamos el score siguiendo la forma del modelo 
  q <- median(score_latente) # el punto de corte para ser asignado como tratado o no es la mediana para que las bases queden balanceadas
  
  d <- as.numeric(m0(x,u, media_x, sd_x) > q)
  
  # Modelo de resultados
  e <- rnorm(n, mean = 0, sd = 1) # definimos error del modelo de resultado como una normal
  y <- theta_0 * d + g0(x) + e # definimos y en base al modelo de resultado
  y_2 <- theta_0 * d + g0(x)
  
  data.frame(y = y, d = d, x)
}

min_mmax <- function(df){
  min1 <- min(df$x1+df$x2+df$x3)
  max1 <- max(df$x1+df$x2+df$x3)
  return(c(min1, max1))
}


# Aplicamos la función  para cada uno de los escenarios y obtenemos los datasets simulados para el n definido antes
df_esc1 <- generar_dgp(1, n)  # Escenario 1: g no lineal y m logística no lineal
df_esc2 <- generar_dgp(2, n)  # Escenario 2: g lineal y m logistica no lineal
df_esc3 <- generar_dgp(3, n)  # Escenario 3: g no lineal y m logistica lineal
df_esc4 <- generar_dgp(4, n)  # Escenario 4: g lineal y m logistica lineal

# max(df_esc1$x1)
# min_mmax(df_esc1)
# min_mmax(df_esc2)
# min_mmax(df_esc3)
# min_mmax(df_esc4)
# En los otros escenarios el rango es similar

##################### Estimaciones del ATE ########################

# 1) Estimamos el ATE mediante el DR paramétrico
# 2) Estimamos el ATE mediante el DR plug in con RF
# 3) DML

############## 1) DR paramétrico ############
df_DR_parametrico <- function(df){
  
  # Outcome Regression (OR): se hacen las regresiones lineales separadas por grupo de tratamiento
  modelo_y_tratado <- lm(y ~ x1 + x2 + x3, data = df[df$d == 1, ])
  modelo_y_control <- lm(y ~ x1 + x2 + x3, data = df[df$d == 0, ])
  
  # Predicciones del OR
  mu_1 <- predict(modelo_y_tratado, newdata = df)
  mu_0 <- predict(modelo_y_control, newdata = df)
  
  # Propensity Score (PS) - modelo logit lineal
  modelo_ps <- glm(d ~ x1 + x2 + x3, data = df, family = binomial)
  ps_hat <- predict(modelo_ps, newdata = df, type = "response")
  
  # Obs: fijamos un valor epsilon de 0.01 para los casos donde potencialmente el ps sea 1 o 0 asi no da error
  eps <- 0.01
  
  # Nos cubrimos de los casos que tengan una probabilidad de casi 0 o 1
  ps_hat[ps_hat <= eps] <- eps
  ps_hat[ps_hat >= 1 - eps] <- 1 - eps
  
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

set.seed(123)

# Aplicamos la función de DR por escenario
df_DR_parametrico_esc1 <- df_DR_parametrico(df_esc1)
df_DR_parametrico_esc2 <- df_DR_parametrico(df_esc2)
df_DR_parametrico_esc3 <- df_DR_parametrico(df_esc3)
df_DR_parametrico_esc4 <- df_DR_parametrico(df_esc4)


############## 2) DML ############

############# 2.1) Estimación Modelos auxiliares ############

# Paralelizamos (dejamos 1 núcleo libre y usamos los demás para los entrenamientos)
cl <- makeCluster(max(1, parallel::detectCores() - 1))
registerDoParallel(cl)

set.seed(123)

################################################.
####### Entrenamiento y evaluación OR ##########
################################################.

# Como primer paso vamos a entrenar diferentes modelos para ambos ecenarios
# de OR que tenemos mediante caret y vamos a elegir en base a esto a la familia
# de modelos que vamos a utilizar para la estimación de modelos intermedios OR y PS
# para cada escenario.

#   - Escenario 4: OR lineal
#   - Escenario 1: OR no lineal

# Usamos una base de n=10.000 para entrenar y testear los
# modelos en esta etapa intermedia en donde vamos a seleccionar la familia del modelo
# a elegir y los hiperparámetros óptimos
n_val <- 10000

# Generamos las bases en donde vamos a estimar los modelos

# Modelo lineal
df_val_lineal <- generar_dgp(4, n_val) %>%
  dplyr::select(y, x1, x2, x3)

# Para el modelo no lineal
df_val_no_lineal <- generar_dgp(1, n_val) %>%
  dplyr::select(y, x1, x2, x3)

# Obs: estas bases las partiremos en test y train para el entrenamiento y testeo de los modelos

# Personalizamos las métricas a considerar para los modelos de OR que vamos a considerar
# dentro de la instancia de caret. Vamos a utilizar el RMSE, NRMSE y el R2
metricas_OR <- function(data, lev = NULL, model = NULL) {
  y_obs  <- data$obs
  y_pred <- data$pred
  
  rmse_val  <- sqrt(mean((y_obs - y_pred)^2))
  r2_val    <- cor(y_obs, y_pred)^2
  nrmse_val <- rmse_val / sd(y_obs)  # NRMSE relativo a la dispersión de y
  
  c(
    RMSE     = rmse_val,
    Rsquared = r2_val,
    NRMSE    = nrmse_val
  )
}

# Configuramos la instancia de train con cv repetida 2 veces con 5 folds y con las métricas definidas antes
ctrl_reg <- trainControl(
  method = "repeatedcv", # aplicamos reppeated cv
  number = 5, # usamos 5 folds
  repeats = 2, # repetimos 2 veces el proceso de cv
  summaryFunction = metricas_OR, # usamos las métricas definidas antes
  savePredictions = "final", # solo guardamos los valores finales
  allowParallel = TRUE # permitimos la paralelización para aprovechar todo el poder del equipo y hacer el train más rápido
)

# Definimos los modelos a usar en las evaluaciones con caret
modelos_reg <- c(
  "lm", # regresión lineal clásica
  "glmnet", # regresión penalizada con LAsso/Ridge L1 L2
  "rpart", # árbol de decisión
  "rf", # random forest
  "xgbTree", # extreme gradient boosting
  "knn", # k means para regresión
  "svmRadial" # SVM
)

# Definimos una función de entrenamiento que no se corte con errores usando tryCatch para que,
# en caso de no contar con alguno de los modelos o que alguno de un error, no se corte el flujo de evaluación de caret y siga con los demás
safe_train_reg <- function(method, formula, data, trControl,
                           tuneLength = 10) {
  tryCatch(
    {
      train(
        formula, # Fórmula del modelo (y ~ x1 + x2 + x3)
        data = data, # datos a considerar para el train
        method     = method, # métodos a evaluar en los entrenamientos
        trControl  = trControl, # forma de control para la cv
        tuneLength = tuneLength, # lo fijamos en 10 por defecto para evaluar 10 instancias de tuning de hiperparámetros
        metric     = "RMSE" # caret minimizará el RMSE
      )
    },
    error = function(e) {
      NULL # obs: en caso de error no se corta el código sino que devuelve null y pasa al siguiente
    }
  )
}

##################### Modelo lineal de OR ########################

# Partimos la base de entrenamiento del OR lineal en train/test
set.seed(123) # seguimos usando esta seed para reproducibilidad del ejercicio

# Obtenemos los índices aleatorios para hacer la separación de 80/20 para train/test
idx_lin  <- sample.int(nrow(df_val_lineal), size = floor(0.8 * nrow(df_val_lineal)))

# Separamos los subconjuntos de train y test
train_lin <- df_val_lineal[idx_lin, ]
test_lin  <- df_val_lineal[-idx_lin, ]

# Formula del modelo a estimar: y en función de las x's
form_or <- y ~ x1 + x2 + x3

# Entrenamiento con caret para el OR lineal
ajustes_lin <- lapply(
  modelos_reg, # aplicamos los modelos definidos antes
  safe_train_reg, # usamos la función de entrenamiento que contempla errores
  formula    = form_or, # y~x1+x2+x3
  data       = train_lin, # usamos el subconjunto de train
  trControl  = ctrl_reg  # usamos la instancia del train con cv definida en un inicio
)

# Eliminamos los modelos que hayan dado errores y que por lo tanto queden con valores nulos
ajustes_lin <- Filter(Negate(is.null), ajustes_lin)

# Asignamos los nombres a cada modelo restante entrenado
names(ajustes_lin) <- modelos_reg[seq_along(ajustes_lin)][seq_along(ajustes_lin)]

#######################################.
# Evaluamos los modelos obtenidos 
#######################################.

# Evaluamos los modelos obtenidos en mediante las métricas calculadas para el OR lineal
comp_lin <- resamples(ajustes_lin) # unificamos todos los modelos en un solo objeto de caret para su comparación de resultados
summary(comp_lin)

# Hacemos la comparativa por el RMSE
lb_lin <- summary(comp_lin)$statistics$RMSE
leaderboard_lin <- data.frame(
  model     = rownames(lb_lin),
  RMSE_Mean = lb_lin[,"Mean"],
  row.names = NULL
)

# Ordenamos el listado en función del RMSE
leaderboard_lin <- leaderboard_lin[order(leaderboard_lin$RMSE_Mean), ]
leaderboard_lin

# Comparamos gráficamente las métricas
dotplot(comp_lin, metric = "RMSE")
dotplot(comp_lin, metric = "NRMSE")
dotplot(comp_lin, metric = "Rsquared")

# Guardamos el mejor modelo lineal
mejor_or_lin <- ajustes_lin[["xgbTree"]] #Obs: este es el mejor considerando que no queremos un modelo perfecto lineal
# ya que este se ajusta perfecto al DGP y es un método paramétrico. En el caso no lineal tiene muco error, lo que no lo
#hace fiable para cuando el DGP es no lineal. Agregamos el lm a la comparativa para tenerlo de referencia.
# Nos quedamos por lo tanto con el mejor de los métodos flexibles

################### Modelo OR no lineal ##################

# Repetimos el proceso para el caso de DGP no lineal.

# Partimos train/test
set.seed(123)

# índices aleatorios para la partición 80/20 de train y test
idx_nl  <- sample.int(nrow(df_val_no_lineal), size = floor(0.8 * nrow(df_val_no_lineal)))
train_nl <- df_val_no_lineal[idx_nl, ]
test_nl  <- df_val_no_lineal[-idx_nl, ]

# Entrenamos con caret  el modelo OR no lineal manteniendo los mismos modelos a evaluar,
# misma función de entrenamiento segura, formula y configuración del train mediante cv con 5 folds repetida 2 veces
set.seed(123)

ajustes_nl <- lapply(
  modelos_reg, 
  safe_train_reg,
  formula = form_or,
  data = train_nl,
  trControl = ctrl_reg 
)

ajustes_nl <- Filter(Negate(is.null), ajustes_nl)
names(ajustes_nl) <- modelos_reg[seq_along(ajustes_nl)][seq_along(ajustes_nl)]

#########################################################################.
# Evaluamos los resultados de los modelos obtenidos para el OR no lineal)
#########################################################################.

# Unificamos en un objeto de caret y comparamos
comp_nl <- resamples(ajustes_nl)
summary(comp_nl)

# Ordenamos por el RMSE medio de los entrenamientos
lb_nl <- summary(comp_nl)$statistics$RMSE
leaderboard_nl <- data.frame(
  model = rownames(lb_nl),
  RMSE_Mean = lb_nl[,"Mean"],
  row.names = NULL
)

# Ordenamos por el RMSE
leaderboard_nl <- leaderboard_nl[order(leaderboard_nl$RMSE_Mean), ]
leaderboard_nl

# En este caso como es previsible los modelos de ML performan mejor que el lm que 
# muestra significativamente más sesgo que en el caso lineal

# Comparamos de forma visual todos los modelos
dotplot(comp_nl, metric = "RMSE")
dotplot(comp_nl, metric = "NRMSE")
dotplot(comp_nl, metric = "Rsquared")

# Nos quedamos con el xgboost
mejor_or_nl <- ajustes_nl[["xgbTree"]]

# Podemos ver que en el caso de los modelos OR a nivel general los modelos de boosting
# son los que performan mejor y con mayor estabilidad en cada escenario, tanto en el lineal como en el o lineal.
# xgboost muestra mejor resultado en RMSE como en R2 que rf. A su vez, como es de esperarse, el mdoelo lm performa mejor
# en el escenario lineal al matchear de forma perfecta la forma funcional y~x1+x2+x3 pero no asi en el escenario no lineal en donde
# muestra el mayor error de todos los modelos usados. Ante esto, no sería conveniente usarlo enn un escenario con una forma funcional
# desconocida que no pueda modelarse con fundamento teórico que lo respalde, siendo una alternativa más robusta el xgboost
# para estimar el OR si el tamaño muestral es lo sificientemente grande.

###############################################################################.
########################## Propensity Score Model ##############################
###############################################################################.

# Repetimos el proceso hecho para OR pero para este modelo de clasificación: la idea es 
# comparar como performan diferentes modelos de clasificación para estimar el PS para elegir el mejor modelos posible 
# para estimar la relación d~f(X) a usar en las secciones siguientes.

# Obtenemos bases para hacer el train y test de los modelos.

#   - Escenario 1: índice no lineal (sin(z))
#   - Escenario 4: índice lineal (z)

# Base de PS con índice no lineal
df_val_ps_nl <- generar_dgp(1, n_val) %>%
  dplyr::select(d, x1, x2, x3)

# Base de PS con índice lineal
df_val_ps_lin <- generar_dgp(4, n_val) %>%
  dplyr::select(d, x1, x2, x3)

# Validamos que d tenga de ambas clases y que esté balanceada como se define en el DGP
table(df_val_ps_nl$d) # cumple con ser 5000/5000 los 0/1
table(df_val_ps_lin$d) # cumple con ser 5000/5000 los 0/1

# Modificamos d como factor "No"/"Si" (negativa/positiva) para que la pueda tomar caret
# en sus instacias de entrenamiento
df_val_ps_nl <- df_val_ps_nl %>%
  mutate(d = factor(ifelse(d == 1, "Si", "No"),
                    levels = c("No","Si"))  )

df_val_ps_lin <- df_val_ps_lin %>%
  mutate(d = factor(ifelse(d == 1, "Si", "No"),
                    levels = c("No","Si")))

# Métricas a evaluar en el PS: AUC y Accuracy
# Vamos a centrarnos en el AUC (área bajo la curva de ROC) para la discriminación
# de clases y en la accuracy para precisión general del modelo
metricas_PS <- function(data, lev = NULL, model = NULL) {
  obs <- data$obs
  # clase positiva = lev[2], aca es "Si" por lo definido de antes
  prob_pos <- data[[lev[2]]]
  
  # AUC
  roc_val <- as.numeric(pROC::auc(obs, prob_pos))
  
  # Accuracy
  acc_val <- mean(data$pred == obs)
  
  c(
    ROC = roc_val,
    Accuracy = acc_val
  )
}

# Ajustamos el control del train del modelo para nuestros modelos de clasificacion para el PS
ctrl_cls <- trainControl(
  method  = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = metricas_PS, # referenciamos las métricas que pusimos antes para el ps de AUC y accuracy para que las calcule
  savePredictions = "final",
  allowParallel = TRUE
)

# Usamos la función de entrenamiento seguro para que no se corte con los errores con PS también
safe_train_cls <- function(method, formula, data, trControl,
                           tuneLength = 10) {
  tryCatch(
    {
      train(
        formula, data = data,
        method     = method,
        trControl  = trControl,
        tuneLength = tuneLength,
        metric     = "ROC"  # optimizamos el AUC
      )
    },
    error = function(e) {
      NULL # mantenemos el seguro ante errores como haciamos en el entrenamiento de los OR
    }
  )
}

# Modelo del PS
form_ps <- d ~ x1 + x2 + x3

########################## PS con índice lineal ################################

# Partimos train/test
set.seed(123)

# Obtenemos los índices con los que partiremos la base
idx_ps_lin  <- createDataPartition(df_val_ps_lin$d, p = 0.8, list = FALSE)

# Partimos train/test
train_ps_lin <- df_val_ps_lin[idx_ps_lin, ]
test_ps_lin  <- df_val_ps_lin[-idx_ps_lin, ]

modelos_cls <- c(
  # Lineales
  "glm",         # regresión logística estándar
  "glmnet",      # regresión logística regularizada
  "svmRadial",   # svm radial
  "rf",          # random forest
  "xgbTree",     # gradient boosting extremo
  "knn"         # k vecinos
)


# Asi como para el OR se utilizó el lm para contrastar los resultados con la forma lineal paramétrica,
# aqui en el PS usamos glm para testear la forma paramétrica del logit con los demas modelos.

# Entrenamos con caret tomando las configuraciones previas y los modelos definidos para el PS
set.seed(123)
ajustes_ps_lin <- lapply(
  modelos_cls,
  safe_train_cls,
  formula   = form_ps,
  data      = train_ps_lin,
  trControl = ctrl_cls
)

# Nos quedamos con los modelos que no sean vacíos
ajustes_ps_lin <- Filter(Negate(is.null), ajustes_ps_lin)
names(ajustes_ps_lin) <- modelos_cls[seq_along(ajustes_ps_lin)]

###############################################.
# Evaluación modelos PS lineal con caret
###############################################.

# unificamos resultados en un solo objeto de caret
comp_ps_lin <- resamples(ajustes_ps_lin)
summary(comp_ps_lin)

# Resultados en dotplots por métrica
dotplot(comp_ps_lin, metric = "ROC")
dotplot(comp_ps_lin, metric = "Accuracy")

# Nos quedamos con el xgboost
mejor_ps_lin <- ajustes_ps_lin[["xgbTree"]]

###################################################################.
####################### PS no lineal ##############################
###################################################################.

# Repetimos el proceso para el PS no lineal

# Partimos la base en train/test
set.seed(123)

# Índices para la partición
idx_ps_nl  <- createDataPartition(df_val_ps_nl$d, p = 0.8, list = FALSE)

# Obtenemos train/test data
train_ps_nl <- df_val_ps_nl[idx_ps_nl, ]
test_ps_nl  <- df_val_ps_nl[-idx_ps_nl, ]

# Entrenamos los modelos
set.seed(123)
ajustes_ps_nl <- lapply(
  modelos_cls,
  safe_train_cls,
  formula   = form_ps,
  data      = train_ps_nl,
  trControl = ctrl_cls
)

# Nos quedamos con los modelos no vacíos
ajustes_ps_nl <- Filter(Negate(is.null), ajustes_ps_nl)
names(ajustes_ps_nl) <- modelos_cls[seq_along(ajustes_ps_nl)]

###############################################.
# Evaluación modelos PS no lineal con caret
###############################################.

# Unificamos los resultados en un objeto solo
comp_ps_nl <- resamples(ajustes_ps_nl)
summary(comp_ps_nl)

# Hacemos los dotplots
dotplot(comp_ps_nl, metric = "ROC")
dotplot(comp_ps_nl, metric = "Accuracy")

mejor_ps_nl <- ajustes_ps_nl[["xgbTree"]]

# Con los resultados podemos ver, asi como pasó para el OR que el modelo de boosting
# domina a los demás, siendo este la mejor de las opciones en cuanto a estabilidad
# entre escenarios y precisión para la estimación del modelo PS.

######################## Entrenamiento final ###################################

# En base a los entrenamientos con caret podemos definir que se utilizarán modelos
# de boosting para la estimación de los modelos auxiliares. Procederemos con
# un gbm de regresión para la estimación del OR y con un gbm de clasificación
# para la estimación de PS.
# En esta sección procederemos a tunear los hiperparámetros de los mismos en orden
# de llegar a la versión final de los modelos a utilizar.

# Dotplots de cada modelo
dotplot(comp_ps_nl)
dotplot(comp_ps_lin)

dotplot(comp_lin)
dotplot(comp_lin)

########################################################.
#  Modelos finales y métricas en test (OR)
########################################################.

# OR lineal: métricas en test
pred_lin_test <- predict(mejor_or_lin, newdata = test_lin)
y_lin_test <- test_lin$y
rmse_lin_test <- sqrt(mean((y_lin_test - pred_lin_test)^2))
r2_lin_test <- cor(y_lin_test, pred_lin_test)^2
nrmse_lin_test <- rmse_lin_test / sd(y_lin_test)

# OR no lineal: métricas en test
pred_nl_test <- predict(mejor_or_nl, newdata = test_nl)
y_nl_test <- test_nl$y
rmse_nl_test <- sqrt(mean((y_nl_test - pred_nl_test)^2))
r2_nl_test <- cor(y_nl_test, pred_nl_test)^2
nrmse_nl_test <- rmse_nl_test / sd(y_nl_test)

#############################################.
# Métricas en test PS
#############################################.

# PS logit lineal: métricas en test
pred_ps_lin_test <- predict(mejor_ps_lin, newdata = test_ps_lin)
d_ps_lin_test <- test_ps_lin$d

# PS logit lineal: métricas en test
prob_ps_lin_test <- predict(mejor_ps_lin, newdata = test_ps_lin, type = "prob")[,"Si"]
pred_ps_lin_test <- ifelse(prob_ps_lin_test >= 0.5, "Si", "No")
pred_ps_lin_test <- factor(pred_ps_lin_test, levels = c("No","Si"))

# AUC ps logit lineal
auc_ps_lin_test <- as.numeric(pROC::auc(test_ps_lin$d, prob_ps_lin_test))
# Accuracy ps no lineal
acc_ps_lin_test <- mean(pred_ps_lin_test == test_ps_lin$d)

############# PS logit no lineal #################.

# PS logit no lineal: métricas en test
pred_ps_nl_test <- predict(mejor_ps_nl, newdata = test_ps_nl)
d_ps_nl_test <- test_ps_nl$d

# PS logit no lineal: métricas en test
prob_ps_nl_test <- predict(mejor_ps_nl, newdata = test_ps_nl, type = "prob")[,"Si"]
pred_ps_nl_test <- ifelse(prob_ps_nl_test >= 0.5, "Si", "No")
pred_ps_nl_test <- factor(pred_ps_nl_test, levels = c("No","Si"))

# AUC ps no lineal
auc_ps_nl_test <- as.numeric(pROC::auc(test_ps_nl$d, prob_ps_nl_test))
# Accuracy ps no lineal
acc_ps_nl_test <- mean(pred_ps_nl_test == test_ps_nl$d)

###############
mejor_ps_nl$bestTune
mejor_ps_lin$bestTune
mejor_or_lin$bestTune
mejor_or_nl$bestTune

########## Exportamos los modelos finales ############
# Guardamos un respaldo de los modelos entrenados
#
# saveRDS(mejor_or_lin,
#         file = "C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_or_lineal_final.rds")
# 
# # Exportar OR no lineal
# saveRDS(mejor_or_nl,
#         file = "C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_or_nolineal_final.rds")
# 
# # Exportar PS lineal
# saveRDS(mejor_ps_lin,
#         file = "C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_ps_lineal_final.rds")
# 
# # Exportar PS no lineal
# saveRDS(mejor_ps_nl,
#         file = "C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_ps_nolineal_final.rds")
# 
# 
# # Cargamos los modelos
# mejor_or_lin <- readRDS("C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_or_lineal_final.rds")
# mejor_or_nl <- readRDS("C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_or_nolineal_final.rds")
# mejor_ps_lin <- readRDS("C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_ps_lineal_final.rds")
# mejor_ps_nl <- readRDS("C:/Users/rodri/OneDrive/Desktop/Tesis Ciencia de Datos/modelos_finales/modelo_ps_nolineal_final.rds")

################### Convergencia de modelos ####################

########### Tasa de convergencia de los estimadores ################.

# Para comparar m^ vs el m0 definido en el DGP ponemos u=0 siendo asi
# el caso donde d = m(X) en vez del X+u (usamos u=0). Comparamos los modelos obtnidos para cada n vs
# el modelo teórico perfecto que podríamos llegar para el cálculo de la norma L2. 
sigma_u <- 0

# La idea de este procedimiento es variar el n para las muestras de entrenamiento de los 
# modelos y verificar el L2 de cada uno de ellos en una muestra de validación generada con otra semilla 
# aleatoria diferente a la de entrenamiento y con un tamaño muestra fijo grande. Queremos evaluar
# el comportamiento de como mejoran los modelos al aumentar el n de entrenamiento, 
# no el n de la base en la que se aplican y con esto poder evaluar de manera empírica la tasa
# de convergencia de los modelos en el contexto finito muestral en el que estamos trabajando y
# compararlo con los valores de referencia de la teoría asintótica.

set.seed(2025) # seteamos una semilla aleatoria diferente para el entrenamiento de los modelos con cada n para luego al testearlos sobre la base de validación generada con una seed distinta 
n_pop <- 200000 # tamaño de la base de validación fija donde testearemos cada modelo entrenado con diferentes tamaños muestrales

# Muestra de validación
X_pop <- MASS::mvrnorm(n_pop, mu = rep(0.7, p), Sigma = Sigma) # generamos las x como en el dgp
S_pop <- rowSums(X_pop)  # S = x1 + x2 + x3

# Sacamos la media y sd de esta muestra de validación
media_Spop <- mean(S_pop)
sd_Spop <- sd(S_pop)

# Para el PS no lineal (escenarios 1 y 2):
# score_latente = plogis((sin(S + U) - media_Spop)/sd_Spop)
U_pop <- rnorm(n_pop, mean = 0, sd = sigma_u) #sigma_u queda en 0, esto lo vamos a usar después en la verificación de m0 definida
score_nl_pop <- plogis((sin(S_pop + U_pop) - media_Spop) / sd_Spop) # score logit no lineal definido de validación
q_pop_nl <- median(score_nl_pop)

# Umbral en el índice sin() tal que plogis((sin_thresh - media_Spop)/sd_Spop) = q_pop_nl como se hace en el DGP
sin_thresh <- media_Spop + sd_Spop * qlogis(q_pop_nl) # Obs: lo despejamos para hacer el control sobre el índice sin la media y sd en la comparativa de la función directamente para evitar riesgos
q_pop_lin <- 0.5

######## Definimos los g y m segun el escenario en que se esté ##########.

# Definimos el g
g0_struct <- function(X, escenario) {
  if (escenario %in% c(1, 3)) {
    g0_nolineal(X)
  } else {
    g0_lineal(X)
  }
}

# Definimos el m
m0_real <- function(X, escenario) {
  S <- rowSums(X)
  n <- length(S)
  
  if (escenario %in% c(3, 4)) {
    # PS lineal:
    return(ifelse(plogis(((X[, 1] + X[, 2] + X[, 3])-media_Spop)/sd_Spop)>0.5, 1, 0))
    
  } else if (escenario %in% c(1, 2)) {
    # PS no lineal: m0(X) = P(sin(S) > sin_thresh)
    sin_vals <- sin(S)
    probs <- ifelse(sin_vals > sin_thresh, 1, 0)
    return(probs)
  } 
}

calcular_L2_xgb_DGP <- function(n, escenario, 
                                n_val    = 10000,
                                seed_train = 123, # semilla usada para los entrenamientos de los modelos 
                                seed_val   = 456, # semilla para la validación
                                nrounds_or = 500, # número de árboles a usar en los modelos OR
                                nrounds_ps = 500 # número de árboles a usar en los modelos PS
                                ) {
  
  
  ## Muestra de entrenamiento
  set.seed(seed_train) # seteamos la semilla de entrenamiento para este paso
  df_train <- generar_dgp(escenario, n) # generamos con el dgp la base a usar para el entrenamiento del modelo
  
  # Separamos en vectores cada una de las variables de la base
  X_train <- as.matrix(df_train[, c("x1", "x2", "x3")])
  d_train <- df_train$d
  y_train <- df_train$y
  
  # Obtenemos el g0 estructural E(y | X) = g0(X) como lo tenemos en el DGP
  y_tilde_train <- y_train
  
  dtrain_or <- xgb.DMatrix(data = X_train, label = y_tilde_train)
  dtrain_ps <- xgb.DMatrix(data = X_train, label = d_train)
  
  
  ## Entrenamiento modelo OR
  params_or <- list(
    objective        = "reg:squarederror",
    max_depth        = 3,
    eta              = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  )
  
  modelo_or <- xgb.train(
    params  = params_or,
    data    = dtrain_or,
    nrounds = nrounds_or,
    verbose = 0
  )
  
  ## Entrenamiento modelo PS
  params_ps <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = 3,
    eta              = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  )
  
  modelo_ps <- xgb.train(
    params  = params_ps,
    data    = dtrain_ps,
    nrounds = nrounds_ps,
    verbose = 0
  )
  
  
  ## Generamos la muestra de validación independiente
  set.seed(seed_val) # usamos la semilla especificada diferente a la de train
  df_val <- generar_dgp(escenario, n_val)
  X_val  <- as.matrix(df_val[, c("x1", "x2", "x3")])
  d_val  <- df_val$d
  
  # Obtenemos los g0(X) y  m0(X) de definición
  g_true <- g0_struct(X_val, escenario)
  m_true <- m0_real(X_val, escenario)
  
  ## Obtenemos las predicciones g_hat(X) y m_hat(X)
  g_hat <- predict(modelo_or, newdata = X_val)
  m_hat <- predict(modelo_ps, newdata = X_val)  # prob(D=1|X), objective = "binary:logistic"
  
  ## Hallamos las normas de errores L2: distancia de estimaciones vs formas estructurales
  L2_g <- sqrt(mean((g_hat - g_true)^2))
  L2_m <- sqrt(mean((m_hat - m_true)^2))
  
  # Vemos el AUC también para validar
  auc   <- as.numeric(pROC::auc(d_val, m_hat))
  
  ## Resultado
  list(
    escenario = escenario,
    n         = n,
    L2_g      = L2_g,
    L2_m      = L2_m,
    AUC       = auc,
    modelo_or = modelo_or,
    modelo_ps = modelo_ps
  )
}

# Evaluamos la función de L2 para los diferentes escenarios y tamaños muestrales
# asi podemos usarlo luego para medir la convergencia empírica

# Tamaños muestrales a evaluar
n_vals <- c(100, 1000, 10000, 50000)

# Escenarios de 1 a 4
escs   <- 1:4

# Definimos una lista vacía en donde guardaremos los resultados
resultados_L2_DGP <- list()


# Iteramos entre escs y n_vals para ir calculando los valores de L2 con la función anterior
for (esc in escs) {
  for (n in n_vals) {
    cat("Escenario =", esc, "  n =", n, "\n") # imprimimos el escenario y n para seguimiento
    res <- calcular_L2_xgb_DGP(
      n         = n,
      escenario = esc,
      n_val     = 10000,
      seed_train = 100 + esc + n,
      seed_val   = 999,
      nrounds_or = 500,
      nrounds_ps = 500)
    
    # Guardamos los resultados en un data frame
    resultados_L2_DGP[[length(resultados_L2_DGP) + 1]] <- data.frame(
      escenario = esc,
      n         = n,
      L2_g      = res$L2_g,
      L2_m      = res$L2_m,
      AUC       = res$AUC
    )
  }
}

# Unificamos los resultados en un solo df
resultados_L2_DGP_df <- dplyr::bind_rows(resultados_L2_DGP)
resultados_L2_DGP_df

# Agregamos la norma L2 general como el producto de L2_g y L2_m
resultados_L2_DGP_df <- resultados_L2_DGP_df %>% 
  mutate(L2_general = L2_g*L2_m)

############ Tasas de convergencia ##################

## Función de tasa de convergencia
tasa_convergencia <- function(e1, e2, n1, n2) {
  log(e1 / e2) / log(n2 / n1)
}

# Tamaños muestrales a evaluar
n1 <- 100
n2 <- 1000
n3 <- 10000
n4 <- 50000

# Hallamos las tasas de convergencia para cada pasaje de n al siguiente que queremos mostrar
tasas_convergencia <- resultados_L2_DGP_df %>%
  group_by(escenario) %>%
  summarise(
    # Tasas para la norma general
    alpha_general_100_1000 = tasa_convergencia(
      L2_general[n == n1],
      L2_general[n == n2],
      n1, n2
    ),
    alpha_general_1000_10000 = tasa_convergencia(
      L2_general[n == n2],
      L2_general[n == n3],
      n2, n3
    ),
    alpha_general_10000_50000 = tasa_convergencia(
      L2_general[n == n3],
      L2_general[n == n4],
      n3, n4
    ),
    
    # Tasas para g
    alpha_g_100_1000 = tasa_convergencia(
      L2_g[n == n1],
      L2_g[n == n2],
      n1, n2
    ),
    alpha_g_1000_10000 = tasa_convergencia(
      L2_g[n == n2],
      L2_g[n == n3],
      n2, n3
    ),
    alpha_g_10000_50000 = tasa_convergencia(
      L2_g[n == n3],
      L2_g[n == n4],
      n3, n4
    ),
    
    # Tasas para m
    alpha_m_100_1000 = tasa_convergencia(
      L2_m[n == n1],
      L2_m[n == n2],
      n1, n2
    ),
    alpha_m_1000_10000 = tasa_convergencia(
      L2_m[n == n2],
      L2_m[n == n3],
      n2, n3
    ),
    alpha_m_10000_50000 = tasa_convergencia(
      L2_m[n == n3],
      L2_m[n == n4],
      n3, n4
    ),
    .groups = "drop"
  )

tasas_convergencia
df_tasas_conv_por_n <- data.frame(tasas_convergencia)

#Obs: referencias de benchmark teórico para funciones de score ortogonalizadas: 1/sqrt(n)
valor1 <- 1/sqrt(100)
valor2 <- 1/sqrt(1000)
valor3 <- 1/sqrt(10000)

(log(valor1)-log(valor2))/(log(1000)-log(100))

################## Gráficos ####################.

# Vector de tamaños muestrales
n_vals <- c(100, 1000, 10000, 50000)

# Construimos el benchmark 1/sqrt(n) y lo ponemos en el mismo df
plot_df <- resultados_L2_DGP_df %>%
  mutate(
    L2_bench = 1 / sqrt(n)  # benchmark teórico de scores ortogonales (DML)
  ) %>%
  dplyr::select(escenario, n, L2_general, L2_bench) %>%
  pivot_longer(
    cols      = c(L2_general, L2_bench),
    names_to  = "tipo",
    values_to = "L2"
  ) %>%
  mutate(
    tipo = dplyr::recode(
      tipo,
      L2_general = "L2 Empírico",
      L2_bench   = "Benchmark: n^(-1/2)"
    )
  )

# Gráfico L2 empírico vs benchmark DML en función de n
convergencia_DML <- ggplot(plot_df, aes(x = n, y = L2,
                                        group = tipo, color = tipo)) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(
    ~ escenario,
    nrow = 2,
    labeller = as_labeller(function(x) paste0("Escenario ", x))
  ) +
  scale_x_log10(
    breaks = n_vals,
    labels = n_vals
  ) +
  labs(
    x = "Tamaño muestral n",
    y = "Norma L2",
    color = NULL
    #,title = "Evolución de L2_general y benchmark 1/sqrt(n) por escenario"
  ) +
  theme_minimal(base_size = 13)


convergencia_DML

1/sqrt(100000)
(10000)^(-1/4)


n_vals <- sort(unique(resultados_L2_DGP_df$n))

### L2_m vs n^(-1/4) #######.
plot_m_df <- resultados_L2_DGP_df %>%
  mutate(
    bench_n14 = 1 / (n^(1/4))  # n^(-1/4)
  ) %>%
  dplyr::select(escenario, n, L2_m, bench_n14) %>%
  pivot_longer(
    cols      = c(L2_m, bench_n14),
    names_to  = "tipo",
    values_to = "valor"
  ) %>%
  mutate(
    tipo = dplyr::recode(
      tipo,
      L2_m     = "L2_m empírico",
      bench_n14 = "Benchmark n^(-1/4)"
    )
  )

# Graficamos como es la evolución de los L2 de m
convergencia_L2_m <- ggplot(plot_m_df, aes(x = n, y = valor,
                                           group = tipo, color = tipo)) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(
    ~ escenario,
    nrow = 2,
    labeller = as_labeller(function(x) paste0("Escenario ", x))
  ) +
  scale_x_log10(
    breaks = n_vals,
    labels = n_vals
  ) +
  labs(
    x = "Tamaño muestral n",
    y = "Valor",
    color = NULL
    #,
    #title = "L2_m vs benchmark n^(-1/4) por escenario"
  ) +
  theme_minimal(base_size = 13)

convergencia_L2_m

# Graficamos como es la evolución de los L2 de g
convergencia_L2_g <- ggplot(plot_g_df, aes(x = n, y = valor,
                                           group = tipo, color = tipo)) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(
    ~ escenario,
    nrow = 2,
    labeller = as_labeller(function(x) paste0("Escenario ", x))
  ) +
  scale_x_log10(
    breaks = n_vals,
    labels = n_vals
  ) +
  labs(
    x = "Tamaño muestral n",
    y = "Valor",
    color = NULL
    #,
    #title = "L2_g vs benchmark n^(-1/4) por escenario"
  ) +
  theme_minimal(base_size = 13)

convergencia_L2_g

# Exportamos los gráficos
ggsave("convergencia_L2_g.png", convergencia_L2_g, width = 9, height = 4, dpi = 300, bg = "white")
ggsave("convergencia_L2_m.png", convergencia_L2_m, width = 9, height = 4, dpi = 300, bg = "white")
ggsave("convergencia_L2_general.png", convergencia_DML, width = 9, height = 4, dpi = 300, bg = "white")


####### Guardamos la estructura de los modelos con los hiperparámetros definidos para ser usados en el DML #########

###############################.
# Learners mlr3 con xgbtree
###############################.

# Guardamos la estructura de los learners a ser implementados con ml3 en DML

# OR no lineal
ml_l_nonlineal <- lrn(
  "regr.xgboost",
  nrounds          = 250,
  max_depth        = 2,
  eta              = 0.3,
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.5555556
)

# OR lineal
ml_l_lineal <- lrn(
  "regr.xgboost",
  nrounds          = 100,
  max_depth        = 2,
  eta              = 0.3,
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.9444444
)

# PS no lineal
ml_m_nologit <- lrn(
  "classif.xgboost",
  predict_type     = "prob",
  nrounds          = 100,
  max_depth        = 2,
  eta              = 0.4,
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 1
)

# PS lineal
ml_m_logit <- lrn(
  "classif.xgboost",
  predict_type     = "prob",
  nrounds          = 100,
  max_depth        = 1,
  eta              = 0.4,
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.5
)

# Modelo de resultados
ml_l_lineal$param_set$values
ml_l_nonlineal$param_set$values

# Modelo Propensity Score
ml_m_logit$param_set$values
ml_m_nologit$param_set$values

############# 3.1) Estimación ATE con DML ############

# Citamos a Bach 2022
# El código siguiente lo obtenemos de ajustar el material de Bach (2022) disponible en su sitio web DML
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
  mse  <- (theta_hat - theta_0)^2
  rmse <- sqrt(mse)
  
  # Guardamos los resultados en un df final
  resultado <- data.frame(
    theta = theta_hat,
    sd = sd,
    se = se,
    rmse,
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
# Reutilizamos los hiperparámetros ya usados en la sección anterior de DML.
# Trabajamos sobre los mismos escenarios, por lo que los hiperparámetros de los modelos nos sirven también para este caso

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

# Definimos la función para estimar ATE con ML plug-in con xgboost
# y hacer mas facil las aplicaciones posteriores sobre cada escenario

df_DR_rf <- function(df, escenario) {
  
  # Modelo OR
  ml_l <- if (escenario %in% c(1, 3)) {
    ml_l_nonlineal$clone()
  } else {
    ml_l_lineal$clone()
  }
  
  # PS
  ml_m <- if (escenario %in% c(1, 2)) {
    ml_m_nologit$clone()
  } else {
    ml_m_logit$clone()
  }
  
  
  # Estimamos mu1(X): E[Y|X, D=1]
  # Sacamos a d del dataset y dejamos solo los tratados
  df1 <- df[df$d == 1, c("y", "x1", "x2", "x3")]
  
  # Aplicamos el OR para los tratados
  task_mu1 <- TaskRegr$new(
    id      = "mu1",
    backend = df1,
    target  = "y"
  )
  
  # Fit del modelo
  fit_mu1 <- ml_l$train(task_mu1)
  mu_1 <- fit_mu1$predict_newdata(df[, c("x1", "x2", "x3")])$response
  
  # Modelo OR para los controles:
  # mu_0: E[Y | X, D=0]
  df0 <- df[df$d == 0, c("y", "x1", "x2", "x3")]
  
  # Entrenamos el modelo  
  task_mu0 <- TaskRegr$new(
    id      = "mu0",
    backend = df0,
    target  = "y"
  )
  
  fit_mu0 <- ml_l$train(task_mu0)
  mu_0 <- fit_mu0$predict_newdata(df[, c("x1", "x2", "x3")])$response
  
  # Liberamos memoria intermedia para no sobrecargar la ram
  rm(task_mu1, task_mu0, fit_mu1, fit_mu0); gc()
  
  # Propensity score: m(X) = P(D=1|X)
  df_ps <- df[, c("d", "x1", "x2", "x3")]
  df_ps$d <- factor(df_ps$d, levels = c(0, 1))
  
  task_ps <- TaskClassif$new(
    id      = "ps",
    backend = df_ps,
    target  = "d"
  )
  
  fit_ps  <- ml_m$train(task_ps)
  pred_ps <- fit_ps$predict_newdata(df_ps)
  ps_hat  <- pred_ps$prob[, "1"]
  
  rm(task_ps, fit_ps, pred_ps); gc()
  
  # valor epsilon para truncamiento de los ps
  eps = 0.01
  
  # Evitamos los casos que ps_hat = 0 o 1 que generan divisiones por 0
  ps_hat <- pmin(pmax(ps_hat, eps), 1 - eps) # truncamos los ps para que no sea 0 ni 1
  
  # Estimador Doubly Robust con las predicciones de los modelos
  d <- df$d
  y <- df$y
  
  # Calculamos el dr con ml
  dr_obs_rf <- (d*(y - mu_1) / ps_hat) - ((1 - d) * (y - mu_0) / (1 - ps_hat)) + (mu_1 - mu_0)
  
  # guardamos el valor de theta predicho promedio, sd y se
  theta_hat <- mean(dr_obs_rf)
  sd_hat    <- sd(dr_obs_rf)
  se_hat    <- sd_hat / sqrt(length(dr_obs_rf))
  
  IC <- theta_hat + c(-1.96, 1.96) * se_hat
  
  data.frame(
    theta    = theta_hat,
    sd       = sd_hat,
    se       = se_hat,
    rmse = rmse,
    IC95_inf = IC[1],
    IC95_sup = IC[2]
  )
}

# Aplicamos la función por escenario
df_DR_rf_esc1 <- df_DR_rf(df_esc1, 1)
df_DR_rf_esc2 <- df_DR_rf(df_esc2, 2)
df_DR_rf_esc3 <- df_DR_rf(df_esc3, 3)
df_DR_rf_esc4 <- df_DR_rf(df_esc4, 4)

################## Simulaciones ##########################

# Vamos a repetir los procesos de DGP y posterior estimación para cada método
# variando la seed aleatoria en cada una de las simulaciones para así tener n estimaciones
# diferentes para cada escenario y método y poder hacer comparaciones más robustas
# A partir de estas tendremos la distribución empírica de los theta_hat estimados y
# veremos si se centran en el verdadero valor 0.5 asi como es su dispersión.
# Vamos a poder comparar cada método en cada escenario de forma rápida también con
# el RMSE y el se de las simulaciones de theta.

########### Simulaciones DML ############
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
simulaciones_dml_esc1 <- simulaciones_dml(1, 10000, 100)
simulaciones_dml_esc2 <- simulaciones_dml(2, 10000, 100)
simulaciones_dml_esc3 <- simulaciones_dml(3, 10000, 100)
simulaciones_dml_esc4 <- simulaciones_dml(4, 10000, 100)


mean(simulaciones_dml_esc1$theta)
mean(simulaciones_dml_esc2$theta)
mean(simulaciones_dml_esc3$theta)
mean(simulaciones_dml_esc4$theta)

getwd()

# Exportamos las simulaciones
# write.xlsx(simulaciones_dml_esc1, "simulaciones_dml_esc1_100casos.xlsx")
# write.xlsx(simulaciones_dml_esc2, "simulaciones_dml_esc2_100casos.xlsx")
# write.xlsx(simulaciones_dml_esc3, "simulaciones_dml_esc3_100casos.xlsx")
# write.xlsx(simulaciones_dml_esc4, "simulaciones_dml_esc4_100casos.xlsx")


# Mostramos resultados de simulaciones DML

df_all_dml <- bind_rows(
  simulaciones_dml_esc1 %>% transmute(theta, base = "Escenario 1"),
  simulaciones_dml_esc2 %>% transmute(theta, base = "Escenario 2"),
  simulaciones_dml_esc3 %>% transmute(theta, base = "Escenario 3"),
  simulaciones_dml_esc4 %>% transmute(theta, base = "Escenario 4")
)

theta_0
theta0 <- 0.5
# Gráfico: densidades superpuestas
ggplot(df_all_dml, aes(x = theta, color = base)) +
  geom_density(linewidth = 1.1, adjust = 1, na.rm = TRUE) +
  geom_density(aes(fill = base), alpha = 0.15, color = NA, adjust = 1, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed") +
  labs(
    title = expression("Distribuciones superpuestas de " * hat(theta)),
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Base",
    fill  = "Base"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")


met <- function(df, esc) {
  df %>%
    summarise(
      mean_theta = mean(theta, na.rm=TRUE),
      bias       = mean(theta - theta0, na.rm=TRUE),
      abs_bias   = mean(abs(theta - theta0), na.rm=TRUE),
      sd_theta   = sd(theta, na.rm=TRUE),
      rmse       = sqrt(mean((theta - theta0)^2, na.rm=TRUE))
    ) %>% mutate(escenario = esc, .before = 1)
}

resumen_dml <- bind_rows(
  met(simulaciones_dml_esc1, "Escenario 1"),
  met(simulaciones_dml_esc2, "Escenario 2"),
  met(simulaciones_dml_esc3, "Escenario 3"),
  met(simulaciones_dml_esc4, "Escenario 4")
) %>% arrange(abs_bias)

resumen_dml


######### Simulaciones DR paramétrico ############

# Simulamos el DR para n bases aleatorias distintas generadas con el dgp
simulaciones_DR <- function(escenario, n_base, n_rep){
  theta_dr = rep(NA, n_rep)
  sd_dr = rep(NA, n_rep)
  rmse_dr = rep(NA, n_rep)
  
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

simulaciones_dr_esc1 <- simulaciones_DR(1, 10000, 100)
simulaciones_dr_esc2 <- simulaciones_DR(2, 10000, 100)
simulaciones_dr_esc3 <- simulaciones_DR(3, 10000, 100)
simulaciones_dr_esc4 <- simulaciones_DR(4, 10000, 100)

cant_simulaciones <- length(simulaciones_dr_esc1$theta)
raiz_cant_simulaciones <- sqrt(length(simulaciones_dr_esc1$theta))

######## DR paramétrico #########.
media_dr_esc1 <- mean(simulaciones_dr_esc1$theta)
media_dr_esc2 <- mean(simulaciones_dr_esc2$theta)
media_dr_esc3 <- mean(simulaciones_dr_esc3$theta)
media_dr_esc4 <- mean(simulaciones_dr_esc4$theta)

se_sims_1_dr <- sd(simulaciones_dr_esc1$theta)/raiz_cant_simulaciones
se_sims_2_dr <- sd(simulaciones_dr_esc2$theta)/raiz_cant_simulaciones
se_sims_3_dr <- sd(simulaciones_dr_esc3$theta)/raiz_cant_simulaciones
se_sims_4_dr <- sd(simulaciones_dr_esc4$theta)/raiz_cant_simulaciones

rmse_sims_1_dr <- sqrt(mean((simulaciones_dr_esc1$theta - theta0)^2))
rmse_sims_2_dr <- sqrt(mean((simulaciones_dr_esc2$theta - theta0)^2))
rmse_sims_3_dr <- sqrt(mean((simulaciones_dr_esc3$theta - theta0)^2))
rmse_sims_4_dr <- sqrt(mean((simulaciones_dr_esc4$theta - theta0)^2))


######## DML #########.
media_dml_esc1 <- mean(simulaciones_dml_esc1$theta)
media_dml_esc2 <- mean(simulaciones_dml_esc2$theta)
media_dml_esc3 <- mean(simulaciones_dml_esc3$theta)
media_dml_esc4 <- mean(simulaciones_dml_esc4$theta)

se_sims_1_dml <- sd(simulaciones_dml_esc1$theta)/raiz_cant_simulaciones
se_sims_2_dml <- sd(simulaciones_dml_esc2$theta)/raiz_cant_simulaciones
se_sims_3_dml <- sd(simulaciones_dml_esc3$theta)/raiz_cant_simulaciones
se_sims_4_dml <- sd(simulaciones_dml_esc4$theta)/raiz_cant_simulaciones

rmse_sims_1_dml <- sqrt(mean((simulaciones_dml_esc1$theta - theta0)^2))
rmse_sims_2_dml <- sqrt(mean((simulaciones_dml_esc2$theta - theta0)^2))
rmse_sims_3_dml <- sqrt(mean((simulaciones_dml_esc3$theta - theta0)^2))
rmse_sims_4_dml <- sqrt(mean((simulaciones_dml_esc4$theta - theta0)^2))

######## DR con ML #########.
media_dr_rf_esc1 <- mean(simulaciones_rf_esc1$theta)
media_dr_rf_esc2 <- mean(simulaciones_rf_esc2$theta)
media_dr_rf_esc3 <- mean(simulaciones_rf_esc3$theta)
media_dr_rf_esc4 <- mean(simulaciones_rf_esc4$theta)

se_sims_1_dr_rf <- sd(simulaciones_rf_esc1$theta)/raiz_cant_simulaciones
se_sims_2_dr_rf <- sd(simulaciones_rf_esc2$theta)/raiz_cant_simulaciones
se_sims_3_dr_rf <- sd(simulaciones_rf_esc3$theta)/raiz_cant_simulaciones
se_sims_4_dr_rf <- sd(simulaciones_rf_esc4$theta)/raiz_cant_simulaciones

rmse_sims_1_dr_rf <- sqrt(mean((simulaciones_rf_esc1$theta - theta0)^2))
rmse_sims_2_dr_rf <- sqrt(mean((simulaciones_rf_esc2$theta - theta0)^2))
rmse_sims_3_dr_rf <- sqrt(mean((simulaciones_rf_esc3$theta - theta0)^2))
rmse_sims_4_dr_rf <- sqrt(mean((simulaciones_rf_esc4$theta - theta0)^2))

# Construimos el data frame final con todas las métricas
tabla_resultados <- data.frame(
  Metodo = rep(c("DML", "DR plug-in ML", "DR paramétrico"), each = 4),
  Escenario = rep(1:4, times = 3),
  
  ### DML ###.
  Media = c(
    media_dml_esc1, media_dml_esc2, media_dml_esc3, media_dml_esc4,
    media_dr_rf_esc1, media_dr_rf_esc2, media_dr_rf_esc3, media_dr_rf_esc4,
    media_dr_esc1, media_dr_esc2, media_dr_esc3, media_dr_esc4
  ),
  
  SE = c(
    se_sims_1_dml, se_sims_2_dml, se_sims_3_dml, se_sims_4_dml,
    se_sims_1_dr_rf, se_sims_2_dr_rf, se_sims_3_dr_rf, se_sims_4_dr_rf,
    se_sims_1_dr, se_sims_2_dr, se_sims_3_dr, se_sims_4_dr
  ),
  
  RMSE = c(
    rmse_sims_1_dml, rmse_sims_2_dml, rmse_sims_3_dml, rmse_sims_4_dml,
    rmse_sims_1_dr_rf, rmse_sims_2_dr_rf, rmse_sims_3_dr_rf, rmse_sims_4_dr_rf,
    rmse_sims_1_dr, rmse_sims_2_dr, rmse_sims_3_dr, rmse_sims_4_dr
  )
)

tabla_resultados


############################################################.

sd(simulaciones_dml_esc1$theta)
sd(simulaciones_dml_esc2$theta)
sd(simulaciones_dml_esc3$theta)
sd(simulaciones_dml_esc4$theta)

sd(simulaciones_rf_esc4$theta)/sqrt(100)


simulaciones_dr_esc1$sesgo_abs <- abs(simulaciones_dr_esc1$theta-0.5)
mean(simulaciones_dr_esc1$sesgo_abs)

simulaciones_dr_esc4$sesgo_abs <- abs(simulaciones_dr_esc4$theta-0.5)
mean(simulaciones_dr_esc4$sesgo_abs)

# Tablas nuevas de resultados
theta0 <- 0.5

# Unimos y etiquetamos
df_all <- bind_rows(
  simulaciones_dr_esc1 %>% transmute(theta, base = "Escenario 1"),
  simulaciones_dr_esc2 %>% transmute(theta, base = "Escenario 2"),
  simulaciones_dr_esc3 %>% transmute(theta, base = "Escenario 3"),
  simulaciones_dr_esc4 %>% transmute(theta, base = "Escenario 4")
)

# Gráfico de densidades superpuestas
ggplot(df_all, aes(x = theta, color = base)) +
  geom_density(linewidth = 1.1, adjust = 1, na.rm = TRUE) +
  geom_density(aes(fill = base), alpha = 0.15, color = NA, adjust = 1, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed") +
  labs(
    title = expression("Distribuciones superpuestas de " * hat(theta)),
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Base",
    fill  = "Base"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")


met <- function(df, esc) {
  df %>%
    summarise(
      mean_theta = mean(theta, na.rm=TRUE),
      bias       = mean(theta - theta0, na.rm=TRUE),
      abs_bias   = mean(abs(theta - theta0), na.rm=TRUE),
      sd_theta   = sd(theta, na.rm=TRUE),
      rmse       = sqrt(mean((theta - theta0)^2, na.rm=TRUE))
    ) %>% mutate(escenario = esc, .before = 1)
}

resumen <- bind_rows(
  met(simulaciones_dr_esc1, "Escenario 1"),
  met(simulaciones_dr_esc2, "Escenario 2"),
  met(simulaciones_dr_esc3, "Escenario 3"),
  met(simulaciones_dr_esc4, "Escenario 4")
) %>% arrange(abs_bias)

resumen

# Evaluamos los theta obtenidos
mean(simulaciones_dr_esc1$theta)
mean(simulaciones_dr_esc2$theta)
mean(simulaciones_dr_esc3$theta)
mean(simulaciones_dr_esc4$theta)

mean(simulaciones_dr_esc1$sd)
mean(simulaciones_dr_esc2$sd)
mean(simulaciones_dr_esc3$sd)
mean(simulaciones_dr_esc4$sd)

########## DR plug in ML #########

# Repetimos el proceso pero ahora para el DR plug in con ML
simulaciones_DR_rf <- function(escenario, n_base, n_rep){
  theta_rf = rep(NA, n_rep)
  sd_rf = rep(NA, n_rep)
  
  for (i_rep in seq_len(n_rep)){
    set.seed(i_rep)
    df <- generar_dgp(escenario, n_base)
    resultados_rf <- df_DR_rf(df, escenario)
    theta_rf[i_rep] <- resultados_rf$theta
    sd_rf[i_rep] <- resultados_rf$sd
    cat(i_rep)
  }
  
  df_combinado <- data.frame(theta = theta_rf, sd = sd_rf)
  return(df_combinado)
}

# Obtenemos las simulaciones con la función de dr con ml para n=10.000
simulaciones_rf_esc1 <- simulaciones_DR_rf(1, 10000, 100)
simulaciones_rf_esc2 <- simulaciones_DR_rf(2, 10000, 100)
simulaciones_rf_esc3 <- simulaciones_DR_rf(3, 10000, 100)
simulaciones_rf_esc4 <- simulaciones_DR_rf(4, 10000, 100)

mean(simulaciones_rf_esc1$theta)
mean(simulaciones_rf_esc2$theta)
mean(simulaciones_rf_esc3$theta)
mean(simulaciones_rf_esc4$theta)

mean(simulaciones_rf_esc1$sd)
mean(simulaciones_rf_esc2$sd)
mean(simulaciones_rf_esc3$sd)
mean(simulaciones_rf_esc4$sd)


####### Resumen DR plug in ML #########.
df_all_dr_ml <- bind_rows(
  simulaciones_rf_esc1 %>% transmute(theta, base = "Escenario 1"),
  simulaciones_rf_esc2 %>% transmute(theta, base = "Escenario 2"),
  simulaciones_rf_esc3 %>% transmute(theta, base = "Escenario 3"),
  simulaciones_rf_esc4 %>% transmute(theta, base = "Escenario 4")
)

# Gráfico de densidades superpuestas
ggplot(df_all_dr_ml, aes(x = theta, color = base)) +
  geom_density(linewidth = 1.1, adjust = 1, na.rm = TRUE) +
  geom_density(aes(fill = base), alpha = 0.15, color = NA, adjust = 1, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed") +
  labs(
    title = expression("Distribuciones superpuestas de " * hat(theta)),
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Base",
    fill  = "Base"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

resumen_dr_ml <- bind_rows(
  met(simulaciones_rf_esc1, "Escenario 1"),
  met(simulaciones_rf_esc2, "Escenario 2"),
  met(simulaciones_rf_esc3, "Escenario 3"),
  met(simulaciones_rf_esc4, "Escenario 4")
) %>% arrange(abs_bias)

################# Resumen cálculos ATE #################.

# DML
mean(simulaciones_dml_esc1$theta)
mean(simulaciones_dml_esc2$theta)
mean(simulaciones_dml_esc3$theta)
mean(simulaciones_dml_esc4$theta)

mean(simulaciones_dml_esc1$se)
mean(simulaciones_dml_esc2$se)
mean(simulaciones_dml_esc3$se)
mean(simulaciones_dml_esc4$se)

# DR plug in ML
mean(simulaciones_rf_esc1$theta)
mean(simulaciones_rf_esc2$theta)
mean(simulaciones_rf_esc3$theta)
mean(simulaciones_rf_esc4$theta)

mean(simulaciones_rf_esc1$sd)/sqrt(10000)
mean(simulaciones_rf_esc2$sd)/sqrt(10000)
mean(simulaciones_rf_esc3$sd)/sqrt(10000)
mean(simulaciones_rf_esc4$sd)/sqrt(10000)

# DR paramétrico
mean(simulaciones_dr_esc1$theta)
mean(simulaciones_dr_esc2$theta)
mean(simulaciones_dr_esc3$theta)
mean(simulaciones_dr_esc4$theta)

mean(simulaciones_dr_esc1$sd)/sqrt(10000)
mean(simulaciones_dr_esc2$sd)/sqrt(10000)
mean(simulaciones_dr_esc3$sd)/sqrt(10000)
mean(simulaciones_dr_esc4$sd)/sqrt(10000)

# Calulamos los p-values de cada uno de los sesgos contra theta real de 0.5

############################################################.
# ttest sesgo DML
############################################################.

# Escenario 1
bias_dml_esc1 <- simulaciones_dml_esc1$theta - theta_0
mean(bias_dml_esc1); sd(bias_dml_esc1)
ttest_dml_esc1 <- t.test(bias_dml_esc1, mu = 0)
ttest_dml_esc1$p.value

# Escenario 2
bias_dml_esc2 <- simulaciones_dml_esc2$theta - theta_0
mean(bias_dml_esc2); sd(bias_dml_esc2)
ttest_dml_esc2 <- t.test(bias_dml_esc2, mu = 0)
ttest_dml_esc2$p.value

# Escenario 3
bias_dml_esc3 <- simulaciones_dml_esc3$theta - theta_0
mean(bias_dml_esc3); sd(bias_dml_esc3)
ttest_dml_esc3 <- t.test(bias_dml_esc3, mu = 0)
ttest_dml_esc3$p.value

# Escenario 4
bias_dml_esc4 <- simulaciones_dml_esc4$theta - theta_0
mean(bias_dml_esc4); sd(bias_dml_esc4)
ttest_dml_esc4 <- t.test(bias_dml_esc4, mu = 0)
ttest_dml_esc4$p.value


############################################################.
# ttest sesgo DR plug in ML
############################################################.

# Escenario 1
bias_rf_esc1 <- simulaciones_rf_esc1$theta - theta_0
mean(bias_rf_esc1); sd(bias_rf_esc1)
ttest_rf_esc1 <- t.test(bias_rf_esc1, mu = 0)
ttest_rf_esc1$p.value

# Escenario 2
bias_rf_esc2 <- simulaciones_rf_esc2$theta - theta_0
mean(bias_rf_esc2); sd(bias_rf_esc2)
ttest_rf_esc2 <- t.test(bias_rf_esc2, mu = 0)
ttest_rf_esc2$p.value

# Escenario 3
bias_rf_esc3 <- simulaciones_rf_esc3$theta - theta_0
mean(bias_rf_esc3); sd(bias_rf_esc3)
ttest_rf_esc3 <- t.test(bias_rf_esc3, mu = 0)
ttest_rf_esc3$p.value

# Escenario 4
bias_rf_esc4 <- simulaciones_rf_esc4$theta - theta_0
mean(bias_rf_esc4); sd(bias_rf_esc4)
ttest_rf_esc4 <- t.test(bias_rf_esc4, mu = 0)
ttest_rf_esc4$p.value


############################################################.
# ttest sesgo DR paramétrico
############################################################.

# Escenario 1
bias_dr_esc1 <- simulaciones_dr_esc1$theta - theta_0
mean(bias_dr_esc1); sd(bias_dr_esc1)
ttest_dr_esc1 <- t.test(bias_dr_esc1, mu = 0)
ttest_dr_esc1$p.value

# Escenario 2
bias_dr_esc2 <- simulaciones_dr_esc2$theta - theta_0
mean(bias_dr_esc2); sd(bias_dr_esc2)
ttest_dr_esc2 <- t.test(bias_dr_esc2, mu = 0)
ttest_dr_esc2$p.value

# Escenario 3
bias_dr_esc3 <- simulaciones_dr_esc3$theta - theta_0
mean(bias_dr_esc3); sd(bias_dr_esc3)
ttest_dr_esc3 <- t.test(bias_dr_esc3, mu = 0)
ttest_dr_esc3$p.value

# Escenario 4
bias_dr_esc4 <- simulaciones_dr_esc4$theta - theta_0
mean(bias_dr_esc4); sd(bias_dr_esc4)
ttest_dr_esc4 <- t.test(bias_dr_esc4, mu = 0)
ttest_dr_esc4$p.value


# p-values por escenario y método
tabla_pvalues <- data.frame(
  Metodo = c(
    "DML", "DML", "DML", "DML",
    "DR plug-in ML", "DR plug-in ML", "DR plug-in ML", "DR plug-in ML",
    "DR paramétrico", "DR paramétrico", "DR paramétrico", "DR paramétrico"
  ),
  Escenario = rep(1:4, 3),
  p_value = c(
    ttest_dml_esc1$p.value,
    ttest_dml_esc2$p.value,
    ttest_dml_esc3$p.value,
    ttest_dml_esc4$p.value,
    ttest_rf_esc1$p.value,
    ttest_rf_esc2$p.value,
    ttest_rf_esc3$p.value,
    ttest_rf_esc4$p.value,
    ttest_dr_esc1$p.value,
    ttest_dr_esc2$p.value,
    ttest_dr_esc3$p.value,
    ttest_dr_esc4$p.value
  )
)

tabla_pvalues

############## Simulaciones con distinto n #############

# Para hacer el análisis de como varía cada método según el tamaño muestral disponible con el
# que se entrenan los modelos y sobre los que se aplica cada método realizamos las simulaciones
# variando el n en diferentes tamaños para evaluar los resultados por escenario y método para cada n y
# ver la sensibilidad de cada uno de ellos ante variaciones en el tamaño muestral.

# Se hacen 100 repeticiones por cada combinación de escenario, tamaño muestral y método.

########## DML ###########

# Escenario 1
simulaciones_DML_nbase10_1000reps <- simulaciones_dml(1, 10, 100)
simulaciones_DML_nbase50_1000reps <- simulaciones_dml(1, 50, 100)
simulaciones_DML_nbase100_1000reps <- simulaciones_dml(1, 100, 100)
simulaciones_DML_nbase500_1000reps <- simulaciones_dml(1, 500, 100)
simulaciones_DML_nbase1000_1000reps <- simulaciones_dml(1, 1000, 100)
simulaciones_DML_nbase5000_1000reps <- simulaciones_dml(1, 5000, 100)
simulaciones_DML_nbase10000_1000reps <- simulaciones_dml(1, 10000, 100)

# Escenario 2
simulaciones_DML_nbase10_1000reps_esc2 <- simulaciones_dml(2, 10, 100)
simulaciones_DML_nbase50_1000reps_esc2 <- simulaciones_dml(2, 50, 100)
simulaciones_DML_nbase100_1000reps_esc2 <- simulaciones_dml(2, 100, 100)
simulaciones_DML_nbase500_1000reps_esc2 <- simulaciones_dml(2, 500, 100)
simulaciones_DML_nbase1000_1000reps_esc2 <- simulaciones_dml(2, 1000, 100)
simulaciones_DML_nbase5000_1000reps_esc2 <- simulaciones_dml(2, 5000, 100)
simulaciones_DML_nbase10000_1000reps_esc2 <- simulaciones_dml(2, 10000, 100)

# Escenario 3
simulaciones_DML_nbase10_1000reps_esc3 <- simulaciones_dml(3, 10, 100)
simulaciones_DML_nbase50_1000reps_esc3 <- simulaciones_dml(3, 50, 100)
simulaciones_DML_nbase100_1000reps_esc3 <- simulaciones_dml(3, 100, 100)
simulaciones_DML_nbase500_1000reps_esc3 <- simulaciones_dml(3, 500, 100)
simulaciones_DML_nbase1000_1000reps_esc3 <- simulaciones_dml(3, 1000, 100)
simulaciones_DML_nbase5000_1000reps_esc3 <- simulaciones_dml(3, 5000, 100)
simulaciones_DML_nbase10000_1000reps_esc3 <- simulaciones_dml(3, 10000, 100)

# Escenario 4
simulaciones_DML_nbase10_1000reps_esc4 <- simulaciones_dml(4, 10, 100)
simulaciones_DML_nbase50_1000reps_esc4 <- simulaciones_dml(4, 50, 100)
simulaciones_DML_nbase100_1000reps_esc4 <- simulaciones_dml(4, 100, 100)
simulaciones_DML_nbase500_1000reps_esc4 <- simulaciones_dml(4, 500, 100)
simulaciones_DML_nbase1000_1000reps_esc4 <- simulaciones_dml(4, 1000, 100)
simulaciones_DML_nbase5000_1000reps_esc4 <- simulaciones_dml(4, 5000, 100)
simulaciones_DML_nbase10000_1000reps_esc4 <- simulaciones_dml(4, 10000, 100)

################### Simulaciones DR por n #########################.

# Escenario 1
simulaciones_DR_nbase10_1000reps_esc1 <- simulaciones_DR(1, 10, 100)
simulaciones_DR_nbase50_1000reps_esc1 <- simulaciones_DR(1, 50, 100)
simulaciones_DR_nbase100_1000reps_esc1 <- simulaciones_DR(1, 100, 100)
simulaciones_DR_nbase500_1000reps_esc1 <- simulaciones_DR(1, 500, 100)
simulaciones_DR_nbase1000_1000reps_esc1 <- simulaciones_DR(1, 1000, 100)
simulaciones_DR_nbase5000_1000reps_esc1 <- simulaciones_DR(1, 5000, 100)
simulaciones_DR_nbase10000_1000reps_esc1 <- simulaciones_DR(1, 10000, 100)

# Escenario 2
simulaciones_DR_nbase10_1000reps_esc2 <- simulaciones_DR(2, 10, 100)
simulaciones_DR_nbase50_1000reps_esc2 <- simulaciones_DR(2, 50, 100)
simulaciones_DR_nbase100_1000reps_esc2 <- simulaciones_DR(2, 100, 100)
simulaciones_DR_nbase500_1000reps_esc2 <- simulaciones_DR(2, 500, 100)
simulaciones_DR_nbase1000_1000reps_esc2 <- simulaciones_DR(2, 1000, 100)
simulaciones_DR_nbase5000_1000reps_esc2 <- simulaciones_DR(2, 5000, 100)
simulaciones_DR_nbase10000_1000reps_esc2 <- simulaciones_DR(2, 10000, 100)

# Escenario 3
simulaciones_DR_nbase10_1000reps_esc3 <- simulaciones_DR(3, 10, 100)
simulaciones_DR_nbase50_1000reps_esc3 <- simulaciones_DR(3, 50, 100)
simulaciones_DR_nbase100_1000reps_esc3 <- simulaciones_DR(3, 100, 100)
simulaciones_DR_nbase500_1000reps_esc3 <- simulaciones_DR(3, 500, 100)
simulaciones_DR_nbase1000_1000reps_esc3 <- simulaciones_DR(3, 1000, 100)
simulaciones_DR_nbase5000_1000reps_esc3 <- simulaciones_DR(3, 5000, 100)
simulaciones_DR_nbase10000_1000reps_esc3 <- simulaciones_DR(3, 10000, 100)

# Escenario 4
simulaciones_DR_nbase10_1000reps_esc4 <- simulaciones_DR(4, 10, 100)
simulaciones_DR_nbase50_1000reps_esc4 <- simulaciones_DR(4, 50, 100)
simulaciones_DR_nbase100_1000reps_esc4 <- simulaciones_DR(4, 100, 100)
simulaciones_DR_nbase500_1000reps_esc4 <- simulaciones_DR(4, 500, 100)
simulaciones_DR_nbase1000_1000reps_esc4 <- simulaciones_DR(4, 1000, 100)
simulaciones_DR_nbase5000_1000reps_esc4 <- simulaciones_DR(4, 5000, 100)
simulaciones_DR_nbase10000_1000reps_esc4 <- simulaciones_DR(4, 10000, 100)


############## Simulaciones para distinto n DR plug in ML ##################

# Escenario 1
simulaciones_DR_rf_nbase10_1000reps_esc1 <- simulaciones_DR_rf(1, 10, 100)
simulaciones_DR_rf_nbase50_1000reps_esc1 <- simulaciones_DR_rf(1, 50, 100)
simulaciones_DR_rf_nbase100_1000reps_esc1 <- simulaciones_DR_rf(1, 100, 100)
simulaciones_DR_rf_nbase500_1000reps_esc1 <- simulaciones_DR_rf(1, 500, 100)
simulaciones_DR_rf_nbase1000_1000reps_esc1 <- simulaciones_DR_rf(1, 1000, 100)
simulaciones_DR_rf_nbase5000_1000reps_esc1 <- simulaciones_DR_rf(1, 5000, 100)
simulaciones_DR_rf_nbase10000_1000reps_esc1 <- simulaciones_DR_rf(1, 10000, 100)


# Escenario 2
simulaciones_DR_rf_nbase10_1000reps_esc2 <- simulaciones_DR_rf(2, 10, 100)
simulaciones_DR_rf_nbase50_1000reps_esc2 <- simulaciones_DR_rf(2, 50, 100)
simulaciones_DR_rf_nbase100_1000reps_esc2 <- simulaciones_DR_rf(2, 100, 100)
simulaciones_DR_rf_nbase500_1000reps_esc2 <- simulaciones_DR_rf(2, 500, 100)
simulaciones_DR_rf_nbase1000_1000reps_esc2 <- simulaciones_DR_rf(2, 1000, 100)
simulaciones_DR_rf_nbase5000_1000reps_esc2 <- simulaciones_DR_rf(2, 5000, 100)
simulaciones_DR_rf_nbase10000_1000reps_esc2 <- simulaciones_DR_rf(2, 10000, 100)

# Escenario 3
simulaciones_DR_rf_nbase10_1000reps_esc3 <- simulaciones_DR_rf(3, 10, 100)
simulaciones_DR_rf_nbase50_1000reps_esc3 <- simulaciones_DR_rf(3, 50, 100)
simulaciones_DR_rf_nbase100_1000reps_esc3 <- simulaciones_DR_rf(3, 100, 100)
simulaciones_DR_rf_nbase500_1000reps_esc3 <- simulaciones_DR_rf(3, 500, 100)
simulaciones_DR_rf_nbase1000_1000reps_esc3 <- simulaciones_DR_rf(3, 1000, 100)
simulaciones_DR_rf_nbase5000_1000reps_esc3 <- simulaciones_DR_rf(3, 5000, 100)
simulaciones_DR_rf_nbase10000_1000reps_esc3 <- simulaciones_DR_rf(3, 10000, 100)

# Escenario 4
simulaciones_DR_rf_nbase10_1000reps_esc4 <- simulaciones_DR_rf(4, 10, 100)
simulaciones_DR_rf_nbase50_1000reps_esc4 <- simulaciones_DR_rf(4, 50, 100)
simulaciones_DR_rf_nbase100_1000reps_esc4 <- simulaciones_DR_rf(4, 100, 100)
simulaciones_DR_rf_nbase500_1000reps_esc4 <- simulaciones_DR_rf(4, 500, 100)
simulaciones_DR_rf_nbase1000_1000reps_esc4 <- simulaciones_DR_rf(4, 1000, 100)
simulaciones_DR_rf_nbase5000_1000reps_esc4 <- simulaciones_DR_rf(4, 5000, 100)
simulaciones_DR_rf_nbase10000_1000reps_esc4 <- simulaciones_DR_rf(4, 10000, 100)

######################## Análisis Resultados #############################

median(simulaciones_dr_esc1$theta)
median(simulaciones_dr_esc2$theta)
median(simulaciones_dr_esc3$theta)
median(simulaciones_dr_esc4$theta)

mean(simulaciones_dml_esc1$theta)
mean(simulaciones_dml_esc2$theta)
mean(simulaciones_dml_esc3$theta)
mean(simulaciones_dml_esc4$theta)

median(simulaciones_rf_esc1$theta)
median(simulaciones_rf_esc2$theta)
median(simulaciones_rf_esc3$theta)
median(simulaciones_rf_esc4$theta)

mean(simulaciones_rf_esc1$theta)
mean(simulaciones_rf_esc2$theta)
mean(simulaciones_rf_esc3$theta)
mean(simulaciones_rf_esc4$theta)

mean(simulaciones_rf_esc1$sd)
mean(simulaciones_rf_esc2$sd)
mean(simulaciones_rf_esc3$sd)
mean(simulaciones_rf_esc4$sd)


############## Densidades simuladas para DR paramétrico ##################.
theta0 <- 0.5

# Unimos y etiquetamos los resultados por escenario
df_all <- bind_rows(
  simulaciones_dr_esc1 %>% transmute(theta, base = "Escenario 1"),
  simulaciones_dr_esc2 %>% transmute(theta, base = "Escenario 2"),
  simulaciones_dr_esc3 %>% transmute(theta, base = "Escenario 3"),
  simulaciones_dr_esc4 %>% transmute(theta, base = "Escenario 4")
)

# Gráfico de densidades superpuestas
ggplot(df_all, aes(x = theta, color = base)) +
  geom_density(linewidth = 1.1, adjust = 1, na.rm = TRUE) +
  geom_density(aes(fill = base), alpha = 0.15, color = NA, adjust = 1, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed") +
  labs(
    title = expression("Distribuciones superpuestas de " * hat(theta)),
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Base",
    fill  = "Base"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")


met <- function(df, esc) {
  df %>%
    summarise(
      mean_theta = mean(theta, na.rm=TRUE),
      bias       = mean(theta - theta0, na.rm=TRUE),
      abs_bias   = mean(abs(theta - theta0), na.rm=TRUE),
      sd_theta   = sd(theta, na.rm=TRUE),
      rmse       = sqrt(mean((theta - theta0)^2, na.rm=TRUE))
    ) %>% mutate(escenario = esc, .before = 1)
}

resumen <- bind_rows(
  met(simulaciones_dr_esc1, "Escenario 1"),
  met(simulaciones_dr_esc2, "Escenario 2"),
  met(simulaciones_dr_esc3, "Escenario 3"),
  met(simulaciones_dr_esc4, "Escenario 4")
) %>% arrange(abs_bias)

resumen

################## Densidades individuales ##################################

###### Distribuciones de theta estimado ####.
# A partir de los df con las n simulaciones vamos a graficar la distribución
# de las estimaciones de theta

# Distribución de estimaciones. Obs: código en base a Bach (2022) para este gráfico, fuente anteriormente citada en estimación de DML y proyecto escrito
graficar_densidad_theta_estimado <- function(theta_vals) {
  ggplot(data.frame(theta = theta_vals), aes(x = theta)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "darkgreen", alpha = 0.3) +
    geom_vline(xintercept = theta_0, color = "grey20", linetype = "dashed", linewidth = 1.2) +
    xlim(min(theta_vals) - 0.02, max(theta_vals) + 0.02) +
    xlab(expression(hat(theta))) +
    ylab("Densidad") +
    theme_minimal()
}

# Gráfico 2
graficar_densidad_theta_estimado <- function(theta_vals, theta_0 = 0.5, bins = 30) {
  mu_emp <- mean(theta_vals)
  sd_emp <- sd(theta_vals)
  
  ggplot(data.frame(theta = theta_vals), aes(x = theta)) +
    geom_histogram(
      aes(y = after_stat(density), 
          fill = "Empirical θ̂ DML", 
          colour = "Empirical θ̂ DML"),
      bins = bins, alpha = 0.3
    ) +
    stat_function(
      fun = dnorm, 
      args = list(mean = mu_emp, sd = sd_emp), 
      aes(linetype = "Normal empírica"),
      linewidth = 1, color = "grey60"
    ) +
    geom_vline(xintercept = theta_0, color = "grey20", linetype = "dashed", linewidth = 1.2) + # valor verdadero
    geom_vline(xintercept = mu_emp, color = "grey60", linetype = "dashed", linewidth = 0.9) +  # media empírica
    scale_color_manual(
      name = '',
      breaks = c("Empirical θ̂ DML", "Normal empírica"),
      values = c("Empirical θ̂ DML" = "darkgreen", "Normal empírica" = "grey60")
    ) +
    scale_fill_manual(
      name = '',
      breaks = c("Empirical θ̂ DML", "Normal empírica"),
      values = c("Empirical θ̂ DML" = "darkgreen", "Normal empírica" = NA)
    ) +
    scale_linetype_manual(
      name = '', values = c("Normal empírica" = "solid")
    ) +
    xlim(c(min(theta_vals) - 0.05, max(theta_vals) + 0.05)) +
    xlab(expression(hat(theta))) +
    ylab("Densidad") +
    theme_minimal()
}

# Graficamos y exportamos los resultados
dist_1_DML <- graficar_densidad_theta_estimado(simulaciones_dml_esc1$theta)
dist_2_DML <- graficar_densidad_theta_estimado(simulaciones_dml_esc2$theta)
dist_3_DML <- graficar_densidad_theta_estimado(simulaciones_dml_esc3$theta)
dist_4_DML <- graficar_densidad_theta_estimado(simulaciones_dml_esc4$theta)

# Exportamos los gráficos
ggsave("densidad_theta_esc1.png", dist_1_DML, width = 5.5, height = 4, dpi = 300, bg = "white")
ggsave("densidad_theta_esc2.png", dist_2_DML, width = 5.5, height = 4, dpi = 300, bg = "white")
ggsave("densidad_theta_esc3.png", dist_3_DML, width = 5.5, height = 4, dpi = 300, bg = "white")
ggsave("densidad_theta_esc4.png", dist_4_DML, width = 5.5, height = 4, dpi = 300, bg = "white")


############### Boxplots de las simulaciones por n #####################

# Añadimos una columna que identifique el tamaño muestral
simulaciones_DML_nbase10_1000reps$n <- 10
simulaciones_DML_nbase50_1000reps$n <- 50
simulaciones_DML_nbase100_1000reps$n <- 100
simulaciones_DML_nbase500_1000reps$n <- 500
simulaciones_DML_nbase1000_1000reps$n <- 1000
simulaciones_DML_nbase5000_1000reps$n <- 5000
simulaciones_DML_nbase10000_1000reps$n <- 10000

# Unimos todos en un solo dataframe para el grafico
simulaciones_todas_dml <- bind_rows(
  simulaciones_DML_nbase10_1000reps,
  simulaciones_DML_nbase50_1000reps,
  simulaciones_DML_nbase100_1000reps,
  simulaciones_DML_nbase500_1000reps,
  simulaciones_DML_nbase1000_1000reps,
  simulaciones_DML_nbase5000_1000reps,
  simulaciones_DML_nbase10000_1000reps
)

# Hacemos los boxplots
boxplot_escenario1 <- ggplot(simulaciones_todas_dml, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(x = "n", y = expression(hat(theta)),
       title = "g(x) no lineal y m(x) logística no lineal") +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario1

####### Boxplot Escenario 2 #######.

# Añadimos el n a las simulaciones
simulaciones_DML_nbase10_1000reps_esc2$n <- 10
simulaciones_DML_nbase50_1000reps_esc2$n <- 50
simulaciones_DML_nbase100_1000reps_esc2$n <- 100
simulaciones_DML_nbase500_1000reps_esc2$n <- 500
simulaciones_DML_nbase1000_1000reps_esc2$n <- 1000
simulaciones_DML_nbase5000_1000reps_esc2$n <- 5000
simulaciones_DML_nbase10000_1000reps_esc2$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc2_dml <- bind_rows(
  simulaciones_DML_nbase10_1000reps_esc2,
  simulaciones_DML_nbase50_1000reps_esc2,
  simulaciones_DML_nbase100_1000reps_esc2,
  simulaciones_DML_nbase500_1000reps_esc2,
  simulaciones_DML_nbase1000_1000reps_esc2,
  simulaciones_DML_nbase5000_1000reps_esc2,
  simulaciones_DML_nbase10000_1000reps_esc2
)

# Hacemos el boxplot
boxplot_escenario2 <- ggplot(simulaciones_esc2_dml, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística no lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario2

####### Boxplot Escenario 3 #######.

# Añadimos el n a las simulaciones
simulaciones_DML_nbase10_1000reps_esc3$n <- 10
simulaciones_DML_nbase50_1000reps_esc3$n <- 50
simulaciones_DML_nbase100_1000reps_esc3$n <- 100
simulaciones_DML_nbase500_1000reps_esc3$n <- 500
simulaciones_DML_nbase1000_1000reps_esc3$n <- 1000
simulaciones_DML_nbase5000_1000reps_esc3$n <- 5000
simulaciones_DML_nbase10000_1000reps_esc3$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc3_dml <- bind_rows(
  simulaciones_DML_nbase10_1000reps_esc3,
  simulaciones_DML_nbase50_1000reps_esc3,
  simulaciones_DML_nbase100_1000reps_esc3,
  simulaciones_DML_nbase500_1000reps_esc3,
  simulaciones_DML_nbase1000_1000reps_esc3,
  simulaciones_DML_nbase5000_1000reps_esc3,
  simulaciones_DML_nbase10000_1000reps_esc3
)

# Hacemos el boxplot
boxplot_escenario3 <- ggplot(simulaciones_esc3_dml, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) no lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario3

####### Boxplot Escenario 4 #######.

# Añadimos el n a las simulaciones
simulaciones_DML_nbase10_1000reps_esc4$n <- 10
simulaciones_DML_nbase50_1000reps_esc4$n <- 50
simulaciones_DML_nbase100_1000reps_esc4$n <- 100
simulaciones_DML_nbase500_1000reps_esc4$n <- 500
simulaciones_DML_nbase1000_1000reps_esc4$n <- 1000
simulaciones_DML_nbase5000_1000reps_esc4$n <- 5000
simulaciones_DML_nbase10000_1000reps_esc4$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc4_dml <- bind_rows(
  simulaciones_DML_nbase10_1000reps_esc4,
  simulaciones_DML_nbase50_1000reps_esc4,
  simulaciones_DML_nbase100_1000reps_esc4,
  simulaciones_DML_nbase500_1000reps_esc4,
  simulaciones_DML_nbase1000_1000reps_esc4,
  simulaciones_DML_nbase5000_1000reps_esc4,
  simulaciones_DML_nbase10000_1000reps_esc4
)

# Hacemos el boxplot
boxplot_escenario4 <- ggplot(simulaciones_esc4_dml, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario4

# Exportamos los boxplots
ggsave("boxplot_escenario1_DML.png", boxplot_escenario1, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario2_DML.png", boxplot_escenario2, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario3_DML.png", boxplot_escenario3, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario4_DML.png", boxplot_escenario4, width = 8, height = 5, dpi = 300, bg = "white")


########## Hacemos boxplots de las simulaciones por n ###########

# Añadimos una columna que identifique el tamaño muestral
simulaciones_DR_rf_nbase10_1000reps_esc1$n <- 10
simulaciones_DR_rf_nbase50_1000reps_esc1$n <- 50
simulaciones_DR_rf_nbase100_1000reps_esc1$n <- 100
simulaciones_DR_rf_nbase500_1000reps_esc1$n <- 500
simulaciones_DR_rf_nbase1000_1000reps_esc1$n <- 1000
simulaciones_DR_rf_nbase5000_1000reps_esc1$n <- 5000
simulaciones_DR_rf_nbase10000_1000reps_esc1$n <- 10000

# Unimos todos en un solo dataframe para el grafico
simulaciones_todas <- bind_rows(
  simulaciones_DR_rf_nbase10_1000reps_esc1,
  simulaciones_DR_rf_nbase50_1000reps_esc1,
  simulaciones_DR_rf_nbase100_1000reps_esc1,
  simulaciones_DR_rf_nbase500_1000reps_esc1,
  simulaciones_DR_rf_nbase1000_1000reps_esc1,
  simulaciones_DR_rf_nbase5000_1000reps_esc1,
  simulaciones_DR_rf_nbase10000_1000reps_esc1
)

# Hacemos los boxplots
boxplot_escenario1 <- ggplot(simulaciones_todas, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(x = "n", y = expression(hat(theta)),
       title = "g(x) no lineal y m(x) logística no lineal") +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario1

####### Boxplot Escenario 2 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_rf_nbase10_1000reps_esc2$n <- 10
simulaciones_DR_rf_nbase50_1000reps_esc2$n <- 50
simulaciones_DR_rf_nbase100_1000reps_esc2$n <- 100
simulaciones_DR_rf_nbase500_1000reps_esc2$n <- 500
simulaciones_DR_rf_nbase1000_1000reps_esc2$n <- 1000
simulaciones_DR_rf_nbase5000_1000reps_esc2$n <- 5000
simulaciones_DR_rf_nbase10000_1000reps_esc2$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc2 <- bind_rows(
  simulaciones_DR_rf_nbase10_1000reps_esc2,
  simulaciones_DR_rf_nbase50_1000reps_esc2,
  simulaciones_DR_rf_nbase100_1000reps_esc2,
  simulaciones_DR_rf_nbase500_1000reps_esc2,
  simulaciones_DR_rf_nbase1000_1000reps_esc2,
  simulaciones_DR_rf_nbase5000_1000reps_esc2,
  simulaciones_DR_rf_nbase10000_1000reps_esc2
)

# Hacemos el boxplot
boxplot_escenario2 <- ggplot(simulaciones_esc2, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística no lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario2

####### Boxplot Escenario 3 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_rf_nbase10_1000reps_esc3$n <- 10
simulaciones_DR_rf_nbase50_1000reps_esc3$n <- 50
simulaciones_DR_rf_nbase100_1000reps_esc3$n <- 100
simulaciones_DR_rf_nbase500_1000reps_esc3$n <- 500
simulaciones_DR_rf_nbase1000_1000reps_esc3$n <- 1000
simulaciones_DR_rf_nbase5000_1000reps_esc3$n <- 5000
simulaciones_DR_rf_nbase10000_1000reps_esc3$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc3 <- bind_rows(
  simulaciones_DR_rf_nbase10_1000reps_esc3,
  simulaciones_DR_rf_nbase50_1000reps_esc3,
  simulaciones_DR_rf_nbase100_1000reps_esc3,
  simulaciones_DR_rf_nbase500_1000reps_esc3,
  simulaciones_DR_rf_nbase1000_1000reps_esc3,
  simulaciones_DR_rf_nbase5000_1000reps_esc3,
  simulaciones_DR_rf_nbase10000_1000reps_esc3
)

# Hacemos el boxplot
boxplot_escenario3 <- ggplot(simulaciones_esc3, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) no lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario3

####### Boxplot Escenario 4 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_rf_nbase10_1000reps_esc4$n <- 10
simulaciones_DR_rf_nbase50_1000reps_esc4$n <- 50
simulaciones_DR_rf_nbase100_1000reps_esc4$n <- 100
simulaciones_DR_rf_nbase500_1000reps_esc4$n <- 500
simulaciones_DR_rf_nbase1000_1000reps_esc4$n <- 1000
simulaciones_DR_rf_nbase5000_1000reps_esc4$n <- 5000
simulaciones_DR_rf_nbase10000_1000reps_esc4$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc4 <- bind_rows(
  simulaciones_DR_rf_nbase10_1000reps_esc4,
  simulaciones_DR_rf_nbase50_1000reps_esc4,
  simulaciones_DR_rf_nbase100_1000reps_esc4,
  simulaciones_DR_rf_nbase500_1000reps_esc4,
  simulaciones_DR_rf_nbase1000_1000reps_esc4,
  simulaciones_DR_rf_nbase5000_1000reps_esc4,
  simulaciones_DR_rf_nbase10000_1000reps_esc4
)

# Hacemos el boxplot
boxplot_escenario4 <- ggplot(simulaciones_esc4, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )

boxplot_escenario4

# Exportamos los boxplots
ggsave("boxplot_escenario1_DR_rf.png", boxplot_escenario1, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario2_DR_rf.png", boxplot_escenario2, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario3_DR_rf.png", boxplot_escenario3, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario4_DR_rf.png", boxplot_escenario4, width = 8, height = 5, dpi = 300, bg = "white")


#############################################################################.
############## Análisis DR paramétrico por tamaño muestral ##################
#############################################################################.

########## Hacemos boxplots de las simulaciones por n ###########

# Añadimos una columna que identifique el tamaño muestral
simulaciones_DR_nbase10_1000reps_esc1$n <- 10
simulaciones_DR_nbase50_1000reps_esc1$n <- 50
simulaciones_DR_nbase100_1000reps_esc1$n <- 100
simulaciones_DR_nbase500_1000reps_esc1$n <- 500
simulaciones_DR_nbase1000_1000reps_esc1$n <- 1000
simulaciones_DR_nbase5000_1000reps_esc1$n <- 5000
simulaciones_DR_nbase10000_1000reps_esc1$n <- 10000

# Unimos todos en un solo dataframe para el grafico
simulaciones_todas <- bind_rows(
  simulaciones_DR_nbase10_1000reps_esc1,
  simulaciones_DR_nbase50_1000reps_esc1,
  simulaciones_DR_nbase100_1000reps_esc1,
  simulaciones_DR_nbase500_1000reps_esc1,
  simulaciones_DR_nbase1000_1000reps_esc1,
  simulaciones_DR_nbase5000_1000reps_esc1,
  simulaciones_DR_nbase10000_1000reps_esc1
)

# Hacemos los boxplots
boxplot_escenario1 <- ggplot(simulaciones_todas, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(x = "n", y = expression(hat(theta)),
       title = "g(x) no lineal y m(x) logística no lineal") +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  ) + scale_y_continuous(limits = c(-5, 6))

boxplot_escenario1

####### Boxplot Escenario 2 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_nbase10_1000reps_esc2$n <- 10
simulaciones_DR_nbase50_1000reps_esc2$n <- 50
simulaciones_DR_nbase100_1000reps_esc2$n <- 100
simulaciones_DR_nbase500_1000reps_esc2$n <- 500
simulaciones_DR_nbase1000_1000reps_esc2$n <- 1000
simulaciones_DR_nbase5000_1000reps_esc2$n <- 5000
simulaciones_DR_nbase10000_1000reps_esc2$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc2 <- bind_rows(
  simulaciones_DR_nbase10_1000reps_esc2,
  simulaciones_DR_nbase50_1000reps_esc2,
  simulaciones_DR_nbase100_1000reps_esc2,
  simulaciones_DR_nbase500_1000reps_esc2,
  simulaciones_DR_nbase1000_1000reps_esc2,
  simulaciones_DR_nbase5000_1000reps_esc2,
  simulaciones_DR_nbase10000_1000reps_esc2
)

# Hacemos el boxplot
boxplot_escenario2 <- ggplot(simulaciones_esc2, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística no lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )+ scale_y_continuous(limits = c(-5, 6))

boxplot_escenario2

####### Boxplot Escenario 3 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_nbase10_1000reps_esc3$n <- 10
simulaciones_DR_nbase50_1000reps_esc3$n <- 50
simulaciones_DR_nbase100_1000reps_esc3$n <- 100
simulaciones_DR_nbase500_1000reps_esc3$n <- 500
simulaciones_DR_nbase1000_1000reps_esc3$n <- 1000
simulaciones_DR_nbase5000_1000reps_esc3$n <- 5000
simulaciones_DR_nbase10000_1000reps_esc3$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc3 <- bind_rows(
  simulaciones_DR_nbase10_1000reps_esc3,
  simulaciones_DR_nbase50_1000reps_esc3,
  simulaciones_DR_nbase100_1000reps_esc3,
  simulaciones_DR_nbase500_1000reps_esc3,
  simulaciones_DR_nbase1000_1000reps_esc3,
  simulaciones_DR_nbase5000_1000reps_esc3,
  simulaciones_DR_nbase10000_1000reps_esc3
)

# Hacemos el boxplot
boxplot_escenario3 <- ggplot(simulaciones_esc3, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) no lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  ) + scale_y_continuous(limits = c(-5, 6))

boxplot_escenario3

####### Boxplot Escenario 4 #######.

# Añadimos el n a las simulaciones
simulaciones_DR_nbase10_1000reps_esc4$n <- 10
simulaciones_DR_nbase50_1000reps_esc4$n <- 50
simulaciones_DR_nbase100_1000reps_esc4$n <- 100
simulaciones_DR_nbase500_1000reps_esc4$n <- 500
simulaciones_DR_nbase1000_1000reps_esc4$n <- 1000
simulaciones_DR_nbase5000_1000reps_esc4$n <- 5000
simulaciones_DR_nbase10000_1000reps_esc4$n <- 10000

# Pasamos todas las simulaciones en una sola base
simulaciones_esc4 <- bind_rows(
  simulaciones_DR_nbase10_1000reps_esc4,
  simulaciones_DR_nbase50_1000reps_esc4,
  simulaciones_DR_nbase100_1000reps_esc4,
  simulaciones_DR_nbase500_1000reps_esc4,
  simulaciones_DR_nbase1000_1000reps_esc4,
  simulaciones_DR_nbase5000_1000reps_esc4,
  simulaciones_DR_nbase10000_1000reps_esc4
)

# Hacemos el boxplot
boxplot_escenario4 <- ggplot(simulaciones_esc4, aes(x = factor(n), y = theta)) +
  geom_boxplot(fill = "#ff9966", alpha = 0.7, outlier.shape = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray30", linewidth = 1) +
  labs(
    x = "n",
    y = expression(hat(theta)),
    title = "g(x) lineal y m(x) logística lineal"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(angle = 0, vjust = 0.5)
  )+ scale_y_continuous(limits = c(-5, 6))

boxplot_escenario4

# Exportamos los boxplots
ggsave("boxplot_escenario1_DR.png", boxplot_escenario1, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario2_DR.png", boxplot_escenario2, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario3_DR.png", boxplot_escenario3, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("boxplot_escenario4_DR.png", boxplot_escenario4, width = 8, height = 5, dpi = 300, bg = "white")


######################## Anexo: distribuciones superpuestas ########################

# Paleta de colores para los 4 escenarios
colores_pastel <- c(
  "#8ECFC9",  # Escenario 1
  "#FFBE7A",  # Escenario 2
  "lightgreen",  # Escenario 3
  "grey80"   # Escenario 4
)

### DML ###.
densidades_dml <- ggplot(df_all_dml, aes(x = theta, color = base, fill = base)) +
  geom_density(linewidth = 1.1, adjust = 1, alpha = 0.05, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = colores_pastel) +
  scale_fill_manual(values = colores_pastel) +
  labs(
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Escenario",
    fill  = "Escenario"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )
densidades_dml

#### DR ####.
densidades_DR <- ggplot(df_all, aes(x = theta, color = base, fill = base)) +
  geom_density(linewidth = 1.1, adjust = 1, alpha = 0.05, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = colores_pastel) +
  scale_fill_manual(values = colores_pastel) +
  labs(
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Escenario",
    fill  = "Escenario"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )

densidades_DR

#### DR ML ####.
densidades_dr_ml <- ggplot(df_all_dr_ml, aes(x = theta, color = base, fill = base)) +
  geom_density(linewidth = 1.1, adjust = 1, alpha = 0.05, na.rm = TRUE) +
  geom_vline(xintercept = theta0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = colores_pastel) +
  scale_fill_manual(values = colores_pastel) +
  labs(
    x = expression(hat(theta)),
    y = "Densidad",
    color = "Escenario",
    fill  = "Escenario"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )


densidades_dr_ml


ggsave("densidades_superpuestas_dml.png", densidades_dml, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("densidades_superpuestas_DR.png", densidades_DR, width = 8, height = 5, dpi = 300, bg = "white")
ggsave("densidades_superpuestas_dr_ml.png", densidades_dr_ml, width = 8, height = 5, dpi = 300, bg = "white")


############## Gráficos estilo de poder - RMSE por método y n ##################

########################## Agregamos RMSE #################################.
# DML
# Agregamos el sd

agregar_rmse_DML <- function(simulaciones_DML_nbase1000_1000reps){
  simulaciones_DML_nbase1000_1000reps <- simulaciones_DML_nbase1000_1000reps %>% 
    mutate(sd = se*sqrt(n))
  
  r = length(simulaciones_DML_nbase1000_1000reps$theta)
  
  # Agregamos el RMSE
  simulaciones_DML_nbase1000_1000reps <- simulaciones_DML_nbase1000_1000reps %>% 
    mutate(MSE = (theta - theta_0)^2)
  
  #sqrt(mean(simulaciones_DML_nbase1000_1000reps$MSE))
  
  RMSE_promedio_DML_nbase1000_1000reps <- sqrt(sum(simulaciones_DML_nbase1000_1000reps$MSE)/r)
  return(RMSE_promedio_DML_nbase1000_1000reps)
}


#### DR paramétrico ####

agregar_rmse_DRs <- function(simulaciones_DR_nbase1000_1000reps_esc1){
  
  r = length(simulaciones_DR_nbase1000_1000reps_esc1$theta)
  
  # Agregamos el RMSE
  simulaciones_DR_nbase1000_1000reps_esc1 <- simulaciones_DR_nbase1000_1000reps_esc1 %>% 
    mutate(MSE = (theta - theta_0)^2)
  
  #sqrt(mean(simulaciones_DML_nbase1000_1000reps$MSE))
  
  RMSE_promedio_DR_nbase1000_1000reps <- sqrt(sum(simulaciones_DR_nbase1000_1000reps_esc1$MSE)/r)
  return(RMSE_promedio_DR_nbase1000_1000reps)
}



# RMSE de DML
rmse_dml_10 <- agregar_rmse_DML(simulaciones_DML_nbase10_1000reps)
rmse_dml_50 <- agregar_rmse_DML(simulaciones_DML_nbase50_1000reps)
rmse_dml_100 <- agregar_rmse_DML(simulaciones_DML_nbase100_1000reps)
rmse_dml_500 <- agregar_rmse_DML(simulaciones_DML_nbase500_1000reps)
rmse_dml_1000 <- agregar_rmse_DML(simulaciones_DML_nbase1000_1000reps)
rmse_dml_5000 <- agregar_rmse_DML(simulaciones_DML_nbase5000_1000reps)
rmse_dml_10000 <- agregar_rmse_DML(simulaciones_DML_nbase10000_1000reps)

# RMSE DR
rmse_dr_10 <- agregar_rmse_DRs(simulaciones_DR_nbase10_1000reps_esc1)
rmse_dr_50 <-agregar_rmse_DRs(simulaciones_DR_nbase50_1000reps_esc1)
rmse_dr_100 <-agregar_rmse_DRs(simulaciones_DR_nbase100_1000reps_esc1)
rmse_dr_500 <-agregar_rmse_DRs(simulaciones_DR_nbase500_1000reps_esc1)
rmse_dr_1000 <-agregar_rmse_DRs(simulaciones_DR_nbase1000_1000reps_esc1)
rmse_dr_5000 <-agregar_rmse_DRs(simulaciones_DR_nbase5000_1000reps_esc1)
rmse_dr_10000 <-agregar_rmse_DRs(simulaciones_DR_nbase10000_1000reps_esc1)

# RMSE DR rf
rmse_dr_rf_10 <- agregar_rmse_DRs(simulaciones_DR_nbase10_1000reps_esc1)
rmse_dr_rf_50 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase50_1000reps_esc1)
rmse_dr_rf_100 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase100_1000reps_esc1)
rmse_dr_rf_500 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase500_1000reps_esc1)
rmse_dr_rf_1000 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase1000_1000reps_esc1)
rmse_dr_rf_5000 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase5000_1000reps_esc1)
rmse_dr_rf_10000 <-agregar_rmse_DRs(simulaciones_DR_rf_nbase10000_1000reps_esc1)

# Agregamos los resultados en una tabla de resumen
rmse_resumen <- data.frame(
  Metodo = c(
    rep("DML", 7),
    rep("DR", 7),
    rep("DR_rf", 7)
  ),
  n = c(
    10, 50, 100, 500, 1000, 5000, 10000,   # DML
    10, 50, 100, 500, 1000, 5000, 10000,   # DR
    10, 50, 100, 500, 1000, 5000, 10000    # DR_rf
  ),
  RMSE = c(
    rmse_dml_10,  rmse_dml_50,  rmse_dml_100,  rmse_dml_500,  rmse_dml_1000,  rmse_dml_5000,  rmse_dml_10000,
    rmse_dr_10,   rmse_dr_50,   rmse_dr_100,   rmse_dr_500,   rmse_dr_1000,   rmse_dr_5000,   rmse_dr_10000,
    rmse_dr_rf_10,rmse_dr_rf_50,rmse_dr_rf_100,rmse_dr_rf_500,rmse_dr_rf_1000,rmse_dr_rf_5000,rmse_dr_rf_10000
  ),
  stringsAsFactors = FALSE
)

# Ordenamos los resultados por Método y n
rmse_resumen <- rmse_resumen[order(rmse_resumen$Metodo, rmse_resumen$n), ]
row.names(rmse_resumen) <- NULL # sacamos los rownames


# Graficamos
p_rmse_unico <- ggplot(rmse_resumen, aes(x = n, y = RMSE, color = Metodo)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = c(10, 50, 100, 500, 1000, 5000, 10000)) +
  labs(x = "Tamaño muestral (n)", y = "RMSE", color = "Método",
       title = "RMSE vs n por método") +
  theme_minimal(base_size = 12)

p_rmse_unico

########### Gráfico RMSE por escenario #############

rmse_plot_por_escenario <- function(escenario = 1,
                                    ns = c(100, 500, 1000, 5000, 10000)) {
  stopifnot(escenario %in% 1:4)
  
  # Helper para traer el objeto correcto según método y n
  traer_obj <- function(n, metodo, esc) {
    nombre <- switch(
      metodo,
      "DML" = if (esc == 1) {
        sprintf("simulaciones_DML_nbase%d_1000reps", n)
      } else {
        sprintf("simulaciones_DML_nbase%d_1000reps_esc%d", n, esc)
      },
      "DR"   = sprintf("simulaciones_DR_nbase%d_1000reps_esc%d", n, esc),
      "DR_ml"= sprintf("simulaciones_DR_rf_nbase%d_1000reps_esc%d", n, esc)
    )
    if (!exists(nombre, envir = .GlobalEnv)) {
      warning("No se encontró el objeto: ", nombre)
      return(NULL)
    }
    get(nombre, envir = .GlobalEnv)
  }
  
  # Definimos la funcion para el calculo del rmse por el método y n
  rmse_de <- function(n, metodo, esc) {
    obj <- traer_obj(n, metodo, esc)
    if (is.null(obj)) return(NA_real_)
    if (metodo == "DML") agregar_rmse_DML(obj) else agregar_rmse_DRs(obj)
  }
  
  # Armamos la base con métoodo, n y RMSE
  metodos <- c("DML", "DR", "DR_ml")
  df_list <- lapply(metodos, function(m) {
    data.frame(
      Metodo = m,
      n = ns,
      RMSE = vapply(ns, rmse_de, numeric(1), metodo = m, esc = escenario),
      stringsAsFactors = FALSE
    )
  })
  rmse_resumen <- do.call(rbind, df_list)
  rmse_resumen <- rmse_resumen[order(rmse_resumen$Metodo, rmse_resumen$n), ]
  row.names(rmse_resumen) <- NULL
  
  # Colores y tipos de línea personalizados
  colores <- c(
    "DML" = "#FFA64D",
    "DR" = "#7FB3D5",
    "DR_ml"= "grey60"
  )
  lineas <- c(
    "DML" = "solid",
    "DR" = "solid",
    "DR_ml"= "dashed"
  )
  
  # Gráfico
  p <- ggplot(rmse_resumen, aes(x = n, y = RMSE, color = Metodo, linetype = Metodo)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = ns) +
    scale_color_manual(values = colores) +
    scale_linetype_manual(values = lineas) +
    labs(
      x = "Tamaño muestral (n)",
      y = "RMSE",
      color = "Método",
      linetype = "Método"
      #,
      #title = paste0("RMSE por método y tamaño muestral | Escenario ", escenario)
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")
  
  attr(p, "rmse_resumen") <- rmse_resumen
  return(p)
}

# Obtenemos los gráficos para cada escenario 
p1 <- rmse_plot_por_escenario(escenario = 1)
p2 <- rmse_plot_por_escenario(escenario = 2)
p3 <- rmse_plot_por_escenario(escenario = 3)
p4 <- rmse_plot_por_escenario(escenario = 4)

# Guardamos los gráficos
ggsave("grafico_poder_esc1.png", p1, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc2.png", p2, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc3.png", p3, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc4.png", p4, width = 9, height = 5, dpi = 300, bg = "white")

############### Gráficos de sesgo por n y método #################

# Agregamos el sesgo absoluto
sesgo_theta_abs <- function(df, theta_true = theta0, col_theta = "theta") {
  stopifnot(col_theta %in% names(df))
  vals <- suppressWarnings(as.numeric(df[[col_theta]]))
  mean(abs(vals - theta_true), na.rm = TRUE)
}

agregar_sesgo_abs_DML <- function(df, theta_true = theta0) sesgo_theta_abs(df, theta_true)
agregar_sesgo_abs_DRs <- function(df, theta_true = theta0) sesgo_theta_abs(df, theta_true)

# Agregamos el sesgo
bias_plot_por_escenario <- function(escenario = 1,
                                    ns = c(100, 500, 1000, 5000, 10000)) {
  stopifnot(escenario %in% 1:4)
  
  traer_obj <- function(n, metodo, esc) {
    nombre <- switch(
      metodo,
      "DML"  = if (esc == 1) sprintf("simulaciones_DML_nbase%d_1000reps", n)
      else          sprintf("simulaciones_DML_nbase%d_1000reps_esc%d", n, esc),
      "DR"   = sprintf("simulaciones_DR_nbase%d_1000reps_esc%d", n, esc),
      "DR_ml"= sprintf("simulaciones_DR_rf_nbase%d_1000reps_esc%d", n, esc)
    )
    if (!exists(nombre, envir = .GlobalEnv)) {
      warning("No se encontró el objeto: ", nombre)
      return(NULL)
    }
    get(nombre, envir = .GlobalEnv)
  }
  
  # Sesgo absoluto por método para una n
  sesgo_abs_de <- function(n, metodo, esc) {
    obj <- traer_obj(n, metodo, esc)
    if (is.null(obj)) return(NA_real_)
    if (metodo == "DML") agregar_sesgo_abs_DML(obj) else agregar_sesgo_abs_DRs(obj)
  }
  
  metodos <- c("DML", "DR", "DR_ml")
  df_list <- lapply(metodos, function(m) {
    data.frame(
      Metodo = m,
      n = ns,
      Bias = vapply(ns, sesgo_abs_de, numeric(1), metodo = m, esc = escenario),
      stringsAsFactors = FALSE
    )
  })
  
  bias_df <- do.call(rbind, df_list)
  bias_df <- bias_df[order(bias_df$Metodo, bias_df$n), ]
  row.names(bias_df) <- NULL
  
  colores <- c("DML" = "#FFA64D", "DR" = "#7FB3D5", "DR_ml" = "grey60")
  lineas  <- c("DML" = "solid",   "DR" = "solid",  "DR_ml" = "dashed")
  
  p <- ggplot(bias_df, aes(x = n, y = Bias, color = Metodo, linetype = Metodo)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = ns) +
    scale_color_manual(values = colores) +
    scale_linetype_manual(values = lineas) +
    labs(
      x = "Tamaño muestral (n)",
      y = "Sesgo absoluto",
      color = "Método", linetype = "Método"
      #,
      #title = paste0("Sesgo absoluto vs n por método — Escenario ", escenario)
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")
  
  attr(p, "bias_resumen") <- bias_df
  p
}

# Aplicamos la función para obtener los gráficos de sesgo por escenario
b1 <- bias_plot_por_escenario(1); b1
b2 <- bias_plot_por_escenario(2); b2
b3 <- bias_plot_por_escenario(3); b3
b4 <- bias_plot_por_escenario(4); b4


ggsave("grafico_poder_esc1_sesgo_abs.png", b1, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc2_sesgo_abs.png", b2, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc3_sesgo_abs.png", b3, width = 9, height = 5, dpi = 300, bg = "white")
ggsave("grafico_poder_esc4_sesgo_abs.png", b4, width = 9, height = 5, dpi = 300, bg = "white")



################ Anexo: Análisis descriptivo de bases generadas #####################

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
analisis_descriptivo(df_esc1, "Escenario1_g_nolineal_m_nologit")
analisis_descriptivo(df_esc2, "Escenario2_g_lineal_m_nologit")
analisis_descriptivo(df_esc3, "Escenario3_g_nolineal_m_logit")
analisis_descriptivo(df_esc4, "Escenario4_g_lineal_m_logit")

