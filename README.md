# Computo_distribuido

Proyecto desarrollado en python que realiza una implementación de cómputo    
paralelo y distribuido en el análisis del registro de accidentes de tránsito
en el estado de Chihuahua entre los años 2015-2024.

## Fuentes de información

Se obtuvieron datos del inegi del informe de datos abiertos de accidentes de 
tránsito en zonas urbanas y suburbanas en formato csv del año 1997 a 2024.

## Limpieza de datos

Se utilizo ray data para realizar el filtro de datos de todas las fuentes de
informacción para usar unicamente los años del 2015 al 2024 y el id del estado
de Chihuahua para realizar nuestro modelo de predicción ese periodo de tiempo
y espacio.

## Modelo para predecir el número de personas fallecidas en accidentes de tránsito de Chihuahua

El proyecto incorpora un modelo de Machine Learning orientado a la predicción del número de fallecidos en accidentes de tránsito utilizando XGBoost.
El objetivo principal es estimar la cantidad de víctimas fatales considerando variables relacionadas con las características del accidente, tipo de vehículos involucrados, ubicación, condiciones del conductor y factores del entorno.

## Procesamiento y preparación de datos
El sistema realiza una etapa de preprocesamiento donde las variables numéricas y categóricas son transformadas a formatos adecuados para el entrenamiento del modelo.
La variable objetivo MUERTOS es convertida a formato numérico y utilizada para predecir la cantidad de fallecidos en un accidente de tránsito. Además, se aplican procesos de limpieza, normalización y codificación de variables categóricas mediante LabelEncoder para optimizar el análisis de los datos.

## Modelo utilizado
El sistema utiliza XGBoost, un algoritmo basado en Gradient Boosting especializado en tareas de regresión y clasificación con alto rendimiento sobre grandes volúmenes de datos.
```
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        7,
    "eta":              0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "alpha":            0.1,
    "lambda":           1.0,
    "seed":             42,
    "nthread":          4,
}
```
## Entrenamiento distribuido

Para optimizar el rendimiento computacional, el entrenamiento del modelo se implementó mediante procesamiento paralelo y distribuido utilizando Ray.

El conjunto de entrenamiento es dividido entre múltiples workers, permitiendo entrenar varios modelos de manera simultánea y seleccionar posteriormente el modelo con mejor desempeño.

Esta estrategia permite reducir significativamente los tiempos de procesamiento en comparación con un entrenamiento secuencial tradicional.

## Evaluación del modelo

El modelo es evaluado utilizando distintas métricas de regresión para medir la calidad de las predicciones realizadas.

### Métricas utilizadas
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)

## Variables utilizadas
### Variables categóricas
- DIASEMANA
- URBANA
- SUBURBANA
- TIPACCID
- CAUSAACCI
- CAPAROD
- SEXO
- ALIENTO
- CINTURON
- etc.

### Variables numéricas
- AÑO
- MES
- ID_HORA
- ID_MINUTO
- CAMPASAJ
- MICROBUS
- PASCAMON
- OMNIBUS
- TRANVIA
- etc.
## Interpretabilidad del modelo
Importancia de variables

El modelo calcula automáticamente la relevancia de cada característica utilizando la métrica gain, permitiendo identificar qué variables influyen más en la predicción del número de fallecidos.

Entre las variables más relevantes pueden encontrarse:

- Tipo de accidente
- Hora del accidente
- Presencia de alcohol
- Tipo de vehículo
- Condiciones del conductor

## Almacenamiento del modelo

El modelo entrenado se almacena automáticamente para su reutilización y posteriores pruebas de inferencia.
```
best_booster.save_model(
    "model.ubj"
)
```

Además, el sistema genera automáticamente archivos con las predicciones realizadas sobre el conjunto de prueba.

```
predicciones_test.csv
```

## Ejecución del proyecto
