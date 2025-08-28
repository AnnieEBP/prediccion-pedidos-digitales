# Predicción de Pedidos Digitales (eB2B)

## 1) Resumen de la lógica del análisis
Esta sección resume la lógica detrás de la implementación de los modelos, la elección de métricas de evaluación y el enfoque de ajuste de hiper-parámetros.

### 1.1. Notas iniciales
- El dataset posee un tamaño considerable (1.25 millones), por lo que se decidió usar la librería `polars` para su manejo.
-	Para no emplear una cantidad significativa de memoria, se optó por optimizar, transformando variables. Por ejemplo, las variables float64 fueron cambiadas a float32. 66 MB se logró reducir a 29 MB.

### 1.2.	Creación de nuevas variables
-	Se observó que cada registro representa un pedido; sin embargo solo existen, aproximadamente, 150 mil clientes únicos. Debido a que el objetivo es predecir a nivel cliente, se optó por agregar variables. 
-	Como se encontró que los clientes pertenecen a un solo país, así como a un tipo de visita, se determina que estas variables están a nivel cliente y pueden ser agregadas sin mayor complejidad.
-	Se propone construir una variable que indique la cantidad de pedidos que se hicieron a través del medio digital. Además, es posible calcular la proporción de pedidos hechos digitalmente sobre el total de pedidos.
-	El EDA reveló que la facturación posee una forma de campana, relativamente. El histograma presenta valores atípicos del lado derecho. Por lo tanto para poder agregar la fx por cliente, minimizando el efecto de los valores "anormales", se opta por usar la mediana y no la media.
-	La distribución de 'materiales_distintos' es relativamente uniforme. Esto sugiere que los pedidos se distribuyen de manera equitativa en cuanto a “cuántos materiales diferentes” compran.
-	Se estimó valioso crear una variable que indique recencia por cliente.
-	Luego de las consideraciones arriba colocadas, se crean las variables agregadas:
-	  pais
-	  total_pedidos
-	  pedidos_digital
-	  prop_pedidos_digital
-	  mediana_fx_usd
-	  sum_fx_usd
-	  mediana_materiales
-	  ult_pedido
-	  freq_visitas
-	  recencia
-	  ult_pedidos (indica si la última compra realizada fue a través del medio digital)
  
-	Se contruye una variable respuesta dicotómica considerando si la última compra fue realizada a través de un medio digital `ult_digital`

### 1.3.	Preparación de los datos 
- Codificación de variables
-     Las variables nominales se codificaron mediante codificación ficticia. Este método se prefirió a la codificación One-Hot, ya que la codificación ficticia omite una de las coordenadas, evitando así la multicolinealidad.
-     Para la variable ordinal `freq_visitas`, primero fue necesario ordenar las clases y, a continuación, se asignó un número a cada una (1, 2, 3, etc.). 

- Escalado
  La decisión de escalar los datos depende del tipo de modelo utilizado. En este caso, se consideraron dos tipos de modelos: un modelo del tipo lineal (logistic regression) y otro basados en conjunto de árboles, XGBoost. Los datos se escalaron utilizando el método de z-score.

- Detección de anomalías
  Para identificar y remover valores atípicos se empleó Chebyshevs con un k=3.  Para un rango de tres desviaciones estándar alrededor de la media, el teorema de Chebyshev establece que al menos el 89 % de las observaciones se encuentran dentro de ese rango. En este caso, el 98.6 % de los datos se encuentra dentro de tres desviaciones estándar, lo que indica que no existe un número significativo de anomalías; por lo tanto, se descartaron los valores fuera del rango.

- Importancia de variables (Feature Importance)
-     La matriz de correlación indicó las características con alta asociación con la variable objetivo; sin embargo, para conocer cuantitativamente cuáles son las variables que tienen mayor influencia en el cliente para pedir digitalmente, se ajustó un Random Forest. Este algoritmo tiene la ventaja que calcula la varianza aportada por cada variable. Los resultados demostraron que tres variables explican aproximadamente el 91% de la varianza total, se decidió trabajar únicamente con las siguientes variables: ["prop_pedidos_digital", "pedidos_digital", "sum_fx_usd"]
-     **Proporción del conjunto de prueba:** 0,25

### 1.4. Selección de modelo
- Logistic Regression
- XGBoost
