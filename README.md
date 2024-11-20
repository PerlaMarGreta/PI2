Exploratory Data Analysis (EDA): Acceso a Internet en Argentina
------------

Descripción del Proyecto

Este proyecto realiza un análisis exploratorio de datos (EDA) sobre el sector de telecomunicaciones en Argentina, específicamente en relación con el acceso a internet. El objetivo principal es identificar desigualdades en la conectividad, analizar patrones de distribución y ofrecer insights clave para la toma de decisiones estratégicas.

Objetivos
--------------------

Evaluar las diferencias regionales en términos de acceso a internet y velocidad de conexión.
Identificar valores atípicos y tendencias relevantes en los datos.
Explorar correlaciones entre las principales métricas, como velocidad y accesos.
Detectar oportunidades de mejora en infraestructura y calidad del servicio.

Datos Utilizados
Dataset: Información sobre acceso a internet y velocidad por provincia.
Variables principales:
Provincia: Distribución de accesos por Provincia.
Partido: Distribución de accesos por Partido.
Localidades: Distribución de accesos por localidad.
Accesos: Número total de accesos a internet.
Velocidad: Velocidad promedio de conexión (Mbps).


Metodología y Estructura del EDA

1) Exploración inicial HOJA 1 Acc_vel_loc_sinrangos 
2) Analisis de Valores Faltantes 
3) Deteccion de Outliers 
4) Analisis de Distribucion de Variables 
5) Analisis de Correlaciones 
6) Busqueda de Registros de Duplicados
7) Conclusion General del EDA
8) Visualizacion General hoja por hoja del DataSet

Resultados Clave
------------

1. Distribución de Accesos por Provincia

Buenos Aires representa más del 38% de los accesos totales, seguida por Córdoba (11.7%) y Santa Fe (10%).
Provincias menos representadas incluyen Tierra del Fuego y La Rioja, con menos del 1% de los accesos totales.

2. Velocidad de Conexión

La mayoría de las velocidades están por debajo de 50 Mbps, con una fuerte concentración en valores bajos.
Los valores extremos incluyen velocidades superiores a 1 Gbps, correspondientes a infraestructuras avanzadas.

3. Correlaciones

Existe poca correlación entre el número de accesos y la velocidad de conexión.
Esto sugiere que la infraestructura influye más en la velocidad que la densidad de usuarios.


Conclusiones
---------------

Desigualdad regional:

 Provincias más pequeñas tienen menor acceso a internet y velocidades más bajas.

Impacto de la infraestructura: Las regiones urbanas concentran la mayor parte de las conexiones de alta velocidad.

Outliers: 

Los valores extremos en velocidad y accesos representan áreas prioritarias para inversiones en telecomunicaciones.

Herramientas Utilizadas

Python:
pandas y numpy para manipulación de datos.
matplotlib y seaborn para visualizaciones.
plotly para gráficos interactivos.
scipy.stats para análisis estadístico.
sklearn.preprocessing para normalización y transformación de datos.

Instrucciones para Reproducir
Clona el repositorio:
bash
Copiar código
git clone <URL-del-repositorio>

Instala las dependencias:

bash
Copiar código
pip install -r requirements.txt