# computo_distribuido

Proyecto desarrollado en python que realiza una implementación de cómputo    
paralelo y distribuido en el análisis del registro de accidentes de tránsito
en el estado de Chihuahua entre los años 2015-2024.

#Fuentes de información

Se obtuvieron datos del inegi del informe de datos abiertos de accidentes de 
tránsito en zonas urbanas y suburbanas en formato csv del año 1997 a 2024.

#Limpieza de datos

Se utilizo ray data para realizar el filtro de datos de todas las fuentes de
informacción para usar unicamente los años del 2015 al 2024 y el id del estado
de Chihuahua para realizar nuestro modelo de predicción ese periodo de tiempo
y espacio.
