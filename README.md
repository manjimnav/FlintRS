El objetivo de este proyecto es realizar un A/B testing usando para ello el framework Apache Flint. Este proyecto proyecto contiene una red neuronal que se encargará de clasificar los datos en tiempo real para determinar si un tipo anuncio debe enseñarse. Un sistema externo se encargará de evaluar los resultados y determinar si el sistema implementado mejora al base teniendo en cuenta la que debe evaluarlo con el número de población significante. 
Para ejecutar el programa debe ejecutar el script execute.sh. 
Este script enviará resultados a bigdatamaster2019.dataspartan.com:19093 con el token que se le pase por parámetro. 
Asegurese de tener la carpeta bin de flink añadida a las variables de entorno para la correcta ejecución.
