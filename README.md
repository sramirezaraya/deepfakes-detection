# *Sistema Clasificador de Deepfakes por medio de Redes Neuronales Artificiales*
Este repositorio tiene el objetivo de explicar la implementacion de un clasificador de deepfakes, el cual es capaz de tomar como entrada un video de una persona y determinar si el video es real o es falso (deepfake). Para esto, se van a detallar cada una de las etapas que conforman el proceso de creacion del clasificador y sirva para futuras investigaciones en el tema. 

## Base de Datos - "*Deepfake Detection Challenge*"
La base de datos que se utiliza para realizar el entrenamiento de la red neuronal es "Deepfake Detection Challenge", la cual es una base de datos de la competencia de Facebook con más de 100k videos (reales y falsos). 
El tamaño total de este dataset es de 471gb, por lo que se divide en distintas partes para realizar el entrenamiento por carpeta y asi facilitar el proceso. 
Estos datos fueron descargados de la pagina oficial de kaggle de la competencia en el siguiente enlace: 
https://www.kaggle.com/c/deepfake-detection-challenge/data

## Preprocesamiento y Reconocimiento de Rostros con "MTCNN"
Para la lectura de videos se utiliza una libreria muy conocida como OpenCV, la cual puede leer videos, imagenes, realizar redimensiones, etc. Cada una de las imagenes que se extraen en los videos se puede apreciar una persona en cuerpo completo, sin embargo, para que la red neuronal pueda entrenarse de mejor forma solo necesita el rostro de la persona. Es por esto, que se utiliza un detector MTCNN, el cual toma cada una de estas imagenes y a traves de "boxes" o cajas, puede detectarlas y encerrarlas para posteriormente extraer esta region y guardarla como una imagen. 

## Extracción de Rostros

Para esto, primero se define una funcion "crop", la cual tomará como entrada una imagen (un frame extraido del video leido) y calculará el boundingbox a partir del detector MTCNN, con el objetivo de crear el "cuadro" en el rostro de la persona y posteriormente redimensiona el tamaño de la imagen a 224x224. Luego se crea una funcion que toma cada video presente en la carpeta 'data_path' y realiza una extracción de imagenes que será regulada por el parámetro count, en donde en este caso extraerá imagenes cada 30 fps para los videos reales y cada 100 fps para los videos fakes(los videos de esta base de datos son de 30 fps y duran 10 segundos lo que da un total de 300 fps, por lo que se extraeran 10 imagenes por video real y 3 imagenes por video fake).

## Modelo de Red Neuronal 
Se utilizan diversos modelos pre entrenados, los cuales fueron extraídos con los pesos de Imagenet (dataset de imágenes de rostros de personas). Entre los modelos utilizados se encuentra:
- EfficientNetB7
- InceptionResnetV2
- InceptionV3
- Resnet50
- VGG16

Se entrenan cada una de las imágenes en los modelos, en donde se van entrenando cada 20000 imágenes, con un batch size de 64 y 50 iteraciones. El optimizador utilizado es adam y binary crossentropy con metrica accuracy. 

## Demo 
Se muestra un ejemplo del algoritmo funcionando, en donde se implementa una interfaz gráfica y se selecciona el video en formato mp4 a predecir. Una vez se elige el video, el algoritmo realiza la predicción, en donde se ve un recuadro verde si la etiqueta es real y un recuadro rojo si la etiqueta es falsa. 
