# *Sistema Clasificador de Deepfakes por medio de Redes Neuronales Artificiales*
Este repositorio tiene el objetivo de explicar la implementacion de un clasificador de deepfakes, el cual es capaz de tomar como entrada un video de una persona y determinar si el video es real o es falso (deepfake). Para esto, se detallan cada una de las etapas que conforman el proceso de creacion del clasificador y sirva para futuras investigaciones en el tema. 

## Base de Datos - "*Deepfake Detection Challenge*"
La base de datos que se utilizó para realizar el entrenamiento de la red neuronal fue "Deepfake Detection Challenge", la cual es una base de datos de la competencia de Facebook con más de 100k videos (reales y falsos). 
El tamaño total de este dataset es de 471gb, por lo que se dividió en distintas partes para realizar el entrenamiento por carpeta y asi facilitar el proceso. 
Estos datos fueron descargados de la pagina oficial de kaggle de la competencia en el siguiente enlace: 
https://www.kaggle.com/c/deepfake-detection-challenge/data

## Preprocesamiento y Reconocimiento de Rostros con "MTCNN"
Para la lectura de videos se utilizó una libreria muy conocida como OpenCV, la cual puede leer videos, imagenes, realizar redimensiones, etc. Cada una de las imagenes que se extraen en los videos se puede apreciar una persona en cuerpo completo, sin embargo, para que la red neuronal pueda entrenarse de mejor forma solo necesita el rostro de la persona. Es por esto, que se utilizó un detector MTCNN, el cual toma cada una de estas imagenes y a traves de "boxes" o cajas, puede detectar y encerrarlas para posteriormente extraer esta region y guardarla como una imagen. 

<p align="center">
  <img src="https://github.com/sramirezaraya/deepfakes-detection/blob/main/images/img2.png" width="700" height="1500" title="hover text">
</p>

## Extracción de Rostros

Para esto, primero se define una funcion "crop", la cual toma como entrada una imagen (un frame extraido del video leido) y calcula el boundingbox a partir del detector MTCNN, con el objetivo de crear el "cuadro" en el rostro de la persona y posteriormente redimensiona el tamaño de la imagen a 224x224. Luego se creó una funcion que toma cada video presente y realiza una extracción de imagenes que será regulada por el parámetro count, en donde en este caso se extraen imagenes cada 30 fps para los videos reales y cada 100 fps para los videos fakes(los videos de esta base de datos son de 30 fps y duran 10 segundos lo que da un total de 300 fps, por lo que se extraen 10 imagenes por video real y 3 imagenes por video fake).

<p align="center">
  <img src="https://github.com/sramirezaraya/deepfakes-detection/blob/main/images/img3.png" width="350" height="350" title="hover text">
</p>

## Modelo de Red Neuronal 
Se utilizan diversos modelos pre entrenados, los cuales fueron extraídos con los pesos de Imagenet (dataset de imágenes de rostros de personas). Entre los modelos utilizados se encuentra:
- EfficientNetB7
- InceptionResnetV2
- InceptionV3
- Resnet50
- VGG16

Se entrenaron cada una de las imágenes en los modelos, en donde se van entrenaron cada 20000 imágenes, con un batch size de 64 y 50 iteraciones. El optimizador que se utilizó fue "adam" y "binary crossentropy" con métrica "accuracy". <br>
Los modelos se pueden encontrar en el siguiente link:
https://drive.google.com/drive/folders/1y4c2cRrEf_iaf4Vq3QaDIBwoSexLjdGj?usp=sharing

## Demo 
Se muestra un ejemplo del algoritmo funcionando, en donde se implementa una interfaz gráfica y se selecciona el video en formato mp4 a predecir. Una vez se elige el video, el algoritmo realiza la predicción, en donde se ve un recuadro verde si la etiqueta es real y un recuadro rojo si la etiqueta es falsa. 

<p align="center">
  <img src="https://github.com/sramirezaraya/deepfakes-detection/blob/main/images/demo.gif" title="hover text">
</p>

## Requisitos
- Python 3.7.6 <br>
- Archivo requirements.txt (contiene las librerias con sus respectivas versiones para el uso)

## Uso 
*Clonar el repositorio y acceder a carpeta* <br>
```python 
git clone https://github.com/sramirezaraya/deepfakes-detection.git
cd deepfakes-detection
```
*Crear un ambiente virtual y activar* <br>
```python
python3 -m venv venv
source venv/bin/activate
```
*Instalar librerias con requirements.txt*
```python
pip3 install -r requirements.txt
```
*Ejecutar programa*
```python
python3 deepfakes-clasificador.py
```

## Licencia
La implementación proporcionada es estrictamente para fines académicos. Si está interesado en utilizar esta tecnología para cualquier uso comercial, no dude en contactarme.
