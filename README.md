# PRÁCTICA 2: OPEN CV DETECTOR DE PERSONAS


## Resumen del Proyecto
Este proyecto correspondiente a la asignatura de Fundamentos de los Sistemas Inteligentes titulado "OpenCV detector de personas", está realizado por Raquel Almeida Quesada y Jorge Morales Llerandi, alumnos del Grado De Ciencia e Ingeniería de Datos en Universidad de Las Palmas de Gran Canaria. 
Esta práctica consiste en la creación de un código en pyhton utilizando la biblioteca OpenCV que nos permita detectar de un vídeo a las personas que pasan caminando por la calle.

## Estructura
Este proyecto esta formado por 5 archivos, tres de ellos son .py, otro .xml y el último.mp4:
- person.py: este archivo define una clase Person en Python para el seguimiento de personas en videos. Cada instancia representa a una persona con un identificador único, posición (rectángulo de detección), y plantilla para reconocimiento en frames sucesivos. Incluye métodos para actualizar su estado y verificar la coincidencia con nuevas detecciones usando la técnica de comparación de plantillas de OpenCV. Las banderas matched y found indican si la persona ha sido emparejada o encontrada en el frame actual.
- run.py: este archivo implementa un programa de seguimiento de personas en videos usando OpenCV en Python. Carga un video, utiliza un clasificador Haar para detectar personas, y rastrea cada persona detectada a través de frames sucesivos. Muestra cada persona con un rectángulo y un ID en el video. El programa se ejecuta en un bucle continuo hasta que se interrumpe manualmente.
- people_tracking.py: este archivo implementa un sistema de seguimiento de personas en videos usando OpenCV y NumPy. Utiliza la técnica de supresión de no máximos para filtrar detecciones superpuestas de personas. La clase PeopleTracker gestiona el seguimiento, actualizando o añadiendo personas basándose en la superposición de detecciones. Cada Person se actualiza con información de ubicación y se rastrea a lo largo del tiempo. El sistema identifica nuevas personas y mantiene un registro de todas las detectadas.
- haarcascade_fullbody.xml: es un archivo de clasificador en cascada preentrenado utilizado por la biblioteca OpenCV para la detección de cuerpos humanos completos en imágenes o videos. Este archivo contiene los datos de un modelo de aprendizaje automático basado en características Haar.
- people_walking2.mp4: el vídeo sobre el que vamos a hacer la deteción de personas.

## Bibliografía
- chat.openai.com
- Notebooks explicativos de la asignatura "Fundamentos de Sistemas Inteligentes" de la ULPGC.
- https://www.youtube.com/watch?v=cZkpaL36fW4
  
## Autores
- Raquel Almeida Quesada
- Jorge Morales Llerandi
