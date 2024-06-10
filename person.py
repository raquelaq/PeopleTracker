import cv2
import numpy as np

class Person:
    def __init__(self, person_id, rect):
        self.id = person_id # identificador único de la persona
        self.rect = rect # Coordenadas del rectángulo que enmarca a la persona (x, y, ancho, alto)
        self.template = None # Plantilla de imagen para comparación
        self.last_seen = 0 # Último frame en el que se vio a la persona
        self.matched = False  # Añadido para rastrear si se ha emparejado en el frame actual

    def update(self, rect, frame, frame_number):
        self.rect = rect # Actualiza las coordenadas del rectángulo
        self.template = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] # Extrae la plantilla de la imagen
        self.last_seen = frame_number # Actualiza el último frame visto
        self.matched = True # Marca que ha sido emparejada en este frame

    def match_template(self, frame, threshold=0.7):
        if self.template is None or self.template.shape[0] > frame.shape[0] or self.template.shape[1] > frame.shape[1]: # Comprueba si la plantilla es válida
            return False
        
        res = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED) # Realiza la comparación de plantillas
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold # Devuelve si la similitud es mayor que el umbral
    
    def calculate_distance(self, new_detection):
        # Calcula alguna métrica de distancia entre la ubicación actual y la nueva detección
        # Por ejemplo, la distancia euclidiana entre los centros de los rectángulos
        x, y, w, h = self.rect
        center_x, center_y = x + w // 2, y + h // 2 # Centro del rectángulo actual
        new_x, new_y, new_w, new_h = new_detection
        new_center_x, new_center_y = new_x + new_w // 2, new_y + new_h // 2 # Centro de la nueva detección
        return ((center_x - new_center_x) ** 2 + (center_y - new_center_y) ** 2) ** 0.5 # Distancia euclidiana
