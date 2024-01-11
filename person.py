import cv2
import numpy as np

class Person:
    def __init__(self, person_id, rect):
        self.id = person_id
        self.rect = rect
        self.template = None
        self.last_seen = 0
        self.matched = False  # Añadido para rastrear si se ha emparejado en el frame actual

    def update(self, rect, frame, frame_number):
        self.rect = rect
        self.template = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        self.last_seen = frame_number
        self.matched = True

    def match_template(self, frame, threshold=0.7):
        if self.template is None or self.template.shape[0] > frame.shape[0] or self.template.shape[1] > frame.shape[1]:
            return False
        
        res = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold
    
    def calculate_distance(self, new_detection):
        # Calcula alguna métrica de distancia entre la ubicación actual y la nueva detección
        # Por ejemplo, la distancia euclidiana entre los centros de los rectángulos
        x, y, w, h = self.rect
        center_x, center_y = x + w // 2, y + h // 2
        new_x, new_y, new_w, new_h = new_detection
        new_center_x, new_center_y = new_x + new_w // 2, new_y + new_h // 2
        return ((center_x - new_center_x) ** 2 + (center_y - new_center_y) ** 2) ** 0.5
