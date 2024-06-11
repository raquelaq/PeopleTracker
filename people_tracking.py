import cv2
import numpy as np
from person import Person

def non_max_suppression(boxes, overlapThresh):
    # Si no hay cajas, devuelve una lista vacía
    if len(boxes) == 0:
        return []

    # Convierte las coordenadas de las cajas a números de punto flotante si son enteros
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Inicializa la lista de índices seleccionados
    pick = []

    # Obtiene las coordenadas de las cajas
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right
    # y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of
        # picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the computed bounding box and the bounding box
        # in the area list
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the
        # provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

class PeopleTracker:
    def __init__(self):
        self.people = []
        self.next_id = 1

    def track_people(self, frame, detections, frame_number):
        detections = np.array([[x, y, w, h] for (x, y, w, h) in detections])

        # Filtrar detecciones basadas en la relación de aspecto
        filtered_detections = []
        for (x, y, w, h) in detections:
            aspect_ratio = h / float(w)
            if 1.5 <= aspect_ratio <= 3.5:  # Ajustar los valores según sea necesario
                filtered_detections.append((x, y, w, h))

        filtered_detections = np.array(filtered_detections)
        filtered_detections = non_max_suppression(filtered_detections, 0.3)

        for person in self.people:
            person.matched = False

        for detection in filtered_detections:
            x, y, w, h = detection
            best_match = None
            min_distance = float('inf')

            for person in self.people:
                distance = person.calculate_distance(detection)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person

            if best_match and min_distance < 60:
                best_match.update(detection, frame, frame_number)
            else:
                new_person = Person(self.next_id, detection)
                new_person.update(detection, frame, frame_number)
                self.people.append(new_person)
                self.next_id += 1

        self.people = [person for person in self.people if person.matched or (frame_number - person.last_seen) < 40]