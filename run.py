import cv2
from people_tracking import PeopleTracker

def main():
    cap = cv2.VideoCapture('people_walking2.mp4')
    tracker = PeopleTracker()
    haar_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Escalar el frame para acercar el video (factor de escala 1.5)
        scale_factor = 1.5
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Obtener dimensiones del frame escalado
        height, width = frame.shape[:2]

        # Calcular las coordenadas para recortar el centro del frame
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        x_center = width // 2
        y_center = height // 2
        x1 = x_center - (new_width // 2)
        x2 = x_center + (new_width // 2)
        y1 = y_center - (new_height // 2)
        y2 = y_center + (new_height // 2)

        # Recortar el frame para centrarlo
        frame = frame[y1:y2, x1:x2]

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ajustar los parámetros para hacer el clasificador más estricto
        detections = haar_cascade.detectMultiScale(gray_frame, 
                                                   scaleFactor=1.2,  # Más estricto
                                                   minNeighbors=12,   # Más estricto
                                                   minSize=(50, 100)) # Más estricto

        tracker.track_people(frame, detections, frame_number)

        for person in tracker.people:
            cv2.rectangle(frame, (person.rect[0], person.rect[1]), 
                          (person.rect[0] + person.rect[2], person.rect[1] + person.rect[3]), 
                          (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {person.id}', (person.rect[0], person.rect[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if person.matched:
                cv2.circle(frame, (person.rect[0], person.rect[1]), 5, (0, 0, 255), -1)
                
        cv2.imshow('People Tracking', frame)
        frame_number += 1

        # Reducir la velocidad de reproducción aumentando el tiempo de espera
        if cv2.waitKey(50) & 0xFF == ord('q'):  # Cambiado a 50 ms en lugar de 1 ms
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
