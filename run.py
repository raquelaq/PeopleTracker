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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7, minSize=(30, 60))
        # scaleFactor lo que hace es que en cada iteración de la detección, se reduce el tamaño de la imagen en un 20% (1.2^5 = 2.49)
        # minNeighbors es el número de rectángulos vecinos que deben ser aceptados para que se considere una detección
        # minSize es el tamaño mínimo del rectángulo que se considerará una detección
        # maxSize es el tamaño máximo del rectángulo que se considerará una detección

        tracker.track_people(frame, detections, frame_number)

        for person in tracker.people:
            cv2.rectangle(frame, (person.rect[0], person.rect[1]), (person.rect[0] + person.rect[2], person.rect[1] + person.rect[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {person.id}', (person.rect[0], person.rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if person.matched:
                cv2.circle(frame, (person.rect[0], person.rect[1]), 5, (0, 0, 255), -1)
                
        cv2.imshow('People Tracking', frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
