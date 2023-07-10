import threading
import pathlib
import cv2
from deepface import DeepFace

"""
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf  = cv2.CascadeClassifier(str(cascade_path))
"""
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
#cap = cv2.VideoCapture("video_reference")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

reference_img =cv2.imread("FaceRecog_CV2\\Reference\\20200822_104745.jpg")

def check_face(frame):
    global face_match

    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except Exception as e:
        print(f"An error occurred during face verification: {str(e)}")
        face_match = False



def capture_frames():
    global counter

    while True:
        ret, frame = cap.read()

        if frame is None:
            continue

        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 255, 0), 2)
        """

        if ret:
            if counter % 30 == 0:
                try:
                    threading.Thread(target=check_face, args=(frame.copy())).start()
                except ValueError:
                    pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    threading.Thread(target=capture_frames).start()