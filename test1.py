import cv2
import mediapipe as mp
import numpy as np
import winsound

cap = cv2.VideoCapture(0) 


mp_face = mp.solutions.face_mesh 
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,   
    refine_landmarks=True, 
)    
LEFT_EYE = [33,160,158,133,153,144] 
RIGHT_EYE = [362,385,387,263,373,380]   

def distance(p1,p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    vertical1 = distance(eye_points[1],eye_points[5])
    vertical2 = distance(eye_points[2],eye_points[4])
    horizontal = distance(eye_points[0],eye_points[3])

    ear = (vertical1+vertical2)/(horizontal*2.0)
    return ear
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:   
        for face in results.multi_face_landmarks: 
            # for landmark in face.landmark: 
                h, w, _ = frame.shape      
                # x = int(landmark.x * w)  
                left_eye = []
                right_eye = []
                # y = int(landmark.y * h)
                # cv2.circle(frame,(x,y),1,(0,232,255),-1) 
                for idx in LEFT_EYE:
                    landmark = face.landmark[idx]
                    x = int(landmark.x*w)
                    y = int(landmark.y*h)
                    left_eye.append((x,y))

                for idx in RIGHT_EYE:
                     landmark = face.landmark[idx]  #index of right eye
                     x = int(landmark.x*w)
                     y = int(landmark.y*h)
                     right_eye.append((x,y))

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                ear = (left_ear+right_ear)/2

                for point in left_eye:
                  cv2.circle(frame, point, 2, (0,255,0), -1) #frame - image, point- where circle will be drown, 2 - radius of circle , -1 thicknes of dot 

                for point in right_eye:
                  cv2.circle(frame, point, 2, (0,255,0), -1)
                if ear < 0.20 :
                     closed_frames +=1
                     cv2.putText(frame,'Eyes Closed',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    closed_frames = 0
                if closed_frames > 20:
                    winsound.Beep(1000,200)
                

    cv2.imshow("Blink Detection", frame) 

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()  
cv2.destroyAllWindows()