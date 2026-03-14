import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) # (0)-> default camera  (1)-> means external camera
    

mp_face = mp.solutions.face_mesh  #collection of models
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,    # maximum face allowed
    refine_landmarks=True,  #add extra accuracy for eyes and lips
    min_detection_confidence=0.5, #required to continousaly detect face
    min_tracking_confidence=0.5
)         # detect 468 points on face including eyes, nose, eyebrows,lips

while True:
    ret, frame = cap.read()  # reading frame or capturing image    frame->numpy array( image matrix)  ,  ret -> true/false , ki capture hui frame ya nahi 
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting color bgr to rgb because mediapipe  cannot detect bgr 

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:    #if face detect run code (show landmarks) or just skip  
        for face in results.multi_face_landmarks: # if multiple faces in camera loop runs for each face
            for landmark in face.landmark:
                h, w, _ = frame.shape      #frame.shape returns (height,width,channels)->(480,640,3) 480,640->pixels, 3->rgb
                x = int(landmark.x * w)    #x,y are normalized values (0.72,0.34) but screen uses pixel thats why we convert them by multiplyig as per width and hegiht of face 
                y = int(landmark.y * h)
                cv2.circle(frame,(x,y),1,(0,255,0),-1)  # frame-> image , (X,y)-> location , (1)-> radius, (greeen color), -1 -> filled circle

    cv2.imshow("Face Landmarks", frame) # it shows the live webcam recording ,  eg. we open camera in our phone and we can see what is getting record 

    if cv2.waitKey(1) & 0xFF == 27: # wait for 1 mili secand and 0xff== 27 is esc key 
        break

cap.release()  # free the cam
cv2.destroyAllWindows() # destory all screens