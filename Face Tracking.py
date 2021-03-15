import cv2
import datetime
import numpy as np
import time

def head_tracking():
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #function to get coordinates
    def get_coords(p1):
        try: return int(p1[0][0][0]), int(p1[0][0][1])
        except: return int(p1[0][0]), int(p1[0][1])

    # Adding the Face and Eye Cascades .xml file 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    m_cap = cv2.VideoCapture(0)                                 # Live Video Capture (0 for WebCam, 'Output.avi' for a sample file)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    #out = cv2.VideoWriter('Output', fourcc, 20.0, (640,480))   # Saves the file as output file

    frame_width = m_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = m_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))
    #out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

    while (m_cap.isOpened()):
        ret, img = m_cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)                           # makes the rectangle for the FACE by adding the dimensions 
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            old_gray = frame_gray.copy()

            face_center = x+w/2, y+h/3                                      # makes a dot on the forehead for positioning analysis (X & Y coordinates)
            #print ("X: %f, Y: %f" % (x+w/2, y+h/3))

            p0 = np.array([[face_center]], np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            cv2.circle(img, get_coords(p1), 4, (0,0,255), -1)                                   # draws the circles (dot) to track the movement in each frame
            cv2.circle(img, get_coords(p0), 4, (255,0,0))

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)                 # draws the rectangle around the EYE

        cv2.putText(img, "X: {}".format(x+w/2), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)            #  Displays the X & Y values on the feed
        cv2.putText(img, "Y: {}".format(y+h/3), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        date_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))                                      # Live date and time w/o milliseconds
                  
        img = cv2.putText(img, date_time, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  # change "date_time" to "text" if you want to display frame width and height
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff                                  # breaks the code with escape key
        if k == 27:
            break
        '''
        # ARRAY CODE
        head_position = []
        #while True:
        head_position.append(face_center)
        print(head_position)
        #time.sleep(1)
        # END OF ARRAY CODE
        '''
    m_cap.release()
    #out.release()
    cv2.destroyAllWindows()

head_tracking()
