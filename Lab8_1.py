import cv2
import numpy as np

def convert_to_hsv():
    img = cv2.imread("images/variant-3.jpeg")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("images/variant-3_hsv.jpg", hsv_img)

convert_to_hsv()

# Parte 2. 3
def track_marker():

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
      
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            marker_corners = corners[0][0]
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
        
            #Coordenada
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"X: {center_x}, Y: {center_y}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            height, width = frame.shape[:2]
            center_rect_start = (width//2 - 100, height//2 - 100)
            center_rect_end = (width//2 + 100, height//2 + 100)
            is_inside = (center_rect_start[0] < center_x < center_rect_end[0] and
                        center_rect_start[1] < center_y < center_rect_end[1])
            color = (0, 255, 0) if is_inside else (0, 0, 255)
            cv2.rectangle(frame, center_rect_start, center_rect_end, color, 2)
            
        cv2.imshow("Marker Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

track_marker()