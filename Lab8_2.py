import cv2
import numpy as np

def track_marker_with_fly():

    fly_img = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)
    if fly_img is None:
        print("Error: No se encontró fly64.png")
        return
    
    fly_bgr = fly_img[:, :, :3]
    fly_alpha = fly_img[:, :, 3]
    fly_height, fly_width = fly_img.shape[:2]
    
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
            # Calcular centro del marcador
            marker_corners = corners[0][0]
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # coordenadas
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"X: {center_x}, Y: {center_y}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculo de posición 
            x1 = center_x - fly_width // 2
            y1 = center_y - fly_height // 2
            x2 = x1 + fly_width
            y2 = y1 + fly_height
            
            frame_height, frame_width = frame.shape[:2]
            x_start = max(0, x1)
            y_start = max(0, y1)
            x_end = min(x2, frame_width)
            y_end = min(y2, frame_height)
            
            if x_end > x_start and y_end > y_start:
                frame_roi = frame[y_start:y_end, x_start:x_end]
                fly_x_offset = x_start - x1
                fly_y_offset = y_start - y1
                fly_roi = fly_bgr[fly_y_offset:fly_y_offset + (y_end - y_start),
                                  fly_x_offset:fly_x_offset + (x_end - x_start)]
                alpha_roi = fly_alpha[fly_y_offset:fly_y_offset + (y_end - y_start),
                                       fly_x_offset:fly_x_offset + (x_end - x_start)]
                
                alpha_mask = alpha_roi[:, :, np.newaxis] / 255.0
                frame_roi[:] = frame_roi * (1 - alpha_mask) + fly_roi * alpha_mask
        
        cv2.imshow("Tracking con Mosca", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

track_marker_with_fly()