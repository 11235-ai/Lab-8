import cv2
import numpy as np

def track_circle_quadrants():
   
    reference_img = cv2.imread("ref-point.jpg", cv2.IMREAD_GRAYSCALE)
    if reference_img is None:
        print("Error: No se encontró circle_quadrants.jpg")
        return
    
    orb = cv2.ORB_create(1000) 
    kp_ref, des_ref = orb.detectAndCompute(reference_img, None)

    fly_img = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)
    if fly_img is None:
        print("Error: No se encontró fly64.png")
        return
    
    fly_bgr = fly_img[:, :, :3]
    fly_alpha = fly_img[:, :, 3]
    fly_h, fly_w = fly_img.shape[:2]
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
        
        if des_frame is not None and des_ref is not None:
            
            matches = bf.match(des_ref, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 10:
               
                num_best_matches = 20 if len(matches) >= 20 else len(matches)
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches[:num_best_matches]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches[:num_best_matches]]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None and mask is not None:
                    inliers = mask.ravel().sum()
                    ratio_inliers = inliers / float(len(mask))
                    
                    if ratio_inliers > 0.5:
                       
                        h_ref, w_ref = reference_img.shape[:2]
                        pts = np.float32([
                            [0, 0],
                            [w_ref, 0],
                            [w_ref, h_ref],
                            [0, h_ref]
                        ]).reshape(-1, 1, 2)
                        
                        dst = cv2.perspectiveTransform(pts, M)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)
                        center_x = int(np.mean(dst[:, 0, 0]))
                        center_y = int(np.mean(dst[:, 0, 1]))
                        
                        x1 = center_x - fly_w // 2
                        y1 = center_y - fly_h // 2
                        x2 = x1 + fly_w
                        y2 = y1 + fly_h
                        
                        frame_h, frame_w = frame.shape[:2]
                        x_start = max(0, x1)
                        y_start = max(0, y1)
                        x_end = min(x2, frame_w)
                        y_end = min(y2, frame_h)
                        
                        if x_end > x_start and y_end > y_start:
                            frame_roi = frame[y_start:y_end, x_start:x_end]
                            fly_x = x_start - x1
                            fly_y = y_start - y1
                            
                            fly_roi = fly_bgr[fly_y:fly_y+(y_end-y_start), fly_x:fly_x+(x_end-x_start)]
                            alpha_roi = fly_alpha[fly_y:fly_y+(y_end-y_start), fly_x:fly_x+(x_end-x_start)]
                            alpha_mask = alpha_roi[:, :, np.newaxis] / 255.0
                            frame_roi[:] = frame_roi * (1 - alpha_mask) + fly_roi * alpha_mask
        
        cv2.imshow("Seguimiento del círculo con cuadrantes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
track_circle_quadrants()
