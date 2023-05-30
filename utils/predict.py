# The SAM functions essential to run the detection algorithm

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def SAM_setup(model_type, model_path, device_id):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    if device_id != "cpu":
        sam.to(device=device_id)
    else:
        print("Warning: Running on CPU. This will be slow.")
    return SamPredictor(sam)

def SAM_prediction(image, points, predictor, img_height, img_width, mask_array=[]):
        # The points are in the format [x, y, color, label]
        input_point = []
        input_label = []
        for i in range(len(points)):
            input_point.append([points[i][0], points[i][1]])
            input_label.append(points[i][3])
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        # Estimate the mask
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        # Convert the mask to an image
        h, w = masks.shape[-2:]
        #mask_color = np.array([1,1,1])
        #make a list of mask colors: red,blue,green,yellow,orange,purple,turquoise,white
        mask_color = np.array([1,1,1])
        mask_image = masks.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)

        # The mask to be saved with different colors for different objects
        mask_save_colors = np.array([[1,0,0],[0,0,1],[0,1,0],[1,1,0],[1,0.5,0],[0.5,0,1],[0,1,1],[1,1,1]])
        mask_save_color = mask_save_colors[len(mask_array)]
        mask_save_image = masks.reshape(h, w, 1) * mask_save_color.reshape(1, 1, -1)
        mask_save_image = (mask_save_image * 255).astype(np.uint8)

        # Morphological operations to enhance detections
        kernel = np.ones((5, 5), np.uint8)
        mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

        # Morphological operations on the mask to be saved
        mask_save_image = cv2.morphologyEx(mask_save_image, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cv2.Canny(mask_image, 30, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Finding the best contour
        overlay = image.copy()
        best_contour = []
        # Find the contours which include the any of the input_points
        for i in range(len(contours)):
            for j in range(len(input_point)):
                if cv2.pointPolygonTest(contours[i],  tuple([int(input_point[j][0]), int(input_point[j][1])]), False) > 0:
                    best_contour.append(contours[i])
                    break
        # Overlay the controur
        cv2.drawContours(overlay, best_contour, -1, (0, 0, 255), thickness=cv2.FILLED)

        # Overlay the mask on the image        
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        if len(mask_array) > 0:
            for m in mask_array:
                image = cv2.addWeighted(m, 0.3, image, 1, 0)

        return image, mask_save_image
