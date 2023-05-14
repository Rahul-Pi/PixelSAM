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

def SAM_prediction(image, points, predictor, img_height, img_width):
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
        mask_color = np.array([1,1,1])
        mask_image = masks.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)

        # Morphological operations to enhance detections
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(mask_image, kernel, iterations=2)
        mask_image = cv2.dilate(eroded_img, kernel, iterations=2)
        
        # Get the edges of the mask
        img_data = np.asarray(mask_image[:, :, 0])
        gy, gx = np.gradient(img_data)
        temp_edge = gy * gy + gx * gx
        gy, gx = np.where(temp_edge != 0.0)

        # Overlay the mask on the image        
        image = cv2.addWeighted(mask_image, 0.3, image, 0.7, 0)

        # Plot the gx and gy on the image
        for i in range(len(gx)):
            image = cv2.circle(image, (gx[i], gy[i]), int((img_height+img_width)/400), (0, 0, 255), -1)
        return image
