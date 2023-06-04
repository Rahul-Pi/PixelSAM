# The SAM functions essential to run the detection algorithm

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Function to convert the mask to a polygon
def mask_to_polygon(mask_image, input_point):
    # Add padding to the mask_image to capture the outer edge
    padded_mask_image = cv2.copyMakeBorder(mask_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Get the contours of the mask
    contours, _ = cv2.findContours(cv2.Canny(padded_mask_image, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Set the best contour and polygon to be empty
    best_contour = []
    best_polygon = []

    # Find the contours which include the any of the input_points
    for contour in contours:

        # Check if the number of points in the contour is less than 10: Skip the contour
        if len(contour) < 10:
            continue

        # Subtract the padding from the contour
        contour = np.subtract(contour,(1,1))

        # Approximate the contour to a polygon
        epsilon = 0.001 * cv2.arcLength(contour,True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Check if it is a closed contour
        if not np.array_equal(approx_contour[0], approx_contour[-1]):
            approx_contour = np.vstack((approx_contour, approx_contour[0][np.newaxis, :]))
        
        best_contour.append(approx_contour)

        # To avoid the polygon having negative coordinates
        polygon = [0 if i < 0 else i for i in approx_contour.flatten().tolist()]
        best_polygon.append(polygon)
    
    return best_contour, best_polygon


# Function to setup the SAM model
def SAM_setup(model_type, model_path, device_id):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    if device_id != "cpu":
        sam.to(device=device_id)
    else:
        print("Warning: Running on CPU. This will be slow.")
    return SamPredictor(sam)


# Function to predict the mask
def SAM_prediction(image, points, predictor, img_height, img_width, mask_array=[], outer_edge=0, bounding_box=0):
        # The points are in the format [x, y, color, label]
        input_point = np.array([[p[0], p[1]] for p in points])
        input_label = np.array([p[3] for p in points])

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
        
        # Add the previous masks to the image
        if len(mask_array) > 0:
            for m in mask_array:
                image = cv2.addWeighted(m, 0.3, image, 1, 0)

        # Check if the outer edge is to be detected
        if outer_edge==1:

            # Create a copy of the image
            overlay = image.copy()

            # Get the best contours in the mask and its corresponding polygon
            best_contour, best_polygon = mask_to_polygon(mask_image, input_point)

            # Overlay the controur
            cv2.drawContours(overlay, best_contour, -1, (0, 0, 255), thickness=cv2.FILLED)

            # Add the overlay to the image
            image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

            # Marking the bounding box for each of the contours
            if bounding_box==1:
                for cnt_cur in best_contour:
                    x, y, w, h = cv2.boundingRect(cnt_cur)
                    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), int((img_height+img_width)/400))

            gx, gy = [], []
            # Extract x and y coordinates into separate arrays
            if len(best_polygon) > 0:
                gx = np.array(np.hstack(best_polygon)).reshape(-1, 2)[:, 0]
                gy = np.array(np.hstack(best_polygon)).reshape(-1, 2)[:, 1]
        
        else:
            # Overlay the mask on the image        
            image = cv2.addWeighted(mask_image, 0.3, image, 0.7, 0)
        
            # Get the edges of the mask
            edges = cv2.Canny(mask_image[:, :, 0], 100, 200)

            # Plot the edges on the image
            gy, gx = np.where(edges != 0)
            for i in range(len(gx)):
                image = cv2.circle(image, (gx[i], gy[i]), int((img_height+img_width)/400), (0, 0, 255),-1)

        # Find the bounding box for the object
        if len(gx) > 0:
            # Calculate bounding box dimensions and center coordinates in YOLO format
            bbox_width = (np.max(gx) - np.min(gx)) / img_width
            bbox_height = (np.max(gy) - np.min(gy)) / img_height
            bbox_center_x = (np.max(gx) + np.min(gx)) / (2 * img_width)
            bbox_center_y = (np.max(gy) + np.min(gy)) / (2 * img_height)

            bbox_corners = [bbox_center_x, bbox_center_y, bbox_width, bbox_height]
            bbox_corners = [round(x, 6) for x in bbox_corners]
        else:
            bbox_corners = [0, 0, 0, 0]

        # If bounding_box is 1, plot the bounding box for the object
        if bounding_box==1:
            image = cv2.rectangle(image, (int((bbox_corners[0] - bbox_corners[2]/2) * img_width), int((bbox_corners[1] - bbox_corners[3]/2) * img_height)), (int((bbox_corners[0] + bbox_corners[2]/2) * img_width), int((bbox_corners[1] + bbox_corners[3]/2) * img_height)), (255, 0, 0), int((img_height+img_width)/400))
        
        return image, mask_save_image, bbox_corners
