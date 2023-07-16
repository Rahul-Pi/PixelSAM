# The SAM functions essential to run the detection algorithm

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# Function to filter the contour by area
def filter_contour_by_area(contour, min_area, max_area):
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    return True

# Function to approximate the polygon
def approx_contour(contour, percentage, epsilon_step=0.005):
    if percentage < 0 or percentage >= 1:
        raise ValueError("Percentage must be in the range [0, 1).")
    
    target_points = max(int(contour.shape[0] * (1 - percentage)), 3)

    epsilon = 0
    while True:
        epsilon += epsilon_step
        approximated_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        if approximated_contour.shape[0] <= target_points:
            break

    return approximated_contour

# Function to convert the mask to a polygon
def mask_to_polygon(mask_image, approximation_percentage = 0.75):
    
    # Add padding to the mask_image to capture the outer edge
    padded_mask_image = cv2.copyMakeBorder(mask_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Get the contours of the mask
    contours, _ = cv2.findContours(cv2.Canny(padded_mask_image, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Remove the contours which have less than 10 points
    contours = [contour for contour in contours if contour.shape[0] >= 10]

    # Remove the previously added padding from the contours
    contours = [np.subtract(contour,(1,1)) for contour in contours]

    # Filter the contours by area
    height, width = mask_image.shape[-2:]
    contours = [contour for contour in contours if filter_contour_by_area(
            contour=contour,
            min_area=0.005*height*width,
            max_area=1*height*width)]

    # Reduce the complexity of contours
    best_contour = [approx_contour(contour=contour, percentage=0.75) for contour in contours]
    
    # Ensure that the contour is closed
    best_contour = [np.vstack((contour, contour[0][np.newaxis, :])) for contour in best_contour if not np.array_equal(contour[0], contour[-1])]

    # Convert the contours to polygons
    best_polygon = [contour.flatten().tolist() for contour in best_contour]
    
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
            best_contour, best_polygon = mask_to_polygon(mask_image)

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
