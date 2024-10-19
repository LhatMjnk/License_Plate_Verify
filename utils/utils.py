import os
import shutil
import tqdm
import time

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the character set for license plates
DECODE_PLATE = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z'
}

def read_yolo_label_file(label_path):
    """
    Reads a YOLO label file and returns the bounding box coordinates and class ids.

    Args:
        label_path (str): Path to the YOLO label file (.txt)

    Returns:
        labels (list): A list of dictionaries, each containing 'class_id', 'x_center', 'y_center', 'width', and 'height'
    """
    labels = []

    with open(label_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Each line contains: class_id, x_center, y_center, width, height
            data = line.strip().split()
            class_id = int(data[0])  # class_id is the first element (as an integer)
            x_center, y_center, width, height = map(float, data[1:])  # All other values are floats

            # Append as dictionary to the labels list
            labels.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    return labels

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Converts YOLO format (x_center, y_center, width, height) to bounding box pixel coordinates.

    Args:
        x_center (float): Normalized x center of the bounding box (0 to 1)
        y_center (float): Normalized y center of the bounding box (0 to 1)
        width (float): Normalized width of the bounding box (0 to 1)
        height (float): Normalized height of the bounding box (0 to 1)
        img_width (int): Width of the image in pixels
        img_height (int): Height of the image in pixels

    Returns:
        bbox (tuple): Pixel coordinates of the bounding box (x_min, y_min, x_max, y_max)
    """
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)

    return x_min, y_min, x_max, y_max

def draw_bounding_box_on_image(image_path, labels, evaluate=False):
    """
    Draws bounding boxes on an image based on YOLO labels.

    Args:
        image_path (str): Path to the image file
        labels (list): List of bounding box dictionaries with class_id, x_center, y_center, width, height

    Returns:
        image_with_bbox: Image with bounding boxes drawn
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    if evaluate:
        # Draw the image width and height
        print(f"Image width: {img_width}, Image height: {img_height}")

    # Iterate through labels and draw bounding boxes
    for label in labels:
        class_id = label['class_id']
        x_center, y_center, width, height = label['x_center'], label['y_center'], label['width'], label['height']

        # Convert YOLO format to pixel coordinates
        x_min, y_min, x_max, y_max = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)

        if evaluate:
            # Print the bounding box coordinates
            print(f"Class ID: {class_id}, x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height}")
            print(f"Class ID: {class_id}, BBox: ({x_min}, {y_min}, {x_max}, {y_max})")

        # Draw the bounding box (red for example)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)
        draw.text((x_min, y_min), f"{class_id}", fill="red")

    return image

def read_plate_characters(label_path):
    """
    Reads a license plate label file and returns the characters.

    Args:
        label_path (str): Path to the license plate label file (.txt)

    Returns:
        characters (str): A string of characters on the license plate
    """
    with open(label_path, 'r') as file:
        characters = file.readline().strip()

    return characters

def copy_matching_images(image_folder, label_filenames, destination_folder):
    # Get all image files in the image_folder
    image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    # Copy images that have a corresponding filename in the tex_label folder
    for image in image_filenames:
        image_name = os.path.splitext(image)[0]
        if image_name in label_filenames:
            shutil.copy(os.path.join(image_folder, image), os.path.join(destination_folder, image))
            print(f"Copied {image} to {destination_folder}")

def crop_images_with_labels(image_folder, label_folder, tex_label_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of label files in the tex_label folder
    tex_label_files = [f for f in os.listdir(tex_label_folder) if f.endswith('.txt')]

    for tex_label_file in tqdm.tqdm(tex_label_files):
        # Get the base name without extension
        base_name = os.path.splitext(tex_label_file)[0]

        # Construct paths
        image_path = os.path.join(image_folder, base_name + '.jpg')
        label_path = os.path.join(label_folder, base_name + '.txt')
        text_label_path = os.path.join(tex_label_folder, tex_label_file)

        # Check if the image and label files exist
        if os.path.exists(image_path) and os.path.exists(label_path):
            # Read the bounding box coordinates from the label file
            labels = read_yolo_label_file(label_path)

            # Open the image
            try:
                image = Image.open(image_path)
                img_width, img_height = image.size

                for label in labels:
                    x_center, y_center, width, height = label['x_center'], label['y_center'], label['width'], label['height']
                    x_min, y_min, x_max, y_max = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)

                    # Crop the image using the bounding box coordinates
                    cropped_image = image.crop((x_min, y_min, x_max, y_max))

                    # Save the cropped image
                    cropped_image_path = os.path.join(output_folder, base_name + '.jpg')
                    cropped_image.save(cropped_image_path)

                    shutil.copy(text_label_path, os.path.join(output_folder, base_name + '.txt'))
            except:
                os.remove(image_path)

def organize_tex_label_folder(image_folder, label_folder, tex_label_folder):
    # Create tex_label folder if it doesn't exist
    os.makedirs(tex_label_folder, exist_ok=True)

    # Create a single labels.txt file
    labels_txt_path = os.path.join(tex_label_folder, 'labels.txt')
    with open(labels_txt_path, 'w') as labels_txt:
        # Get list of label files in the label folder
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

        for label_file in label_files:
            # Get the base name without extension
            base_name = os.path.splitext(label_file)[0]

            # Construct paths
            image_path = os.path.join(image_folder, base_name + '.jpg')
            label_path = os.path.join(label_folder, label_file)

            # Check if the image and label files exist
            if os.path.exists(image_path) and os.path.exists(label_path):
                # Move the image to the tex_label folder
                shutil.copy(image_path, tex_label_folder)

                # Read the label file and write to labels.txt
                with open(label_path, 'r') as lf:
                    label = lf.readline().strip()
                    labels_txt.write(f"{base_name}.jpg {label}\n")

    print(f"Organized tex_label folder and created {labels_txt_path}")

def read_image(image_path):
    """Read an image from a file and return it as a numpy array."""
    image = Image.open(image_path)
    return np.array(image)

def read_label(label_path):
    """Read a label from a file and return it as a string."""
    with open(label_path, 'r') as file:
        label = file.readline().strip()
    return label

# Define a function to sort bounding boxes top-to-down, left-to-right
def sort_boxes_top_to_down_left_to_right(boxes, classes):
    # Step 1: Sort by the `y_min` (top edge of bounding box) to get top-to-down order
    sorted_indices = np.argsort(boxes[:, 1])  # Sort by `y_min` (index 1)
    boxes = boxes[sorted_indices]
    classes = classes[sorted_indices]
    
    # Step 2: Within each row (roughly same y_min), sort by `x_min` (left edge of bounding box)
    sorted_boxes = []
    sorted_classes = []
    
    # Define a threshold to consider "rows" (adjust based on your images)
    row_threshold = 0.1 * (np.max(boxes[:, 3]) - np.min(boxes[:, 1]))  # 10% of plate height
    
    current_row = []
    current_classes = []
    
    last_y_min = boxes[0][1]
    
    for i, box in enumerate(boxes):
        if abs(box[1] - last_y_min) > row_threshold:  # Start a new row
            # Sort the current row by `x_min` (left edge of bounding box)
            sorted_row_indices = np.argsort(np.array(current_row)[:, 0])
            sorted_boxes.extend(np.array(current_row)[sorted_row_indices])
            sorted_classes.extend(np.array(current_classes)[sorted_row_indices])
            
            # Reset for the new row
            current_row = [box]
            current_classes = [classes[i]]
            last_y_min = box[1]
        else:
            # Add to the current row
            current_row.append(box)
            current_classes.append(classes[i])
    
    # Sort the last row
    sorted_row_indices = np.argsort(np.array(current_row)[:, 0])
    sorted_boxes.extend(np.array(current_row)[sorted_row_indices])
    sorted_classes.extend(np.array(current_classes)[sorted_row_indices])
    
    return sorted_boxes, sorted_classes

# Function to smooth the bounding box coordinates
def smooth_boxes(current_boxes, prev_boxes, smoothing_factor):
    if len(prev_boxes) == 0:
        return current_boxes  # If no previous boxes, return the current boxes

    smoothed_boxes = []
    for curr_box, prev_box in zip(current_boxes, prev_boxes):
        smoothed_box = smoothing_factor * prev_box + (1 - smoothing_factor) * curr_box
        smoothed_boxes.append(smoothed_box)

    return np.array(smoothed_boxes)

# Function to process frames for license plate detection and recognition
def process_frame(frame, detector, recognizer, smoothing_factor=0.8, prev_boxes = []):
    start_time = time.time()
    # Detect the license plates using the first YOLO model
    detection_results = detector(frame, verbose=False, conf=0.4, device=device)[0]
    current_boxes = detection_results.boxes.xyxy.cpu().numpy()

    # Smooth the bounding boxes with temporal smoothing
    if len(prev_boxes) > 0:
        current_boxes = smooth_boxes(current_boxes, prev_boxes, smoothing_factor)

    # Update previous bounding boxes
    prev_boxes = current_boxes
    plate_text = ""

    for result in current_boxes:
        # Get the bounding box coordinates for each detected license plate
        x1, y1, x2, y2 = map(int, result[:4])  # Ensure coordinates are integers

        # Crop the license plate region from the frame
        plate = frame[y1:y2, x1:x2]

        # Recognize the license plate characters using the second YOLO model
        recog_results = recognizer(plate, verbose=False, device=device)[0]
        recog_boxes = recog_results.boxes.xyxy.cpu()  # Bounding boxes for characters
        recog_classes = recog_results.boxes.cls.cpu()  # Character classes

        if len(recog_boxes) > 0:
            # Sort characters from top to down and left to right
            sorted_boxes, sorted_classes = sort_boxes_top_to_down_left_to_right(
                np.array(recog_boxes), np.array(recog_classes)
            )

            # Decode the plate text
            plate_text = ''.join([DECODE_PLATE[int(cls)] for cls in sorted_classes])
        else:
            plate_text = ""

        # If text is detected
        if len(plate_text.strip()) > 0:
            # Draw the bounding box around the detected license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the recognized text above the bounding box
            cv2.putText(frame, plate_text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Calculate the processing time
    end_time = time.time()
    fps_display = int(1 / (end_time - start_time))

    # Display the processing time on the frame
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, plate_text