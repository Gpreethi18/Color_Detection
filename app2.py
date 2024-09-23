import cv2
import numpy as np
import webcolors
import matplotlib
from sklearn.cluster import KMeans
import time  # Import time module for timing
from collections import Counter  # Import Counter for counting colors

# Load YOLOv4 model
def load_yolo():
    weights_path = r"yolov4.weights"
    config_path = r"yolov4.cfg"

    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()

    if isinstance(out_layer_indices, np.ndarray):
        out_layer_indices = out_layer_indices.flatten()

    output_layers = [layer_names[i - 1] for i in out_layer_indices]
    return net, output_layers

# Detect persons in the frame using YOLOv4
def detect_persons(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            obj = np.array(detection)
            if len(obj.shape) == 1 and len(obj) >= 7:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Confidence threshold for 'person'
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_boxes = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            detected_boxes.append(box)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Bounding box
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_boxes

# Find the dominant color in the ROI
def find_dominant_color(image, k=1):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)

# Find closest color name
def closest_color(requested_color, css4_colors):
    min_distance = float('inf')
    closest_color_name = None
    for color_name, hex_value in css4_colors.items():
        r_c, g_c, b_c = matplotlib.colors.hex2color(hex_value)
        r_c, g_c, b_c = (r_c * 255, g_c * 255, b_c * 255)
        distance = np.linalg.norm(np.array([r_c, g_c, b_c]) - np.array(requested_color))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name
    return closest_color_name

def get_color_name_from_rgb(rgb_value, css4_colors):
    try:
        return webcolors.rgb_to_name(tuple(rgb_value))
    except ValueError:
        return closest_color(rgb_value, css4_colors)

# Filter the ROI to improve color accuracy
def filter_roi(roi):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_h, low_s, low_v = 0, 50, 50  # lower bounds
    high_h, high_s, high_v = 180, 255, 255  # upper bounds
    mask = cv2.inRange(roi_hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
    filtered_roi = cv2.bitwise_and(roi, roi, mask=mask)
    return filtered_roi

# Initialize video capture and YOLO model
cap = cv2.VideoCapture(0)
net, output_layers = load_yolo()
css4_colors = matplotlib.colors.CSS4_COLORS

start_time = time.time()  # Start the timer
detected_colors = []  # List to store detected colors

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Detect persons in the frame
    boxes = detect_persons(frame, net, output_layers)

    # Process each detected person
    for box in boxes:
        x, y, w, h = box
        # Define ROI for the lower half of the bounding box (dress area)
        roi = frame[y + int(h/2):y + h, x:x + w]  # Use lower half for dress color

        if roi.size == 0:  # Check if ROI is valid
            continue

        # Filter the ROI to improve color accuracy
        filtered_roi = filter_roi(roi)
        if np.count_nonzero(filtered_roi) == 0:  # Ensure ROI has color
            continue

        dominant_color = find_dominant_color(filtered_roi, k=5)  # Increased clusters for accuracy

        # Convert BGR to RGB for matching
        dominant_color_rgb = tuple(dominant_color[::-1])

        # Get the color name based on the dominant color RGB value
        color_name = get_color_name_from_rgb(dominant_color_rgb, css4_colors)

        # Store the detected color
        detected_colors.append(color_name)

        # Draw a rectangle around the detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Optional: another box

        # Display RGB color values and color name on the frame
        rgb_text = f"RGB: {dominant_color_rgb[0]}, {dominant_color_rgb[1]}, {dominant_color_rgb[2]}"
        color_text = f"Color: {color_name}"
        cv2.putText(frame, rgb_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, color_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the color information to the console
        print(f"Detected Color: {color_name} (RGB: {dominant_color_rgb})")

    # Show the video feed
    cv2.imshow('Video Feed', frame)

    # Break the loop after 20 seconds
    if time.time() - start_time > 20:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After exiting the loop, calculate and print the majority color
if detected_colors:
    majority_color = Counter(detected_colors).most_common(1)[0]
    print(f"Majority Color: {majority_color[0]} (Detected {majority_color[1]} times)")
else:
    print("No colors were detected.")
