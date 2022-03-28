~~~Python
import cv2
import numpy as np

# Initilization
LABELS = open('weights/coco.txt').read().strip().split("\n")
image = cv2.imread('media/calling_man.jpg')
model_path = 'weights/coco.onnx'
net = cv2.dnn.readNet(model_path)
layer_name = net.getLayerNames()
layer_name = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
YOLO_SIZE = 640
# Box colors
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
#-----

CONFIDENCE_THRESHOLD = 0.5
def make_final_boxes(image, boxes, probabilities, classIDs):
    (original_image_height, original_image_width) = image.shape[:2] # Take the height and width of the original image.
    
    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    NMS_THRESHOLD = 0.3
    filtered_indices = cv2.dnn.NMSBoxes(boxes, probabilities, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    (original_image_height, original_image_width) = image.shape[:2] # Take the height and width of the original image.
    # Factors to recover the image size to before being decreased to fit YOLO 640 size
    x_factor = original_image_height / YOLO_SIZE
    y_factor = original_image_width / YOLO_SIZE

    # Draw boxes
    for i in filtered_indices:
        color = [int(c) for c in COLORS[classIDs[i]]]

        # Factors to recover the image size to before being decreased to fit YOLO 640 size
        x_factor = original_image_width / YOLO_SIZE
        y_factor = original_image_height / YOLO_SIZE

        (left_x, top_y) = ( int(boxes[i][0] * x_factor), int(boxes[i][1] * y_factor) )
        (width, height) = ( int(boxes[i][2] * x_factor), int(boxes[i][3] * y_factor) )

        cv2.rectangle(image, (left_x, top_y, width, height), color, 2) # Bounding box
        text = "{}: {:.2f}%".format(LABELS[classIDs[i]], probabilities[i]*100)
        #text = "{}: {:.2f}%".format('INCHEON UNIVERSITY', probabilities[i]*100)
        cv2.putText(image, text, (left_x, top_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        coordinate = "Coordinate: (X={}, Y={})".format(left_x, top_y)
        cv2.putText(image, coordinate, (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def image_into_neural_network(image):
    # Image normalization
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (YOLO_SIZE, YOLO_SIZE), swapRB=True, crop=False)

    net.setInput(blob)
    return net.forward(layer_name)[0][0] # Return anchor boxes

def initial_detection(image):
    # Initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    probabilites = []
    classIDs = []

    # Process the anchor boxes
    anchor_boxes = image_into_neural_network(image)
    for anchor_box in anchor_boxes:
        candidate_probabilites = anchor_box[5:] # An anchor box has class probabilites from index 5.
        classID = np.argmax(candidate_probabilites) # Determine the most likely class.
        probability_of_most_likely_class = candidate_probabilites[classID]
        confidence = anchor_box[4] # Confidence means the probability that the found class is correct.
        # It turns out each grid box gets an anchor box 3 times thssat gets bigger in each creation.
        # Anchor boxes are the biggest in the 1st layer and get small in each layer.
        # and it turns out that each anchor box has probabilities of all the classes.

        # Filter out weak predictions by ensuring the detected confidence is greater than the minimum confidence.
        if confidence > CONFIDENCE_THRESHOLD: #and classID == 43:
            # Update our list of bounding boxes, confidences, and class IDs
            (center_x, center_y, width, height) = anchor_box[0:4]
            # Use the center coordinates to derive the top and left corner of the bounding box.
            left_x = center_x - (width / 2)
            top_y = center_y - (height / 2)
            boxes.append([left_x, top_y, width, height])
            probabilites.append(probability_of_most_likely_class)
            classIDs.append(classID)
        
            make_final_boxes(image, boxes, probabilites, classIDs)

initial_detection(image)
cv2.imwrite('detection.png', image)
~~~
## Output
![detection](https://user-images.githubusercontent.com/67142421/160309318-121d888a-aca1-452a-aecd-499c6a04905c.png)
