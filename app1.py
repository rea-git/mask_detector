import cv2

from keras.models import load_model
import numpy as np

mask_model = load_model('mask_detector.keras')

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
         # Crop the face region from the frame
        face_region = vid[y:y+h, x:x+w]
        
        # Preprocess the face region (resizing and normalization)
        face_region_resized = cv2.resize(face_region, (100, 100))  # Assuming the model uses 224x224 input
        face_region_resized = face_region_resized / 255.0  # Normalize to [0, 1]
        face_region_resized = np.expand_dims(face_region_resized, axis=0)  # Add batch dimension
        
        # Get predictions from the model
        predictions = mask_model.predict(face_region_resized)
        predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
        labels = ['WithoutMask', 'WithMask'] 
        # Map the predicted class to a label (Mask or No Mask)
        predicted_label = labels[predicted_class]
        
        # Display the prediction result on the image
        label_position = (x, y - 10)  # Position the label above the face
        cv2.putText(vid, predicted_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

