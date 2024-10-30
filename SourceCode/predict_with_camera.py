import numpy as np
np.object = np.object_              # Resolve deprecation warnings for numpy data types
np.bool = np.bool_
np.int = np.int32
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("best_model.keras")

# Load Haar Cascade for face detection
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    image = cv2.resize(image, (48, 48))  # Ensure the image is resized to 48x48
    feature = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    feature = np.array(feature)
    feature = feature.reshape(1, 48, 48, 3)
    return feature / 255.0

# Initialize webcam
webcam=cv2.VideoCapture(0)
# Define emotion labels corresponding to the model's output classes
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

# Loop for real-time emotion detection
while True:
    # Capture frame from webcam
    i,im=webcam.read() 
    # Convert frame to grayscale                             
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
    # Detect faces in the frame       
    faces=face_cascade.detectMultiScale(im,1.3,5)   
    try: 
        # Iterate over detected faces
        for (p,q,r,s) in faces:
             # Extract the face region
            image = gray[q:q+s,p:p+r]
            # Draw rectangle around face
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            # Resize face to 48x48
            image = cv2.resize(image,(48,48))
            # Extract features from the face
            img = extract_features(image)
            # Predict emotion
            pred = model.predict(img)
            # Get the predicted emotion label
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        # Display the frame with predictions
        cv2.imshow("Output",im)
        # Break the loop if 'Esc' key is pressed
        cv2.waitKey(27)
    except cv2.error:
        pass