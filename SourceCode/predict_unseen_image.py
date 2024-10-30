import os
import cv2
import numpy as np
np.object = np.object_              # Resolve deprecation warnings for numpy data types
np.bool = np.bool_
np.int = np.int32
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model = load_model("best_model.keras")
print("Model loaded successfully")

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the base folder containing all emotion folders
base_folder = 'images2/validation'

def predict_emotion(base_folder, label):
    # Initialize counters
    angry_count = 0
    disgust_count = 0
    fear_count = 0
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    surprise_count = 0
    total_count = 0

    # Define the folder for the given label
    label_folder = os.path.join(base_folder, label)
    
    # Loop through all images in the specified label folder
    for filename in os.listdir(label_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(label_folder, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image {filename}")
                continue
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize to match model input size
            resized = cv2.resize(gray, (48, 48))
            # Convert grayscale image to 3-channel image
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            # Normalize pixel values
            normalized = rgb_image / 255.0
            # Make prediction
            prediction = loaded_model.predict(np.expand_dims(normalized, axis=0))
            # Get the predicted emotion label
            predicted_label = emotion_labels[np.argmax(prediction)]
            
            print(f"Image: {filename}, Predicted Emotion: {predicted_label}")
            total_count += 1
            if predicted_label == 'angry':
                angry_count += 1
            if predicted_label == 'disgust':
                disgust_count += 1
            if predicted_label == 'fear':
                fear_count += 1
            if predicted_label == 'happy':
                happy_count += 1
            if predicted_label == 'neutral':
                neutral_count += 1
            if predicted_label == 'sad':
                sad_count += 1
            if predicted_label == 'surprise':
                surprise_count += 1

    # Return the counts
    return total_count, angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count


# Predict unseen images for each emotion category
angry_total, angry_angry, angry_disgust, angry_fear, angry_happy, angry_neutral, angry_sad, angry_surprise = predict_emotion(base_folder, 'angry')
disgust_total, disgust_angry, disgust_disgust, disgust_fear, disgust_happy, disgust_neutral, disgust_sad, disgust_surprise = predict_emotion(base_folder, 'disgust')
fear_total, fear_angry, fear_disgust, fear_fear, fear_happy, fear_neutral, fear_sad, fear_surprise = predict_emotion(base_folder, 'fear')
happy_total, happy_angry, happy_disgust, happy_fear, happy_happy, happy_neutral, happy_sad, happy_surprise = predict_emotion(base_folder, 'happy')
neutral_total, neutral_angry, neutral_disgust, neutral_fear, neutral_happy, neutral_neutral, neutral_sad, neutral_surprise = predict_emotion(base_folder, 'neutral')
sad_total, sad_angry, sad_disgust, sad_fear, sad_happy, sad_neutral, sad_sad, sad_surprise = predict_emotion(base_folder, 'sad')
surprise_total, surprise_angry, surprise_disgust, surprise_fear, surprise_happy, surprise_neutral, surprise_sad, surprise_surprise = predict_emotion(base_folder, 'surprise')

# Print the results
print(f"In the Angry folder:\nTotal Angry Images: {angry_total}\nAngry: {angry_angry}\nDisgust: {angry_disgust}\nFear: {angry_fear}\nHappy: {angry_happy}\nNeutral: {angry_neutral}\nSad: {angry_sad}\nSurprise: {angry_surprise}\n")
print(f"In the Disgust folder:\nTotal Disgust Images: {disgust_total}\nAngry: {disgust_angry}\nDisgust: {disgust_disgust}\nFear: {disgust_fear}\nHappy: {disgust_happy}\nNeutral: {disgust_neutral}\nSad: {disgust_sad}\nSurprise: {disgust_surprise}\n")
print(f"In the Fear folder:\nTotal Fear Images: {fear_total}\nAngry: {fear_angry}\nDisgust: {fear_disgust}\nFear: {fear_fear}\nHappy: {fear_happy}\nNeutral: {fear_neutral}\nSad: {fear_sad}\nSurprise: {fear_surprise}\n")
print(f"In the Happy folder:\nTotal Happy Images: {happy_total}\nAngry: {happy_angry}\nDisgust: {happy_disgust}\nFear: {happy_fear}\nHappy: {happy_happy}\nNeutral: {happy_neutral}\nSad: {happy_sad}\nSurprise: {happy_surprise}\n")
print(f"In the Neutral folder:\nTotal Neutral Images: {neutral_total}\nAngry: {neutral_angry}\nDisgust: {neutral_disgust}\nFear: {neutral_fear}\nHappy: {neutral_happy}\nNeutral: {neutral_neutral}\nSad: {neutral_sad}\nSurprise: {neutral_surprise}\n")
print(f"In the Sad folder:\nTotal Sad Images: {sad_total}\nAngry: {sad_angry}\nDisgust: {sad_disgust}\nFear: {sad_fear}\nHappy: {sad_happy}\nNeutral: {sad_neutral}\nSad: {sad_sad}\nSurprise: {sad_surprise}\n")
print(f"In the Surprise folder:\nTotal Surprise Images: {surprise_total}\nAngry: {surprise_angry}\nDisgust: {surprise_disgust}\nFear: {surprise_fear}\nHappy: {surprise_happy}\nNeutral: {surprise_neutral}\nSad: {surprise_sad}\nSurprise: {surprise_surprise}\n")
