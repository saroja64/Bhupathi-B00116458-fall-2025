import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load the trained model
saved_model = load_model("model/Monkeypox.h5")

# Define disease labels
labels = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']


def check(input_img):
    """Predicts the disease type for the given image."""
    print("Your image is:", input_img)

    img_path = os.path.join("images", input_img)
    print("Full image path:", img_path)

    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Load image
    except Exception as e:
        print("Error loading image:", e)
        return "Error: Unable to load image"

    img_array = np.asarray(img) / 255.0  # Normalize image to match training
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

    # Make prediction
    output = saved_model.predict(img_array)
    confidence = np.max(output) * 100  # Confidence percentage
    predicted_index = np.argmax(output)
    predicted_class = labels[predicted_index]

    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")

    return predicted_class, confidence  # Return class & confidence


# Example usage
pred_class, conf = check("chickenpox1.png")
print(f"Final Prediction: {pred_class} with {conf:.2f}% confidence")
