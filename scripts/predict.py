import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
model = load_model("../models/digit_model.keras")

# Load and preprocess custom test image
img_path = "C:\\Users\\Manjunath R Gowda\\Desktop\\Handwritten_Recognition_Project\\samples\\two.jpg"  # Replace with your image path
img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = img_array.reshape(1, 28, 28) / 255.0  # Normalize

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

# Display
plt.imshow(img_array.reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis("off")
plt.show()
