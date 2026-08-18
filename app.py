import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Project Title
st.title("German Traffic Sign Recognition")
st.write("This application uses a CNN model to classify German traffic signs.")

# Dictionary of 43 classes
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


# Loading the model 
@st.cache_resource
def load_model():
  return tf.keras.models.load_model("traffic_sign_model.h5")


model = load_model()

# Upload the picture
uploaded_file = st.file_uploader(
    "Choose a traffic sign image...", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image.", use_column_width=True)

  # Preprocessing
  img = image.resize((30, 30))
  img_array = np.array(img).astype("float32") / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # Predict button
  if st.button("Predict"):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_name = classes.get(predicted_class, "Unknown")

    st.write(f"### Predicted Class ID: {predicted_class}")
    st.write(f"### Sign Name: {class_name}")
    st.success("Prediction completed!")