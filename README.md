## Anand Mohan Hear Disease Detection

# Heart Disease Detection
A deep learning binary classification model built with Keras and trained on 12 input features.  
The model predicts a binary outcome (e.g., presence or absence of a condition) from basic health indicators.

Input Features

The model expects *12 numerical input features*, normalized or standardized in the same way as during training:

| Feature Name | Description |
|---------------|--------------|
| id | Record or subject identifier (may be ignored during prediction) |
| age | Age of the individual (standardized) |
| gender | Gender encoded numerically (e.g., 1 = male, 2 = female) |
| height | Height in cm (standardized) |
| weight | Weight in kg (standardized) |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | Cholesterol level (1–3 scale, standardized) |
| gluc | Glucose level (1–3 scale, standardized) |
| smoke | Smoking status (0 = no, 1 = yes) |
| alco | Alcohol intake (0 = no, 1 = yes) |
| active | Physical activity status (0 = no, 1 = yes) |

---

Model Details

- *Framework:* TensorFlow / Keras  
- *Architecture:* 12 → 10 → 8 → 6 → 4 → 2 → 1  
- *Activations:* Swish (hidden layers), Sigmoid (output)  
- *Loss:* Binary Crossentropy  
- *Optimizer:* Adam  
- *Metric:* Accuracy  

---

How to Use

You can directly download and load the model from this repository for inference.

```python
import requests
from tensorflow.keras.models import load_model
import numpy as np

# Download the model
url = "https://github.com/feethub/heart-disease-prediction.git"
open("heart-disease-prediction.keras", "wb").write(requests.get(url).content)

# Load the model
model = load_model("heart-disease-prediction.keras")

# Example input (12 features)
x = np.array([[-1.732080, -0.436062, 1.364055, 0.443452, -0.847873,
               -0.122182, -0.088238, -0.539322, -0.395720, -0.310879,
               -0.238384, 0.494167]])

# Make a prediction
pred = model.predict(x)
print("Prediction:", float(pred[0][0]))
