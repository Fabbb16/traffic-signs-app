import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Step 1: Load CSV
traffic_test = pd.read_csv("Test.csv")

# Step 2: Prepare images and labels
images = []
labels = []

for i in range(min(500, len(traffic_test))):
    image_path = "dataset_trf/" + traffic_test['Path'][i]
    class_id = traffic_test['ClassId'][i]
    img = cv2.imread(image_path)

    if img is None:
        print("No image found")
        continue

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x1 = traffic_test.loc[i, 'Roi.X1']
    x2 = traffic_test.loc[i, 'Roi.X2']
    y1 = traffic_test.loc[i, 'Roi.Y1']
    y2 = traffic_test.loc[i, 'Roi.Y2']

    cropped_image = image[y1:y2, x1:x2]
    resized_image = cv2.resize(cropped_image, (32, 32))

    images.append(resized_image)
    labels.append(class_id)

# Step 3: Show first few images
plt.figure(figsize=(15,5))
for i in range(len(images)):
    plt.subplot(10,50,i+1)
    plt.imshow(images[i])
    plt.title(f"Class id: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 4: Prepare for model
images = np.array(images) / 255.0
labels = np.array(labels)

encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)

print(images.shape)
print(labels.shape)

# Step 5: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 6: Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 7: Train model
model.fit(
    X_train, Y_train,
    epochs=10,
    validation_data=(X_test, Y_test),
    batch_size=32
)

# Step 8: Evaluate
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# -------------------------
# Step 9: Add class names
class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Step 10: Predict a few random images with explanation
num_samples = 5  # how many random images to predict
indices = np.random.choice(len(X_test), num_samples)

for idx in indices:
    image = X_test[idx]
    label = Y_test[idx]

    image_expanded = np.expand_dims(image, axis=0)
    prediction = model.predict(image_expanded)
    predicted_class_idx = np.argmax(prediction)
    true_class_idx = np.argmax(label)
    true_class_id = encoder.classes_[true_class_idx]
    predicted_class_id = encoder.classes_[predicted_class_idx]

    true_name = class_names.get(true_class_id, "Unknown")
    predicted_name = class_names.get(predicted_class_id, "Unknown")

    plt.imshow(image)
    plt.axis('off')
    plt.title(f"True: {true_name} ({true_class_id})\nPredicted: {predicted_name} ({predicted_class_id})")
    plt.show()

    print(f"True label: {true_name} ({true_class_id}), Predicted: {predicted_name} ({predicted_class_id})")
