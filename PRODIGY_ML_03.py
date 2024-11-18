import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Step 1: Define directories
train_dir = 'train'  # Your 'train' folder containing images
test_dir = 'test1/test1'  # Your 'test1/test1' subfolder containing test images

# Step 2: Load and preprocess the images from your train set
image_data = []
labels = []

# Get the list of files in the train directory
for file_name in os.listdir(train_dir):
    image_path = os.path.join(train_dir, file_name)

    if file_name.endswith(".jpg"):  # Only process .jpg files
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {file_name}")  # Debugging line to see which images fail to load
            continue

        image = cv2.resize(image, (64, 64))  # Resize for feature extraction
        image_data.append(image.flatten())  # Flatten image to a vector of pixels

        # Assign label based on the filename containing 'cat' or 'dog'
        if 'cat' in file_name.lower():  # Check if 'cat' is in the filename
            labels.append(0)  # 0 for cats
        elif 'dog' in file_name.lower():  # Check if 'dog' is in the filename
            labels.append(1)  # 1 for dogs

# Debugging: Check if images are loaded and labeled
print(f"Number of images processed: {len(image_data)}")
print(f"Labels: {labels[:10]}")  # Print first 10 labels to verify

# Step 3: Convert image data and labels to a numpy array
if len(image_data) == 0:
    print("No images loaded. Please check the image paths and filenames.")
else:
    image_data = np.array(image_data)
    labels = np.array(labels)

    # Step 4: Normalize the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(image_data)

    # Step 5: Train a Support Vector Machine (SVM)
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, labels)  # Train the model

    # Step 6: Load and preprocess the test images from the 'test1/test1' subfolder
    test_image_data = []
    test_image_names = []

    for file_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, file_name)

        if file_name.endswith(".jpg"):  # Only process .jpg files
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading test image: {file_name}")
                continue

            image = cv2.resize(image, (64, 64))  # Resize for feature extraction
            test_image_data.append(image.flatten())  # Flatten image to a vector of pixels
            test_image_names.append(file_name)

    # Debugging: Check if test images are loaded
    print(f"Number of test images: {len(test_image_data)}")

    # Step 7: Normalize the test data using the same scaler
    if len(test_image_data) > 0:
        X_test_scaled = scaler.transform(test_image_data)

        # Step 8: Make predictions on the test images using the trained model
        predictions = svm_model.predict(X_test_scaled)

        # Step 9: Prepare the submission file
        submission = pd.DataFrame({'id': test_image_names, 'label': predictions})

        # Step 10: Save the submission file
        submission.to_csv('submission.csv', index=False)

        print("Submission file saved as 'submission.csv'")
    else:
        print("No test images loaded. Please check the image paths and filenames.")

