import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths and Parameters
base_dir = 'C:/Users/KIIT/Desktop/Programs/Dig Bhem'
train_dir = os.path.join(base_dir, 'train')

img_size = 224
batch_size = 50

# Initialization of ImageDataGenerator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Train generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Initialization of VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Function to extract features using VGG16
def extract_features(generator, sample_count):
    features = np.zeros((sample_count, 7, 7, 512))
    labels = np.zeros((sample_count))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg16_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# Extracting features and labels
sample_count = len(train_generator.filenames)
features, labels = extract_features(train_generator, sample_count)

# Reshaping features for SVM
features_flat = features.reshape((sample_count, 7 * 7 * 512))

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_flat, labels, test_size=0.2, random_state=42)

# SVM model Training
start_time = time.time()

svm_model = SVC(kernel='linear', verbose=True)
svm_model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time for SVM on {sample_count} images: {training_time:.2f} seconds")

import joblib

# Save the SVM model
joblib.dump(svm_model, 'svm_model.pkl')

# Save the VGG16 model (optional, if already saved elsewhere)
vgg16_model.save('vgg16_model.h5')


# # Model Evaluation
# y_pred = svm_model.predict(X_val)
# accuracy = accuracy_score(y_val, y_pred)
# print(f'Validation Accuracy: {accuracy:.4f}')
# print('Classification Report:')
# print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))

# def classify_images_in_folder(folder_path, svm_model, vgg16_model):
#     class_names = ['Cat', 'Dog']
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(folder_path, filename)
#             img = load_img(image_path, target_size=(img_size, img_size))
#             img_array = img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)
            
#             features = vgg16_model.predict(img_array)
#             features_flat = features.reshape((1, 7 * 7 * 512))
            
#             prediction = svm_model.predict(features_flat)
#             predicted_class = class_names[int(prediction[0])]
            
#             plt.imshow(img)
#             plt.axis('off')
#             plt.title(f'Predicted: {predicted_class}')
#             plt.show()
            
#             print(f'The image {filename} is classified as: {predicted_class}')

# # Test the classify_images_in_folder function with a folder of test images
# test_folder_path = r'C:\Users\KIIT\Desktop\Programs\Dig Bhem\test'
# classify_images_in_folder(test_folder_path, svm_model, vgg16_model)
