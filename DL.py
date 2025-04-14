import os

# List all available datasets
base_path = "/kaggle/input/"

print("Datasets available:")
print(os.listdir(base_path))  # This should show both dataset folders

dataset1_path = "/kaggle/input/new-dataset"  # Change 'dataset1' to the actual folder name
dataset2_path = "/kaggle/input/finetune"  # Change 'dataset2' to the actual folder name

# List files inside each dataset
print("Dataset 1 contents:", os.listdir(dataset1_path))
print("Dataset 2 contents:", os.listdir(dataset2_path))

import os
import shutil

# Define dataset paths
source_dirs = ["/kaggle/input/new-dataset/augmented", "/kaggle/input/finetune/finetune"]  # List of datasets to merge
combined_dir = "/kaggle/working/combined_dataset"  # Temporary folder for combined dataset

# Create the combined dataset directory
os.makedirs(combined_dir, exist_ok=True)

# Merge datasets
for source_dir in source_dirs:
    for category in ["cancerous", "non_cancerous"]:
        src_folder = os.path.join(source_dir, category)
        dest_folder = os.path.join(combined_dir, category)
        
        os.makedirs(dest_folder, exist_ok=True)  # Create category folders if they don't exist
        
        # Copy images
        for file in os.listdir(src_folder):
            src_file = os.path.join(src_folder, file)
            dest_file = os.path.join(dest_folder, file)
            shutil.copy(src_file, dest_file)  # Use shutil.copy instead of shutil.copy2

print("✅ Datasets combined successfully!")

import os
import random
import shutil

def split_data_kaggle(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Paths to cancerous and non-cancerous directories
    cancerous_dir = os.path.join(source_dir, 'cancerous')
    non_cancerous_dir = os.path.join(source_dir, 'non_cancerous')
    
    # Create train, validation, and test directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(dest_dir, split, 'cancerous'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'non_cancerous'), exist_ok=True)

    # Function to shuffle and split images
    def shuffle_and_split(images, train_ratio, val_ratio):
        random.shuffle(images)
        train_end = int(train_ratio * len(images))
        val_end = train_end + int(val_ratio * len(images))
        return images[:train_end], images[train_end:val_end], images[val_end:]

    # Process cancerous images
    cancerous_images = os.listdir(cancerous_dir)
    train_cancer, val_cancer, test_cancer = shuffle_and_split(cancerous_images, train_ratio, val_ratio)

    # Move cancerous images to respective folders
    for img in train_cancer:
        shutil.copy(os.path.join(cancerous_dir, img), os.path.join(dest_dir, 'train', 'cancerous', img))
    for img in val_cancer:
        shutil.copy(os.path.join(cancerous_dir, img), os.path.join(dest_dir, 'validation', 'cancerous', img))
    for img in test_cancer:
        shutil.copy(os.path.join(cancerous_dir, img), os.path.join(dest_dir, 'test', 'cancerous', img))

    # Process non-cancerous images
    non_cancerous_images = os.listdir(non_cancerous_dir)
    train_non_cancer, val_non_cancer, test_non_cancer = shuffle_and_split(non_cancerous_images, train_ratio, val_ratio)

    # Move non-cancerous images to respective folders
    for img in train_non_cancer:
        shutil.copy(os.path.join(non_cancerous_dir, img), os.path.join(dest_dir, 'train', 'non_cancerous', img))
    for img in val_non_cancer:
        shutil.copy(os.path.join(non_cancerous_dir, img), os.path.join(dest_dir, 'validation', 'non_cancerous', img))
    for img in test_non_cancer:
        shutil.copy(os.path.join(non_cancerous_dir, img), os.path.join(dest_dir, 'test', 'non_cancerous', img))

    print(f"Data split complete! Check the {dest_dir} directory.")

# Example usage for Kaggle
source_directory = '/kaggle/working/combined_dataset'
destination_directory = '/kaggle/working/split-datasetcombined'

split_data_kaggle(source_directory, destination_directory)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" dimport tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet101  # Replace ResNet50 with ResNet101 or ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

# Paths to your directories
train_dir = '/kaggle/working/split-datasetcombined/train'
val_dir = '/kaggle/working/split-datasetcombined/validation'
test_dir = '/kaggle/working/split-datasetcombined/test'

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,          # Increase rotation range
    width_shift_range=0.1,      # Increase width shift
    height_shift_range=0.1,     # Increase height shift
    shear_range=0.1,            # Increase shear range
    zoom_range=0.2,             
    brightness_range=[0.8, 1.2],  # Increase brightness range
    horizontal_flip=True, 
    fill_mode='nearest'
)


val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                    batch_size=16, class_mode='binary')
val_generator = val_test_datagen.flow_from_directory(val_dir, target_size=(224, 224),
                                                     batch_size=16, class_mode='binary')
test_generator = val_test_datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                                      batch_size=1, class_mode='binary', shuffle=False)

# Base Model: ResNet101 with pre-trained weights from ImageNet
base_model = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))  # Use ResNet152 instead if desired

# Fine-tuning: Unfreeze the top ResNet blocks
for layer in base_model.layers[-150:]:
    layer.trainable = True

# Adding custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Regularization
# Add L2 regularization# Sigmoid activation for binary classification
predictions = Dense(1, activation='sigmoid')(x)
# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with binary_crossentropy for binary classification
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('/kaggle/working/best_resnetcombined_model.keras', save_best_only=True, monitor='val_accuracy', mode='min')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=1)



# Train the model
history = model.fit(train_generator, epochs=15, validation_data=val_generator, 
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# Save the final model
model.save('/kaggle/working/final_resnetcombined_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# ROC AUC Score
test_generator.reset()  # Ensure no shuffling for prediction
Y_pred = model.predict(test_generator)
roc_auc = roc_auc_score(test_generator.classes, Y_pred)
print(f'Test ROC AUC: {roc_auc:.2f}')

# Plot accuracy and loss graphs
def plot_accuracy_loss(history):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')

    plt.show()

# Call the function to plot graphs
plot_accuracy_loss(history)

# Confusion Matrix and Classification Report
def plot_confusion_matrix(generator, model):
    Y_pred = model.predict(generator)
    y_pred = np.round(Y_pred).astype(int).flatten()  # Convert predictions to binary 0/1
    y_true = generator.classes

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_labels = ['Normal', 'Cancer']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=class_labels))

# Call the function to plot confusion matrix and print classification report
plot_confusion_matrix(test_generator, model)