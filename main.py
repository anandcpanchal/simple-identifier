# ==============================================================================
#      Terminal-Based AI Part Identifier using Feature Extraction
# ==============================================================================
#
# Description:
# This script implements a complete part identification system that runs in the
# terminal. It uses a pre-trained deep learning model (ResNet50) to extract
# unique visual features from images.
#
# Workflow:
# 1.  Generate Sample Data (Optional): Creates a dummy dataset of 100 parts
#     with sample images to simulate a real-world inventory.
# 2.  Indexing: The script processes these images, extracts a feature vector
#     for each, and saves these vectors along with their part names. This only
#     needs to be done once, or when new parts are added.
# 3.  Identification: A user provides a path to an unknown image. The script
#     extracts its features and compares them to the indexed database to find
#     the most visually similar part.
#
# This approach does NOT require re-training the model to add new parts.
# You simply add the new part's images and re-run the indexing.
#
# Required Libraries:
# - tensorflow
# - Pillow (PIL)
# - numpy
# - scikit-learn
#
# To install, run:
# pip install tensorflow Pillow numpy scikit-learn
#
# ==============================================================================

import os
import numpy as np
import sys
from PIL import Image, ImageDraw

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuration ---
INVENTORY_DIR = "./asset/inventory"
FEATURES_FILE = "./output/features/part_features.npy"
NAMES_FILE = "./asset/parts/part_names.npy"
IMG_SIZE = (224, 224)
NUM_PARTS_TO_GENERATE = 10


def create_empty_file(filepath):
    """Creates an empty file at the specified path, creating directories if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, 'a') as f:
        pass  # 'a' mode will create the file if it doesn't exist


# --- 1. Feature Extraction Setup ---

def get_feature_extractor():
    """
    Loads a pre-trained ResNet50 model and prepares it for feature extraction.
    We remove the final classification layer to get the raw feature vectors.
    """
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ Feature extractor (ResNet50) loaded successfully.")
        return feature_extractor
    except Exception as e:
        print(
            f"❌ Model Load Error: Could not load TensorFlow model. Ensure you have internet access for the first download. Error: {e}",
            file=sys.stderr)
        sys.exit(1)


def extract_features(img_path, model):
    """
    Loads an image, preprocesses it for ResNet50, and returns its feature vector.
    """
    if not os.path.exists(img_path):
        print(f"⚠️ Warning: Image path not found: {img_path}", file=sys.stderr)
        return None
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=0)
    return features.flatten()


# --- 2. Indexing and Data Generation ---

def generate_dummy_inventory():
    """
    Creates a dummy directory of parts with sample images of various shapes and colors.
    """
    if os.path.exists(INVENTORY_DIR):
        print(f"ℹ️ Dummy inventory directory '{INVENTORY_DIR}' already exists.")
        return

    print(f"⚙️ Generating a dummy inventory of {NUM_PARTS_TO_GENERATE} parts at '{INVENTORY_DIR}'...")
    os.makedirs(INVENTORY_DIR, exist_ok=True)

    shapes = ['rectangle', 'ellipse', 'triangle']

    for i in range(1, NUM_PARTS_TO_GENERATE + 1):
        part_name = f"part_{i:03d}"
        part_dir = os.path.join(INVENTORY_DIR, part_name)
        os.makedirs(part_dir, exist_ok=True)

        # Each part will be consistently assigned one shape type based on its number
        shape_for_this_part = shapes[i % len(shapes)]

        for j in range(3):  # Create 3 sample images for each part
            img = Image.new('RGB', (100, 100), color='white')
            draw = ImageDraw.Draw(img)

            # Generate a distinct color for the shape variant
            shape_color = ((i * 2) % 255, (i * 5) % 255, (i * 10 + j * 20) % 255)

            # Draw the chosen shape with an outline
            if shape_for_this_part == 'rectangle':
                draw.rectangle([20, 20, 80, 80], fill=shape_color, outline='black')
            elif shape_for_this_part == 'ellipse':
                draw.ellipse([20, 20, 80, 80], fill=shape_color, outline='black')
            elif shape_for_this_part == 'triangle':
                # Defines the three points of the triangle
                points = [(50, 20), (80, 80), (20, 80)]
                draw.polygon(points, fill=shape_color, outline='black')

            img.save(os.path.join(part_dir, f"sample_{j}.png"))

    print("✅ Dummy inventory created successfully.")


def index_inventory(model):
    """
    Iterates through the inventory directory, extracts features for each image,
    and saves the features and corresponding names to .npy files.
    """
    if not os.path.exists(INVENTORY_DIR):
        print(f"❌ Error: Inventory directory '{INVENTORY_DIR}' not found. Please generate it first.", file=sys.stderr)
        return

    print("⚙️ Starting indexing process... This may take a moment.")
    all_features = []
    all_part_names = []

    part_folders = sorted([d for d in os.listdir(INVENTORY_DIR) if os.path.isdir(os.path.join(INVENTORY_DIR, d))])
    total_images = sum(len(files) for _, _, files in os.walk(INVENTORY_DIR))
    processed_images = 0

    for part_name in part_folders:
        part_dir = os.path.join(INVENTORY_DIR, part_name)
        image_files = [f for f in os.listdir(part_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            img_path = os.path.join(part_dir, img_file)
            features = extract_features(img_path, model)
            if features is not None:
                all_features.append(features)
                all_part_names.append(part_name)
            processed_images += 1
            # Simple progress bar
            progress = int((processed_images / total_images) * 50)
            sys.stdout.write(
                f"\r\tProcessing: [{'=' * progress}{' ' * (50 - progress)}] {processed_images}/{total_images}")
            sys.stdout.flush()

    sys.stdout.write("\n")  # Newline after progress bar
    if not all_features:
        print("❌ Error: No images found to index.", file=sys.stderr)
        return

    feature_vectors = np.array(all_features)
    part_names_array = np.array(all_part_names)

    if not os.path.exists(FEATURES_FILE):
        create_empty_file(FEATURES_FILE)
    np.save(FEATURES_FILE, feature_vectors)

    if not os.path.exists(NAMES_FILE):
        create_empty_file(NAMES_FILE)

    np.save(NAMES_FILE, part_names_array)

    print(f"✅ Indexing complete. Saved {len(feature_vectors)} feature vectors to '{FEATURES_FILE}' and '{NAMES_FILE}'.")


# --- 3. Identification ---

def identify_part(unknown_image_path, model):
    """
    Identifies an unknown part by comparing its features to the indexed database.
    """
    if not (os.path.exists(FEATURES_FILE) and os.path.exists(NAMES_FILE)):
        print("❌ Error: Index files not found. Please run the indexing first.", file=sys.stderr)
        return

    indexed_features = np.load(FEATURES_FILE)
    indexed_names = np.load(NAMES_FILE)

    unknown_features = extract_features(unknown_image_path, model)
    if unknown_features is None:
        return

    similarities = cosine_similarity(unknown_features.reshape(1, -1), indexed_features)
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[0, best_match_index]
    identified_part_name = indexed_names[best_match_index]

    print("\n--- Identification Result ---")
    print(f"✅ Identified as:   {identified_part_name}")
    print(f"📈 Confidence:      {best_match_score:.2%}")


# --- 4. Main Command-Line Interface ---

def main_cli():
    """
    Runs the main command-line interface loop for the application.
    """
    # Load the model once at the start
    feature_extractor = get_feature_extractor()

    while True:
        print("\n" + "=" * 50)
        print("          AI Part Identifier CLI")
        print("=" * 50)
        print(f"1. Generate Sample Inventory ({NUM_PARTS_TO_GENERATE} Parts)")
        print("2. Index All Parts in Inventory")
        print("3. Identify a Part from an Image")
        print("4. Exit")

        choice = input(">> Enter your choice: ")

        if choice == '1':
            generate_dummy_inventory()
        elif choice == '2':
            index_inventory(feature_extractor)
        elif choice == '3':
            img_path = input(">> Enter the full path to the image file: ")
            if os.path.exists(img_path):
                identify_part(img_path, feature_extractor)
            else:
                print("❌ Error: File path does not exist.", file=sys.stderr)
        elif choice == '4':
            print("👋 Exiting application. Goodbye!")
            break
        else:
            print("⚠️ Invalid choice, please try again.", file=sys.stderr)


if __name__ == "__main__":
    main_cli()