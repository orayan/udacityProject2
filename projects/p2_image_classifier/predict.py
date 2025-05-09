# predict.py

import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

# ----------- Process Image -----------
def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

# ----------- Predict Top K Classes -----------
def predict(image_path, model, top_k):
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    return top_probs, top_indices

# ----------- Command Line Interface -----------
def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('model_path', help='Path to trained Keras model (.h5)')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', default='label_map.json', help='Path to label map JSON file')

    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False
    )

    # Load label map
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    # Predict
    probs, classes = predict(args.image_path, model, args.top_k)

    # Print results
    print("\nPrediction Results:")
    for prob, cls in zip(probs, classes):
        label = class_names.get(str(cls), f"Class {cls}")
        print(f"{label}: {prob:.4f}")

if __name__ == '__main__':
    main()
