import os
import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "landmark_model.keras"
TEST_DIR = "data/test"

def load_class_names():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    total = 0
    correct = 0

    for true_label in sorted(os.listdir(TEST_DIR)):
        class_path = os.path.join(TEST_DIR, true_label)
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_path, fname)
            img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]

            total += 1
            if pred_label == true_label:
                correct += 1

    acc = correct / total if total else 0
    print(f"Test images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
