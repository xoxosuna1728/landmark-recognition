import tensorflow as tf
import numpy as np
import sys

IMG_SIZE = (224, 224)
MODEL_PATH = "landmark_model.keras"

def load_class_names():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]
    results = [(class_names[i], float(probs[i])) for i in top3]
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    results = predict_image(sys.argv[1])
    for name, conf in results:
         print(f"{name}  |  {conf:.3f}")

