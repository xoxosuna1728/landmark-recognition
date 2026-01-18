import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
MODEL_PATH = "landmark_model.keras"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

def load_class_names():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(exts):
            files.append(os.path.join(folder, name))
    return files

def predict_top1(model, img_path, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx])

def show_prediction_grid(test_image_path, k=4):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    # actual class = parent folder name
    actual_class = os.path.basename(os.path.dirname(test_image_path))

    pred_class, conf = predict_top1(model, test_image_path, class_names)

    # pick k random images from predicted class in TRAIN
    pred_train_folder = os.path.join(TRAIN_DIR, pred_class)
    pred_images = list_images(pred_train_folder) if os.path.isdir(pred_train_folder) else []
    random.shuffle(pred_images)
    pred_images = pred_images[:k]

    # plot 1 row: test + k retrieved
    total_cols = 1 + k
    plt.figure(figsize=(3.5 * total_cols, 4))

    # test image
    plt.subplot(1, total_cols, 1)
    img = tf.keras.utils.load_img(test_image_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Image to predict\n\nPredicted: {pred_class}\nConf: {conf:.2f}\nActual: {actual_class}"
    )

    # predicted-class examples
    for i in range(k):
        plt.subplot(1, total_cols, 2 + i)
        if i < len(pred_images):
            img2 = tf.keras.utils.load_img(pred_images[i], target_size=IMG_SIZE)
            plt.imshow(img2)
            plt.title(f"Image of class {pred_class}")
        else:
            plt.text(0.5, 0.5, "No image", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pick ONE test image path here (edit this line)
    test_image_path = "data/test/landmark_1/tm1.jpg"
    show_prediction_grid(test_image_path, k=4)
