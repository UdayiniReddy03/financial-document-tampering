import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering.keras"
)

IMAGE_SIZE = (224, 224)

TARGET_LAYER_NAME = "efficientnetv2-b0"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 50)
print("Loading Grad-CAM model...")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# FIND TARGET EFFICIENTNET LAYER
# ============================================================

target_layer = model.get_layer(
    TARGET_LAYER_NAME
)

print()
print("Candidate layer:", target_layer.name)
print("Output shape:", target_layer.output_shape)

print()
print("=" * 50)
print("Grad-CAM target layer:")
print(target_layer.name)
print("Output shape:", target_layer.output_shape)
print("=" * 50)


# ============================================================
# FIND CLASSIFIER LAYERS
# ============================================================

print()
print("Model layers:")

for i, layer in enumerate(model.layers):
    print(
        i,
        layer.name,
        layer.output_shape
        if hasattr(layer, "output_shape")
        else ""
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_layer_by_name(name):

    try:
        return model.get_layer(name)

    except Exception:

        return None


# ============================================================
# GET CLASSIFIER LAYERS
# ============================================================

global_pool = get_layer_by_name(
    "global_average_pooling2d"
)

batch_norm = get_layer_by_name(
    "batch_normalization"
)

dropout = get_layer_by_name(
    "dropout"
)

dense = get_layer_by_name(
    "dense"
)

dropout_1 = get_layer_by_name(
    "dropout_1"
)

dense_1 = get_layer_by_name(
    "dense_1"
)


print()
print("Classifier layers found:")

print(
    "global_average_pooling2d:",
    global_pool is not None
)

print(
    "batch_normalization:",
    batch_norm is not None
)

print(
    "dropout:",
    dropout is not None
)

print(
    "dense:",
    dense is not None
)

print(
    "dropout_1:",
    dropout_1 is not None
)

print(
    "dense_1:",
    dense_1 is not None
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def load_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    original = Image.open(
        image_path
    ).convert("RGB")

    resized = original.resize(
        IMAGE_SIZE
    )

    image = np.asarray(
        resized,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    return original, image


# ============================================================
# MANUAL FORWARD PASS
# ============================================================

def forward_pass(image):

    """
    Instead of creating a separate Grad-CAM Model,
    we manually pass the image through the same layers.

    This avoids the Keras 3 Functional graph KeyError.
    """

    x = image

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    augmentation = get_layer_by_name(
        "data_augmentation"
    )

    if augmentation is not None:

        x = augmentation(
            x,
            training=False
        )

    # --------------------------------------------------------
    # EFFICIENTNET
    # --------------------------------------------------------

    conv_outputs = target_layer(
        x,
        training=False
    )

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    x = conv_outputs

    if global_pool is not None:

        x = global_pool(x)

    if batch_norm is not None:

        x = batch_norm(
            x,
            training=False
        )

    if dropout is not None:

        x = dropout(
            x,
            training=False
        )

    if dense is not None:

        x = dense(x)

    if dropout_1 is not None:

        x = dropout_1(
            x,
            training=False
        )

    if dense_1 is not None:

        predictions = dense_1(x)

    else:

        # fallback
        predictions = x

    return conv_outputs, predictions


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image_path):

    print()
    print("=" * 50)
    print("Generating Grad-CAM...")
    print(
        "Image:",
        image_path
    )
    print("=" * 50)

    original, image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # GRADIENT TAPE
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = forward_pass(
            image
        )

        print(
            "Feature map shape:",
            conv_outputs.shape
        )

        print(
            "Prediction shape:",
            predictions.shape
        )

        # ----------------------------------------------------
        # TAMpered probability
        # ----------------------------------------------------

        # Model output is (None, 1)
        probability = predictions[:, 0]

    # --------------------------------------------------------
    # CALCULATE GRADIENTS
    # --------------------------------------------------------

    gradients = tape.gradient(
        probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            """
Gradients are None.

The model is not connected correctly
to the EfficientNet feature layer.
"""
        )

    print(
        "Gradient shape:",
        gradients.shape
    )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # REMOVE BATCH DIMENSION
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_gradients,
        axis=-1
    )

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    heatmap = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probability_value = float(
        probability.numpy()[0]
    )

    if probability_value >= 0.5:

        prediction = "Tampered"

    else:

        prediction = "Genuine"

    return (
        original,
        heatmap,
        prediction,
        probability_value
    )


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(image_path):

    (
        original,
        heatmap,
        prediction,
        probability
    ) = generate_gradcam(
        image_path
    )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    original_array = np.asarray(
        original,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )

    plt.axis("off")

    plt.title(
        f"Prediction: {prediction} | "
        f"Tampering Probability: {probability:.2%}",
        fontsize=14
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    filename = os.path.basename(
        image_path
    )

    name, ext = os.path.splitext(
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{name}.png"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    print()
    print("=" * 50)
    print("GRAD-CAM RESULT")
    print("=" * 50)

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Tampering probability: "
        f"{probability:.2%}"
    )

    print(
        f"Saved to:\n{output_path}"
    )

    print("=" * 50)

    return output_path


# ============================================================
# FIND TEST IMAGE
# ============================================================

def find_test_image():

    # Your actual folder
    test_folder = os.path.join(
        BASE_DIR,
        "patches",
        "test",
        "tampered"
    )

    print()
    print(
        "Looking for test images in:"
    )

    print(
        test_folder
    )

    if not os.path.exists(
        test_folder
    ):

        raise FileNotFoundError(
            f"""
Test folder not found:

{test_folder}

Make sure your project contains:

patches/
    test/
        tampered/
"""
        )

    valid_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp"
    )

    files = [

        f

        for f in os.listdir(
            test_folder
        )

        if f.lower().endswith(
            valid_extensions
        )

    ]

    if not files:

        raise FileNotFoundError(
            f"""
No image files found in:

{test_folder}
"""
        )

    files.sort()

    image_path = os.path.join(
        test_folder,
        files[0]
    )

    return image_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("GRAD-CAM TEST")
    print("=" * 50)

    try:

        image_path = find_test_image()

        print()
        print(
            "Testing:",
            image_path
        )

        output_path = save_gradcam(
            image_path
        )

        print()
        print("SUCCESS!")
        print()
        print(
            "Open this file:"
        )

        print(
            output_path
        )

    except Exception as e:

        print()
        print("=" * 50)
        print("ERROR")
        print("=" * 50)

        print(
            str(e)
        )

        import traceback

        traceback.print_exc()