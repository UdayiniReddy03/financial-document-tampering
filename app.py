import os
import uuid
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from PIL import Image
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "findocai-secret-key"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetv2b0_patch_tampering.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

GRADCAM_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "gradcam"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

os.makedirs(
    GRADCAM_FOLDER,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading FinDocAI model...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "Make sure the model exists inside the models folder."
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# GET MODEL LAYERS
# ============================================================

efficientnet = model.get_layer(
    "efficientnetv2-b0"
)

augmentation = model.get_layer(
    "data_augmentation"
)

global_pool = model.get_layer(
    "global_average_pooling2d"
)

batch_norm = model.get_layer(
    "batch_normalization"
)

dropout = model.get_layer(
    "dropout"
)

dense = model.get_layer(
    "dense"
)

dropout_1 = model.get_layer(
    "dropout_1"
)

dense_1 = model.get_layer(
    "dense_1"
)


print("EfficientNet layer found.")
print(
    "EfficientNet output:",
    efficientnet.output.shape
)


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# LOAD IMAGE
# ============================================================

def prepare_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    resized = image.resize(
        IMAGE_SIZE
    )

    array = np.array(
        resized,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return image, tf.convert_to_tensor(
        array,
        dtype=tf.float32
    )


# ============================================================
# MODEL FORWARD PASS
# ============================================================

def forward_pass(image):

    # Data augmentation
    augmented = augmentation(
        image,
        training=False
    )

    # EfficientNet
    features = efficientnet(
        augmented,
        training=False
    )

    # Classifier
    x = global_pool(
        features
    )

    x = batch_norm(
        x,
        training=False
    )

    x = dropout(
        x,
        training=False
    )

    x = dense(
        x
    )

    x = dropout_1(
        x,
        training=False
    )

    prediction = dense_1(
        x
    )

    return features, prediction


# ============================================================
# NORMAL PREDICTION
# ============================================================

def predict_image(image_path):

    original, image = prepare_image(
        image_path
    )

    features, prediction = forward_pass(
        image
    )

    probability = float(
        prediction.numpy()[0][0]
    )

    if probability >= 0.5:

        label = "Tampered"

        confidence = probability

    else:

        label = "Genuine"

        confidence = 1.0 - probability


    return {
        "label": label,
        "tampering_probability": probability,
        "confidence": confidence,
        "original": original
    }


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_path):

    print()
    print("=" * 60)
    print("Generating Grad-CAM...")
    print("Image:", image_path)
    print("=" * 60)

    original, image = prepare_image(
        image_path
    )


    # --------------------------------------------------------
    # GRADIENT TAPE
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Augmentation
        augmented = augmentation(
            image,
            training=False
        )

        # EfficientNet feature map
        conv_outputs = efficientnet(
            augmented,
            training=False
        )

        # Watch feature map
        tape.watch(
            conv_outputs
        )

        # Classifier
        x = global_pool(
            conv_outputs
        )

        x = batch_norm(
            x,
            training=False
        )

        x = dropout(
            x,
            training=False
        )

        x = dense(
            x
        )

        x = dropout_1(
            x,
            training=False
        )

        predictions = dense_1(
            x
        )

        # Tampered class
        tampered_probability = predictions[:, 0]


    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    gradients = tape.gradient(
        tampered_probability,
        conv_outputs
    )


    if gradients is None:

        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


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
        max_value + 1e-8
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
        original.size
    )

    heatmap = np.array(
        heatmap_image
    ) / 255.0


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probability = float(
        tampered_probability.numpy()[0]
    )


    if probability >= 0.5:

        label = "Tampered"

    else:

        label = "Genuine"


    return (
        original,
        heatmap,
        label,
        probability
    )


# ============================================================
# SAVE GRAD-CAM IMAGE
# ============================================================

def save_gradcam(
    image_path,
    output_filename
):

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

    original_array = (
        np.array(original)
        / 255.0
    )


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )

    plt.axis(
        "off"
    )


    plt.title(
        f"Prediction: {prediction} | "
        f"Tampering Probability: "
        f"{probability:.2%}"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        GRADCAM_FOLDER,
        output_filename
    )


    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=180
    )

    plt.close()


    print()
    print("=" * 60)
    print("GRAD-CAM RESULT")
    print("=" * 60)

    print(
        "Prediction:",
        prediction
    )

    print(
        "Tampering probability:",
        f"{probability:.2%}"
    )

    print(
        "Saved:",
        output_path
    )


    return (
        output_filename,
        prediction,
        probability
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# DETECT PAGE
# ============================================================

@app.route(
    "/detect",
    methods=["GET", "POST"]
)
def detect():

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "detect.html"
        )


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    if "document" not in request.files:

        flash(
            "Please select a document."
        )

        return redirect(
            url_for("detect")
        )


    file = request.files[
        "document"
    ]


    if file.filename == "":

        flash(
            "Please select an image."
        )

        return redirect(
            url_for("detect")
        )


    if not allowed_file(
        file.filename
    ):

        flash(
            "Only PNG, JPG, JPEG and WEBP files are supported."
        )

        return redirect(
            url_for("detect")
        )


    # --------------------------------------------------------
    # UNIQUE FILENAME
    # --------------------------------------------------------

    extension = file.filename.rsplit(
        ".",
        1
    )[1].lower()


    filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(
        upload_path
    )


    print()
    print("=" * 60)
    print("DOCUMENT RECEIVED")
    print("=" * 60)

    print(
        "Saved:",
        upload_path
    )


    # --------------------------------------------------------
    # PREDICTION + GRAD-CAM
    # --------------------------------------------------------

    try:

        (
            gradcam_filename,
            prediction,
            probability
        ) = save_gradcam(
            upload_path,
            "gradcam_" + filename
        )


    except Exception as e:

        print()
        print("=" * 60)
        print("GRAD-CAM ERROR")
        print("=" * 60)

        print(e)

        flash(
            "Analysis failed. Check the terminal for details."
        )

        return redirect(
            url_for("detect")
        )


    # --------------------------------------------------------
    # RESULT DATA
    # --------------------------------------------------------

    confidence = (
        probability
        if prediction == "Tampered"
        else 1.0 - probability
    )


    original_url = url_for(
        "static",
        filename="uploads/" + filename
    )


    gradcam_url = url_for(
        "static",
        filename="gradcam/" + gradcam_filename
    )


    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(
        "result.html",

        image_url=original_url,

        original_image=original_url,

        gradcam_url=gradcam_url,

        gradcam_image=gradcam_url,

        prediction=prediction,

        result=prediction,

        label=prediction,

        probability=probability,

        tampering_probability=probability,

        confidence=confidence,

        confidence_percent=confidence * 100,

        filename=filename
    )


# ============================================================
# RESULT ROUTE
# ============================================================

@app.route(
    "/result",
    methods=["GET", "POST"]
)
def result():

    # If someone manually opens /result,
    # send them to Detect instead of showing 404.

    if request.method == "GET":

        return redirect(
            url_for("detect")
        )


    # If a form submits directly to /result,
    # process it exactly like /detect.

    if "document" not in request.files:

        return redirect(
            url_for("detect")
        )


    file = request.files[
        "document"
    ]


    if file.filename == "":

        return redirect(
            url_for("detect")
        )


    if not allowed_file(
        file.filename
    ):

        flash(
            "Invalid image format."
        )

        return redirect(
            url_for("detect")
        )


    extension = file.filename.rsplit(
        ".",
        1
    )[1].lower()


    filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(
        upload_path
    )


    try:

        (
            gradcam_filename,
            prediction,
            probability
        ) = save_gradcam(
            upload_path,
            "gradcam_" + filename
        )


    except Exception as e:

        print(
            "Error:",
            e
        )

        flash(
            "Could not analyze the document."
        )

        return redirect(
            url_for("detect")
        )


    confidence = (
        probability
        if prediction == "Tampered"
        else 1.0 - probability
    )


    original_url = url_for(
        "static",
        filename="uploads/" + filename
    )


    gradcam_url = url_for(
        "static",
        filename="gradcam/" + gradcam_filename
    )


    return render_template(
        "result.html",

        image_url=original_url,

        original_image=original_url,

        gradcam_url=gradcam_url,

        gradcam_image=gradcam_url,

        prediction=prediction,

        result=prediction,

        label=prediction,

        probability=probability,

        tampering_probability=probability,

        confidence=confidence,

        confidence_percent=confidence * 100,

        filename=filename
    )


# ============================================================
# HOW IT WORKS
# ============================================================

@app.route(
    "/how-it-works"
)
def how_it_works():

    return render_template(
        "how_it_works.html"
    )


# Also support underscore URL
@app.route(
    "/how_it_works"
)
def how_it_works_alt():

    return render_template(
        "how_it_works.html"
    )


# ============================================================
# PERFORMANCE
# ============================================================

@app.route(
    "/performance"
)
def performance():

    return render_template(
        "performance.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route(
    "/about"
)
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# UPLOAD ALIAS
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "GET":

        return redirect(
            url_for("detect")
        )


    return detect()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return {
        "status": "running",
        "model": "loaded",
        "model_path": MODEL_PATH
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FinDocAI - Page Not Found</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 80px;
                background: #f5f7fb;
                color: #14264d;
            }

            h1 {
                font-size: 60px;
                margin-bottom: 10px;
            }

            a {
                display: inline-block;
                margin-top: 20px;
                padding: 14px 25px;
                background: #3567d6;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }
        </style>
    </head>

    <body>

        <h1>404</h1>

        <h2>Page not found</h2>

        <p>
            The requested page does not exist.
        </p>

        <a href="/">
            Go to FinDocAI Home
        </a>

    </body>
    </html>
    """, 404


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("FINdocAI SERVER")
    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Upload folder:",
        UPLOAD_FOLDER
    )

    print()
    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )