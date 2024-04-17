import io
import base64
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from src import metrics
from src.config import color_map
from tensorflow.keras.models import load_model

app = FastAPI(title="MyApp", description="Design Autonomous Car")


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Load vgg16_model manually
vgg16_model = load_model(
    "/home/hao/repositories/design-autonomous-car/models/vgg16.keras",
    custom_objects={
        "dice_coeff": metrics.dice_coeff,
        "dice_loss": metrics.dice_loss,
        "total_loss": metrics.total_loss,
        "jaccard": metrics.jaccard,
    },
)


@app.post("/segment_image")
def segment_image(image: dict):
    # Retrieve the serialized image from the request
    image_base64 = image["image"]
    image_bytes = base64.b64decode(image_base64)
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for segmentation
    image_array = np.array(image_pil) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Perform segmentation with the VGG16 model
    segmented_image = vgg16_model.predict(image_array)

    segmented_image = np.argmax(segmented_image, axis=-1)
    segmented_image = np.squeeze(segmented_image, axis=0)

    # Créer une image colorée basée sur les classes prédites
    colored_image = np.zeros((224, 224, 3), dtype=np.uint8)

    for class_index in range(8):
        mask = segmented_image == class_index
        colored_image[mask] = color_map[str(class_index)]

    # Create PIL Image from numpy array
    colored_image_pil = Image.fromarray(colored_image)

    # Serialize colored image to send back to Streamlit
    buffered = io.BytesIO()
    colored_image_pil.save(buffered, format="PNG")
    colored_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": colored_image_base64}
