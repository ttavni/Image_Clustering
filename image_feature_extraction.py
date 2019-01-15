from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

model_vgg16 = VGG16(weights='imagenet', include_top=False)

def ExtractImageFeature(img):
    image = load_img(img, target_size=(224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_vgg16.predict(x)
    flat_features = features.flatten()
    return flat_features