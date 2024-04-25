import numpy as np
import keras

def predict_class(model_path, img_path, class_names):
    # Load the saved model
    model = load_model(model_path)

    # Load the input image as a PIL image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the PIL image to a numpy array
    x = image.img_to_array(img)

    # Preprocess the input image data
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Use the loaded model to predict the class probabilities of the input image
    predictions = model.predict(x)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the predicted class index to the corresponding class name
    predicted_class_name = class_names[predicted_class_index]

    # Return the predicted class name
    return predicted_class_name