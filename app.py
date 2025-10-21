from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import tensorflow as tf  
model_path = "crops_disease_predication.keras"
load_model = tf.keras.models.load_model(model_path)
# Flask App
app = Flask(__name__)
CORS(app)



# ========== MAIN PREDICT ROUTE ==========

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file'][0]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        img = tf.keras.preprocessing.image.load_img(file, target_size=(160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
        img_array = img_array / 255.0  # Normalize the image

        predictions = load_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        class_labels = {
            0: "sugarcane_Bacterial_Blight",
            1: "Corn_(maize)___Common_Rust",
            1: "Corn_(maize)___Common_Grey_Leaf_Spot",
            2: "Corn_(maize)___healthy",
            3: "Corn_(maize)___Northern_Leaf_Blight",
            11: "sugarcane___Healthy",
            20: "Potato___Early_blight",
            22: "Potato___healthy",
            21: "Potato___Late_blight",
            23: "Raspberry___healthy",
            24: "Sugarcane_red_rot",
            28: "Rice___Brown_spot",
            29: "Rice___Healthy",
            31: "Rice___Leaf_Blast",
            32: "Rice___Neck_Blast",
            31: "Rice___Brown_Rust",
            29: "Wheat___Healthy",
            32: "Rice___Yellow_Rust",
            
        }
        result = {
            "predicted_class": class_labels.get(predicted_class, "Unknown"),
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== RUN APP ==========
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)