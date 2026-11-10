import os
import io
import uuid
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# NEW FIX: Manually build the architecture to avoid version crashes
# ---------------------------------------------------------
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3), name="input_layer"),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d"),
        tf.keras.layers.BatchNormalization(name="batch_normalization"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_1"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.5, name="dropout"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
    ])
    return model

MODEL_PATH = os.path.join(BASE_DIR, 'model\\best_model10.h5')
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = build_model()
        model.load_weights(MODEL_PATH) # Load ONLY the weights!
        print("✅ SUCCESS: Model loaded perfectly using manual architecture!")
    else:
        print(f"❌ ERROR: Cannot find model file at {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL ERROR loading model: {e}")

# ---------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64)) # FIXED: Changed from 224 to 64 to match your model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
            return layer.name
    return None

def generate_gradcam(img_array, image_path, save_path):
    last_conv_layer_name = get_last_conv_layer(model)
    if not last_conv_layer_name:
        return None 

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0]) if preds.shape[1] > 1 else 0 # Handle binary securely
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Superimpose heatmap
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224)) # Keep visual output large for UI
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img
    
    cv2.imwrite(save_path, superimposed_img)
    return save_path

@app.route('/')
def index():
    sample_folder = os.path.join(app.root_path, 'static', 'samples')
    sample_images = sorted(
        filename for filename in os.listdir(sample_folder)
        if os.path.isfile(os.path.join(sample_folder, filename))
        and os.path.splitext(filename)[1].lower() in {'.png', '.jpg', '.jpeg'}
        and not filename.lower().startswith('gradcam_')
    ) if os.path.isdir(sample_folder) else []
    return render_template('index.html', sample_images=sample_images)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Please upload a valid image (JPG/PNG)'}), 400

    try:
        if model is None:
            return jsonify({'error': 'Server Error: The AI model failed to load.'}), 500
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)[0][0]
        
        # Determine class (Using your > 0.5 logic)
        is_parasitized = prediction > 0.5
        result_label = "Parasitized (Malaria Detected)" if is_parasitized else "Uninfected"
        confidence = float(prediction if is_parasitized else (1 - prediction)) * 100

        # Generate Grad-CAM
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        generate_gradcam(img_array, filepath, gradcam_path)

        return jsonify({
            'result': result_label,
            'confidence': round(confidence, 2),
            'original_image': f"/static/uploads/{filename}",
            'gradcam_image': f"/static/uploads/{gradcam_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Malaria Cell Classification Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 710, f"Result: {data.get('result', 'N/A')}")
    p.drawString(100, 690, f"Confidence Score: {data.get('confidence', 'N/A')}%")
    p.drawString(100, 670, "Note: AI-generated report; please verify with a medical professional.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="malaria_report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)