import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

app = Flask(__name__)
CORS(app)


# VGG16
base_vgg = VGG16(weights='imagenet', include_top=False)
x_vgg = base_vgg.output
x_vgg = GlobalAveragePooling2D()(x_vgg)
vgg_model = Model(inputs=base_vgg.input, outputs=x_vgg)

# ResNet50
base_resnet = ResNet50(weights='imagenet', include_top=False)
x_resnet = base_resnet.output
x_resnet = GlobalAveragePooling2D()(x_resnet)
resnet_model = Model(inputs=base_resnet.input, outputs=x_resnet)

# MobileNetV2
base_mobilenet = MobileNetV2(weights='imagenet', include_top=False)
x_mobilenet = base_mobilenet.output
x_mobilenet = GlobalAveragePooling2D()(x_mobilenet)
mobilenet_model = Model(inputs=base_mobilenet.input, outputs=x_mobilenet)




def preprocess_image_for_model(img_pil, model_type):
    target_size = (224, 224) #image size
    
    img_pil = img_pil.resize(target_size)
    img_array = image.img_to_array(img_pil)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'vgg16':
        return vgg_preprocess(expanded_img_array)
    elif model_type == 'resnet50':
        return resnet_preprocess(expanded_img_array)
    elif model_type == 'mobilenet_v2':
        return mobilenet_preprocess(expanded_img_array)
    return None

def extract_features(img_pil, model_type):
    preprocessed_img = preprocess_image_for_model(img_pil, model_type)
    
    if model_type == 'vgg16':
        features = vgg_model.predict(preprocessed_img)
    elif model_type == 'resnet50':
        features = resnet_model.predict(preprocessed_img)
    elif model_type == 'mobilenet_v2':
        features = mobilenet_model.predict(preprocessed_img)
    else:
        return None 
            
    return features.flatten()

def calculate_ssim(img_pil1, img_pil2):
    img1 = np.array(img_pil1.convert('L'))
    img2 = np.array(img_pil2.convert('L'))
    
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    min_h, min_w = min(h1, h2), min(w1, w2)
    
    img1 = cv2.resize(img1, (min_w, min_h))
    img2 = cv2.resize(img2, (min_w, min_h))
    
    score, _ = ssim(img1, img2, full=True)
    return score

def generate_heatmap(img_pil1, img_pil2):
    img1 = np.array(img_pil1.convert('L'))
    img2 = np.array(img_pil2.convert('L'))

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    min_h, min_w = min(h1, h2), min(w1, w2)

    img1 = cv2.resize(img1, (min_w, min_h))
    img2 = cv2.resize(img2, (min_w, min_h))

    diff = cv2.absdiff(img1, img2)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    is_success, buffer = cv2.imencode(".png", heatmap)
    if is_success:
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    return None

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    model_type = request.form.get('model', 'ssim') 

    try:
        img1_pil = Image.open(file1.stream).convert('RGB')
        img2_pil = Image.open(file2.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    score = 0.0
    

    if model_type == 'ssim':
        score = calculate_ssim(img1_pil, img2_pil)
    elif model_type in ['vgg16', 'resnet50', 'mobilenet_v2']:
        features1 = extract_features(img1_pil, model_type)
        features2 = extract_features(img2_pil, model_type)
        score = cosine_similarity([features1], [features2])[0][0]
    

    heatmap_base64 = generate_heatmap(img1_pil, img2_pil)


    return jsonify({
        'similarity_score': float(score),
        'model_used': model_type,
        'heatmap_image': heatmap_base64
    })

if __name__ == '__main__':
    app.run(debug=True)