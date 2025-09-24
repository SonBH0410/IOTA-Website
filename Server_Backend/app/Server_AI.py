import base64

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
from transformers import SamProcessor, SamModel, SamConfig

#Định nghĩa
app = Flask(__name__)
CORS(app)
# ==============================================HÀM SAM============================================
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def create_mask_image(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB and scale to [0, 255]
    return mask_image

def merge_image_and_mask(image, mask_image, alpha=0.6):
    mask_overlay = cv2.addWeighted(image, 0.8, mask_image, alpha, 0)
    return mask_overlay

def show_boxes_on_image(image, samOriginal_seg):
    mask_image = create_mask_image(samOriginal_seg)
    img_merge = merge_image_and_mask(image, mask_image)

    return img_merge

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=3))

# =================================================================================================
# Định nghĩa hàm để xử lý ảnh từ PIL
def preprocess_image_from_pil(image, img_size=640, stride=32, auto=True):
    # Chuyển đổi từ PIL Image sang NumPy array
    im0 = np.array(image)
    im0 = im0[:, :, ::-1]  # Chuyển đổi từ RGB sang BGR

    # Áp dụng letterbox để thay đổi kích thước và padding
    im = letterbox(im0, new_shape=(img_size, img_size), stride=stride, auto=auto)[0]

    # Chuyển đổi từ HWC sang CHW
    im = im.transpose((2, 0, 1))  # HWC to CHW

    # Chuyển đổi từ BGR sang RGB
    im = im[::-1]  # BGR to RGB

    # Tạo mảng liền kề
    im = np.ascontiguousarray(im)

    return im

# ===========================================THIẾT LẬP ĐƯỜNG TRUYỀN============================
#==============SAM================
@app.route('/api/segment', methods=['POST'])
def segment_image():
    # if 'file' not in request.files:
    #     return {'error': 'No file part'}, 400
    #
    # Đọc dữ liệu ảnh từ request và chuyển thành đối tượng Image của Pillow
    # Nhận hình ảnh từ client
    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data))
    x1 = request.form.get('x1')
    x2 = request.form.get('x2')
    y1 = request.form.get('y1')
    y2 = request.form.get('y2')

    try:
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
    except ValueError:
        return jsonify({'error': 'Invalid coordinates'}), 400

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model_SAM = SamConfig.from_pretrained("facebook/sam-vit-base")
    SAMMA = SamModel(config=model_SAM)

    model_weights_path = "weight/weight_Manual.pt"
    SAMMA.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda:0')))

    # Lấy kích thước ảnh gốc
    original_width, original_height = image.size

    # Chuyển đổi đối tượng hình ảnh Pillow thành mảng NumPy
    image_np = np.array(image)

    # Chuyển đổi hình ảnh từ định dạng RGB (của Pillow) sang BGR (của OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Thay đổi kích thước hình ảnh bằng OpenCV
    image = cv2.resize(image_np, (256, 256))

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SAMMA.to(device)

    promptMA = [x1, y1, x2, y2]

    # prepare image + box prompt for the model
    inputsMA = processor(image, input_boxes=[[promptMA]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_SAMMA = SAMMA(**inputsMA, multimask_output=False)

    samMA_seg_prob = torch.sigmoid(outputs_SAMMA.pred_masks.squeeze(1))
    samMA_seg_prob = samMA_seg_prob.cpu().numpy().squeeze()
    samMA_seg = (samMA_seg_prob > 0.5).astype(np.uint8)
    # samMA_seg = cv2.medianBlur(samMA_seg, 7)
    img_v = cv2.resize(samMA_seg, (original_width, original_height))
    img_v = cv2.medianBlur(img_v, 7)
    # img_v = img_v / 255.0
    # (thresh, img_v) = cv2.threshold(img_v, 0, 1, cv2.THRESH_BINARY)
    # print(img_v)

    # Show image with mask and return the merged image
    img_merge = show_boxes_on_image(image_np, img_v)
    # Chuyển ảnh về kích thước ban đầu
    # restored_image = cv2.resize(img_merge, (original_width, original_height))
    # # Chuyển đổi mảng NumPy thành hình ảnh PIL
    restored_image_pil = Image.fromarray(img_merge)
    # Lưu ảnh đen trắng vào một đối tượng BytesIO để gửi về
    img_io = io.BytesIO()
    restored_image_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

#====================YOLOV5==========================
@app.route('/api/detect', methods=['POST'])
def detect_image():
    #Nhận hình ảnh từ client
    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data))

    weights = 'weight/best.pt'
    conf_thres = 0.4
    iou_thres = 0.45
    device = ''

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    # Gọi hàm xử lý
    img = preprocess_image_from_pil(image, img_size=640, stride=32, auto=True)
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # print(f"Shape from PIL: {img.shape}")
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

    # Process detections
    detections = []
    img0 = np.array(image)  # Convert PIL image to numpy array
    annotator = Annotator(img0, line_width=3, example=str(names))
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                xyxy = [float(coord) for coord in xyxy]
                detections.append({'box': xyxy, 'conf': float(conf), 'cls': c})

    img_annotated = annotator.result()
    # # Chuyển đổi mảng NumPy thành hình ảnh PIL
    restored_image_pil = Image.fromarray(img_annotated)
    # Lưu ảnh đen trắng vào một đối tượng BytesIO để gửi về
    img_io = io.BytesIO()
    restored_image_pil.save(img_io, 'PNG')
    img_io.seek(0)

    # Tạo phản hồi JSON chứa hình ảnh và danh sách detection
    response_data = {
        'detections': str(detections),
        'image': base64.b64encode(img_io.getvalue()).decode('utf-8')  # Encode image as base64 string
    }

    # return send_file(img_io, mimetype='image/png')
    return jsonify(response_data)

#====================YOLOV5&SAM==========================
@app.route('/api/segmentauto', methods=['POST'])
def segment_auto_image():
    # Nhận hình ảnh từ client
    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data))

    weights = 'weight/best.pt'
    conf_thres = 0.4
    iou_thres = 0.45
    device = ''

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    # Gọi hàm xử lý
    img = preprocess_image_from_pil(image, img_size=640, stride=32, auto=True)
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # print(f"Shape from PIL: {img.shape}")
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

    # Process detections
    detections = []
    img0 = np.array(image)  # Convert PIL image to numpy array
    annotator = Annotator(img0, line_width=3, example=str(names))
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                xyxy = [float(coord) for coord in xyxy]
                detections.append({'box': xyxy, 'conf': float(conf), 'cls': c})

    #CHỌN CÁI NÀO CÓ ĐỘ TIN CẬY LỚN NHẤT RỒI BIẾN ĐỔI TỌA ĐỘ
    # Tìm phần tử có conf lớn nhất
    # Lấy kích thước ảnh gốc
    temp = []
    original_width, original_height = image.size
    try:
        max_conf_detection = max(detections, key=lambda x: x['conf'])
        temp.append(max_conf_detection)
        # print(max_conf_detection)
        # Lấy giá trị của 'box'
        box = max_conf_detection['box']

        # Tính toán x_min, y_min, x_max, y_max
        x_min = float(256*box[0]/original_width)
        y_min = float(256*box[1]/original_height)
        x_max = float(256*box[2]/original_width)
        y_max = float(256*box[3]/original_height)
        #Tìm x_min, y_min, x_max, y_max
    except:
        x_min = 0
        y_min = 0
        x_max = 255
        y_max = 255

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model_SAM = SamConfig.from_pretrained("facebook/sam-vit-base")
    SAMMA = SamModel(config=model_SAM)

    model_weights_path = "weight/weight_Manual.pt"
    SAMMA.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda:0')))

    # Chuyển đổi đối tượng hình ảnh Pillow thành mảng NumPy
    image_np = np.array(image)
    # Chuyển đổi hình ảnh từ định dạng RGB (của Pillow) sang BGR (của OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Thay đổi kích thước hình ảnh bằng OpenCV
    image = cv2.resize(image_np, (256, 256))
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SAMMA.to(device)

    promptMA = [x_min, y_min, x_max, y_max]
    # print(promptMA)

    # prepare image + box prompt for the model
    inputsMA = processor(image, input_boxes=[[promptMA]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_SAMMA = SAMMA(**inputsMA, multimask_output=False)

    samMA_seg_prob = torch.sigmoid(outputs_SAMMA.pred_masks.squeeze(1))
    samMA_seg_prob = samMA_seg_prob.cpu().numpy().squeeze()
    samMA_seg = (samMA_seg_prob > 0.5).astype(np.uint8)
    # samMA_seg = cv2.medianBlur(samMA_seg, 7)
    img_v = cv2.resize(samMA_seg, (original_width, original_height))
    img_v = cv2.medianBlur(img_v, 7)
    # img_v = img_v / 255.0
    # (thresh, img_v) = cv2.threshold(img_v, 0, 1, cv2.THRESH_BINARY)

    # Show image with mask and return the merged image
    img_merge = show_boxes_on_image(image_np, img_v)
    # Chuyển ảnh về kích thước ban đầu
    # restored_image = cv2.resize(img_merge, (original_width, original_height))
    # # Chuyển đổi mảng NumPy thành hình ảnh PIL
    restored_image_pil = Image.fromarray(img_merge)
    # Lưu ảnh đen trắng vào một đối tượng BytesIO để gửi về
    img_io = io.BytesIO()
    restored_image_pil.save(img_io, 'PNG')
    img_io.seek(0)

    # Tạo phản hồi JSON chứa hình ảnh và danh sách detection
    response_data = {
        'detections': str(temp),
        'image': base64.b64encode(img_io.getvalue()).decode('utf-8')  # Encode image as base64 string
    }

    # return send_file(img_io, mimetype='image/png')
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)