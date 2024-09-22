from flask import Flask, request, render_template, Response, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model_path = 'model/best.onnx'
model = YOLO(model_path,task='detect')

upload_folder = 'uploads'
output_folder = 'outputs'
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file:
        file_extension = file.filename.rsplit('.', 1)[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Save the input image
            input_img_path = os.path.join(upload_folder, file.filename)
            file.save(input_img_path)

            # Process the saved image
            img_array = np.fromfile(input_img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return 'Invalid image file', 400
            
            output_img = process_image(img)
            
            # Save the output image
            output_img_path = os.path.join(output_folder, 'output_img.png')
            cv2.imwrite(output_img_path, output_img)
            
            return render_template('result.html', original=file.filename, output='output_img.png')
        
        elif file_extension in ['mp4', 'avi']:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            return Response(process_video(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        return 'Unsupported file type', 400

    return 'No file uploaded', 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_folder, filename)

@app.route('/outputs/<filename>')
def processed_file(filename):
    return send_from_directory(output_folder, filename)

def process_image(img):
    results = model(img)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        class_id = int(box.cls[0])

        if class_id == 0:
            box_color = (0, 255, 0)
            text_color = (0, 255, 0)
        elif class_id == 1:
            box_color = (0, 0, 255)
            text_color = (0, 0, 255)
        else:
            box_color = (255, 255, 255)
            text_color = (255, 255, 255)

        label = f'{class_id}: {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return img

def process_video(file_path):
    video_capture = cv2.VideoCapture(file_path)

    def generate_frames():
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            processed_frame = process_image(frame)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_stream = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_stream + b'\r\n')

    return generate_frames()

if __name__ == '__main__':
    app.run(debug=True)



















































# from flask import Flask, request, render_template, Response, send_from_directory
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)
# model_path = 'model/best.onnx'  # Path to your YOLO ONNX model
# model = YOLO(model_path)

# upload_folder = 'uploads'
# output_folder = 'outputs'
# os.makedirs(upload_folder, exist_ok=True)
# os.makedirs(output_folder, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files.get('file')
#     if file:
#         file_extension = file.filename.rsplit('.', 1)[-1].lower()
        
#         if file_extension in ['jpg', 'jpeg', 'png']:
#             # Save the input image
#             input_img_path = os.path.join(upload_folder, file.filename)
#             file.save(input_img_path)

#             # Process the saved image
#             img_array = np.fromfile(input_img_path, np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 return 'Invalid image file', 400
            
#             output_img = process_image(img)
            
#             # Save the output image
#             output_img_path = os.path.join(output_folder, 'output_img.png')
#             cv2.imwrite(output_img_path, output_img)
            
#             return render_template('result.html', original=file.filename, output='output_img.png')
        
#         elif file_extension in ['mp4', 'avi']:
#             # Save the input video
#             input_video_path = os.path.join(upload_folder, file.filename)
#             file.save(input_video_path)
            
#             # Process the saved video
#             output_video_path = os.path.join(output_folder, 'output_video.mp4')
#             process_video(input_video_path, output_video_path)

#             return render_template('result.html', original=file.filename, output='output_video.mp4')
        
#         return 'Unsupported file type', 400

#     return 'No file uploaded', 400

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(upload_folder, filename)

# @app.route('/outputs/<filename>')
# def processed_file(filename):
#     return send_from_directory(output_folder, filename)

# def process_image(img):
#     results = model(img)[0]

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = box.conf[0]
#         class_id = int(box.cls[0])

#         if class_id == 0:
#             box_color = (0, 255, 0)
#             text_color = (0, 255, 0)
#         elif class_id == 1:
#             box_color = (0, 0, 255)
#             text_color = (0, 0, 255)
#         else:
#             box_color = (255, 255, 255)
#             text_color = (255, 255, 255)

#         label = f'{class_id}: {conf:.2f}'
#         cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

#     return img

# def process_video(input_path, output_path):
#     video_capture = cv2.VideoCapture(input_path)
#     frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = video_capture.get(cv2.CAP_PROP_FPS)

#     # Define the codec and create VideoWriter object to save the video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     while True:
#         success, frame = video_capture.read()
#         if not success:
#             break

#         processed_frame = process_image(frame)
#         output_video.write(processed_frame)

#     video_capture.release()
#     output_video.release()

# @app.route('/video_feed/<filename>')
# def video_feed(filename):
#     video_path = os.path.join(output_folder, filename)
#     return Response(process_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# def process_video_stream(file_path):
#     video_capture = cv2.VideoCapture(file_path)

#     def generate_frames():
#         while True:
#             success, frame = video_capture.read()
#             if not success:
#                 break

#             processed_frame = process_image(frame)

#             _, buffer = cv2.imencode('.jpg', processed_frame)
#             frame_stream = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_stream + b'\r\n')

#     return generate_frames()

# if __name__ == '__main__':
#     app.run(debug=True)