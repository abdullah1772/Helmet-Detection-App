from flask import Flask, request, render_template, Response, url_for
import cv2
import os
import numpy as np
import threading
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('../Helemet_Detection/Helmet.pt')



# Initialize with a black frame
initial_frame = np.zeros((500, 500, 3), dtype=np.uint8)
ret, buffer = cv2.imencode('.jpg', initial_frame)
current_frame = buffer.tobytes()
video_feed_active = False



def process_video(video_path):
    global current_frame, video_feed_active
    video_feed_active = True  # Video processing starts
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.05)

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        current_frame = buffer.tobytes()

    cap.release()
    video_feed_active = False  # Video processing ends

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        filename = secure_filename(video.filename)
        video_path = os.path.join('uploads', filename)
        video.save(video_path)

        # Start video processing in a separate thread
        threading.Thread(target=process_video, args=(video_path,)).start()

        return render_template('index.html', frame_url=url_for('get_frame'))
    return render_template('index.html', frame_url=None)

@app.route('/frame')
def get_frame():
    global current_frame
    return Response(current_frame, mimetype='image/jpeg')
    # if current_frame is not None and video_feed_active:
    #     return Response(current_frame, mimetype='image/jpeg')
    # else:
    #     return Response('', status=204)

if __name__ == '__main__':
    app.run(debug=True)
