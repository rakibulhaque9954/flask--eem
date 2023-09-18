from flask import Flask, render_template, Response
import cv2
import numpy as np
import onnxruntime as rt
import tensorflow as tf

app = Flask(__name__)


providers = ['CPUExecutionProvider']
output_path = 'vit_quantized.onnx'
output_names = ['dense']
m = rt.InferenceSession(output_path, providers=providers)

# Define the emotion class labels
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Function to generate frames with predictions
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            input_size = (256, 256)
            face = cv2.resize(face, input_size)
            face_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
            face_tensor = np.expand_dims(face_tensor, axis=0)
            emotion_prediction = m.run(output_names, {'input_image': face_tensor})
            predicted_class = np.argmax(emotion_prediction)
            predicted_emotion = CLASS_NAMES[predicted_class]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 125, 255), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (38, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
