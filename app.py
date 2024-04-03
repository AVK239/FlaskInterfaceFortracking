from flask import Flask, render_template, Response
import cv2
import torch
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
# Для SQLite: 'sqlite:///yourdatabase.db'
# Для PostgreSQL: 'postgresql://username:password@localhost/yourdatabase'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/mydatabase'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class PeopleCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

# Инициализация и загрузка модели YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.15  # порог доверия
model.iou = 0.45  # порог IOU

video_path = 0  # Используйте 0 для веб-камеры
cap = cv2.VideoCapture(video_path)

# Инициализация переменных для подсчета
last_hour = datetime.now()
hourly_count = 0

def generate_frames():
    global last_hour, hourly_count
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            num_people = sum([1 for det in results.xyxy[0] if int(det[5]) == 0])
            hourly_count += num_people

            if datetime.now() - last_hour >= timedelta(hours=1):
                new_count = PeopleCount(count=hourly_count)
                db.session.add(new_count)
                db.session.commit()
                last_hour = datetime.now()
                hourly_count = 0

            for *xyxy, conf, cls in results.xyxy[0]:
                if cls == 0:  # класс 'человек'
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('project.html')

if __name__ == '__main__':
    with app.app_context():
        # Эта проверка не нужна для PostgreSQL, но оставлена для демонстрации.
        if db.engine.url.drivername == 'sqlite':
            db.create_all()
    app.run(debug=True)
