from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)

camera=cv2.VideoCapture(0)

def gen_frames():
    while True:
        success,frame=camera.read() #read the camera frame
        if not success:
            break
        else:
            f_detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            faces=f_detector.detectMultiScale(frame,1.1,7)

            e_detector=cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            eyes=e_detector.detectMultiScale(frame,1.1,7)

            s_detector=cv2.CascadeClassifier('Haarcascades/haarcascade_smile.xml')
            smiles=s_detector.detectMultiScale(frame,1.8,35)

            for (x,y,w,h) in eyes:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            for (x,y,w,h) in smiles:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            ret,buffer=cv2.imencode('.jpg',frame)

            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host='face-eye-smile-detector.herokuapp.com')
