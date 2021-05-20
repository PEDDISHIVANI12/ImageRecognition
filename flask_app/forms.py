from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
import cv2
import numpy as np
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/shivani/Desktop/image/training/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'shivani', 'Paula', 'Ilza', 'Z', 'W']
#img =  cv2.imread('/home/shivani/Desktop/image/pics/profile.png')
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
        while True:
            file = request.files["file"]
            file_name = os.path.join('/home/shivani/Desktop/flask_app/uploads/',secure_filename(file.filename))
            file.save(file_name)
            img = cv2.imread(file_name)
            #img = cv2.imread('/home/shivani/' + Downloads/personal_details/my.jpeg)
            if(img is not None):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                print(img)
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 15,
      #  minSize = (int(minW), int(minH)),
            )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
        
                cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                        )
                cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                        )
            c = cv2.imshow('camera',img)
        #return c
            k = cv2.waitKey(0) & 0xff # Press 'ESC' for exiting video
            if k == 1:
                break
        return c
    return render_template("forms.html")

if __name__=='__main__':
    app.run()
