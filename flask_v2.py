import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

from predict import predict

print(' ---- here1 -----')



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        print(' --image-- :',image)
        input_file = image.read()
        with open("input.jpg", "wb") as f:
            f.write(input_file)
        print(' ---- here2 -----')

        
        
        try:
            output,data = predict("input.jpg")
            print('--- detected.jpg is created ----')
            #print('---- output -----',output,type(output))
        except:
            print(' --- Exception Happened --- ')
        

        

        #image_1 = 'detected.jpg'
        #output_file = image_1.read()
        #image_array_1 = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_UNCHANGED)

        _, buffer = cv2.imencode('.jpg', output)
        jpg_as_text = base64.b64encode(buffer).decode()
        return render_template("show_image.html", image=jpg_as_text,data=data)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,port=8001)
