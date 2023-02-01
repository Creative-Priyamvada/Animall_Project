from flask import Flask, request, render_template
from pred_image import predict1

'''app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        file = request.files['file']
        class_name,confidence_score=predict1(file)
        return class_name

if __name__ == '__main__':
    app.run(debug=True, port=8001)'''


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        return file.filename
    return render_template('index.html', title='Prediction of pose in Animal')

@app.route('/style.css')
def stylesheet():
    return render_template('style.css')

if __name__ == '__main__':
    app.run(debug=True)


