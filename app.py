import os
from flask import Flask, render_template, request
from helper import Helper
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'.jpg'}

MODEL = load_model("models/vgg16_model_scen2confv1.h5")

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

# @app.route('/predict')
# def upload_page():
#     return render_template('predict.html')

@app.route('/predict', methods=['POST','GET'])
def upload_files():
    prediction = None
    if request.method == 'POST':
        uploaded_file = request.files['imageFile']
        if uploaded_file.filename == '':
            return render_template('index.html', prediction='No file selected')
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            message = Helper().check_file(filename,ALLOWED_EXTENSIONS)
            if message:
                return render_template('predict.html', message=message)
            else:
                filepath = "static/img_uploaded/" + filename
                uploaded_file.save(filepath)

        prediction = Helper().tumor_predict(filename, MODEL)
        try:
            os.remove(filepath)
        except:
            pass
        return render_template('predict.html', prediction = prediction, filename=filename)
    return render_template('predict.html', prediction = prediction)

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=False)
