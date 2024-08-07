from flask import Flask, render_template, request
from helper import Helper
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'.jpg'}

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
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            message = Helper().check_file(filename,ALLOWED_EXTENSIONS)
            if message:
                return render_template('predict.html', message=message)
            else:
                uploaded_file.save("static/img_uploaded/"+filename)
        
        prediction = Helper().tumor_predict(filename)    
        return render_template('predict.html', prediction = prediction, filename=filename)
    return render_template('predict.html', prediction = prediction)

app.run(host="localhost", port=8000, debug=True)