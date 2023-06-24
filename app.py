import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.save', 'rb'))


@app.route('/')
def home():
    return 'Hello World'
    # return render_template('home.html')
    # return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    test = pd.DataFrame(data)
    A = model.predict(test)
    output = np.reshape(A, (1, -1))
    output = scaler.inverse_transform(output)
    output = output.flatten()
    df = pd.DataFrame(output)
    return df.to_json()


if __name__ == '__main__':
    app.run(debug=True)
