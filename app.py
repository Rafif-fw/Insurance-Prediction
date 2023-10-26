from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("linear.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = [
            float(request.form['age']),
            float(request.form['bmi']),
            int(request.form['children']),
            1 if request.form['smoker'] == 'Yes' else 0,
            1 if request.form['region'] == 'Northwest' else 0,
            1 if request.form['sex'] == 'Male' else 0
        ]
        input_data = np.array(input_data).reshape(1, -1)
        estimated_cost = model.predict(input_data)
        return render_template('result.html', estimated_cost=estimated_cost[0])

if __name__ == '__main__':
    app.run(debug=True)
