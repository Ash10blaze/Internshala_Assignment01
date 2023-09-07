import pickle
from flask import Flask, render_template, request
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    location = request.form.get('location')
    subscription_length = int(request.form.get('subscription_length'))
    monthly_bill = float(request.form.get('monthly_bill'))
    total_usage_gb = float(request.form.get('total_usage_gb'))
    features = [[age, gender, location, subscription_length, monthly_bill, total_usage_gb]]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'Result is {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
