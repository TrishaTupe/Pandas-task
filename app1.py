from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("linear_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    prediction = model.predict([[area]])
    return render_template('index1.html', prediction_text=f'Predicted House Price: ₹{prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)