from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('linear_model.pkl', 'rb'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST']) 
def predict(): 
    try: 
        x = float(request.form['x'])  # Change as per your model inputs 
        
 
        features = np.array([[x]]) 
        prediction = model.predict(features) 
 
        return render_template('index.html', result=f'Predicted Value: {prediction[0]:.2f}') 
    except Exception as e: 
        return render_template('index.html', result=f'Error: {str(e)}') 
 
if __name__ == '__main__': 
    app.run(debug=True)