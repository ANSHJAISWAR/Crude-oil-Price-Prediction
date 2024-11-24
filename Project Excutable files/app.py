import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Load the model
try:
    model = load_model('crude_oil.h5')  # Adjust path if necessary
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# About page route
@app.route('/about')
def about():
    return render_template("index.html")

# Prediction page route
@app.route('/predict')
def predict():
    print("Predict route accessed")  # Debugging print statement
    return render_template("web.html")

# Form submission for prediction
@app.route('/login', methods=['POST'])
def login():
    try:
        x_input = str(request.form['year'])
        x_input = x_input.split(',')
        print(f"Raw input: {x_input}")

        x_input = [float(i) for i in x_input]  # Convert to float
        print(f"Formatted input: {x_input}")

        x_input = np.array(x_input).reshape(1, -1)
        temp_input = list(x_input[0])  # Flatten
        print(f"Temp input initialized: {temp_input}")

        lst_output = []
        n_steps = 10
        i = 0

        while i < 1:
            if len(temp_input) > 10:
                x_input = np.array(temp_input[-n_steps:])
                print(f"{i} day input: {x_input}")
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(f"{i} day output: {yhat}")
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
            else:
                x_input = np.array(temp_input).reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(f"{i} day output: {yhat}")
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
            i += 1

        print(f"Final predictions: {lst_output}")
        return render_template("web.html", showcase=f'The next day predicted value is: {lst_output[0][0]}')

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("web.html", showcase="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
