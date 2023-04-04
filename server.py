# Description: This file contains the code for the Flask server that will be used to serve the model.

# Importing Libraries
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
# Creating Flask App
app = Flask(__name__)
# Creating route for the prediction page
@app.route('/predictions', methods=['GET'])
# Creating function for the prediction page
def predict():
    # Creating a variable to store the values X to predict Y (comes from in/felicidad.csv first row )
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    # Creating a variable to store the prediction
    prediction = model.predict(X_test.reshape(1, -1))
    # Returning the prediction
    return jsonify({'prediction': list(prediction)})    

if __name__ == "__main__":    
    # Loading the model
    model = joblib.load('./models/best_model.pkl')
    # Running the app
    app.run(port=8080, debug=True)

