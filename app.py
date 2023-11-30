from flask import Flask, render_template, request, send_file
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict_rainfall():
    min_temp = float(request.form.get('MinTemp'))
    max_temp = float(request.form.get('MaxTemp'))
    rainfall = float(request.form.get('Rainfall'))
    evaporation = float(request.form.get('Evaporation'))
    sunshine = float(request.form.get('Sunshine'))
    wind_gust_speed = float(request.form.get('WindGustSpeed'))
    wind_speed_9am = float(request.form.get('WindSpeed9am'))
    wind_speed_3pm = float(request.form.get('WindSpeed3pm'))
    humidity_9am = float(request.form.get('Humidity9am'))
    humidity_3pm = float(request.form.get('Humidity3pm'))
    pressure_9am = float(request.form.get('Pressure9am'))
    pressure_3pm = float(request.form.get('Pressure3pm'))
    cloud_9am = float(request.form.get('Cloud9am'))
    cloud_3pm = float(request.form.get('Cloud3pm'))
    temp_9am = float(request.form.get('Temp9am'))
    temp_3pm = float(request.form.get('Temp3pm'))

    # prediction
    result = model.predict(np.array([min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, wind_speed_9am, wind_speed_3pm, humidity_9am,
                            humidity_3pm, pressure_9am, pressure_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm]).reshape(1,16))

    if result[0] == 1:
        image_path = "rain.JPG"
    else:
        image_path = "no_rain.JPG"

    return send_file(image_path, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)