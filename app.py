import os
import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

from models import db
from models.user import User
from config import Config

# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)
app.config.from_object(Config)
app.config["SECRET_KEY"] = "FarmerGuideAI@2026"


# ==========================================
# UPLOAD CONFIGURATION
# ==========================================

UPLOAD_FOLDER = os.path.join("static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# DISEASE DETECTION MODEL
# ==========================================

DISEASE_MODEL_PATH = (
    r"D:\Capstone Project\Farmer---Guide---AI"
    r"\Model\disease_mobilenetv2_best.keras"
)

disease_model = load_model(DISEASE_MODEL_PATH)

print("Disease Detection Model Loaded Successfully")


# ==========================================
# 27 DISEASE CLASSES
# EXACT ORDER USED DURING MODEL TRAINING
# ==========================================

DISEASE_CLASSES = [

    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",

    "Cherry - Healthy",
    "Cherry - Powdery Mildew",

    "Corn - Common Rust",
    "Corn - Gray Leaf Spot",
    "Corn - Healthy",
    "Corn - Northern Leaf Blight",

    "Grape - Black Rot",
    "Grape - Esca",
    "Grape - Healthy",
    "Grape - Leaf Blight",

    "Peach - Bacterial Spot",
    "Peach - Healthy",

    "Pepper - Bacterial Spot",
    "Pepper - Healthy",

    "Potato - Early Blight",
    "Potato - Healthy",
    "Potato - Late Blight",

    "Strawberry - Healthy",
    "Strawberry - Leaf Scorch",

    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Tomato - Late Blight"
]


# ==========================================
# CONFIDENCE THRESHOLD
# ==========================================

CONFIDENCE_THRESHOLD = 60.0


# ==========================================
# DISEASE INFORMATION
# ==========================================

DISEASE_INFO = {

    # --------------------------------------
    # APPLE
    # --------------------------------------

    "Apple - Apple Scab": {
        "symptoms":
            "Olive-green to dark lesions may appear on apple leaves and fruit.",
        "treatment":
            "Remove affected leaves and fruit and follow recommended disease management practices.",
        "prevention":
            "Maintain orchard sanitation and improve air circulation."
    },

    "Apple - Black Rot": {
        "symptoms":
            "Brown or black circular lesions may develop on apple leaves and fruit.",
        "treatment":
            "Remove infected plant material and maintain good orchard sanitation.",
        "prevention":
            "Remove diseased debris and maintain proper pruning."
    },

    "Apple - Cedar Apple Rust": {
        "symptoms":
            "Yellow-orange spots may appear on apple leaves.",
        "treatment":
            "Remove severely affected leaves and follow suitable disease management practices.",
        "prevention":
            "Maintain orchard hygiene and monitor plants regularly."
    },

    "Apple - Healthy": {
        "symptoms":
            "The apple leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition, pruning and regular monitoring."
    },


    # --------------------------------------
    # CHERRY
    # --------------------------------------

    "Cherry - Healthy": {
        "symptoms":
            "The cherry leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition and regular monitoring."
    },

    "Cherry - Powdery Mildew": {
        "symptoms":
            "White powder-like fungal growth may appear on leaves.",
        "treatment":
            "Remove severely affected leaves and follow recommended fungal disease management practices.",
        "prevention":
            "Improve air circulation and avoid excessive humidity."
    },


    # --------------------------------------
    # CORN
    # --------------------------------------

    "Corn - Common Rust": {
        "symptoms":
            "Small reddish-brown rust-colored pustules may appear on corn leaves.",
        "treatment":
            "Remove severely affected plant material and follow suitable disease management practices.",
        "prevention":
            "Use resistant varieties where appropriate and maintain proper crop management."
    },

    "Corn - Gray Leaf Spot": {
        "symptoms":
            "Gray or tan rectangular lesions may develop on corn leaves.",
        "treatment":
            "Remove infected crop debris and follow recommended disease management practices.",
        "prevention":
            "Maintain field sanitation and use suitable resistant varieties."
    },

    "Corn - Healthy": {
        "symptoms":
            "The corn leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper irrigation, nutrition and regular crop monitoring."
    },

    "Corn - Northern Leaf Blight": {
        "symptoms":
            "Long gray-green or tan cigar-shaped lesions may appear on corn leaves.",
        "treatment":
            "Remove severely affected plant material and follow recommended disease control practices.",
        "prevention":
            "Maintain crop sanitation and consider resistant varieties."
    },


    # --------------------------------------
    # GRAPE
    # --------------------------------------

    "Grape - Black Rot": {
        "symptoms":
            "Brown circular lesions may develop on grape leaves and fruit.",
        "treatment":
            "Remove infected plant material and maintain vineyard sanitation.",
        "prevention":
            "Improve air circulation and remove infected debris."
    },

    "Grape - Esca": {
        "symptoms":
            "Leaves may develop irregular discoloration and affected vines may show decline.",
        "treatment":
            "Remove severely affected plant material and follow suitable vineyard disease management practices.",
        "prevention":
            "Maintain vineyard hygiene and monitor vines regularly."
    },

    "Grape - Healthy": {
        "symptoms":
            "The grape leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper irrigation, nutrition and vineyard monitoring."
    },

    "Grape - Leaf Blight": {
        "symptoms":
            "Dark or brown lesions may develop on grape leaves.",
        "treatment":
            "Remove affected leaves and follow recommended disease management practices.",
        "prevention":
            "Maintain good air circulation and reduce prolonged leaf moisture."
    },


    # --------------------------------------
    # PEACH
    # --------------------------------------

    "Peach - Bacterial Spot": {
        "symptoms":
            "Small dark spots may appear on peach leaves and fruit.",
        "treatment":
            "Remove severely affected plant material and maintain field sanitation.",
        "prevention":
            "Avoid overhead watering and maintain proper plant spacing."
    },

    "Peach - Healthy": {
        "symptoms":
            "The peach leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition and regular monitoring."
    },


    # --------------------------------------
    # PEPPER
    # --------------------------------------

    "Pepper - Bacterial Spot": {
        "symptoms":
            "Small dark spots and lesions may appear on pepper leaves.",
        "treatment":
            "Remove severely affected leaves and maintain proper field hygiene.",
        "prevention":
            "Avoid overhead watering and maintain good spacing between plants."
    },

    "Pepper - Healthy": {
        "symptoms":
            "The pepper leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition and regular monitoring."
    },


    # --------------------------------------
    # POTATO
    # --------------------------------------

    "Potato - Early Blight": {
        "symptoms":
            "Dark spots with concentric ring patterns may appear on older leaves.",
        "treatment":
            "Remove affected leaves and follow suitable disease management practices.",
        "prevention":
            "Maintain field sanitation and avoid excessive moisture on leaves."
    },

    "Potato - Healthy": {
        "symptoms":
            "The potato leaf appears healthy without visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue regular monitoring and proper crop management."
    },

    "Potato - Late Blight": {
        "symptoms":
            "Dark irregular lesions may develop on potato leaves.",
        "treatment":
            "Remove severely affected plant material and follow recommended disease control practices.",
        "prevention":
            "Avoid prolonged leaf wetness and monitor plants regularly."
    },


    # --------------------------------------
    # STRAWBERRY
    # --------------------------------------

    "Strawberry - Healthy": {
        "symptoms":
            "The strawberry leaf appears healthy without major visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition and regular monitoring."
    },

    "Strawberry - Leaf Scorch": {
        "symptoms":
            "Reddish-purple spots and scorched areas may appear on strawberry leaves.",
        "treatment":
            "Remove severely affected leaves and follow suitable disease management practices.",
        "prevention":
            "Maintain field sanitation and avoid prolonged leaf moisture."
    },


    # --------------------------------------
    # TOMATO
    # --------------------------------------

    "Tomato - Bacterial Spot": {
        "symptoms":
            "Small dark spots may appear on tomato leaves and other plant parts.",
        "treatment":
            "Remove severely affected leaves and maintain good field hygiene.",
        "prevention":
            "Avoid overhead irrigation and use proper plant spacing."
    },

    "Tomato - Early Blight": {
        "symptoms":
            "Dark lesions with concentric rings commonly appear on older tomato leaves.",
        "treatment":
            "Remove affected leaves and follow suitable disease management practices.",
        "prevention":
            "Maintain sanitation and avoid prolonged leaf moisture."
    },

    "Tomato - Healthy": {
        "symptoms":
            "The tomato leaf appears healthy without visible disease symptoms.",
        "treatment":
            "No disease treatment is required.",
        "prevention":
            "Continue proper watering, nutrition and regular monitoring."
    },

    "Tomato - Late Blight": {
        "symptoms":
            "Dark irregular lesions can spread rapidly across tomato leaves.",
        "treatment":
            "Remove infected plant material and follow recommended disease control practices.",
        "prevention":
            "Monitor plants frequently and reduce prolonged leaf wetness."
    }
}


# ==========================================
# IMAGE VALIDATION
# ==========================================

def validate_image(filepath):

    try:

        img = load_img(
            filepath,
            target_size=(224, 224)
        )

        img_array = img_to_array(img)

        # Very dark image
        if img_array.mean() < 20:
            return False

        # Almost completely uniform image
        if img_array.std() < 10:
            return False

        return True

    except Exception:

        return False


# ==========================================
# DATABASE
# ==========================================

db.init_app(app)

with app.app_context():

    db.create_all()

    crop_model = joblib.load(
        r"D:\Capstone Project\Farmer---Guide---AI\Model\crop_model.pkl"
    )


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# ABOUT
# ==========================================

@app.route("/about")
def about():

    return render_template("about.html")


# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(
            email=email
        ).first()

        if user:

            if user.password == password:

                flash(
                    "Welcome " + user.name,
                    "success"
                )

                return redirect("/dashboard")

            else:

                flash(
                    "Incorrect Password",
                    "danger"
                )

        else:

            flash(
                "User Not Found",
                "warning"
            )

    return render_template("login.html")


# ==========================================
# REGISTER
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        password = request.form.get("password")

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "Email already registered!",
                "danger"
            )

            return redirect("/register")

        user = User(
            name=name,
            email=email,
            mobile=mobile,
            password=password
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration Successful",
            "success"
        )

        return redirect("/login")

    return render_template("register.html")


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


# ==========================================
# CROP RECOMMENDATION
# ==========================================

@app.route("/crop", methods=["GET", "POST"])
def crop():

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])

        temperature = float(
            request.form["temperature"]
        )

        humidity = float(
            request.form["humidity"]
        )

        ph = float(
            request.form["ph"]
        )

        rainfall = float(
            request.form["rainfall"]
        )

        data = np.array([
            [
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]
        ])

        prediction = crop_model.predict(data)

        return render_template(
            "crop.html",
            prediction=prediction[0]
        )

    return render_template("crop.html")


# ==========================================
# DISEASE DETECTION
# ==========================================

@app.route("/disease", methods=["GET", "POST"])
def disease():

    image = None
    prediction = None
    confidence = None
    disease_info = None

    if request.method == "POST":

        file = request.files.get("image")

        # ----------------------------------
        # Check uploaded file
        # ----------------------------------

        if file and file.filename:

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            image = filename

            # ----------------------------------
            # Validate image
            # ----------------------------------

            if not validate_image(filepath):

                prediction = "Invalid Image"

                confidence = 0.0

                disease_info = None

                return render_template(
                    "disease.html",
                    image=image,
                    prediction=prediction,
                    confidence=confidence,
                    disease_info=disease_info
                )

            # ----------------------------------
            # Load image
            # ----------------------------------

            img = load_img(
                filepath,
                target_size=(224, 224)
            )

            # ----------------------------------
            # Convert image to array
            # ----------------------------------

            img_array = img_to_array(img)

            # ----------------------------------
            # Normalize
            # Same preprocessing used during
            # model training
            # ----------------------------------

            img_array = img_array / 255.0

            # ----------------------------------
            # Add batch dimension
            # ----------------------------------

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ----------------------------------
            # Model prediction
            # ----------------------------------

            predictions = disease_model.predict(
                img_array,
                verbose=0
            )

            # ----------------------------------
            # Find predicted class
            # ----------------------------------

            predicted_index = np.argmax(
                predictions[0]
            )

            # ----------------------------------
            # Confidence
            # ----------------------------------

            confidence = float(
                predictions[0][predicted_index] * 100
            )

            # ----------------------------------
            # Apply confidence threshold
            # ----------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction = DISEASE_CLASSES[
                    predicted_index
                ]

                disease_info = DISEASE_INFO.get(
                    prediction
                )

            else:

                prediction = (
                    "Unable to identify disease "
                    "with sufficient confidence"
                )

                disease_info = None

        else:

            prediction = "Please upload an image."

            confidence = 0.0

            disease_info = None

    return render_template(
        "disease.html",
        image=image,
        prediction=prediction,
        confidence=confidence,
        disease_info=disease_info
    )


# ==========================================
# WEATHER
# ==========================================

@app.route("/weather", methods=["GET", "POST"])
def weather():

    weather_data = None
    error = None
    city = ""

    if request.method == "POST":

        city = request.form.get("city", "").strip()

        if not city:
            error = "Please enter a city name."

            return render_template(
                "weather.html",
                weather=None,
                error=error,
                city=city
            )

        try:

            # -----------------------------------------
            # STEP 1: CITY -> LATITUDE/LONGITUDE
            # -----------------------------------------

            geo_url = "https://geocoding-api.open-meteo.com/v1/search"

            geo_params = {
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            }

            geo_response = requests.get(
                geo_url,
                params=geo_params,
                timeout=10
            )

            geo_response.raise_for_status()

            geo_data = geo_response.json()

            if not geo_data.get("results"):

                error = f"Location '{city}' not found."

                return render_template(
                    "weather.html",
                    weather=None,
                    error=error,
                    city=city
                )

            location = geo_data["results"][0]

            latitude = location["latitude"]
            longitude = location["longitude"]

            location_name = location["name"]
            country = location.get("country", "Unknown")

            # -----------------------------------------
            # STEP 2: GET WEATHER DATA
            # -----------------------------------------

            weather_url = "https://api.open-meteo.com/v1/forecast"

            weather_params = {

                "latitude": latitude,
                "longitude": longitude,

                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "weather_code,"
                    "wind_speed_10m"
                ),

                "daily": (
                    "weather_code,"
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max,"
                    "precipitation_sum,"
                    "sunrise,"
                    "sunset"
                ),

                "forecast_days": 7,
                "timezone": "auto",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh"
            }

            weather_response = requests.get(
                weather_url,
                params=weather_params,
                timeout=10
            )

            weather_response.raise_for_status()

            data = weather_response.json()

            # -----------------------------------------
            # WEATHER CODE FUNCTION
            # -----------------------------------------

            def weather_condition(code):

                weather_codes = {

                    0: ("☀️", "Clear Sky"),

                    1: ("🌤️", "Mainly Clear"),
                    2: ("⛅", "Partly Cloudy"),
                    3: ("☁️", "Overcast"),

                    45: ("🌫️", "Fog"),
                    48: ("🌫️", "Depositing Rime Fog"),

                    51: ("🌦️", "Light Drizzle"),
                    53: ("🌦️", "Moderate Drizzle"),
                    55: ("🌧️", "Dense Drizzle"),

                    56: ("🌧️", "Light Freezing Drizzle"),
                    57: ("🌧️", "Dense Freezing Drizzle"),

                    61: ("🌦️", "Slight Rain"),
                    63: ("🌧️", "Moderate Rain"),
                    65: ("🌧️", "Heavy Rain"),

                    66: ("🌧️", "Light Freezing Rain"),
                    67: ("🌧️", "Heavy Freezing Rain"),

                    71: ("🌨️", "Slight Snow"),
                    73: ("🌨️", "Moderate Snow"),
                    75: ("❄️", "Heavy Snow"),

                    77: ("❄️", "Snow Grains"),

                    80: ("🌦️", "Slight Rain Showers"),
                    81: ("🌧️", "Moderate Rain Showers"),
                    82: ("⛈️", "Violent Rain Showers"),

                    85: ("🌨️", "Slight Snow Showers"),
                    86: ("❄️", "Heavy Snow Showers"),

                    95: ("⛈️", "Thunderstorm"),

                    96: ("⛈️", "Thunderstorm with Hail"),
                    99: ("⛈️", "Severe Thunderstorm with Hail")
                }

                return weather_codes.get(
                    code,
                    ("🌦️", "Unknown Weather")
                )

            # -----------------------------------------
            # CURRENT WEATHER
            # -----------------------------------------

            current = data["current"]

            current_icon, current_condition = weather_condition(
                current["weather_code"]
            )

            current_weather = {

                "temperature": round(
                    current["temperature_2m"], 1
                ),

                "condition": current_condition,

                "humidity": round(
                    current["relative_humidity_2m"]
                ),

                "wind_speed": round(
                    current["wind_speed_10m"], 1
                ),

                "icon": current_icon
            }

            # -----------------------------------------
            # 7-DAY FORECAST
            # -----------------------------------------

            daily = data["daily"]

            forecast = []

            for i in range(len(daily["time"])):

                icon, condition = weather_condition(
                    daily["weather_code"][i]
                )

                forecast.append({

                    "date": daily["time"][i],

                    "icon": icon,

                    "condition": condition,

                    "max_temp": round(
                        daily["temperature_2m_max"][i], 1
                    ),

                    "min_temp": round(
                        daily["temperature_2m_min"][i], 1
                    ),

                    "rain_probability": (
                        daily["precipitation_probability_max"][i]
                        if daily["precipitation_probability_max"][i]
                        is not None
                        else 0
                    ),

                    "rain": round(
                        daily["precipitation_sum"][i], 1
                    ),

                    "sunrise": daily["sunrise"][i],

                    "sunset": daily["sunset"][i]
                })

            # -----------------------------------------
            # FARMER WEATHER ADVISORY
            # -----------------------------------------

            advisories = []

            temperature = current_weather["temperature"]
            humidity = current_weather["humidity"]
            wind_speed = current_weather["wind_speed"]

            today_rain_probability = forecast[0]["rain_probability"]

            # Temperature advisory
            if temperature >= 35:

                advisories.append(
                    "☀️ High temperature detected. "
                    "Provide adequate irrigation and avoid unnecessary water loss."
                )

            elif temperature <= 15:

                advisories.append(
                    "🥶 Low temperature detected. "
                    "Monitor crops for cold stress and protect sensitive plants."
                )

            else:

                advisories.append(
                    "🌡️ Temperature is currently suitable "
                    "for normal crop activities."
                )

            # Humidity advisory
            if humidity >= 80:

                advisories.append(
                    "💧 High humidity detected. "
                    "Monitor crops for fungal diseases and maintain proper field ventilation."
                )

            elif humidity <= 40:

                advisories.append(
                    "🏜️ Low humidity detected. "
                    "Crops may require additional irrigation depending on soil moisture."
                )

            else:

                advisories.append(
                    "💧 Humidity is within a moderate range. "
                    "Continue regular crop monitoring."
                )

            # Rain advisory
            if today_rain_probability >= 70:

                advisories.append(
                    "🌧️ High chance of rainfall today. "
                    "Consider postponing irrigation and avoid unnecessary watering."
                )

            elif today_rain_probability >= 40:

                advisories.append(
                    "🌦️ Moderate chance of rainfall. "
                    "Monitor weather conditions before irrigation."
                )

            else:

                advisories.append(
                    "🌤️ Low chance of rainfall. "
                    "Check soil moisture before deciding on irrigation."
                )

            # Wind advisory
            if wind_speed >= 30:

                advisories.append(
                    "💨 Strong winds detected. "
                    "Protect young plants and avoid spraying pesticides during strong winds."
                )

            elif wind_speed >= 15:

                advisories.append(
                    "💨 Moderate wind conditions. "
                    "Use caution while spraying fertilizers or pesticides."
                )

            # -----------------------------------------
            # FINAL WEATHER OBJECT
            # -----------------------------------------

            weather_data = {

                "location": location_name,

                "country": country,

                "current": current_weather,

                "forecast": forecast,

                "advisories": advisories
            }

        except requests.exceptions.RequestException as e:

            print("Weather API Error:", e)

            error = (
                "Unable to connect to weather service. "
                "Please try again later."
            )

        except Exception as e:

            print("Weather Error:", e)

            error = (
                "Something went wrong while fetching weather data."
            )

    return render_template(
        "weather.html",
        weather=weather_data,
        error=error,
        city=city
    )


# ==========================================
# FERTILIZER
# ==========================================

@app.route("/fertilizer")
def fertilizer():

    return render_template("fertilizer.html")


# ==========================================
# MARKET
# ==========================================

@app.route("/market")
def market():

    return render_template("market.html")


# ==========================================
# CHATBOT
# ==========================================

@app.route("/chatbot")
def chatbot():

    return render_template("chatbot.html")


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)