import os
from dotenv import load_dotenv
import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from datetime import date
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

from models import db
from models.user import User
from config import Config

load_dotenv()

# FLASK APPLICATION

app = Flask(__name__)
app.config.from_object(Config)
app.config["SECRET_KEY"] = "FarmerGuideAI@2026"

# UPLOAD CONFIGURATION

UPLOAD_FOLDER = os.path.join("static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DISEASE DETECTION MODEL

DISEASE_MODEL_PATH = (
    r"D:\Capstone Project\Farmer---Guide---AI"
    r"\Model\disease_mobilenetv2_best.keras")
disease_model = load_model(DISEASE_MODEL_PATH)
print("Disease Detection Model Loaded Successfully")

# 27 DISEASE CLASSES
# EXACT ORDER USED DURING MODEL TRAINING

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
# CONFIDENCE THRESHOLD
CONFIDENCE_THRESHOLD = 60.0

# DISEASE INFORMATION
DISEASE_INFO = {
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

    # CORN
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

    # GRAPE
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

    # PEACH
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

    # PEPPER
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

    # POTATO
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

    # STRAWBERRY
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

    # TOMATO
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

# IMAGE VALIDATION
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

# DATABASE
db.init_app(app)
with app.app_context():
    db.create_all()
    crop_model = joblib.load(
        r"D:\Capstone Project\Farmer---Guide---AI\Model\crop_model.pkl"
    )

# HOME
@app.route("/")
def home():
    return render_template("index.html")

# ABOUT
@app.route("/about")
def about():

    return render_template("about.html")

# LOGIN
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

# REGISTER
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

# DASHBOARD
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# CROP RECOMMENDATION
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

# DISEASE DETECTION
@app.route("/disease", methods=["GET", "POST"])
def disease():

    image = None
    prediction = None
    confidence = None
    disease_info = None

    if request.method == "POST":

        file = request.files.get("image")
        # Check uploaded file

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

            # Validate image
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
            # Load image
            img = load_img(
                filepath,
                target_size=(224, 224)
            )
            # Convert image to array
            img_array = img_to_array(img)

            # Normalize
            # Same preprocessing used during
            # model training

            img_array = img_array / 255.0
            # Add batch dimension

            img_array = np.expand_dims(
                img_array,
                axis=0
            )
            # Model prediction
            predictions = disease_model.predict(
                img_array,
                verbose=0
            )

            # Find predicted class

            predicted_index = np.argmax(
                predictions[0]
            )
            # Confidence
            confidence = float(
                predictions[0][predicted_index] * 100)
            
            # Apply confidence threshold
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
# WEATHER
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
            # STEP 1: CITY -> LATITUDE/LONGITUDE
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

            # STEP 2: GET WEATHER DATA
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

            # WEATHER CODE FUNCTION

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

            # CURRENT WEATHER
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
            # 7-DAY FORECAST

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

            # FARMER WEATHER ADVISORY
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

            # FINAL WEATHER OBJECT
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

# FERTILIZER RECOMMENDATION
FERTILIZER_PROFILES = {
    "Urea": {
        "N": 46,
        "P": 0,
        "K": 0,
        "use": "Nitrogen requirement"
    },

    "DAP": {
        "N": 18,
        "P": 46,
        "K": 0,
        "use": "Nitrogen + Phosphorus requirement"
    },

    "10-26-26": {
        "N": 10,
        "P": 26,
        "K": 26,
        "use": "Balanced NPK with higher Potassium"
    },

    "14-35-14": {
        "N": 14,
        "P": 35,
        "K": 14,
        "use": "Higher Phosphorus requirement"
    },

    "17-17-17": {
        "N": 17,
        "P": 17,
        "K": 17,
        "use": "Balanced NPK requirement"
    },

    "20-20": {
        "N": 20,
        "P": 20,
        "K": 0,
        "use": "Nitrogen + Phosphorus requirement"
    },

    "28-28": {
        "N": 28,
        "P": 28,
        "K": 0,
        "use": "Higher Nitrogen + Phosphorus requirement"
    }
}


def recommend_fertilizers(
    nitrogen,
    phosphorus,
    potassium,
    soil_type,
    crop_type,
    top_n=3
):

    recommendations = []

    # Nutrient requirement
    n_need = max(0, 40 - nitrogen)
    p_need = max(0, 40 - phosphorus)
    k_need = max(0, 20 - potassium)

    # Soil adjustment
    soil_bonus = {
        "Sandy": {
            "N": 1.20,
            "P": 1.00,
            "K": 1.15
        },

        "Loamy": {
            "N": 1.00,
            "P": 1.00,
            "K": 1.00
        },

        "Clayey": {
            "N": 0.90,
            "P": 1.00,
            "K": 0.95
        },

        "Black": {
            "N": 0.95,
            "P": 1.00,
            "K": 1.00
        },

        "Red": {
            "N": 1.10,
            "P": 1.05,
            "K": 1.00
        }
    }

    soil_factor = soil_bonus.get(
        soil_type,
        {
            "N": 1.00,
            "P": 1.00,
            "K": 1.00
        }
    )

    for fertilizer, profile in FERTILIZER_PROFILES.items():

        n_score = (
            n_need
            * profile["N"]
            * soil_factor["N"]
        )

        p_score = (
            p_need
            * profile["P"]
            * soil_factor["P"]
        )

        k_score = (
            k_need
            * profile["K"]
            * soil_factor["K"]
        )

        total_score = (
            n_score
            + p_score
            + k_score
        )

        # Crop compatibility
        crop_bonus = 0

        if crop_type in [
            "Paddy",
            "Wheat",
            "Maize",
            "Barley"
        ]:

            if fertilizer in [
                "Urea",
                "DAP",
                "28-28"
            ]:
                crop_bonus = 15

        elif crop_type in [
            "Pulses",
            "Millets",
            "Oil seeds"
        ]:

            if fertilizer in [
                "17-17-17",
                "10-26-26",
                "14-35-14"
            ]:
                crop_bonus = 15

        elif crop_type in [
            "Cotton",
            "Sugarcane",
            "Tobacco"
        ]:

            if fertilizer in [
                "10-26-26",
                "17-17-17",
                "28-28"
            ]:
                crop_bonus = 15

        final_score = total_score + crop_bonus

        recommendations.append({
            "Fertilizer": fertilizer,
            "Score": round(final_score, 2),
            "N Contribution": profile["N"],
            "P Contribution": profile["P"],
            "K Contribution": profile["K"],
            "Reason": profile["use"]
        })

    recommendations.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    return recommendations[:top_n]


@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():

    recommendations = None
    error = None

    if request.method == "POST":

        try:

            temperature = float(
                request.form["temperature"]
            )

            humidity = float(
                request.form["humidity"]
            )

            moisture = float(
                request.form["moisture"]
            )

            nitrogen = float(
                request.form["nitrogen"]
            )

            phosphorus = float(
                request.form["phosphorus"]
            )

            potassium = float(
                request.form["potassium"]
            )

            soil_type = request.form["soil_type"]

            crop_type = request.form["crop_type"]

            recommendations = recommend_fertilizers(
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                soil_type=soil_type,
                crop_type=crop_type,
                top_n=3
            )

        except (ValueError, KeyError):

            error = "Please enter valid fertilizer input values."

    return render_template(
        "fertilizer.html",
        recommendations=recommendations,
        error=error
    )

# ============================================================
# MARKET PRICES - AGMARKNET
# ============================================================

AGMARKNET_BASE_URL = "https://api.agmarknet.gov.in/v1"

AGMARKNET_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "User-Agent": "Mozilla/5.0"
}
def normalize_text(value):
    """Normalize text for reliable market-name matching."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_id(value):
    """Normalize IDs before comparison."""
    if value is None:
        return ""
    return str(value).strip()

def get_agmarknet_filters():
    """
    Fetch Agmarknet filter data.
    Uses local cache if Agmarknet temporarily returns an error.
    """

    import os
    import json

    url = f"{AGMARKNET_BASE_URL}/daily-price-arrival/filters"

    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Data",
        "Market"
    )

    cache_file = os.path.join(
        cache_dir,
        "agmarknet_filters_cache.json"
    )

    os.makedirs(cache_dir, exist_ok=True)

    try:
        response = requests.get(
            url,
            headers=AGMARKNET_HEADERS,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("status"):
            raise Exception("Agmarknet filter API failed.")

        data = result.get("data", {})

        # Save successful official response locally
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("AGMARKNET FILTERS: Fresh data loaded and cached.")

        return data

    except Exception as e:

        print(
            "AGMARKNET FILTER API ERROR:",
            type(e).__name__,
            str(e)
        )

        # Try local cache
        if os.path.exists(cache_file):

            try:
                with open(
                    cache_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    cached_data = json.load(f)

                print(
                    "AGMARKNET FILTERS: Using local cached data."
                )

                return cached_data

            except Exception as cache_error:

                print(
                    "AGMARKNET CACHE ERROR:",
                    str(cache_error)
                )

        print(
            "AGMARKNET FILTERS: No local cache available."
        )

        return {}

def get_market_filter_data():
    """
    Prepare clean dropdown/filter data from Agmarknet.
    District names are loaded separately because the
    filters API is currently returning empty district_data.
    """

    data = get_agmarknet_filters()

    commodities = data.get("cmdt_data", [])
    states = data.get("state_data", [])
    markets = data.get("market_data", [])

    print("FILTER DATA KEYS:", list(data.keys()))
    print("FIRST MARKET RAW:", markets[:3])

    # ========================================================
    # STATES
    # ========================================================

    state_list = []

    for item in states:

        state_id = item.get("state_id")

        state_name = str(
            item.get("state_name") or ""
        ).strip()

        if state_id is None or not state_name:
            continue

        state_list.append({
            "id": state_id,
            "name": state_name
        })

    # Remove duplicate states
    state_unique = {}

    for item in state_list:
        state_unique[str(item["id"])] = item

    state_list = list(state_unique.values())

    # ========================================================
    # COMMODITIES
    # ========================================================

    # ------------------------------------------------------------
    # FARMER / AGRICULTURE FOCUSED COMMODITY LIST
    # ------------------------------------------------------------

    commodity_list = []

    # Clearly NON-agricultural / unwanted mandi items
    excluded_keywords = [
        # Animals / livestock
        "bull",
        "calf",
        "camel hair",
        "cock",
        "cow",
        "duck",
        "egg",
        "fish",
        "goat",
        "goat hair",
        "hen",
        "hilsa",
        "ox",
        "pig",
        "pigs",
        "prawn",
        "ram",
        "she buffalo",
        "she goat",
        "sheep",
        "shrimp",
        "silk cocoon",
        "skin and hide",
        "wool",

        # Industrial / wood / fuel
        "engineered wood",
        "processed wood",
        "firewood",
        "imarti wood",
        "popular wood",
        "resinwood",
        "sandalwood",
        "stone pulverizer",
        "wood veneer",
        "wooden stick",
        "wooden pole",

        # Industrial materials
        "bamboo",
        "bamboo shoot",
        "coal",
        "charcoal",
        "cement",
        "fuel",
        "iron",
        "metal",
        "plastic",
        "rubber",
        "scrap",
        "steel",
        "waste",

        # Non-raw / processed products
        "maida atta",
        "wheat atta",
        "gramflour",
        "meal maker",
        "processed products",
        "turmeric powder",
        "mango powder",
        "sabu dan",
        "soji",
        "dalia",
        "beaten rice",
        "broken rice",

        # Animal-derived / non-crop food products
        "butter",
        "ghee",
        "honey",
        "khoya",
        "milk",

        # Tobacco
        "tobacco"
    ]

    # Exact unwanted / unclear entries
    excluded_exact_names = {
        "seek",
        "bop",
        "menetc*3",
        "irish",
        "buttery",
        "cane",
        "wax",
        "lint"
    }

    for item in commodities:

        commodity_name = str(
            item.get("cmdt_name") or ""
        ).strip()

        commodity_id = item.get("cmdt_id")
        group_id = item.get("cmdt_group_id")

        if commodity_id is None or not commodity_name:
            continue

        name_lower = commodity_name.lower().strip()

        # Exact exclusions
        if name_lower in excluded_exact_names:
            continue

        # Keyword exclusions
        if any(
            keyword in name_lower
            for keyword in excluded_keywords
        ):
            continue

        # Keep agriculture/farming commodity
        commodity_list.append({
            "id": commodity_id,
            "name": commodity_name,
            "group_id": group_id
        })


    # ------------------------------------------------------------
    # Remove duplicate commodities
    # ------------------------------------------------------------

    commodity_unique = {}

    for item in commodity_list:
        commodity_unique[str(item["id"])] = item

    commodity_list = list(
        commodity_unique.values()
    )

    # ------------------------------------------------------------
    # Sort alphabetically
    # ------------------------------------------------------------

    commodity_list = sorted(
        commodity_list,
        key=lambda x: x["name"].lower()
    )
    print("\n")
    print("=" * 60)
    print("FINAL SELECT CROP LIST")
    print("=" * 60)

    for i, item in enumerate(commodity_list, start=1):
        print(f"{i}. {item['name']}")

    print("=" * 60)
    print(f"TOTAL CROPS: {len(commodity_list)}")
    print("=" * 60)

    print(
        "TOTAL FARMER COMMODITIES:",
        len(commodity_list)
    )

    # ========================================================
    # DISTRICTS
    # ========================================================

    district_list = []

    try:

        location_url = (
            f"{AGMARKNET_BASE_URL}/location/state"
        )
        page = 1
        all_location_states =[]

        while True:
            response = requests.get(
                location_url,
                params={"page": page},
                headers=AGMARKNET_HEADERS,
                timeout=20
            )

            print(
                f"LOCATION STATE PAGE{page} STATUS:",
                response.status_code
            )

            response.raise_for_status()

            location_data = response.json()
            if page == 1:
                print(
                    "LOCATION STATE RESPONSE KEYS:",
                    list(location_data.keys())
                    if isinstance(location_data, dict)
                    else type(location_data).__name__
                )

            page_states = location_data.get(
                "states",
                []
            )

            print(
                f"LOCATION STATE PAGE {page} STATES:",
                len(page_states)
            )

            if not page_states:
                break

            all_location_states.extend(
                page_states
            )

            pagination = location_data.get(
                "pagination",
                {}
            )

            total_pages = pagination.get(
                "total_pages"
            )

            if total_pages is not None:
                if page >= int(total_pages):
                    break
            else:
                if len(page_states) < 10:
                    break

            page += 1

        print(
            "TOTAL LOCATION STATES RECEIVED:",
            len(all_location_states)
        )

        for state_item in all_location_states:

            state_id = (
                state_item.get("state_id")
                or state_item.get("id")
            )

            state_name = (
                state_item.get("state_name")
                or state_item.get("name")
            )

            state_districts = (
                state_item.get("districts")
                or state_item.get("district_data")
                or []
            )

            for district_item in state_districts:

                district_id = (
                    district_item.get("district_id")
                    or district_item.get("id")
                )

                district_name = (
                    district_item.get("district_name")
                    or district_item.get("name")
                )

                if (
                    district_id is not None
                    and district_name
                ):
                    district_list.append({
                        "id": district_id,
                        "name": str(
                            district_name
                        ).strip(),
                        "state_id": state_id,
                        "state_name": state_name
                    })

    except Exception as e:

        print(
            "AGMARKNET DISTRICT API ERROR:",
            type(e).__name__,
            str(e)
        )

    # ========================================================
    # FALLBACK DISTRICTS
    # ========================================================

    if not district_list:

        raw_districts = data.get(
            "district_data",
            []
        )

        print(
            "FILTER DISTRICTS RECEIVED:",
            len(raw_districts)
        )

        for item in raw_districts:

            district_id = item.get(
                "district_id"
            )

            district_name = item.get(
                "district_name"
            )

            state_id = item.get(
                "state_id"
            )

            if (
                district_id is not None
                and district_name
            ):
                district_list.append({
                    "id": district_id,
                    "name": str(
                        district_name
                    ).strip(),
                    "state_id": state_id
                })

    # Remove duplicate districts
    district_unique = {}

    for item in district_list:

        key = (
            str(item.get("state_id")),
            str(item.get("id"))
        )

        district_unique[key] = item

    district_list = list(
        district_unique.values()
    )

    # ========================================================
    # MARKETS
    # ========================================================

    market_list = []

    for item in markets:

        market_id = item.get("id")

        market_name = str(
            item.get("mkt_name") or ""
        ).strip()

        district_id = item.get(
            "district_id"
        )

        state_id = item.get(
            "state_id"
        )

        if (
            market_id is None
            or not market_name
        ):
            continue

        market_list.append({
            "id": market_id,
            "name": market_name,
            "district_id": district_id,
            "state_id": state_id
        })

    # ========================================================
    # SORT
    # ========================================================

    state_list.sort(
        key=lambda x: x["name"].lower()
    )

    commodity_list.sort(
        key=lambda x: x["name"].lower()
    )

    district_list.sort(
        key=lambda x: x["name"].lower()
    )

    market_list.sort(
        key=lambda x: x["name"].lower()
    )

    # ========================================================
    # FINAL DEBUG
    # ========================================================

    print("TOTAL STATES:", len(state_list))
    print("TOTAL DISTRICTS:", len(district_list))
    print("TOTAL MARKETS:", len(market_list))

    print(
        "FIRST 5 DISTRICTS:",
        district_list[:5]
    )

    return (
        state_list,
        commodity_list,
        district_list,
        market_list
    )

def get_agmarknet_month_data(year, month, state_id, commodity_id):
    """
    Fetch official Agmarknet price data for
    one state + one commodity + one month.
    """

    url = (
        f"{AGMARKNET_BASE_URL}/"
        "prices-and-arrivals/"
        "date-wise/specific-commodity"
    )

    params = {
        "year": year,
        "month": month,
        "stateId": state_id,
        "commodityId": commodity_id,
        "includeExcel": "false"
    }

    print("\n-----------------------------------")
    print("AGMARKNET MONTH DATA REQUEST")
    print("Year:", year)
    print("Month:", month)
    print("State ID:", state_id)
    print("Commodity ID:", commodity_id)
    print("-----------------------------------")

    response = requests.get(
        url,
        params=params,
        headers=AGMARKNET_HEADERS,
        timeout=30
    )

    print("Agmarknet Status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    if not result.get("success"):
        raise Exception(
            result.get(
                "message",
                "Agmarknet price API failed."
            )
        )

    print(
        "Markets Returned:",
        len(result.get("markets", []))
    )

    return result

@app.route("/market", methods=["GET", "POST"])
def market():

    prices = []
    error = None
    fallback_message  = None

    states = []
    commodities = []
    districts = []
    markets = []

    selected_crop = ""
    selected_state = ""
    selected_district = ""
    selected_market = ""

    try:

        (
            states,
            commodities,
            districts,
            markets
        ) = get_market_filter_data()
        print("TOTAL STATES:", len(states))
        print("TOTAL DISTRICTS:", len(districts))
        print("TOTAL MARKETS:", len(markets))
        print("FIRST 5 DISTRICTS:", districts[:5])
    except Exception as e:

        print("AGMARKNET FILTER ERROR:", type(e).__name__)
        print("AGMARKNET FILTER ERROR:", str(e))

        error = "Unable to load government mandi filters."

    # ========================================================
    # SEARCH
    # ========================================================

    if request.method == "POST":

        selected_crop = request.form.get(
            "crop", ""
        ).strip()

        selected_state = request.form.get(
            "state", ""
        ).strip()

        selected_district = request.form.get(
            "district", ""
        ).strip()

        selected_market = request.form.get(
            "market", ""
        ).strip()

        print("\n===================================")
        print("MARKET PRICE SEARCH")
        print("Crop:", selected_crop)
        print("State:", selected_state)
        print("District:", selected_district)
        print("Market:", selected_market)
        print("===================================")

        try:

            # ------------------------------------------------
            # Find selected commodity
            # ------------------------------------------------

            commodity_obj = next(
                (
                    item
                    for item in commodities
                    if str(item["id"]) == selected_crop
                ),
                None
            )

            # ------------------------------------------------
            # Find selected state
            # ------------------------------------------------

            state_obj = next(
                (
                    item
                    for item in states
                    if str(item["id"]) == selected_state
                ),
                None
            )

            if not commodity_obj:
                raise Exception("Invalid commodity selected.")

            if not state_obj:
                raise Exception("Invalid state selected.")

            commodity_id = commodity_obj["id"]
            state_id = state_obj["id"]

            # ------------------------------------------------
            # Current year and month
            # ------------------------------------------------

            today = date.today()

            year = today.year
            month = today.month

            # ------------------------------------------------
            # Official Agmarknet date-wise API
            # ------------------------------------------------

            url = (
                f"{AGMARKNET_BASE_URL}/"
                "prices-and-arrivals/"
                "date-wise/specific-commodity"
            )

            params = {
                "year": year,
                "month": month,
                "stateId": state_id,
                "commodityId": commodity_id,
                "includeExcel": "false"
            }

            print("Sending request to Agmarknet...")
            print("URL:", url)
            print("Params:", params)

            response = requests.get(
                url,
                params=params,
                headers=AGMARKNET_HEADERS,
                timeout=30
            )

            print("Agmarknet Status:", response.status_code)

            response.raise_for_status()

            result = response.json()

            print("Agmarknet Success:", result.get("success"))
            print("Markets Returned:", len(result.get("markets", [])))

            # ------------------------------------------------
            # Build selected district/market matching
            # ------------------------------------------------
            selected_district_id = None

            if selected_district:
                district_obj = next(
                    (
                        item
                        for item in districts
                        if str(item["id"]).strip()
                        == str(selected_district).strip()
                    ),
                    None
                )

                if district_obj:
                    selected_district_id = str(
                        district_obj["id"]
                    ).strip()

                    print(
                        "SELECTED DISTRICT:",
                        district_obj["name"]
                    )

                    print(
                        "SELECTED DISTRICT ID:",
                        selected_district_id
                    )
            
            selected_market_name = None

            if selected_market:

                market_obj = next(
                    (
                        item
                        for item in markets
                        if str(item["id"]) == selected_market
                    ),
                    None
                )

                if market_obj:
                    selected_market_name = market_obj["name"]

            # ------------------------------------------------
            # Process API markets
            # ------------------------------------------------

            for market_data in result.get("markets", []):

                market_name = market_data.get(
                    "marketName",
                    ""
                )

                # --------------------------------------------
                # Find market metadata
                # --------------------------------------------

                market_meta = next(
                    (
                        item
                        for item in markets
                        if normalize_text(item["name"])
                        == normalize_text(market_name)
                    ),
                    None
                )

                # District filter
                if selected_district_id:
                    if not market_meta:
                        print(
                            "DISTRICT DEBUG - MARKET META NOT FOUND:",
                            market_name
                        )
                        continue

                    market_district_id = str(
                        market_meta.get("district_id", "")
                    ).strip()

                    print(
                        "DISTRICT DEBUG:",
                        market_name,
                        "API district:",
                        market_district_id,
                        "Selected district:",
                        selected_district_id
                    )

                    if normalize_id(market_district_id) != normalize_id(
                        selected_district_id
                    ):
                        continue

                # Market filter
                if selected_market_name:

                    if market_name != selected_market_name:
                        continue

                # --------------------------------------------
                # Dates
                # --------------------------------------------

                for day_data in market_data.get(
                    "dates",
                    []
                ):

                    arrival_date = day_data.get(
                        "arrivalDate",
                        ""
                    )

                    total_arrivals = day_data.get(
                        "total_arrivals",
                        0
                    )

                    # ----------------------------------------
                    # Variety-wise prices
                    # ----------------------------------------

                    for item in day_data.get(
                        "data",
                        []
                    ):

                        prices.append({

                            "commodity":
                                commodity_obj["name"],

                            "market":
                                market_name,

                            "district":(
                                next(
                                        (
                                            d["name"]
                                            for d in districts
                                            if market_meta
                                            and normalize_id(d["id"])
                                            == normalize_id(market_meta[
                                                "district_id"
                                            ]
                                            )
                                        ),
                                        ""
                                    )
                                    if market_meta
                                    else ""
                                ),

                            "date":
                                arrival_date,

                            "variety":
                                item.get(
                                    "variety",
                                    ""
                                ),

                            "arrivals":
                                item.get(
                                    "arrivals",
                                    total_arrivals
                                ),

                            "min_price":
                                item.get(
                                    "minimumPrice",
                                    0
                                ),

                            "max_price":
                                item.get(
                                    "maximumPrice",
                                    0
                                ),

                            "modal_price":
                                item.get(
                                    "modalPrice",
                                    0
                                )
                        })

            # ------------------------------------------------
            # Newest first
            # ------------------------------------------------

            prices.sort(
                key=lambda x: x["date"],
                reverse=True
            )
            prices = prices[:50]

            if not prices:
                print("NO EXACT DATA FOUND.")
                print("Trying official fallback mandi data...")

                fallback_prices = []

                # ------------------------------------------------
                # FALLBACK:
                # Same State + Same Commodity
                # ------------------------------------------------

                for market_data in result.get("markets", []):

                    fallback_market_name = market_data.get(
                        "marketName",
                        ""
                    )

                    fallback_market_meta = next(
                        (
                            item
                            for item in markets
                            if str(item.get("name", "")).strip().lower()
                            ==
                            str(fallback_market_name).strip().lower()
                        ),
                        None
                    )

                    # --------------------------------------------
                    # If a specific market was selected,
                    # first prefer another market from same district
                    # --------------------------------------------

                    if selected_market_name:

                        if fallback_market_name == selected_market_name:
                            continue

                        if (
                            fallback_market_meta
                            and selected_district_id
                            and str(
                                fallback_market_meta.get(
                                    "district_id",
                                    ""
                                )
                            ).strip()
                            == selected_district_id
                        ):
                            fallback_priority = 1
                        else:
                            fallback_priority = 2

                    else:
                        fallback_priority = 1

                    for day_data in market_data.get(
                        "dates",
                        []
                    ):

                        arrival_date = day_data.get(
                            "arrivalDate",
                            ""
                        )

                        total_arrivals = day_data.get(
                            "total_arrivals",
                            0
                        )

                        for item in day_data.get(
                            "data",
                            []
                        ):

                            fallback_prices.append({

                                "commodity":
                                    commodity_obj["name"],

                                "market":
                                    fallback_market_name,

                                "district":
                                    (
                                        next(
                                            (
                                                d["name"]
                                                for d in districts
                                                if fallback_market_meta
                                                and str(d["id"]).strip()
                                                ==
                                                str(
                                                    fallback_market_meta.get(
                                                        "district_id",
                                                        ""
                                                    )
                                                ).strip()
                                            ),
                                            ""
                                        )
                                        if fallback_market_meta
                                        else ""
                                    ),

                                "date":
                                    arrival_date,

                                "variety":
                                    item.get(
                                        "variety",
                                        ""
                                    ),

                                "arrivals":
                                    item.get(
                                        "arrivals",
                                        total_arrivals
                                    ),

                                "min_price":
                                    item.get(
                                        "minimumPrice",
                                        0
                                    ),

                                "max_price":
                                    item.get(
                                        "maximumPrice",
                                        0
                                    ),

                                "modal_price":
                                    item.get(
                                        "modalPrice",
                                        0
                                    ),

                                "_fallback_priority":
                                    fallback_priority
                            })

                # ------------------------------------------------
                # Sort fallback data
                # Priority first, newest date second
                # ------------------------------------------------

                fallback_prices.sort(
                    key=lambda x: (
                        x.get("_fallback_priority", 99),
                        x.get("date", "")
                    ),
                    reverse=False
                )

                # Newest date within priority
                fallback_prices.sort(
                    key=lambda x: x.get("date", ""),
                    reverse=True
                )

                # ------------------------------------------------
                # Use fallback records
                # ------------------------------------------------

                if fallback_prices:

                    prices = fallback_prices[:50]

                    # Remove internal helper field
                    for price in prices:
                        price.pop(
                            "_fallback_priority",
                            None
                        )

                    if selected_district:

                        selected_district_name = next(
                            (
                                d["name"]
                                for d in districts
                                if str(d["id"]).strip()
                                ==
                                str(selected_district_id).strip()
                            ),
                            "selected district"
                        )

                        fallback_market = prices[0]["market"]
                        fallback_district = prices[0]["district"]

                        fallback_message = (
                            f"No current {commodity_obj['name']} "
                            f"price was found for {selected_district_name}. "
                            f"Showing available official mandi data "
                            f"from {fallback_market}, "
                            f"{fallback_district}."
                        )

                    elif selected_market:

                        fallback_message = (
                            f"No current {commodity_obj['name']} "
                            f"price was found for the selected market. "
                            f"Showing available official mandi data "
                            f"from {prices[0]['market']}, "
                            f"{prices[0]['district']}."
                        )

                    else:

                        fallback_message = (
                            f"No exact mandi data was found for the "
                            f"selected filters. Showing available "
                            f"official {commodity_obj['name']} "
                            f"mandi data from {prices[0]['market']}, "
                            f"{prices[0]['district']}."
                        )

                    print(
                        "FALLBACK USED:",
                        fallback_message
                    )

                else:
                    error = (
                        "No official mandi price data is "
                        "currently available for the selected "
                        "commodity and state."
                    )
            else:
                error = None

        except Exception as e:

            print(
                "AGMARKNET PRICE ERROR:",
                type(e).__name__
            )

            print(
                "AGMARKNET PRICE ERROR:",
                str(e)
            )

            error = (
                "Unable to fetch government mandi data."
            )

    return render_template(
        "market.html",
        prices=prices,
        error=error,
        fallback_message=fallback_message,
        states=states,
        commodities=commodities,
        districts=districts,
        markets=markets,
        selected_crop=selected_crop,
        selected_state=selected_state,
        selected_district=selected_district,
        selected_market=selected_market
    )

# CHATBOT
@app.route("/chatbot")
def chatbot():

    return render_template("chatbot.html")

# RUN APPLICATION
if __name__ == "__main__":
    app.run(debug=True)