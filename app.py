import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, redirect, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from models import db
from models.user import User
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.config["SECRET_KEY"] = "FarmerGuideAI@2026"

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DISEASE_MODEL_PATH = r"D:\Capstone Project\Farmer---Guide---AI\Model\disease_model_best.keras"
disease_model = load_model(DISEASE_MODEL_PATH)
print("Disease Detection Model Loaded Successfully")
DISEASE_CLASSES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]
CONFIDENCE_THRESHOLD = 60.0
DISEASE_INFO = {
    "Pepper Bell Bacterial Spot": {
        "symptoms": "Small dark spots and lesions may appear on pepper leaves.",
        "treatment": "Remove severely affected leaves and maintain proper field hygiene.",
        "prevention": "Avoid overhead watering and maintain good spacing between plants."
    },
    "Pepper Bell Healthy": {
        "symptoms": "The pepper leaf appears healthy with no major disease symptoms.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue proper watering, nutrition, and regular monitoring."
    },
    "Potato Early Blight": {
        "symptoms": "Dark spots with concentric ring patterns may appear on older leaves.",
        "treatment": "Remove affected leaves and use appropriate disease management practices.",
        "prevention": "Maintain field sanitation and avoid excessive moisture on leaves."
    },
    "Potato Late Blight": {
        "symptoms": "Dark irregular lesions may develop on potato leaves.",
        "treatment": "Remove severely affected plant material and follow recommended disease control practices.",
        "prevention": "Avoid prolonged leaf wetness and monitor plants regularly."
    },
    "Potato Healthy": {
        "symptoms": "The potato leaf appears healthy without visible disease symptoms.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue regular monitoring and proper crop management."
    },
    "Tomato Bacterial Spot": {
        "symptoms": "Small dark spots may appear on tomato leaves and other plant parts.",
        "treatment": "Remove severely affected leaves and maintain good field hygiene.",
        "prevention": "Avoid overhead irrigation and use proper plant spacing."
    },
    "Tomato Early Blight": {
        "symptoms": "Dark lesions with concentric rings commonly appear on older tomato leaves.",
        "treatment": "Remove affected leaves and follow suitable disease management practices.",
        "prevention": "Maintain sanitation and avoid prolonged leaf moisture."
    },
    "Tomato Late Blight": {
        "symptoms": "Dark irregular lesions can spread rapidly across tomato leaves.",
        "treatment": "Remove infected plant material and follow recommended disease control practices.",
        "prevention": "Monitor plants frequently and reduce prolonged leaf wetness."
    },
    "Tomato Leaf Mold": {
        "symptoms": "Yellowish areas may appear on the upper leaf surface with mold growth associated with the underside.",
        "treatment": "Remove affected leaves and improve air circulation around plants.",
        "prevention": "Maintain proper spacing and avoid excessive humidity."
    },
    "Tomato Septoria Leaf Spot": {
        "symptoms": "Small circular spots may develop on tomato leaves.",
        "treatment": "Remove severely affected leaves and maintain field sanitation.",
        "prevention": "Avoid overhead watering and remove infected plant debris."
    },
    "Tomato Spider Mites": {
        "symptoms": "Leaves may show yellow or speckled areas and can become damaged by mite feeding.",
        "treatment": "Remove heavily affected leaves and follow appropriate pest management practices.",
        "prevention": "Regularly inspect leaves and maintain proper plant health."
    },
    "Tomato Target Spot": {
        "symptoms": "Circular spots with target-like patterns may develop on tomato leaves.",
        "treatment": "Remove affected leaves and maintain good crop sanitation.",
        "prevention": "Improve air circulation and avoid prolonged leaf wetness."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "symptoms": "Leaves may curl upward, become yellow, and plant growth may be reduced.",
        "treatment": "Remove severely infected plants and manage insect vectors appropriately.",
        "prevention": "Monitor plants for vector insects and maintain field hygiene."
    },
    "Tomato Mosaic Virus": {
        "symptoms": "Leaves may develop mosaic-like light and dark green patterns.",
        "treatment": "Remove severely infected plants and maintain strict sanitation.",
        "prevention": "Avoid spreading plant sap between healthy and infected plants."
    },
    "Tomato Healthy": {
        "symptoms": "The tomato leaf appears healthy without visible disease symptoms.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue proper watering, nutrition, and regular monitoring."
    }
}
def validate_image(filepath):
    try:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        if img_array.mean() < 20:
            return False
        if img_array.std() < 10:
            return False
        return True
    except Exception:
        return False

app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()
    crop_model = joblib.load(
    r"D:\Capstone Project\Farmer---Guide---AI\Model\crop_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                flash("Welcome " + user.name, "success")
                return redirect("/dashboard")
            else:
                flash("Incorrect Password", "danger")
        else:
            flash("User Not Found", "warning")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        password = request.form.get("password")
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!", "danger")
            return redirect("/register")
        user = User(
            name=name,
            email=email,
            mobile=mobile,
            password=password
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration Successful", "success")
        return redirect("/login")
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/crop", methods=["GET", "POST"])
def crop():
    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(data)
        return render_template(
            "crop.html",
            prediction=prediction[0]
        )
    return render_template("crop.html")

@app.route("/disease", methods=["GET", "POST"])
def disease():
    image = None
    prediction = None
    confidence = None
    disease_info = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename)
        
            file.save(filepath)
            image = filename
            if not validate_image(filepath):
                prediction = "Invalid Image"
                confidence = 0.0
                disease_info = None

                return render_template(
                    "disease.html",
                    image=image,
                    prediction=prediction,
                    confidence=confidence,
                    disease_info=disease_info)
            img = load_img(
                filepath,
                target_size=(224, 224))
            
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(
                img_array,
                axis=0)

            predictions = disease_model.predict(
                img_array,
                verbose=0)
            predicted_index = np.argmax(predictions[0])

            confidence = float(
                predictions[0][predicted_index] * 100)

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = DISEASE_CLASSES[predicted_index]
                disease_info = DISEASE_INFO.get(prediction)
            else:
                prediction = "Unable to identify disease"
                disease_info = None
    return render_template(
        "disease.html",
        image=image,
        prediction=prediction,
        confidence=confidence,
        disease_info=disease_info)
    
@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html")

@app.route("/market")
def market():
    return render_template("market.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(debug=True)