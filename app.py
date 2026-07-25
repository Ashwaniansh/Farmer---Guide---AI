import os
from werkzeug.utils import secure_filename
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, flash
from models import db
from models.user import User
from config import Config
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "FarmerGuideAI@2026"

app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    db.create_all()
    crop_model = joblib.load("D:\Capstone Project\Farmer---Guide---AI\Model\crop_model.pkl")

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
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            image = filename
    return render_template("disease.html", image=image)

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