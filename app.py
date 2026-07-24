from flask import Flask, render_template, request, redirect, flash
from models import db
from models.user import User
from config import Config
app = Flask(__name__)
app.config["SECRET_KEY"] = "FarmerGuideAI@2026"

app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    db.create_all()

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

        flash("Login feature will be connected with database soon.", "info")

        return redirect("/login")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        password = request.form.get("password")
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

@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/disease")
def disease():
    return render_template("disease.html")

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