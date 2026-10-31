from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    session
)

import os
import re
import sqlite3
import numpy as np
import tensorflow as tf

from datetime import datetime, date
from functools import wraps
from PIL import Image

from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ecobasketai-development-secret-change-before-deployment"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR, "ai", "database", "ecobasket.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "ai", "models", "fruit_model.keras"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# AI MODEL CONFIGURATION
# ==========================================

IMG_SIZE = 128

categories = [
    "FreshApple",
    "FreshBanana",
    "FreshGrape",
    "FreshGuava",
    "FreshJujube",
    "FreshOrange",
    "FreshPomegranate",
    "FreshStrawberry",
    "RottenApple",
    "RottenBanana",
    "RottenGrape",
    "RottenGuava",
    "RottenJujube",
    "RottenOrange",
    "RottenPomegranate",
    "RottenStrawberry"
]


# ==========================================
# LOAD AI MODEL
# ==========================================

model = None

try:

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(MODEL_PATH)

        if model.output_shape[-1] != len(categories):
            raise ValueError(
                "Model output does not match categories."
            )

        print("EcoBasketAI Model Loaded Successfully!")

    else:

        print("ERROR: AI model file not found!")

except Exception as e:

    print("Error loading AI model:", e)


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==========================================
# INITIALIZE DATABASE
# ==========================================

def initialize_database():

    conn = get_db_connection()

    # Users table

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT NOT NULL UNIQUE,

            phone TEXT NOT NULL,

            password TEXT NOT NULL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )
    """)

    # Groceries table

    conn.execute("""
        CREATE TABLE IF NOT EXISTS groceries (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            item_name TEXT NOT NULL,

            category TEXT NOT NULL,

            quantity INTEGER NOT NULL,

            expiry_date TEXT NOT NULL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )
    """)

    # Check existing columns

    columns = [
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(groceries)"
        ).fetchall()
    ]

    # Add user_id if it does not exist

    if "user_id" not in columns:

        conn.execute("""
            ALTER TABLE groceries
            ADD COLUMN user_id INTEGER REFERENCES users(id)
        """)

    # Create index

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_groceries_user_id
        ON groceries(user_id)
    """)

    conn.commit()

    conn.close()

    print("Database Initialized Successfully!")


# ==========================================
# LOGIN REQUIRED DECORATOR
# ==========================================

def login_required(route_function):

    @wraps(route_function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash("Please login to access this page.")

            return redirect(url_for("login"))

        return route_function(*args, **kwargs)

    return wrapper


# ==========================================
# EXPIRY STATUS
# ==========================================

def get_expiry_status(expiry_date):

    try:

        expiry = datetime.strptime(
            expiry_date, "%Y-%m-%d"
        ).date()

        today = date.today()

        remaining_days = (expiry - today).days

        if remaining_days < 0:

            return "Expired"

        elif remaining_days <= 7:

            return "Expiring Soon"

        else:

            return "Fresh"

    except (ValueError, TypeError):

        return "Unknown"


# ==========================================
# FOOD WASTE SUGGESTIONS
# ==========================================

def get_food_waste_suggestion(item_name, category):

    item = item_name.strip().lower()

    category = category.strip().lower()

    suggestions = {

        # FRUITS

        "apple": "Prepare apple juice or fruit salad.",

        "banana": "Prepare a banana smoothie or banana pancakes.",

        "orange": "Prepare fresh orange juice.",

        "grape": "Prepare grape juice or fruit salad.",

        "guava": "Prepare guava juice or guava smoothie.",

        "jujube": "Prepare jujube juice.",

        "pomegranate": "Prepare fresh pomegranate juice.",

        "strawberry": "Prepare a strawberry smoothie.",


        # VEGETABLES

        "carrot": "Prepare carrot soup or carrot salad.",

        "tomato": "Prepare tomato soup or tomato sauce.",

        "potato": "Prepare mashed potatoes or potato curry.",

        "onion": "Use it in vegetable curry or soup.",

        "spinach": "Prepare spinach soup or spinach curry.",

        "cabbage": "Prepare cabbage salad or vegetable stir-fry.",

        "broccoli": "Prepare broccoli soup or stir-fry.",

        "cucumber": "Prepare fresh cucumber salad.",


        # DAIRY

        "milk": "Check freshness and storage instructions before use.",

        "curd": "Use in curd rice if it is still fresh.",

        "yogurt": "Use in a smoothie if it is still fresh.",

        "cheese": "Use in sandwiches or pasta if properly stored.",

        "butter": "Use in cooking or baking if still fresh.",

        "paneer": "Prepare paneer curry if it is still fresh.",


        # BAKERY

        "bread": "Prepare bread toast or bread sandwiches.",

        "bun": "Prepare a vegetable bun sandwich.",

        "cake": "Use in a dessert if it is still fresh.",

        "cookies": "Enjoy with milk if they are still fresh.",

        "muffins": "Use as a snack if properly stored.",

        "croissant": "Prepare a croissant sandwich."

    }

    if item in suggestions:

        return suggestions[item]

    category_suggestions = {

        "fruits": "Prepare fruit juice or a fruit salad.",

        "vegetables": "Prepare vegetable soup or a healthy salad.",

        "dairy": "Check freshness and storage instructions before use.",

        "bakery": "Prepare toast or a suitable bakery snack.",

        "other": "Check the item's condition and storage instructions."

    }

    return category_suggestions.get(
        category,
        "Check the item's condition before deciding how to use it."
    )


# ==========================================
# CHECK IMAGE EXTENSION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# USER REGISTRATION
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        phone = request.form.get(
            "phone", ""
        ).strip()

        password = request.form.get("password", "")

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not all([
            name,
            email,
            phone,
            password,
            confirm_password
        ]):

            flash("Please fill in all fields.")

            return redirect(url_for("register"))

        # Validate email

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email
        ):

            flash("Please enter a valid email address.")

            return redirect(url_for("register"))

        # Validate phone

        if not re.fullmatch(r"[0-9]{10}", phone):

            flash(
                "Phone number must contain exactly 10 digits."
            )

            return redirect(url_for("register"))

        # Validate password

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters."
            )

            return redirect(url_for("register"))

        # Confirm password

        if password != confirm_password:

            flash("Passwords do not match.")

            return redirect(url_for("register"))

        # Hash password

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()

        try:

            conn.execute("""
                INSERT INTO users
                (name, email, phone, password)
                VALUES (?, ?, ?, ?)
            """, (
                name,
                email,
                phone,
                hashed_password
            ))

            new_user_id = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]

            user_count = conn.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0]

            # Assign old grocery records to first user

            if user_count == 1:

                conn.execute("""
                    UPDATE groceries
                    SET user_id = ?
                    WHERE user_id IS NULL
                """, (new_user_id,))

            conn.commit()

            flash("Registration successful! Please login.")

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            conn.rollback()

            flash(
                "This email address is already registered."
            )

        except Exception as e:

            conn.rollback()

            print("Registration Error:", e)

            flash("Registration failed. Please try again.")

        finally:

            conn.close()

    return render_template("register.html")


# ==========================================
# USER LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get("password", "")

        if not email or not password:

            flash("Please enter email and password.")

            return redirect(url_for("login"))

        conn = get_db_connection()

        user = conn.execute("""
            SELECT * FROM users
            WHERE email = ?
        """, (email,)).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"], password
        ):

            session.clear()

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            flash("Login successful! Welcome to EcoBasketAI.")

            return redirect(url_for("dashboard"))

        else:

            flash("Invalid email or password.")

            return redirect(url_for("login"))

    return render_template("login.html")


# ==========================================
# USER LOGOUT
# ==========================================

@app.route("/logout")
@login_required
def logout():

    session.clear()

    flash("You have logged out successfully.")

    return redirect(url_for("login"))


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_db_connection()

    groceries = conn.execute("""
        SELECT * FROM groceries
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()

    conn.close()

    fresh_count = 0

    expiring_count = 0

    expired_count = 0

    for item in groceries:

        status = get_expiry_status(
            item["expiry_date"]
        )

        if status == "Fresh":

            fresh_count += 1

        elif status == "Expiring Soon":

            expiring_count += 1

        elif status == "Expired":

            expired_count += 1

    return render_template(
        "dashboard.html",
        groceries=groceries,
        fresh_count=fresh_count,
        expiring_count=expiring_count,
        expired_count=expired_count,
        get_expiry_status=get_expiry_status,
        get_food_waste_suggestion=get_food_waste_suggestion
    )


# ==========================================
# ADD GROCERY
# ==========================================

@app.route("/add_grocery", methods=["POST"])
@login_required
def add_grocery():

    item_name = request.form.get(
        "item_name", ""
    ).strip()

    category = request.form.get(
        "category", ""
    ).strip()

    quantity = request.form.get(
        "quantity", ""
    ).strip()

    expiry_date = request.form.get(
        "expiry_date", ""
    ).strip()

    if not all([
        item_name,
        category,
        quantity,
        expiry_date
    ]):

        flash("Please fill in all grocery details.")

        return redirect(url_for("dashboard"))

    try:

        quantity = int(quantity)

        if quantity <= 0:

            raise ValueError

        datetime.strptime(
            expiry_date, "%Y-%m-%d"
        )

    except ValueError:

        flash(
            "Please enter a valid quantity and expiry date."
        )

        return redirect(url_for("dashboard"))

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO groceries
        (item_name, category, quantity, expiry_date, user_id)
        VALUES (?, ?, ?, ?, ?)
    """, (
        item_name,
        category,
        quantity,
        expiry_date,
        session["user_id"]
    ))

    conn.commit()

    conn.close()

    flash("Grocery added successfully!")

    return redirect(url_for("dashboard"))


# ==========================================
# EDIT GROCERY
# ==========================================

@app.route(
    "/edit_grocery/<int:id>",
    methods=["GET", "POST"]
)
@login_required
def edit_grocery(id):

    conn = get_db_connection()

    grocery = conn.execute("""
        SELECT * FROM groceries
        WHERE id = ? AND user_id = ?
    """, (
        id,
        session["user_id"]
    )).fetchone()

    if grocery is None:

        conn.close()

        flash("Grocery item not found.")

        return redirect(url_for("dashboard"))

    if request.method == "POST":

        item_name = request.form.get(
            "item_name", ""
        ).strip()

        category = request.form.get(
            "category", ""
        ).strip()

        quantity = request.form.get(
            "quantity", ""
        ).strip()

        expiry_date = request.form.get(
            "expiry_date", ""
        ).strip()

        if not all([
            item_name,
            category,
            quantity,
            expiry_date
        ]):

            conn.close()

            flash("Please fill in all grocery details.")

            return redirect(
                url_for("edit_grocery", id=id)
            )

        try:

            quantity = int(quantity)

            if quantity <= 0:

                raise ValueError

            datetime.strptime(
                expiry_date, "%Y-%m-%d"
            )

        except ValueError:

            conn.close()

            flash("Please enter valid grocery details.")

            return redirect(
                url_for("edit_grocery", id=id)
            )

        conn.execute("""
            UPDATE groceries

            SET item_name = ?,
                category = ?,
                quantity = ?,
                expiry_date = ?

            WHERE id = ? AND user_id = ?
        """, (
            item_name,
            category,
            quantity,
            expiry_date,
            id,
            session["user_id"]
        ))

        conn.commit()

        conn.close()

        flash("Grocery updated successfully!")

        return redirect(url_for("dashboard"))

    conn.close()

    return render_template(
        "edit_grocery.html",
        grocery=grocery
    )


# ==========================================
# DELETE GROCERY
# ==========================================

@app.route(
    "/delete_grocery/<int:id>",
    methods=["POST"]
)
@login_required
def delete_grocery(id):

    conn = get_db_connection()

    conn.execute("""
        DELETE FROM groceries
        WHERE id = ? AND user_id = ?
    """, (
        id,
        session["user_id"]
    ))

    conn.commit()

    conn.close()

    flash("Grocery deleted successfully!")

    return redirect(url_for("dashboard"))


# ==========================================
# AI FRUIT FRESHNESS PREDICTION
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        if model is None:

            flash("AI model is not available.")

            return redirect(url_for("predict"))

        if "fruit_image" not in request.files:

            flash("Please select a fruit image.")

            return redirect(url_for("predict"))

        file = request.files["fruit_image"]

        if file.filename == "":

            flash("No image selected.")

            return redirect(url_for("predict"))

        if not allowed_file(file.filename):

            flash(
                "Please upload a JPG, JPEG or PNG image."
            )

            return redirect(url_for("predict"))

        try:

            filename = secure_filename(file.filename)

            timestamp = datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )

            image_filename = timestamp + "_" + filename

            image_path = os.path.join(
                UPLOAD_FOLDER,
                image_filename
            )

            file.save(image_path)

            image = Image.open(image_path).convert("RGB")

            image = image.resize(
                (IMG_SIZE, IMG_SIZE)
            )

            image_array = np.array(image) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_category = categories[
                predicted_index
            ]

            confidence = float(
                predictions[0][predicted_index] * 100
            )

            if predicted_category.startswith("Fresh"):

                freshness = "FRESH"

                fruit_name = predicted_category.replace(
                    "Fresh", ""
                )

            else:

                freshness = "ROTTEN"

                fruit_name = predicted_category.replace(
                    "Rotten", ""
                )

            return render_template(
                "predict.html",
                fruit_name=fruit_name,
                freshness=freshness,
                confidence=round(confidence, 2),
                image_filename=image_filename
            )

        except Exception as e:

            print("Prediction Error:", e)

            flash(
                "Unable to process the image. Please try another image."
            )

            return redirect(url_for("predict"))

    return render_template("predict.html")


# ==========================================
# DISPLAY UPLOADED IMAGE
# ==========================================

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    initialize_database()

    app.run(debug=True)