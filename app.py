from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, g, send_from_directory, jsonify
)
import sqlite3
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import cv2

# ------------------------------------------- App config ------------------------------------------- #



app = Flask(__name__)
app.secret_key = "secret123"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ML globals
IMG_SIZE = (96, 96)
model = None
class_labels = []



# ------------------------------- Template filter ------------------------------------------- #


@app.template_filter('datetimeformat')
def datetimeformat(value):
    try:
        t = datetime.strptime(value, "%H:%M:%S")
        return t.strftime("%I:%M:%S %p")
    except:
        return value




# ------------------------------------ Database config ---------------------------------- #


DATABASE = "users.db"
ADMINS_DB = "admins.db"

# ---------- DB Setup ----------
def ensure_users_table():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            locality TEXT,
            status TEXT DEFAULT 'Inactive',
            role TEXT DEFAULT 'user',
            registered_time TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_admin_db():
    os.makedirs(os.path.dirname(ADMINS_DB), exist_ok=True) if os.path.dirname(ADMINS_DB) else None
    with sqlite3.connect(ADMINS_DB) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'admin'
            )
        """)
        conn.commit()

# Initialize DBs
ensure_users_table()
init_admin_db()



def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn



# -------------------------------------- Routes ------------------------------------------------ #



# Home (root)
@app.route('/')
@app.route('/home')
def home():
    return render_template("user_home.html")  # your landing/home template


# Provide user_home endpoint to avoid BuildError if templates use url_for('user_home')
@app.route('/user_home')
def user_home():
    # Usually same as home or a slightly different page — kept for compatibility
    return render_template("user_home.html")


#----------------------------------------- Register --------------------------------------------#



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form.get("email", "").strip()
        password = request.form["password"]
        mobile = request.form.get("mobile", "").strip()
        locality = request.form.get("locality", "").strip()

        now = datetime.now()
        reg_date = now.strftime("%Y-%m-%d")
        reg_time = now.strftime("%H:%M:%S")

        try:
            with sqlite3.connect(DATABASE) as conn:
                c = conn.cursor()
                cols = [r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()]
                missing_cols = {
                    "mobile": "TEXT",
                    "locality": "TEXT",
                    "status": "TEXT DEFAULT 'Inactive'",
                    "role": "TEXT DEFAULT 'user'",
                    "registered_date": "TEXT",
                    "registered_time": "TEXT",
                    "last_login_date": "TEXT",
                    "last_login_time": "TEXT"
                }
                for col, col_type in missing_cols.items():
                    if col not in cols:
                        try:
                            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
                        except sqlite3.OperationalError:
                            pass
                conn.commit()

                c.execute("""
                    INSERT INTO users (username, email, password, mobile, locality, status, role, registered_date, registered_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (username, email, password, mobile, locality, "Inactive", "user", reg_date, reg_time))
                conn.commit()

            session["username"] = username
            session["role"] = "user"
            session["admin"] = False

            flash("✅ Registered successfully! Wait for admin activation.", "success")
            return redirect(url_for("user_dashboard"))

        except sqlite3.IntegrityError:
            flash("❌ Username or email already exists.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")



@app.route("/go_back_register")
def go_back_register():
    return redirect(url_for("home"))




#-------------------------------------------- Login -----------------------------------------------------#


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        # Hardcoded admin credentials
        if username == "admin" and password == "admin":
            session["admin"] = True
            session["username"] = "admin"
            flash("✅ Admin login successful!", "success")
            return redirect(url_for("admin_view"))

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password)).fetchone()

        if user:
            if user["status"] != "Active":
                flash("⚠️ Account not activated yet. Please wait for admin approval.", "warning")
                return redirect(url_for("login"))

            now = datetime.now()
            last_date = now.strftime("%Y-%m-%d")
            last_time = now.strftime("%H:%M:%S")
            db.execute("UPDATE users SET last_login_date=?, last_login_time=? WHERE id=?", (last_date, last_time, user["id"]))
            db.commit()

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["admin"] = False

            flash("✅ Login successful!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template("login.html")




#------------------------------------------------------- Admin manage view ---------------------------------------------#


@app.route("/admin_view")
def admin_view():
    if not session.get("admin"):
        flash("Access denied! Admins only.", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return render_template("admin_manage.html", users=users)




@app.route("/go_back_login")
def go_back_login():
    return redirect(url_for("home"))



#----------------------------------------- Admin dashboard --------------------------------------#


@app.route("/admin_dashboard")
def admin_dashboard():
    if "username" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")



#------------------------------------------ User dashboard ------------------------------------------#


@app.route("/user_dashboard")
def user_dashboard():
    if "username" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    return render_template("user_dashboard.html")



# -------------------------------------- User List -------------------------------------------------- #


@app.route("/user_list", methods=["GET", "POST"])
def user_list():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT id, username, email, status, role FROM users")
    rows = cur.fetchall()
    users = [{"id": r[0], "username": r[1], "email": r[2], "status": r[3], "role": r[4]} for r in rows]
    conn.close()
    return render_template("user_list.html", users=users)


@app.route("/activate/<int:user_id>")
def activate_user(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET status='Active' WHERE id=?", (user_id,))
        conn.commit()
    flash("Your account has been activated successfully!", "success")
    return redirect(url_for("user_list"))


@app.route("/deactivate/<int:user_id>")
def deactivate_user(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET status='Inactive' WHERE id=?", (user_id,))
        conn.commit()
    flash("Your account has been deactivated.", "warning")
    return redirect(url_for("user_list"))


@app.route("/delete/<int:user_id>")
def delete_user(user_id):
    try:
        with sqlite3.connect(DATABASE) as conn_user:
            cur_user = conn_user.cursor()
            cur_user.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn_user.commit()

        init_admin_db()
        with sqlite3.connect(ADMINS_DB) as conn_admin:
            cur_admin = conn_admin.cursor()
            cur_admin.execute("DELETE FROM admins WHERE user_id=?", (user_id,))
            conn_admin.commit()

        flash("Account deleted permanently from both databases.", "danger")
    except sqlite3.Error as e:
        flash(f"❌ Database error: {e}", "danger")
    return redirect(url_for("user_list"))


@app.route("/update_role/<int:user_id>", methods=["POST"])
def update_role(user_id):
    role = request.form.get("role")

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()

    cursor.execute("SELECT username, password FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"status": "error"})

    # ✅ Send correct password visibility logic back to frontend
    if role == "Active":
        message = f"✅ {user['username']}'s role updated to {role.upper()} successfully!"
        password_display = user["password"]
    else:
        message = f"⚠️ {user['username']}'s role changed to {role.upper()}!"
        password_display = "*****"

    return jsonify({
        "status": "success",
        "message": message,
        "role": role,
        "username": user["username"],
        "password": password_display
    })


# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("home"))




# Upload & Predict (MRI brain tumor)
@app.route("/index", methods=["GET", "POST"])
def upload():
    global model, class_labels
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please select a file.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load model if needed
        if model is None:
            model_path = "model/brain_tumor_model.h5"
            labels_path = "model/class_labels.npy"
            if os.path.exists(model_path) and os.path.exists(labels_path):
                model = tf.keras.models.load_model(model_path)
                class_labels = np.load(labels_path, allow_pickle=True).tolist()
            else:
                flash("Model not trained yet.", "danger")
                return redirect(url_for("upload"))

        # Read image with OpenCV for MRI heuristic check
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            # not a readable image
            try:
                os.remove(filepath)
            except:
                pass
            flash("⚠️ Invalid image file. Please upload a valid MRI brain scan image.", "danger")
            return redirect(url_for("upload"))

        # Convert to grayscale & compute simple heuristics
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        color_std = np.std(img_cv)
        gray_std = np.std(gray_img)
        diff_rg = np.mean(np.abs(img_cv[:, :, 0] - img_cv[:, :, 1]))
        diff_gb = np.mean(np.abs(img_cv[:, :, 1] - img_cv[:, :, 2]))
        avg_diff = (diff_rg + diff_gb) / 2.0

        # Heuristic threshold to reject clearly colorful / non-MRI images
        if avg_diff > 20 or color_std > gray_std * 2.2:
            try:
                os.remove(filepath)
            except:
                pass
            flash("⚠️ Invalid image type. Please upload a valid MRI brain scan image.", "danger")
            return redirect(url_for("upload"))

        # Valid MRI-like image -> predict
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)].strip().lower()
        confidence = float(np.max(predictions))

        flash(f"✅ Prediction: {predicted_class.title()} (Confidence: {confidence:.2f})", "success")
        return render_template("result.html", filename=filename, prediction=predicted_class.title())

    return render_template("index.html")

# Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Train model (admin only)
@app.route("/train_model")
def train_model_route():
    global model, class_labels
    if "admin" not in session or not session.get("admin"):
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))

    try:
        train_dir = "dataset/brain_tumor/train"
        batch_size = 32
        epochs = 3

        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=IMG_SIZE, batch_size=batch_size,
            class_mode="categorical", subset="training", shuffle=True
        )
        val_gen = train_datagen.flow_from_directory(
            train_dir, target_size=IMG_SIZE, batch_size=batch_size,
            class_mode="categorical", subset="validation", shuffle=False
        )

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet"
        )
        input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(input_layer, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        output = Dense(train_gen.num_classes, activation="softmax")(x)

        new_model = Model(inputs=input_layer, outputs=output)
        base_model.trainable = False
        new_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

        callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
        new_model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

        os.makedirs("model", exist_ok=True)
        new_model.save("model/brain_tumor_model.h5")
        np.save("model/class_labels.npy", np.array(list(train_gen.class_indices.keys())))

        model = new_model
        class_labels = list(train_gen.class_indices.keys())
        flash("✅ Model trained and saved!", "success")

    except Exception as e:
        flash(f"Training failed: {e}", "danger")

    return redirect(url_for("admin_dashboard"))

# ---------------- Run ---------------- #
if __name__ == "__main__":
    
    app.run(debug=True)
