from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os
import json
from functools import wraps

app = Flask(__name__)
app.secret_key = 'trafficai_secret_2026'

MODEL_PATH    = 'model/model.pkl'
FEATURES_PATH = 'model/features.pkl'
ENCODERS_PATH = 'model/encoders.pkl'

# ─────────────────────────────────────────────────────────────
# Configure Database (PostgreSQL for Render / SQLite fallback)
# ─────────────────────────────────────────────────────────────
database_url = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    fullname = db.Column(db.String(120), nullable=False)

with app.app_context():
    db.create_all()
    # Create default accounts if empty
    if not User.query.first():
        admin = User(username='admin', password='admin123', fullname='Administrator')
        demo = User(username='user', password='user123', fullname='Demo User')
        db.session.add(admin)
        db.session.add(demo)
        db.session.commit()

# ─────────────────────────────────────────────────────────────
# Severity map
# ─────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    1: {"label": "Low",      "color": "#10b981",
        "desc": "Minor impact — conditions are manageable. Low risk of serious incident."},
    2: {"label": "Medium",   "color": "#f59e0b",
        "desc": "Moderate risk — proceed with caution. Possible delays or minor incidents."},
    3: {"label": "High",     "color": "#f97316",
        "desc": "High risk — dangerous conditions detected. Significant chance of a serious accident."},
    4: {"label": "Critical", "color": "#ef4444",
        "desc": "Critical risk — road closure likely. Immediate safety measures recommended."}
}

# Global variables to store the loaded model, features, and encoders
global_model = None
global_features = []
global_encoders = {}
ml_resources_loaded = False

def load_ml_resources():
    global global_model, global_features, global_encoders, ml_resources_loaded
    if ml_resources_loaded:
        return global_model, global_features, global_encoders
        
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        global_model    = joblib.load(MODEL_PATH)
        global_features = joblib.load(FEATURES_PATH)
        global_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
        ml_resources_loaded = True
        return global_model, global_features, global_encoders
    return None, [], {}

# Attempt to load initially when app starts
load_ml_resources()

# ─────────────────────────────────────────────────────────────
# Auth decorator
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', error='Please sign in to access this page.'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────
# Auth routes
# ─────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))

    error = request.args.get('error')
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['username'] = user.username
            session['fullname'] = user.fullname
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password. Please try again.'

    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))

    error   = None
    success = None

    if request.method == 'POST':
        fullname         = request.form.get('fullname', '').strip()
        username         = request.form.get('username', '').strip()
        password         = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Validation
        if not fullname or not username or not password:
            error = 'All fields are required.'
        elif len(username) < 3 or len(username) > 20:
            error = 'Username must be 3–20 characters long.'
        elif not username.replace('_', '').isalnum():
            error = 'Username may only contain letters, numbers, and underscores.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                error = f'Username "{username}" is already taken. Please choose another.'
            else:
                # Register the user
                new_user = User(username=username, password=password, fullname=fullname)
                db.session.add(new_user)
                db.session.commit()
                success = f'Account created successfully! Welcome, {fullname}. You can now sign in.'

    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# ─────────────────────────────────────────────────────────────
# Protected routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    model, features, encoders = load_ml_resources()

    if request.method == 'POST':
        if not model:
            return jsonify({'error': 'Model not trained yet. Please run train.py first.'}), 500

        try:
            data = request.json
            categorical_cols = ['Road_Type', 'Road_Condition', 'Vehicle_Type',
                                 'Weather_Condition', 'Junction']
            input_dict = {}
            for feat in features:
                if feat in categorical_cols:
                    val = str(data.get(feat, ''))
                    le  = encoders.get(feat)
                    if le:
                        if val not in le.classes_:
                            val = le.classes_[0]
                        input_dict[feat] = [int(le.transform([val])[0])]
                    else:
                        input_dict[feat] = [0]
                else:
                    input_dict[feat] = [float(data.get(feat, 0))]

            input_df = pd.DataFrame(input_dict)
            pred     = int(model.predict(input_df)[0])
            info     = SEVERITY_MAP.get(pred, {"label": "Unknown", "color": "#999",
                                               "desc": "Unable to determine severity."})
            return jsonify({
                'severity_level': pred,
                'label':          info['label'],
                'color':          info['color'],
                'explanation':    info['desc']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return render_template('predict.html', features=features, model_loaded=model is not None)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000)
