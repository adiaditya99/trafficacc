from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os
from functools import wraps
from sklearn.cluster import DBSCAN
import numpy as np
import requests
import urllib.parse

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

@app.route('/predict', methods=['GET'])
@login_required
def predict():
    model, features, encoders = load_ml_resources()

    return render_template('predict.html', features=features, model_loaded=model is not None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
@login_required
def profile():
    user = User.query.filter_by(username=session['username']).first()
    return render_template('profile.html', user=user)

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    np.random.seed(42)
    lats = np.random.uniform(20.0, 21.0, 50) 
    lons = np.random.uniform(78.0, 79.0, 50)
    coords = np.column_stack((lats, lons))
    
    clustering = DBSCAN(eps=0.05, min_samples=3).fit(coords)
    hotspots = [{'lat': float(coords[i][0]), 'lng': float(coords[i][1])} 
                for i, label in enumerate(clustering.labels_) if label != -1]
    return jsonify(hotspots)

@app.route('/api/location', methods=['GET'])
def api_location():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"address": {}, "infra": [], "type": "Unknown"}), 400

    # 1. Nominatim Reverse Geocoding
    geo_data = {}
    try:
        geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        r_geo = requests.get(geo_url, timeout=5, headers={'User-Agent': 'TrafficAccidentAIv1'})
        if r_geo.status_code == 200:
            geo_data = r_geo.json()
    except Exception as e:
        print("Geocoding Error:", e)

    # 2. Overpass Infrastructure
    infra = []
    is_junction = "No"
    s_limit = None
    try:
        overpass_query = f"""[out:json][timeout:5];(node(around:200,{lat},{lon})["highway"~"crossing|stop"];node(around:200,{lat},{lon})["railway"="level_crossing"];node(around:200,{lat},{lon})["traffic_calming"];node(around:200,{lat},{lon})["junction"];way(around:50,{lat},{lon})["highway"]["maxspeed"];);out body;"""
        url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(overpass_query)
        r_over = requests.get(url, timeout=5)
        if r_over.status_code == 200:
            o_data = r_over.json()
            for el in o_data.get('elements', []):
                tags = el.get('tags', {})
                if tags.get('maxspeed'):
                    try:
                        parsed_spd = int(''.join(c for c in str(tags.get('maxspeed')) if c.isdigit()))
                        if parsed_spd > 0: s_limit = parsed_spd
                    except: pass
                if tags.get('highway') == 'crossing': infra.append("Crossing")
                if tags.get('highway') == 'stop': infra.append("Stop")
                if tags.get('railway') == 'level_crossing': infra.append("Railway")
                if tags.get('traffic_calming'): infra.append("Speed Bump")
                if tags.get('junction'):
                    is_junction = "Crossroad" if tags.get('junction') == 'roundabout' else "T-Junction"
                    infra.append("Junction")
    except Exception as e:
        print("Overpass Infra Error:", e)

    return jsonify({
        "address": geo_data.get('address', {}),
        "type": geo_data.get('type', 'unknown'),
        "infra": list(set(infra)),
        "junction": is_junction,
        "speed_limit": s_limit
    })

@app.route('/api/weather', methods=['GET'])
def api_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    api_key = '626d1cf5eb7a23ce015f1a2260033dcd'
    fallback = {"main": "Clear", "temp": 20, "humidity": 60}
    
    if not lat or not lon: return jsonify(fallback)
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            w_data = r.json()
            vis = w_data.get('visibility', 10000) / 100
            return jsonify({
                "main": w_data['weather'][0]['main'],
                "temp": w_data['main']['temp'],
                "humidity": w_data['main']['humidity'],
                "visibility": min(max(vis, 10), 100)
            })
    except Exception as e:
        print("Weather API Error:", e)
        
    return jsonify(fallback)

def haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0 # Earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/api/hospitals', methods=['GET'])
def api_hospitals():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    fallback = {"nearest": None, "government": None}
    if not lat or not lon: return jsonify(fallback)

    try:
        lat_f = float(lat)
        lon_f = float(lon)
        # 50km radius query checking Nodes, Ways, AND Relations (polygons)! Ultra-fast 'out center' limits data transfer.
        hosp_query = f"""[out:json][timeout:15];nwr(around:50000,{lat_f},{lon_f})["amenity"="hospital"];out center;"""
        url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(hosp_query)
        r = requests.get(url, timeout=15)
        
        if r.status_code == 200:
            h_data = r.json()
            hospitals = []
            
            for el in h_data.get('elements', []):
                tags = el.get('tags', {})
                name = tags.get('name', 'General Hospital')
                
                # 'out center' gives lat/lon direct for nodes, but inside a 'center' dict for ways/relations
                h_lat = el.get('lat')
                h_lon = el.get('lon')
                if 'center' in el:
                    h_lat = el['center'].get('lat')
                    h_lon = el['center'].get('lon')
                    
                if not h_lat or not h_lon: continue
                
                # Calculate True Distance!
                dist_km = haversine(lat_f, lon_f, h_lat, h_lon)
                
                # Determine Gov/Private
                h_type = "Private"
                name_low = name.lower()
                op_type = tags.get('operator:type', '').lower()
                op_name = tags.get('operator', '').lower()
                
                if op_type in ['public', 'government'] or 'gov' in name_low or 'public' in name_low or 'general' in name_low or 'civil' in name_low or 'district' in name_low or 'government' in op_name:
                    h_type = "Government"
                
                hospitals.append({
                    "name": name,
                    "type": h_type,
                    "lat": h_lat,
                    "lon": h_lon,
                    "distance": round(dist_km, 1)
                })
            
            if not hospitals:
                return jsonify(fallback)
            
            # Sort by distance
            hospitals.sort(key=lambda x: x['distance'])
            
            nearest_hosp = hospitals[0]
            nearest_gov = next((h for h in hospitals if h['type'] == 'Government'), None)
            
            return jsonify({
                "nearest": nearest_hosp,
                "government": nearest_gov
            })
            
    except Exception as e:
        print("Hospital API Error:", e)

    return jsonify(fallback)

@app.route('/realtime_predict', methods=['POST'])
@login_required
def realtime_predict():
    model, features, encoders = load_ml_resources()
    if not model: return jsonify({'error': 'Model not trained.'}), 500

    data = request.json
    
    speed = float(data.get('Vehicle_Speed', 0))
    limit = float(data.get('Speed_Limit', 60))
    weather = data.get('Weather_Condition', 'Clear')
    road_cond = data.get('Road_Condition', 'Dry')
    
    speed_ratio = speed / limit if limit > 0 else 1.0
    base_risk = min(speed_ratio * 40, 60)
    weather_risk = 25 if weather in ['Rainy', 'Snow', 'Storm', 'Foggy'] else 0
    road_risk = 15 if road_cond in ['Wet', 'Ice', 'Snow', 'Damaged'] else 0
    
    driver_score = min(int(base_risk + weather_risk + road_risk + 5), 100)
    
    input_dict = {}
    categorical_cols = ['Road_Type', 'Road_Condition', 'Weather_Condition', 'Vehicle_Type']
    for feat in features:
        if feat in categorical_cols:
            val = str(data.get(feat, ''))
            le = encoders.get(feat)
            if le:
                val = val if val in le.classes_ else le.classes_[0]
                input_dict[feat] = [int(le.transform([val])[0])]
            else:
                input_dict[feat] = [0]
        else:
            input_dict[feat] = [float(data.get(feat, 0))]
            
    input_df = pd.DataFrame(input_dict)
    
    is_xgb = "XGB" in str(type(model))
    try:
        pred_val = int(model.predict(input_df)[0])
        pred = pred_val + 1 if is_xgb else pred_val
        info = SEVERITY_MAP.get(pred, {"label": "Unknown", "color": "#999", "desc": "N/A"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    emergency_map = {
        1: "Self Care / First Aid Kit Sufficient",
        2: "General Hospital Checkup Recommended",
        3: "Requires Immediate Ambulance Dispatch",
        4: "Critical: Dispatch Ambulance, ICU Preparation & Police Assistance"
    }
    
    final_explanation = ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        
        sys_prompt = """You are an advanced AI assistant integrated into a Traffic Accident Risk Prediction System.
Your role is to help users understand predictions, risks, and safety recommendations in simple and clear language.
Restrictions: Do NOT give medical or legal advice. Do NOT claim 100% accuracy. Do NOT use complex technical terms. Ensure support for India-wide context.
HOW YOU SHOULD RESPOND: Keep answers short but meaningful. Always explain "WHY". If risk is HIGH: Emphasize safety and suggest immediate action. Tone: Helpful, clear, calm, and safety-focused."""

        user_content = f"The ML model predicted a '{info.get('label')}' risk severity level. Context: I am driving at {speed}km/h (Zone limit: {limit}km/h). Weather condition is {weather}. Road condition is {road_cond}. Explain why my risk is {info.get('label')} and give me clear, actionable safety advice."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            timeout=8
        )
        final_explanation = response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI Error:", e)
        # Fallback Mode
        exps_en = []
        if speed > limit + 10: exps_en.append("High vehicle speed over the limit sharply increased the risk.")
        if weather in ['Rainy', 'Snow', 'Storm', 'Foggy']: exps_en.append(f"Adverse weather ({weather}) severely reduced traction and safety.")
        if data.get('Road_Condition') in ['Wet', 'Ice', 'Snow', 'Damaged']: exps_en.append(f"Dangerous road surface ({data.get('Road_Condition')}) destabilized the vehicle.")
        if not exps_en: exps_en.append("Routine conditions detected; baseline risk model applied.")
        final_explanation = " ".join(exps_en) + "\n\n*(Note: Basic fallback explanation used due to network delay)*"

    return jsonify({
        'severity_level': pred,
        'label': info.get('label', 'Unknown'),
        'color': info.get('color', '#999'),
        'explanation': info.get('desc', 'N/A'),
        'driver_score': driver_score,
        'emergency_response': emergency_map.get(pred, "Standard Response"),
        'ai_explanation': final_explanation
    })

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    user_msg = data.get('message', '').strip()
    history = data.get('history', [])
    
    if not user_msg:
        return jsonify({'error': 'Message cannot be empty.'}), 400
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        
        sys_prompt = "You are a helpful, clear, and safety-focused AI Traffic Assistant for an Indian Traffic Severity Predictor system. Answer driving safety, app usage, and traffic-related questions quickly and easily without complex jargon. Do not provide medical or legal advice."
        
        messages = [{"role": "system", "content": sys_prompt}]
        
        # Add a couple of previous messages for brief conversational context
        for msg in history[-4:]: 
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({"role": "user", "content": user_msg})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            timeout=8
        )
        
        ai_reply = response.choices[0].message.content.strip()
        return jsonify({"reply": ai_reply})
        
    except Exception as e:
        error_msg = str(e)
        print("Chatbot Error:", error_msg)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return jsonify({"reply": "⚠️ Your OpenAI API Key has run out of credits (Quota Exceeded)! Please top up your OpenAI billing account to chat with me."})
        return jsonify({"error": "I'm having trouble connecting right now. Please drive safely and try asking again later!"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000)
