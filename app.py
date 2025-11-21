import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId # Keeping the import for context, though not strictly used for simple conversion
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

# ------------------------------
# ADDITION: Import Blueprint
# ------------------------------
from profile_routes import profile_routes

# ------------------------------
# Flask App Initialization
# ------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# MongoDB Connection
# ------------------------------
client = MongoClient(
    "mongodb+srv://creatorpanda26:admin%40123@cluster0.izpl1se.mongodb.net/"
)

db = client["sih_appraisal"]
classification_collection = db["column_classification"]
success_profile_collection = db["success_profiles"] 

# ------------------------------
# Feature Map for Keyword Match
# ------------------------------
feature_map = {
    "performance": ["appraisal", "kpi", "productivity", "error", "projects", "feedback"],
    "potential": ["potential", "learning", "leadership", "problem", "communication"],
    "behavior": ["team", "collab", "manager", "peer", "engagement", "wlb"],
    "skill": ["skill", "training", "certification"],
    "risk": ["attrition", "attendance"]
}
pillar_weights = {
    "Performance": 0.40,
    "Potential":   0.30,
    "Behavior":    0.15,
    "Skill":       0.10,
    "Risk":        0.05
}

# ------------------------------
# Function: Detect Pillar Name
# ------------------------------
def detect_pillar(col_name):
    col_low = col_name.lower()
    for pillar, keywords in feature_map.items():
        if any(keyword in col_low for keyword in keywords):
            return pillar
    return "unclassified"

def task(df,performance,potential,skill,risk,behavior):   
    def safe_minmax(series):
            s = pd.to_numeric(series, errors="coerce")
            mn = s.min(skipna=True)
            mx = s.max(skipna=True)
            if pd.isna(mn) or pd.isna(mx) or mn == mx:
                return pd.Series(0.0, index=s.index)
            return (s - mn) / (mx - mn)

    def invert(series_norm):
            return 1.0 - series_norm

    def compute_composite(df, cols, invert_cols=None, internal_weights=None):
            invert_cols = set(invert_cols or [])
            numeric_cols_present = [c for c in cols if c in df.columns]
            if not numeric_cols_present:
                return pd.Series(0.0, index=df.index)

            norm_df = pd.DataFrame({c: safe_minmax(df[c]) for c in numeric_cols_present})

            for c in numeric_cols_present:
                if c in invert_cols:
                    norm_df[c] = invert(norm_df[c])

            if internal_weights:
                weights = {c: internal_weights.get(c, 0) for c in numeric_cols_present}
                total = sum(weights.values()) or 1.0
                weights = {c: w/total for c,w in weights.items()}
            else:
                n = len(numeric_cols_present)
                weights = {c: 1.0/n for c in numeric_cols_present}

            composite = sum(norm_df[c] * weights[c] for c in numeric_cols_present)
            composite = composite.clip(0,1)
            return composite

    performance_invert = ["ErrorRate(%)"]
    skill_invert = ["SkillGapScore"]
    risk_invert = ["AttritionRiskScore(1-5)"]

    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df["Performance_Composite"] = compute_composite(df, performance, invert_cols=performance_invert)
    df["Potential_Composite"]   = compute_composite(df, potential)
    df["Behavior_Composite"]    = compute_composite(df, behavior)
    df["Skill_Composite"]       = compute_composite(df, skill, invert_cols=skill_invert)
    df["Risk_Composite"]        = compute_composite(df, risk, invert_cols=risk_invert)

    df["FinalScore"] = (
        pillar_weights["Performance"] * df["Performance_Composite"]
    + pillar_weights["Potential"]   * df["Potential_Composite"]
    + pillar_weights["Behavior"]    * df["Behavior_Composite"]
    + pillar_weights["Skill"]       * df["Skill_Composite"]
    + pillar_weights["Risk"]        * df["Risk_Composite"]
    )
    df["FinalScore_0_100"] = (df["FinalScore"] * 100).round(2)

    def cat_3level(s):
        return pd.cut(s, bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low","Medium","High"])

    df["Performance_Category"] = cat_3level(df["Performance_Composite"])
    df["Potential_Category"]   = cat_3level(df["Potential_Composite"])
    df["NineBox"] = df["Potential_Category"].astype(str) + "-" + df["Performance_Category"].astype(str)

    import random
    df.to_csv(f"scored_employees{random.randint(0,10000)}.csv", index=False)

# ------------------------------
# Home Route
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend with Mongo running. Upload at /upload"})

# ------------------------------
# File Upload + Column Classification + Mongo Save
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "File name is empty"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file format (Use CSV or Excel)"}), 400
    except Exception as e:
        return jsonify({"error": f"File reading error: {str(e)}"}), 400

    assigned = {pillar: [] for pillar in feature_map}
    assigned["unclassified"] = []

    for col in df.columns:
        pillar_name = detect_pillar(col)
        assigned[pillar_name].append(col)

    process=[]
    target_role=[["System Analyst"],["Sales Executive"]]

    for i in range(len(target_role)):
        new_df=df[df["Role"]==target_role[i][0]]
        process.append(new_df)

    with ThreadPoolExecutor() as executor:
        executor.map(task,process,
                     repeat(assigned.get("performance")),
                     repeat(assigned.get("potential")),
                     repeat(assigned.get("behavior")),
                     repeat(assigned.get("skill")),
                     repeat(assigned.get("risk")))
    
    result_data = {
        "file_name": file.filename,
        "total_columns": len(df.columns),
        "classified_columns": assigned
    }

    insert_result = classification_collection.insert_one(result_data)
    result_data["_id"] = str(insert_result.inserted_id)

    return jsonify(result_data)

# ------------------------------
# Create New Success Profile Route (POST)
# ------------------------------
@app.route("/create-profile", methods=["POST"])
def create_success_profile():
    try:
        profile_data = request.json
        
        if not profile_data:
            return jsonify({"error": "No JSON data provided"}), 400

        if "RoleTitle" not in profile_data:
            return jsonify({"error": "Missing 'RoleTitle' in request body"}), 400
        if "MinimumExperienceYears" not in profile_data:
            return jsonify({"error": "Missing 'MinimumExperienceYears' in request body"}), 400
        if "RequiredCompetencies" not in profile_data or not isinstance(profile_data.get("RequiredCompetencies"), dict):
            return jsonify({"error": "Missing or invalid 'RequiredCompetencies' (expected dictionary)"}), 400
        
        if "ApplicableRoles" not in profile_data or not isinstance(profile_data.get("ApplicableRoles"), list):
             profile_data["ApplicableRoles"] = [] 
        
        if "FunctionalSkills" not in profile_data or not isinstance(profile_data.get("FunctionalSkills"), list):
            profile_data["FunctionalSkills"] = []
            
        if "GeographicalExperience" not in profile_data or not isinstance(profile_data.get("GeographicalExperience"), list):
            profile_data["GeographicalExperience"] = []

        insert_result = success_profile_collection.insert_one(profile_data)
        inserted_id_str = str(insert_result.inserted_id)
        profile_data["_id"] = inserted_id_str 

        response_data = {
            "message": "Success Profile created successfully",
            "profile_id": inserted_id_str,
            "data_received": profile_data
        }

        return jsonify(response_data), 201

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ------------------------------
# Route: Get All Success Profiles (GET)
# ------------------------------
@app.route("/profiles", methods=["GET"])
def get_all_success_profiles():
    try:
        profiles_cursor = success_profile_collection.find({})
        profiles_list = list(profiles_cursor)
        
        for profile in profiles_list:
            if "_id" in profile:
                profile["_id"] = str(profile["_id"]) 
        
        return jsonify(profiles_list), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ------------------------------
# ADDITION: Attach DB to Each Request
# ------------------------------
@app.before_request
def attach_db():
    request.db = db

# ------------------------------
# ADDITION: Register Blueprint
# ------------------------------
app.register_blueprint(profile_routes)

# ------------------------------
# Start Flask Server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
