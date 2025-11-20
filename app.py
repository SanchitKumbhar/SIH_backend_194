import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId # Keeping the import for context, though not strictly used for simple conversion
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

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
# ADDED: New collection for Success Profiles
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
            """Return normalized 0-1 series, handling constant columns and NaNs."""
            s = pd.to_numeric(series, errors="coerce")
            mn = s.min(skipna=True)
            mx = s.max(skipna=True)
            if pd.isna(mn) or pd.isna(mx) or mn == mx:
                # If constant or all NaN, return zeros
                return pd.Series(0.0, index=s.index)
            return (s - mn) / (mx - mn)

    def invert(series_norm):
            """Invert normalized 0-1 so that higher is better for formerly 'negative' metrics."""
            return 1.0 - series_norm

    def compute_composite(df, cols, invert_cols=None, internal_weights=None):
            """
            - df: dataframe
            - cols: list of column names to use
            - invert_cols: subset of cols that should be inverted after normalization
            - internal_weights: dict col->weight (if None, equal weights)
            Returns: pd.Series (0-1)
            """
            invert_cols = set(invert_cols or [])
            numeric_cols_present = [c for c in cols if c in df.columns]
            if not numeric_cols_present:
                return pd.Series(0.0, index=df.index)

            # Normalize each column
            norm_df = pd.DataFrame({c: safe_minmax(df[c]) for c in numeric_cols_present})

            # Invert negative indicators
            for c in numeric_cols_present:
                if c in invert_cols:
                    norm_df[c] = invert(norm_df[c])

            # Internal weights
            if internal_weights:
                # normalize internal weights to sum 1 over the columns present
                weights = {c: internal_weights.get(c, 0) for c in numeric_cols_present}
                total = sum(weights.values()) or 1.0
                weights = {c: w/total for c,w in weights.items()}
            else:
                # equal weights
                n = len(numeric_cols_present)
                weights = {c: 1.0/n for c in numeric_cols_present}

            # weighted sum across columns
            composite = sum(norm_df[c] * weights[c] for c in numeric_cols_present)
            # ensure 0-1
            composite = composite.clip(0,1)
            return composite

        # -----------------------------
        # 5) Compute each pillar composite
        # -----------------------------
        # specify which cols are "negative" (higher value = worse) so we invert them
    performance_invert = ["ErrorRate(%)"]          # higher error -> worse
    skill_invert = ["SkillGapScore"]               # larger gap -> worse
    risk_invert = ["AttritionRiskScore(1-5)"]      # higher attrition -> worse (we want lower is better)
    print(performance)
    # Compute pillars
    df = df.copy()  # work on a copy
    df.fillna(df.mean(numeric_only=True), inplace=True)  # simple missing handling for numeric
    df["Performance_Composite"] = compute_composite(df, performance, invert_cols=performance_invert)
    df["Potential_Composite"]   = compute_composite(df, potential)
    df["Behavior_Composite"]    = compute_composite(df, behavior)
    df["Skill_Composite"]       = compute_composite(df, skill, invert_cols=skill_invert)
    df["Risk_Composite"]        = compute_composite(df, risk, invert_cols=risk_invert)

        # -----------------------------
        # 6) Final score = weighted sum of pillars
        # -----------------------------
    df["FinalScore"] = (
        pillar_weights["Performance"] * df["Performance_Composite"]
    + pillar_weights["Potential"]   * df["Potential_Composite"]
    + pillar_weights["Behavior"]    * df["Behavior_Composite"]
    + pillar_weights["Skill"]       * df["Skill_Composite"]
    + pillar_weights["Risk"]        * df["Risk_Composite"]
    )
        # FinalScore is in 0-1 range. If you want 0-100:
    df["FinalScore_0_100"] = (df["FinalScore"] * 100).round(2)

        # -----------------------------
        # 7) Map Performance & Potential composites to Low/Med/High for 9-box
        # -----------------------------
    def cat_3level(s):
            # thresholds are customizable. Using 0.4, 0.7 by default
        return pd.cut(s, bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low","Medium","High"])

    df["Performance_Category"] = cat_3level(df["Performance_Composite"])
    df["Potential_Category"]   = cat_3level(df["Potential_Composite"])

        # 9-box label
    df["NineBox"] = df["Potential_Category"].astype(str) + "-" + df["Performance_Category"].astype(str)

        # -----------------------------
        # 8) Save results
        # -----------------------------
    import random

    df.to_csv(f"scored_employees{random.randint(0,10000)}.csv", index=False)
    print(df[["FinalScore_0_100","Performance_Category","Potential_Category","NineBox"]].head())
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

    # Save file to uploads/
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ------------------------------
    # Load CSV or Excel
    # ------------------------------
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file format (Use CSV or Excel)"}), 400

    except Exception as e:
        return jsonify({"error": f"File reading error: {str(e)}"}), 400

    # ------------------------------
    # Classify Columns
    # ------------------------------
    assigned = {pillar: [] for pillar in feature_map}
    assigned["unclassified"] = []

    for col in df.columns:
        pillar_name = detect_pillar(col)
        assigned[pillar_name].append(col)

    process=[]
    target_role=[["System Analyst"],["Sales Executive"]]

    for i in range(len(target_role)):
        print(target_role[i])
        new_df=df[df["Role"]==target_role[i][0]]
        process.append(new_df)

    with ThreadPoolExecutor() as executor:
            executor.map(task,process,repeat(assigned.get("performance")),repeat(assigned.get("potential")),repeat(assigned.get("behavior")),repeat(assigned.get("skill")),repeat(assigned.get("risk")))
    
    # Prepare result JSON (before saving to Mongo to ensure only standard Python objects are used)
    result_data = {
        "file_name": file.filename,
        "total_columns": len(df.columns),
        "classified_columns": assigned
    }

    # ------------------------------
    # Save in MongoDB and get the string ID for the response
    # ------------------------------
    insert_result = classification_collection.insert_one(result_data)
    
    # CONVERSION: Use the inserted_id from the result and convert it to string
    result_data["_id"] = str(insert_result.inserted_id)

    # Return the data, which now has the string ID
    return jsonify(result_data)

# ------------------------------
# Create New Success Profile Route (POST)
# ------------------------------
@app.route("/create-profile", methods=["POST"])
def create_success_profile():
    """
    Handles POST requests to create a new success profile in the database.
    """
    try:
        # Get the JSON data from the request body
        profile_data = request.json
        
        if not profile_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # --- Mandatory Field Validation ---
        if "RoleTitle" not in profile_data:
            return jsonify({"error": "Missing 'RoleTitle' in request body"}), 400
        if "MinimumExperienceYears" not in profile_data:
            return jsonify({"error": "Missing 'MinimumExperienceYears' in request body"}), 400
        # Check if Competencies are present and correctly formatted as a dictionary
        if "RequiredCompetencies" not in profile_data or not isinstance(profile_data.get("RequiredCompetencies"), dict):
            return jsonify({"error": "Missing or invalid 'RequiredCompetencies' (expected dictionary)"}), 400
        
        # --- Handle Optional/List Fields ---
        # Ensure list fields are present and are lists, defaulting to empty list if missing/wrong type
        if "ApplicableRoles" not in profile_data or not isinstance(profile_data.get("ApplicableRoles"), list):
             profile_data["ApplicableRoles"] = [] 
        
        if "FunctionalSkills" not in profile_data or not isinstance(profile_data.get("FunctionalSkills"), list):
            profile_data["FunctionalSkills"] = []
            
        if "GeographicalExperience" not in profile_data or not isinstance(profile_data.get("GeographicalExperience"), list):
            profile_data["GeographicalExperience"] = []

        # Insert the entire, validated profile data object into MongoDB
        # NOTE: This operation mutates profile_data to include the ObjectId
        insert_result = success_profile_collection.insert_one(profile_data)
        
        # --- FIX: Convert the ObjectId in the inserted data to a string ---
        inserted_id_str = str(insert_result.inserted_id)
        # Update the original dict with the string ID, making it safe for JSON serialization
        profile_data["_id"] = inserted_id_str 
        # -----------------------------------------------------------------

        # Prepare the response
        response_data = {
            "message": "Success Profile created successfully",
            "profile_id": inserted_id_str, # Use the converted string ID
            "data_received": profile_data # This is now safe because "_id" is a string
        }

        # Return a JSON response with status 201 (Created)
        return jsonify(response_data), 201

    except Exception as e:
        print(f"Error creating profile: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ------------------------------
# Route: Get All Success Profiles (GET)
# ------------------------------
@app.route("/profiles", methods=["GET"])
def get_all_success_profiles():
    """
    Handles GET requests to retrieve all success profiles from the database.
    """
    try:
        # Query MongoDB to find all documents in the collection
        profiles_cursor = success_profile_collection.find({})
        
        # Convert the Cursor into a list of dictionaries
        profiles_list = list(profiles_cursor)
        
        # Process the list: convert the MongoDB ObjectId to a string
        for profile in profiles_list:
            if "_id" in profile:
                # Convert BSON ObjectId to string for JSON serialization
                profile["_id"] = str(profile["_id"]) 
        
        # Return the list of profiles
        return jsonify(profiles_list), 200

    except Exception as e:
        print(f"Error retrieving profiles: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ------------------------------
# Start Flask Server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)