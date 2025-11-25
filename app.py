import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import json

# Blueprint imports (Ensure these files exist: profile_routes.py, auth_routes.py, employee_routes.py, target_role_routes.py)
from profile_routes import profile_routes
from auth_routes import auth_routes
from employee_routes import employee_routes



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#mongo connection
# NOTE: Replace with your actual connection string if you run this outside the canvas environment
client = MongoClient(
    "mongodb+srv://creatorpanda26:admin%40123@cluster0.izpl1se.mongodb.net/"
)

db = client["sih_appraisal"]
classification_collection = db["column_classification"]

# Attach DB into every request (for blueprint)
@app.before_request
def inject_db():
    request.db = db

# Register Blueprints
app.register_blueprint(profile_routes)
app.register_blueprint(auth_routes)
app.register_blueprint(employee_routes)



# feature mapping for pillars
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

# Function to detect pillar based on keywords
def detect_pillar(col_name):
    col_low = col_name.lower()
    for pillar, keywords in feature_map.items():
        if any(keyword in col_low for keyword in keywords):
            return pillar
    return "unclassified"

# Processing task for each role dataframe (executed in parallel)
def task(df, current_role, target_role_map, performance, potential, skill, risk, behavior): # MODIFIED: Added current_role, target_role_map
    """
    Calculates composite scores, classification, and writes results to MongoDB 
    Appraisal_Score_Analytics collection.
    
    NOTE: Database write is wrapped in app.app_context() to handle 
    thread isolation in ThreadPoolExecutor.
    """
    
    def safe_minmax(series):
        """Normalize series data (Min-Max scaling)."""
        s = pd.to_numeric(series, errors="coerce")
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            # Handle case where min == max or data is invalid
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def invert(series_norm):
        """Invert normalized score (e.g., for error rate or risk score)."""
        return 1.0 - series_norm

    def compute_composite(df, cols, invert_cols=None, internal_weights=None):
        """Computes the weighted composite score for a pillar."""
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

        composite = sum(norm_df[c] * weights[c] * (1/len(df.columns)) for c in numeric_cols_present)
        return composite.clip(0,1)

    performance_invert = ["ErrorRate(%)"]
    skill_invert = ["SkillGapScore"]
    risk_invert = ["AttritionRiskScore(1-5)"]

    df = df.copy()
    # Fill missing values with the mean of the column (Numeric imputation)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 1. Compute Composite Scores
    df["Performance_Composite"] = compute_composite(df, performance, invert_cols=performance_invert)
    df["Potential_Composite"]   = compute_composite(df, potential)
    df["Behavior_Composite"]    = compute_composite(df, behavior)
    df["Skill_Composite"]       = compute_composite(df, skill, invert_cols=skill_invert)
    df["Risk_Composite"]        = compute_composite(df, risk, invert_cols=risk_invert)

    # 2. Compute Final Weighted Score
    df["FinalScore"] = (
        pillar_weights["Performance"] * df["Performance_Composite"] +
        pillar_weights["Potential"]   * df["Potential_Composite"] +
        pillar_weights["Behavior"]    * df["Behavior_Composite"] +
        pillar_weights["Skill"]       * df["Skill_Composite"] +
        pillar_weights["Risk"]        * df["Risk_Composite"]
    )
    df["FinalScore_0_100"] = (df["FinalScore"] * 100).round(2)

    # 3. Compute Categories (for 9-Box Grid)
    def cat_3level(s):
        """Categorize composite scores into Low, Medium, High."""
        return pd.cut(s, bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low","Medium","High"], right=True)

    df["Performance_Category"] = cat_3level(df["Performance_Composite"])
    df["Potential_Category"]   = cat_3level(df["Potential_Composite"])
    df["NineBox"] = df["Potential_Category"].astype(str) + "-" + df["Performance_Category"].astype(str)

    # Determine Target Role based on current role (current_role)
    # This looks up the employee's current role in the map derived from 'ApplicableRoles'
    target_role = target_role_map.get(current_role, None) 


    # 4. Write the scored data to the Appraisal_Score_Analytics collection
    # FIX: Use app.app_context() to access the database from the worker thread
    with app.app_context():
        # Access DB instance using the global 'db' reference
        appraisal_collection = db['Appraisal_Score_Analytics']
        
        records = df.to_dict('records')
        
        for record in records:
            # Create the document structure for the database
            appraisal_doc = {
                "EmployeeID": record.get("EmployeeID"), # Assumes "EmployeeID" is a column in the uploaded sheet
                # MODIFIED: Insert the determined target_role
                "Target_Role": target_role, 
                "Current_Role": current_role, # Optionally store the current role for clarity
                "Performance_Composite": record.get("Performance_Composite", 0.0),
                "Potential_Composite": record.get("Potential_Composite", 0.0),
                "Behavior_Composite": record.get("Behavior_Composite", 0.0),
                "Skill_Composite": record.get("Skill_Composite", 0.0),
                "Risk_Composite": record.get("Risk_Composite", 0.0),
                "FinalScore_0_100": record.get("FinalScore_0_100", 0.0),
                "Performance_Category": str(record.get("Performance_Category", "Low")),
                "Potential_Category": str(record.get("Potential_Category", "Low")),
                "NineBox": str(record.get("NineBox", "Low-Low")),
            }
            
            # Upsert (Update or Insert) the data based on EmployeeID
            emp_id = appraisal_doc.get("EmployeeID")
            if emp_id:
                appraisal_collection.update_one(
                    {"EmployeeID": emp_id},
                    {"$set": appraisal_doc},
                    upsert=True # Creates the document if it doesn't exist
                )

# home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend with Mongo running. Upload at /upload"})

#upload route
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
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": f"File reading error: {str(e)}"}), 400

    # 1. Column Classification
    assigned = {pillar: [] for pillar in feature_map}
    assigned["unclassified"] = []

    for col in df.columns:
        pillar_name = detect_pillar(col)
        assigned[pillar_name].append(col)

    # NEW: Fetch success profiles and create Target Role Map
    success_profiles_collection = db["success_profiles"]
    profiles = list(success_profiles_collection.find({}))
    
    # Map: {Employee's Current Role (from 'ApplicableRoles'): Target Role Title (from 'RoleTitle')}
    target_role_map = {}
    for profile in profiles:
        role_title = profile.get("RoleTitle")
        # Ensure ApplicableRoles is a list and iterate over it
        applicable_roles = profile.get("ApplicableRoles", [])
        if isinstance(applicable_roles, list):
            for app_role in applicable_roles:
                # The employee's current role (app_role) maps to the target role (role_title)
                target_role_map[app_role] = role_title
        elif isinstance(applicable_roles, str):
            # Handle case where ApplicableRoles might be a single string unexpectedly
            target_role_map[applicable_roles] = role_title
            
    
    process_dfs_and_roles = []
    
    # 2. Group data by 'Role' (Job Title) for processing
    if 'Role' not in df.columns:
         return jsonify({"error": "Uploaded file must contain a 'Role' column for employee grouping and target role mapping."}), 400
         
    unique_roles = df['Role'].unique()
    
    for role in unique_roles:
        # Filter the DataFrame for the current role group
        new_df = df[df["Role"] == role].copy()
        process_dfs_and_roles.append((new_df, role)) # Store (DataFrame, Role) tuple

    # 3. Process scores in parallel
    with ThreadPoolExecutor() as executor:
        # Unpack the list of (df, role) tuples for the first two arguments of task
        dfs = [item[0] for item in process_dfs_and_roles]
        roles = [item[1] for item in process_dfs_and_roles]
        
        # The 'task' function calculates scores and saves to DB (Appraisal_Score_Analytics)
        executor.map(
            task,
            dfs, # DataFrames (df)
            roles, # Current Roles (current_role)
            repeat(target_role_map), # Target Role Map (target_role_map)
            repeat(assigned.get("performance")),
            repeat(assigned.get("potential")),
            repeat(assigned.get("behavior")),
            repeat(assigned.get("skill")),
            repeat(assigned.get("risk")),
        )

    # 4. Save classification results to DB
    result_data = {
        "file_name": file.filename,
        "total_columns": len(df.columns),
        "classified_columns": assigned,
        "target_role_mapping_status": "Target Roles assigned based on 'success_profiles' ApplicableRoles."
    }

    insert_result = classification_collection.insert_one(result_data)
    result_data["_id"] = str(insert_result.inserted_id)

    return jsonify(result_data)

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)