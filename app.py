import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
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
    # Prepare result JSON
    result_data = {
        "file_name": file.filename,
        "total_columns": len(df.columns),
        "classified_columns": assigned
    }

    # ------------------------------
    # Save in MongoDB
    # ------------------------------
    insert_id = classification_collection.insert_one(result_data).inserted_id
    result_data["_id"] = str(insert_id)

    return jsonify(result_data)


# ------------------------------
# Start Flask Server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)