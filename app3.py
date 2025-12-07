import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from google import genai
from google.genai import types
import json
import re
import asyncio
from difflib import SequenceMatcher
import time
from string import Template
from collections import Counter, defaultdict
from bson.objectid import ObjectId
from bson.errors import InvalidId

# -------------------- APP SETUP -------------------- #
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 3
API_KEY = "AIzaSyB6jwfjNjiZAApQgi4wxeuarlyusk5reJc" 

mongo_client = MongoClient("mongodb+srv://creatorpanda26:admin%40123@cluster0.izpl1se.mongodb.net/")
db = mongo_client["sih_appraisal"]

# -------------------- NINEBOX MODULE -------------------- #
try:
    from ninebox import NineBox
except ImportError:
    class NineBox:
        @staticmethod
        def apply(data):
            for emp in data:
                analysis = emp.get("employee_analysis", {})
                scores = analysis.get("quantitative_scores", {})
                
                try:
                    p_score = float(scores.get("performance_score", 0))
                    pot_score = float(scores.get("potential_score", 0))
                except (ValueError, TypeError):
                    p_score = 0
                    pot_score = 0
                
                label = "Risk" 
                
                if pot_score >= 8:
                    if p_score >= 8: label = "Stars"
                    elif p_score >= 5: label = "High Potential"
                    else: label = "Enigma"
                elif pot_score >= 5:
                    if p_score >= 8: label = "Core Star"
                    elif p_score >= 5: label = "Core Player"
                    else: label = "Inconsistent Player"
                else:
                    if p_score >= 8: label = "High Performer"
                    elif p_score >= 5: label = "Effective"
                    else: label = "Risk"

                emp["nine_box_label"] = label
                if "employee_analysis" in emp:
                    emp["employee_analysis"]["nine_box_label"] = label
                    
            return data

# -------------------- ANALYTICS ENGINE -------------------- #
class AnalyticsEngine:
    @staticmethod
    def calculate_readiness(scores):
        """Calculates 0-100 score."""
        try:
            perf = float(scores.get("performance_score", 0))
            pot = float(scores.get("potential_score", 0))
            # Weighted: 60% Performance, 40% Potential
            weighted_score = (perf * 0.6) + (pot * 0.4)
            return round(min(weighted_score * 10, 100))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def get_category_label(score):
        """Maps 1-10 score to Low/Medium/High for Frontend Compatibility."""
        try:
            s = float(score)
            if s >= 8: return "high"
            if s >= 5: return "medium"
            return "low"
        except: return "low"

    @staticmethod
    def get_matrix_label(perf_score, pot_score):
        """Maps scores to 'High-High', 'High-Medium' etc."""
        p_label = AnalyticsEngine.get_category_label(perf_score).capitalize() # High
        pot_label = AnalyticsEngine.get_category_label(pot_score).capitalize() # Medium
        return f"{p_label}-{pot_label}"

    @staticmethod
    def determine_status(readiness, scores):
        risk = scores.get("risk_of_attrition", "Low")
        if readiness < 50 or risk == "High":
            return "needs-help" # Lowercase to match React check
        return "on-track"

    @staticmethod
    def generate_dashboard_data(employee_data):
        total_candidates = len(employee_data)
        if total_candidates == 0:
            return {"kpi": {}, "charts": {}, "message": "No data available"}

        # --- Aggregation Containers ---
        readiness_values = []
        status_counts = {"on-track": 0, "needs-help": 0}
        
        # 1. Readiness Distribution (Matches React ranges)
        readiness_dist = {
            "0-50%": 0,
            "50-70%": 0,
            "70-85%": 0,
            "85-100%": 0
        }

        # 2. Performance vs Potential Matrix
        matrix_counts = defaultdict(int)
        
        # 3. Role Aggregation
        role_sums = defaultdict(list)

        high_performers_count = 0

        # --- Main Processing Loop ---
        for emp in employee_data:
            analysis = emp.get("employee_analysis", {})
            scores = analysis.get("quantitative_scores", {})
            
            # Extract Scores
            p_score = float(scores.get("performance_score", 0))
            pot_score = float(scores.get("potential_score", 0))
            
            # Calculate Metrics
            r_score = AnalyticsEngine.calculate_readiness(scores)
            readiness_values.append(r_score)
            
            status = AnalyticsEngine.determine_status(r_score, scores)
            status_counts[status] += 1

            # Bucketing for Distribution Chart
            if r_score < 50: readiness_dist["0-50%"] += 1
            elif r_score < 70: readiness_dist["50-70%"] += 1
            elif r_score < 85: readiness_dist["70-85%"] += 1
            else: readiness_dist["85-100%"] += 1

            # Matrix Label (e.g., "High-High")
            mat_label = AnalyticsEngine.get_matrix_label(p_score, pot_score)
            matrix_counts[mat_label] += 1
            
            if mat_label == "High-High":
                high_performers_count += 1

            # Role Data
            role = analysis.get("target_role", "Unassigned")
            role_sums[role].append(r_score)

        # --- Formatting for Frontend Recharts ---
        
        # 1. Distribution Chart Data
        dist_chart_data = [
            {"range": k, "count": v} for k, v in readiness_dist.items()
        ]

        # 2. Matrix Chart Data (with colors)
        # Colors mapped to specific matrix zones
        matrix_colors = {
            "High-High": "#10B981",    # Green
            "High-Medium": "#3B82F6",  # Blue
            "Medium-High": "#F59E0B",  # Orange
            "Medium-Medium": "#6B7280",# Gray
            "Low-Medium": "#EF4444",   # Red
            "Low-Low": "#EF4444"
        }
        
        matrix_chart_data = []
        for label, value in matrix_counts.items():
            matrix_chart_data.append({
                "name": label,
                "value": value,
                "color": matrix_colors.get(label, "#94a3b8") # Default slate
            })

        # 3. Role Chart Data
        role_chart_data = []
        for role, val_list in role_sums.items():
            avg = round(sum(val_list) / len(val_list))
            role_chart_data.append({"role": role, "readiness": avg})

        # 4. IDP Completion Data
        idp_chart_data = [
            {"status": "On Track", "count": status_counts["on-track"], "color": "#10B981"},
            {"status": "Needs Help", "count": status_counts["needs-help"], "color": "#F59E0B"}
        ]

        # 5. KPI Data
        avg_readiness = round(sum(readiness_values) / total_candidates) if total_candidates else 0
        pipeline_health = round((status_counts["on-track"] / total_candidates) * 100) if total_candidates else 0

        return {
            "kpi": {
                "totalCandidates": total_candidates,
                "avgReadiness": avg_readiness,
                "highPerformers": high_performers_count,
                "pipelineHealth": pipeline_health,
                "onTrackCount": status_counts["on-track"],
                "needsHelpCount": status_counts["needs-help"]
            },
            "charts": {
                "readinessDistribution": dist_chart_data,
                "performancePotentialData": matrix_chart_data,
                "readinessByRole": role_chart_data,
                "idpCompletionData": idp_chart_data
            }
        }

# -------------------- DB HELPERS -------------------- #

def create_user_accounts(employees_data):
    """
    Automatically creates user accounts in MongoDB 'users' collection.
    Enforces unique employee_id.
    """
    users_col = db["users"]
    
    # 1. Enforce Uniqueness at Database Level
    users_col.create_index("employee_id", unique=True)
    
    count = 0
    print("🔄 Starting Auto-Account Creation...")
    
    for emp in employees_data:
        e_id = str(emp.get("employee_id", "")).strip()
        if not e_id: continue
        
        e_name = emp.get("Name") or emp.get("Employee Name") or emp.get("employee_name") or f"Employee_{e_id}"
        e_name = str(e_name).strip()
        
        # Default Password Logic
        password = f"Pass@{e_id}"
        
        user_doc = {
            "username": e_name,
            "employee_id": e_id,
            "password": password, 
            "role": "employee",
            "created_at": time.time(),
            "first_login": True
        }
        
        try:
            # Upsert ensures we update if exists, insert if new, based on unique employee_id
            users_col.update_one(
                {"employee_id": e_id}, 
                {"$set": user_doc}, 
                upsert=True
            )
            count += 1
        except Exception as e:
            print(f"Skipping duplicate or error for {e_id}: {e}")
            
    print(f"✅ Successfully created/updated {count} user accounts.")

def save_idp_results(final_data):
    """
    Saves the final IDP analysis to the 'idp_results' collection.
    """
    idp_col = db["idp_results"]
    count = 0
    print("💾 Saving IDP Results to DB...")

    for record in final_data:
        e_id = str(record.get("employee_id", "")).strip()
        if not e_id: continue

        doc = {
            "employee_id": e_id,
            "employee_name": record.get("employee_name"),
            "target_role": record.get("employee_analysis", {}).get("target_role"),
            "full_analysis": record, 
            "generated_at": time.time(),
            "status": "Active"
        }
        try:
            idp_col.update_one({"employee_id": e_id}, {"$set": doc}, upsert=True)
            count += 1
        except Exception as e:
            print(f"Failed to save IDP for {e_id}: {e}")

    print(f"✅ Successfully saved {count} IDP records.")

def serialize_doc(doc):
    """Converts MongoDB ObjectId to string for JSON serialization."""
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

def serialize_course(course):
    """Converts MongoDB document to the frontend Interface format for Learning Module"""
    return {
        "id": str(course["_id"]),
        "title": course.get("title"),
        "description": course.get("description"),
        "category": course.get("category"),
        "duration": course.get("duration"),
        "format": course.get("format"),
        "enrolledCount": course.get("enrolledCount", 0),
        "thumbnail": course.get("thumbnail", "")
    }

# -------------------- HELPERS & GENAI -------------------- #

def normalize_role_text(s):
    if s is None: return ""
    s = str(s).lower().strip()
    return re.sub(r'[\(\)\[\]\.,/\\\-]', ' ', re.sub(r'\s+', ' ', s))

def fuzzy_ratio(a, b):
    return int(SequenceMatcher(None, a, b).ratio() * 100)

def map_target_roles(employee_role, success_profiles, fuzzy_threshold=85):
    if not employee_role: return []
    norm_emp = normalize_role_text(employee_role)
    matched_roles = set()
    mapping = []
    for profile in success_profiles:
        role_title = profile.get("RoleTitle")
        applicable_roles = profile.get("ApplicableRoles", []) or []
        for ar in applicable_roles:
            mapping.append((normalize_role_text(ar), role_title))

    for nar, role_title in mapping:
        if norm_emp == nar: matched_roles.add(role_title)
        elif set(norm_emp.split()) == set(nar.split()): matched_roles.add(role_title)
        elif fuzzy_ratio(norm_emp, nar) >= fuzzy_threshold: matched_roles.add(role_title)
    return list(matched_roles)

def clean_json_response(text):
    try: return json.loads(text)
    except:
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return None

async def process_batch_async(client, batch_data, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (Analysis)...")
        prompt = Template("""
        You are an Expert HR AI.
        INPUT DATA: $batch_json
        TASK: Compare 'current_role' vs 'target_role'. Return JSON List.
        STRUCTURE:
        [
            {
                "employee_name": "String",
                "employee_id": "String",
                "employee_analysis": {
                    "current_role": "String",
                    "target_role": "String",
                    "quantitative_scores": {
                        "performance_score": 1-10,
                        "potential_score": 1-10,
                        "experience_alignment_score": 1-10,
                        "competency_gap_count": 0,
                        "risk_of_attrition": "Low/Med/High"
                    },
                    "qualitative_analysis": { 
                        "top_skills": [],
                        "behavioral_traits": [],
                        "competency_gaps": []
                     },
                    "reasoning": "String"
                }
            }
        ]
        """).substitute(batch_json=json.dumps(batch_data, indent=2))
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return clean_json_response(response.text)
        except Exception as e:
            print(f"Batch {batch_id} Error:", e)
            return None

async def orchestrate_processing(all_employees, api_key):
    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for i in range(0, len(all_employees), BATCH_SIZE):
        tasks.append(process_batch_async(client, all_employees[i:i + BATCH_SIZE], (i//BATCH_SIZE)+1, semaphore))
    results = await asyncio.gather(*tasks)
    return [item for sublist in results if sublist for item in sublist]

def select_best_role_fits(analyzed_data):
    grouped = {}
    for entry in analyzed_data:
        emp_id = entry.get("employee_id")
        analysis = entry.get("employee_analysis", {})
        scores = analysis.get("quantitative_scores", {})
        try:
            perf = float(scores.get("performance_score", 0))
            pot = float(scores.get("potential_score", 0))
            gap_count = int(scores.get("competency_gap_count", 99)) 
            exp_align = float(scores.get("experience_alignment_score", 0))
        except: perf, pot, gap_count, exp_align = 0, 0, 99, 0
            
        entry["_metrics"] = {"total_score": perf + pot, "gap_count": gap_count, "exp_score": exp_align}
        if emp_id not in grouped: grouped[emp_id] = []
        grouped[emp_id].append(entry)

    final_selection = []
    for emp_id, candidates in grouped.items():
        candidates.sort(key=lambda x: (-x["_metrics"]["total_score"], x["_metrics"]["gap_count"], -x["_metrics"]["exp_score"]))
        best_fit = candidates[0]
        alternatives = []
        for other in candidates[1:]:
            analysis = other.get("employee_analysis", {})
            alternatives.append({
                "target_role": analysis.get("target_role") or other.get("target_role"),
                "nine_box_label": other.get("nine_box_label", analysis.get("nine_box_label", "N/A")),
                "quantitative_scores": analysis.get("quantitative_scores"),
                "reasoning": analysis.get("reasoning")
            })
        best_fit["alternative_role_analysis"] = alternatives
        if "_metrics" in best_fit: del best_fit["_metrics"]
        final_selection.append(best_fit)
    return final_selection

async def second_pass_batch(client, batch, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (IDP)...")
        prompt = Template("""
        Generate IDP for these employees.
        INPUT DATA: $batch_json
        OUTPUT STRUCTURE:
        ... inside "employee_analysis": {
             ...,
             "idp_plan": {
                  "idp_type": "Leadership/Improvement",
                  "focus_areas": ["..."],
                  "action_plan": ["..."],
                  "timeline": "6 months"
             }
        }
        """).substitute(batch_json=json.dumps(batch, indent=2))
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return clean_json_response(response.text)
        except Exception: return None

async def orchestrate_second_pass(data, api_key):
    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    for i in range(0, len(data), BATCH_SIZE):
        tasks.append(second_pass_batch(client, data[i:i + BATCH_SIZE], (i//BATCH_SIZE)+1, semaphore))
    results = await asyncio.gather(*tasks)
    return [item for sublist in results if sublist for item in sublist]

# -------------------- SUCCESS PROFILE CRUD APIs -------------------- #

# 1. CREATE: Add a new profile
@app.route('/api/profiles', methods=['POST'])
def create_profile():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Insert into success_profiles collection
        result = db["success_profiles"].insert_one(data)
        
        return jsonify({
            "message": "Profile created successfully",
            "id": str(result.inserted_id)
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. READ ALL: Get all profiles
@app.route('/api/profiles', methods=['GET'],strict_slashes=False)
def get_all_profiles():
    try:
        profiles = []
        for doc in db["success_profiles"].find():
            profiles.append(serialize_doc(doc))
        return jsonify(profiles), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. READ ONE: Get a specific profile by ID
@app.route('/api/profiles/<id>', methods=['GET'])
def get_profile(id):
    try:
        doc = db["success_profiles"].find_one({"_id": ObjectId(id)})
        if doc:
            return jsonify(serialize_doc(doc)), 200
        return jsonify({"error": "Profile not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Object ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. UPDATE: Update an existing profile by ID
@app.route('/api/profiles/<id>', methods=['PUT'])
def update_profile(id):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Prevent updating the immutable _id field
        if '_id' in data:
            del data['_id']

        result = db["success_profiles"].update_one(
            {"_id": ObjectId(id)},
            {"$set": data}
        )

        if result.matched_count:
            # Fetch and return the updated document
            updated_doc = db["success_profiles"].find_one({"_id": ObjectId(id)})
            return jsonify({
                "message": "Profile updated successfully",
                "data": serialize_doc(updated_doc)
            }), 200
        
        return jsonify({"error": "Profile not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Object ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 5. DELETE: Remove a profile by ID
@app.route('/api/profiles/<id>', methods=['DELETE'])
def delete_profile(id):
    try:
        result = db["success_profiles"].delete_one({"_id": ObjectId(id)})
        
        if result.deleted_count:
            return jsonify({"message": "Profile deleted successfully"}), 200
        
        return jsonify({"error": "Profile not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Object ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- LEARNING MODULE CRUD APIs -------------------- #

# 1. READ ALL COURSES
@app.route('/api/courses', methods=['GET'])
def get_courses():
    try:
        courses = db["learning_courses"].find()
        return jsonify([serialize_course(course) for course in courses]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. CREATE COURSE
@app.route('/api/courses', methods=['POST'])
def add_course():
    try:
        data = request.json
        
        # basic validation
        if not data.get('title') or not data.get('category'):
            return jsonify({"error": "Title and Category are required"}), 400

        new_course = {
            "title": data.get("title"),
            "description": data.get("description", ""),
            "category": data.get("category"),
            "duration": data.get("duration", ""),
            "format": data.get("format", "video"),
            "enrolledCount": 0, # Default to 0
            "thumbnail": "",     # Placeholder for file upload logic
            "created_at": time.time()
        }

        result = db["learning_courses"].insert_one(new_course)
        
        # Return the created object with its new ID
        created_course = db["learning_courses"].find_one({"_id": result.inserted_id})
        return jsonify(serialize_course(created_course)), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. DELETE COURSE
@app.route('/api/courses/<id>', methods=['DELETE'])
def delete_course(id):
    try:
        result = db["learning_courses"].delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 1:
            return jsonify({"message": "Course deleted successfully"}), 200
        else:
            return jsonify({"error": "Course not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Object ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. UPDATE COURSE (Optional helper)
@app.route('/api/courses/<id>', methods=['PUT'])
def update_course(id):
    try:
        data = request.json
        # Filter out fields we don't want to wipe (like _id)
        update_data = {k: v for k, v in data.items() if k != 'id' and k != '_id'}
        
        result = db["learning_courses"].update_one(
            {"_id": ObjectId(id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 1:
            updated_course = db["learning_courses"].find_one({"_id": ObjectId(id)})
            return jsonify(serialize_course(updated_course)), 200
        else:
            return jsonify({"error": "Course not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid Object ID format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- USER MANAGEMENT APIs -------------------- #

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """
    Fetches all user accounts with their IDs and info.
    """
    try:
        users = []
        # Fetch all users, excluding the MongoDB internal _id if you prefer, 
        # or converting it to string as shown below.
        for doc in db["users"].find():
            doc["_id"] = str(doc["_id"]) # Serialize ObjectId
            users.append(doc)
            
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- MAIN ROUTES -------------------- #

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files["file"]
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file.filename))
        if "employee_id" not in df.columns: df["employee_id"] = df.index.astype(str)
        else: df["employee_id"] = df["employee_id"].astype(str)
        employees = df.to_dict(orient="records")
        
        # 1. Create Accounts
        create_user_accounts(employees)

        success_profiles = list(db["success_profiles"].find({}))
        exploded = []
        for emp in employees:
            role_key = next((k for k in emp.keys() if k.lower() in ['role','job title','designation']), None)
            if role_key:
                targets = map_target_roles(emp[role_key], success_profiles)
                if not targets: continue 
                for t in targets:
                    new_emp = emp.copy()
                    new_emp["target_role"] = t
                    exploded.append(new_emp)
                    
        if not exploded: return jsonify({"error": "No roles matched"}), 400

        # 2. Process
        analyzed = asyncio.run(orchestrate_processing(exploded, API_KEY))
        processed_9box = NineBox.apply(analyzed)
        best_fits = select_best_role_fits(processed_9box)
        final_output = asyncio.run(orchestrate_second_pass(best_fits, API_KEY))

        # 3. Save Results
        save_idp_results(final_output)

        return jsonify({"status": "success", "results": final_output})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analytics/dashboard", methods=["GET"])
def get_dashboard_analytics():
    try:
        # Fetch only the analysis data to save memory
        idp_cursor = db["idp_results"].find({}, {"full_analysis": 1, "_id": 0})
        
        employee_data = []
        for doc in idp_cursor:
            if "full_analysis" in doc:
                employee_data.append(doc["full_analysis"])
        
        if not employee_data:
            return jsonify({
                "status": "success", 
                "data": {
                    "kpi": {"totalCandidates": 0, "avgReadiness": 0, "pipelineHealth": 0},
                    "charts": {}
                }
            })

        # Run the Engine
        dashboard_data = AnalyticsEngine.generate_dashboard_data(employee_data)

        return jsonify({
            "status": "success",
            "data": dashboard_data
        })

    except Exception as e:
        print(f"❌ Analytics Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/analytics/generate", methods=["GET", "POST"])
def get_analytics():
    """
    Fetches IDP data directly from MongoDB 'idp_results' and generates dashboard analytics.
    Does NOT require a JSON body anymore.
    """
    try:
        print("📊 Fetching data from DB for analytics...")
        
        # 1. Fetch all processed IDPs from DB
        idp_cursor = db["idp_results"].find({})
        employees = []
        
        # 2. Extract the 'full_analysis' blob we saved earlier
        for doc in idp_cursor:
            if "full_analysis" in doc:
                employees.append(doc["full_analysis"])

        if not employees:
            return jsonify({"error": "No IDP data found in database. Please upload a file first."}), 404

        # 3. Generate Dashboard
        dashboard_data = AnalyticsEngine.generate_dashboard_data(employees)
        
        return jsonify({"status": "success", "dashboard": dashboard_data})

    except Exception as e:
        print(f"Analytics Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)