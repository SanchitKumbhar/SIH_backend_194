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

# -------------------- NINEBOX MODULE -------------------- #
try:
    from ninebox import NineBox
except ImportError:
    # Dummy fallback if file is missing
    class NineBox:
        @staticmethod
        def apply(data):
            for emp in data:
                emp["nine_box_label"] = "Core Players" # Default
            return data

# -------------------- APP SETUP -------------------- #
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- CONFIGURATION -------------------- #
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 3

# -------------------- MONGO CONNECT -------------------- #
mongo_client = MongoClient(
    "mongodb+srv://creatorpanda26:admin%40123@cluster0.izpl1se.mongodb.net/"
)
db = mongo_client["sih_appraisal"]

# -------------------- API KEY -------------------- #
# ⚠️ Ensure this is your valid key
API_KEY = "AIzaSyA_nstKox_CXsjkwp2WJhCUYjSxYFm8p8U"

# -------------------- HELPERS -------------------- #

def normalize_role_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[\(\)\[\]\.,/\\\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def fuzzy_ratio(a, b):
    return int(SequenceMatcher(None, a, b).ratio() * 100)

def map_target_role(employee_role, success_profiles, fuzzy_threshold=85):
    if not employee_role:
        return None

    norm_emp = normalize_role_text(employee_role)
    mapping = []
    
    for profile in success_profiles:
        applicable_roles = profile.get("ApplicableRoles", []) or []
        for ar in applicable_roles:
            nar = normalize_role_text(ar)
            mapping.append((nar, profile.get("RoleTitle")))

    # 1. Exact match
    for nar, role_title in mapping:
        if norm_emp == nar:
            return role_title

    # 2. Token match
    emp_tokens = set(norm_emp.split())
    for nar, role_title in mapping:
        nar_tokens = set(nar.split())
        if emp_tokens == nar_tokens:
            return role_title

    # 3. Fuzzy match
    best_score = 0
    best_role = None
    for nar, role_title in mapping:
        score = fuzzy_ratio(norm_emp, nar)
        if score > best_score:
            best_score = score
            best_role = role_title

    if best_score >= fuzzy_threshold:
        return best_role

    return None

def clean_json_response(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


# -------------------- PHASE 1: ANALYSIS -------------------- #

async def process_batch_async(client, batch_data, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (Analysis)...")

        prompt_template = Template("""
        You are an Expert HR AI.
        
        TASK:
        1. Analyze the "INPUT DATA" (Employees).
        2. Perform an appraisal analysis comparing 'current_role' vs 'target_role'.
        3. Assign quantitative scores (1-10) and qualitative insights.

        INPUT DATA:
        $batch_json

        INSTRUCTIONS:
        Return a strictly formatted JSON LIST of objects.
        
        REQUIRED JSON STRUCTURE:
        [
            {
                "employee_name": "String",
                "employee_id": "String (from input)",
                "employee_analysis": {
                    "current_role": "String",
                    "target_role": "String",
                    "quantitative_scores": {
                        "performance_score": 5,
                        "potential_score": 5,
                        "risk_of_attrition": "Low/Medium/High"
                    },
                    "qualitative_analysis": {
                        "top_skills": ["Skill 1", "Skill 2"],
                        "behavioral_traits": ["Trait 1", "Trait 2"],
                        "competency_gaps": ["Gap 1", "Gap 2"]
                    },
                    "reasoning": "Summary here."
                }
            }
        ]
        """)

        prompt = prompt_template.substitute(
            batch_json=json.dumps(batch_data, indent=2)
        )

        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return clean_json_response(response.text)
        except Exception as e:
            print(f"Batch {batch_id} Error:", e)
            return None

async def orchestrate_processing(all_employees, api_key):
    # 1. Create Client INSIDE the async flow
    client = genai.Client(api_key=api_key)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for i in range(0, len(all_employees), BATCH_SIZE):
        batch = all_employees[i:i + BATCH_SIZE]
        batch_id = (i // BATCH_SIZE) + 1
        # Pass the client to the batch function
        tasks.append(process_batch_async(client, batch, batch_id, semaphore))

    results = await asyncio.gather(*tasks)
    
    final_data = []
    for r in results:
        if r: final_data.extend(r)
    return final_data


# -------------------- PHASE 2: IDP GENERATION -------------------- #

async def second_pass_batch(client, batch, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (IDP Generation)...")

        prompt_template = Template("""
        You are an Expert HR AI.

        TASK:
        You are receiving employee data that includes a 'nine_box_label'.
        Add a new field 'idp_plan' based on that label.

        RULES:
        - If label is top-tier (Stars) -> Leadership IDP
        - If label is mid-tier (Core) -> Improvement IDP
        - If label is bottom-tier (Risk) -> Recovery Plan

        INPUT DATA:
        $batch_json

        OUTPUT STRUCTURE:
        Return the exact same list, but inside 'employee_analysis', add:
        "idp_plan": {
            "idp_type": "Leadership/Improvement/Recovery",
            "focus_areas": ["Area 1", "Area 2"],
            "action_plan": ["Action 1", "Action 2"],
            "timeline": "6 months"
        }
        """)

        prompt = prompt_template.substitute(
            batch_json=json.dumps(batch, indent=2)
        )

        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return clean_json_response(response.text)
        except Exception as e:
            print(f"IDP Batch {batch_id} Error:", e)
            return None

async def orchestrate_second_pass(processed_with_9box, api_key):
    # 1. Create Client INSIDE the async flow
    client = genai.Client(api_key=api_key)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for i in range(0, len(processed_with_9box), BATCH_SIZE):
        batch = processed_with_9box[i:i + BATCH_SIZE]
        batch_id = (i // BATCH_SIZE) + 1
        # Pass the client to the batch function
        tasks.append(second_pass_batch(client, batch, batch_id, semaphore))

    results = await asyncio.gather(*tasks)
    
    final_output = []
    for r in results:
        if r: final_output.extend(r)
    return final_output


# -------------------- MAIN ROUTES -------------------- #

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "HR AI Backend Running"})

@app.route("/upload", methods=["POST"])
def upload_file():
    start_time = time.time()

    try:
        # 1. Check File
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # 2. Save & Read
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path)
        employees = df.to_dict(orient="records")

        # 3. Get Success Profiles
        success_profiles = list(db["success_profiles"].find({}))
        
        # 4. Map Roles (Local Logic)
        mapped_employees = []
        for emp in employees:
            # Case-insensitive column finding
            role_key = next((k for k in emp.keys() if k.lower() in ['role', 'job title', 'designation']), None)
            
            if role_key:
                current_role = emp[role_key]
                target = map_target_role(current_role, success_profiles)
                if target:
                    emp["current_role"] = current_role
                    emp["target_role"] = target
                    mapped_employees.append(emp)

        if not mapped_employees:
            return jsonify({"error": "No employees matched Success Profiles roles."}), 400

        # 5. First Pass: AI Analysis
        print(f"--- Starting Phase 1: Analysis ({len(mapped_employees)} employees) ---")
        # Pass API_KEY here
        processed_data = asyncio.run(orchestrate_processing(mapped_employees, API_KEY))
        
        if not processed_data:
            return jsonify({"error": "AI Phase 1 returned no data"}), 500

        # 6. Apply NineBox Logic
        processed_with_9box = NineBox.apply(processed_data)

        # 7. Second Pass: IDP Generation
        print("--- Starting Phase 2: IDP Generation ---")
        # Pass API_KEY here
        final_output = asyncio.run(orchestrate_second_pass(processed_with_9box, API_KEY))

        if not final_output:
            return jsonify({"error": "AI Phase 2 failed"}), 500

        # 8. Save & Return
        output_file_path = os.path.join(UPLOAD_FOLDER, "results.json")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        return jsonify({
            "status": "success",
            "total_processed": len(final_output),
            "response_time_seconds": duration,
            "results_file": output_file_path,
            "results": final_output
        })

    except Exception as e:
        end_time = time.time()
        print(f"Server Error: {e}")
        return jsonify({
            "error": str(e),
            "response_time_seconds": round(end_time - start_time, 2)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)