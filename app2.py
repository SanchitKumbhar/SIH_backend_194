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
    class NineBox:
        @staticmethod
        def apply(data):
            for emp in data:
                # Safe extraction of scores
                analysis = emp.get("employee_analysis", {})
                scores = analysis.get("quantitative_scores", {})
                
                # Handle potential None or String values safely
                try:
                    p_score = float(scores.get("performance_score", 0))
                    pot_score = float(scores.get("potential_score", 0))
                except (ValueError, TypeError):
                    p_score = 0
                    pot_score = 0
                
                # Standard Nine Box Logic (Can be customized)
                label = "Risk" # Default
                
                # High Potential (8-10)
                if pot_score >= 8:
                    if p_score >= 8: label = "Stars"
                    elif p_score >= 5: label = "High Potential"
                    else: label = "Enigma"
                # Medium Potential (5-7)
                elif pot_score >= 5:
                    if p_score >= 8: label = "Core Star"
                    elif p_score >= 5: label = "Core Player"
                    else: label = "Inconsistent Player"
                # Low Potential (1-4)
                else:
                    if p_score >= 8: label = "High Performer"
                    elif p_score >= 5: label = "Effective"
                    else: label = "Risk"

                # Assign label to root
                emp["nine_box_label"] = label
                
                # Assign label to internal analysis (so it shows in nested JSON)
                if "employee_analysis" in emp:
                    emp["employee_analysis"]["nine_box_label"] = label
                    
            return data

# -------------------- APP SETUP -------------------- #
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- CONFIGURATION -------------------- #
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 3
# ⚠️ Replace with your actual API Key
API_KEY = "AIzaSyDS_Rzc2Nu-aEOrO0ECjj0Bj-1qwa1Tw_U" 

# -------------------- DATABASE -------------------- #
mongo_client = MongoClient("mongodb+srv://creatorpanda26:admin%40123@cluster0.izpl1se.mongodb.net/")
db = mongo_client["sih_appraisal"]

# -------------------- HELPERS -------------------- #

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
        if norm_emp == nar:
            matched_roles.add(role_title)
        elif set(norm_emp.split()) == set(nar.split()):
            matched_roles.add(role_title)
        elif fuzzy_ratio(norm_emp, nar) >= fuzzy_threshold:
            matched_roles.add(role_title)

    return list(matched_roles)

def clean_json_response(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return None

# -------------------- PHASE 1: ANALYSIS -------------------- #

async def process_batch_async(client, batch_data, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (Analysis)...")
        
        prompt = Template("""
        You are an Expert HR AI.
        INPUT DATA: $batch_json
        
        TASK: Compare 'current_role' vs 'target_role'. Return JSON List.
        
        CRITICAL SCORING INSTRUCTIONS:
        1. 'competency_gap_count': Count strictly how many specific skills are missing (Integer).
        2. 'experience_alignment_score': Score 1-10 on how well their past experience fits the target.
        3. 'target_role': You MUST include the target role inside 'employee_analysis'.
        
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

# -------------------- PHASE 1.5: SELECT BEST FIT (FLOWCHART LOGIC) -------------------- #

def select_best_role_fits(analyzed_data):
    """
    Implements Flowchart Decision Tree with FIX for null Target Roles.
    """
    grouped = {}

    # 1. Grouping & Metrics
    for entry in analyzed_data:
        emp_id = entry.get("employee_id")
        
        # Safe extraction
        analysis = entry.get("employee_analysis", {})
        scores = analysis.get("quantitative_scores", {})
        
        try:
            perf = float(scores.get("performance_score", 0))
            pot = float(scores.get("potential_score", 0))
            # Default gap count to high (bad) if missing
            gap_count = int(scores.get("competency_gap_count", 99)) 
            exp_align = float(scores.get("experience_alignment_score", 0))
        except:
            perf, pot, gap_count, exp_align = 0, 0, 99, 0
            
        entry["_metrics"] = {
            "total_score": perf + pot,
            "gap_count": gap_count,
            "exp_score": exp_align
        }

        if emp_id not in grouped: grouped[emp_id] = []
        grouped[emp_id].append(entry)

    final_selection = []
    
    # 2. Sorting & Selection
    for emp_id, candidates in grouped.items():
        
        # Sort Logic (Flowchart):
        # 1. Total Score (Desc)
        # 2. Gap Count (Asc) - Lower gaps are better
        # 3. Experience Score (Desc)
        candidates.sort(key=lambda x: (
            -x["_metrics"]["total_score"],
            x["_metrics"]["gap_count"],
            -x["_metrics"]["exp_score"]
        ))
        
        # Winner
        best_fit = candidates[0]
        
        # Alternatives (Losers)
        alternatives = []
        for other in candidates[1:]:
            analysis = other.get("employee_analysis", {})
            
            # --- FIX: LOOK INSIDE ANALYSIS FOR TARGET ROLE ---
            t_role = analysis.get("target_role") 
            if not t_role:
                # Fallback to root if analysis missed it
                t_role = other.get("target_role")
                
            alternatives.append({
                "target_role": t_role, # Corrected extraction
                "nine_box_label": other.get("nine_box_label", analysis.get("nine_box_label", "N/A")),
                "quantitative_scores": analysis.get("quantitative_scores"),
                "reasoning": analysis.get("reasoning")
            })
            
        best_fit["alternative_role_analysis"] = alternatives
        
        # Cleanup
        if "_metrics" in best_fit: del best_fit["_metrics"]
        
        final_selection.append(best_fit)

    return final_selection

# -------------------- PHASE 2: IDP -------------------- #

async def second_pass_batch(client, batch, batch_id, semaphore):
    async with semaphore:
        print(f"Processing Batch {batch_id} (IDP)...")
        prompt = Template("""
        Generate IDP for these employees.
        INPUT DATA: $batch_json
        
        INSTRUCTIONS:
        1. Keep JSON structure EXACTLY as provided.
        2. DO NOT remove 'alternative_role_analysis'.
        3. Only ADD 'idp_plan' inside 'employee_analysis'.
        
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

# -------------------- ROUTES -------------------- #

@app.route("/upload", methods=["POST"])
def upload_file():
    start_time = time.time()
    try:
        if "file" not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files["file"]
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file.filename))
        if "employee_id" not in df.columns: df["employee_id"] = df.index.astype(str)
        else: df["employee_id"] = df["employee_id"].astype(str)
        
        employees = df.to_dict(orient="records")
        success_profiles = list(db["success_profiles"].find({}))
        
        # 1. Map & Explode
        exploded = []
        for emp in employees:
            role_key = next((k for k in emp.keys() if k.lower() in ['role','job title','designation']), None)
            if role_key:
                targets = map_target_roles(emp[role_key], success_profiles)
                if not targets: continue # Skip if no match
                for t in targets:
                    new_emp = emp.copy()
                    new_emp["target_role"] = t
                    exploded.append(new_emp)
                    
        if not exploded: return jsonify({"error": "No roles matched"}), 400

        # 2. Analyze
        analyzed = asyncio.run(orchestrate_processing(exploded, API_KEY))
        
        # 3. Calc Labels (Pre-Selection)
        processed_9box = NineBox.apply(analyzed)

        # 4. Filter (Flowchart Logic + Fix for Nulls)
        best_fits = select_best_role_fits(processed_9box)

        # 5. IDP
        final_output = asyncio.run(orchestrate_second_pass(best_fits, API_KEY))

        return jsonify({
            "status": "success",
            "results": final_output
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)