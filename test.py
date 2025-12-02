import ollama
import json
import random
import re

# --- 1. CONFIGURATION ---
BATCH_SIZE = 5  # Recommended: 5 for 7B models. 20 might hit context limits or cause JSON errors.
MODEL_NAME = "qwen2:7b" # Ensure you have this pulled in Ollama

# --- 2. DATA GENERATION ---
roles = ["Developer", "QA", "Data Analyst", "Project Manager", "Designer"]
skills = ["Leadership", "Communication", "Time Management", "Technical", "Problem Solving", "Teamwork", "Creativity"]

# Generate 20 employees (Mock Data)
employees = []
for i in range(1, 21):
    name = f"Employee_{i}"
    role = random.choice(roles)
    gaps = random.sample(skills, k=2)
    # We add a fake 'raw_feedback' to give the LLM something to analyze for the "Reasoning" field
    raw_feedback = f"{name} is a decent {role} but struggles with {gaps[0]} and needs to improve {gaps[1]}."
    employees.append({"id": i, "name": name, "role": role, "gaps": gaps, "feedback": raw_feedback})

print(f"[OK] Generated {len(employees)} employee profiles.\n")

# --- 3. HELPER FUNCTION: JSON PARSER ---
def safe_json_parse(text):
    """
    Robust JSON extraction that looks for the first '[' and last ']' 
    to handle cases where LLM adds conversational text.
    """
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        # Regex to find a JSON list structure
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# --- 4. CORE FUNCTION: BATCH LLM CALL ---
def process_batch(employee_batch):
    # Convert the batch of python dicts to a string representation for the prompt
    batch_input_str = json.dumps(employee_batch, indent=2)

    prompt = f"""
    You are an Expert HR AI. Process the following list of employees.
    
    INPUT DATA:
    {batch_input_str}

    INSTRUCTIONS:
    For EACH employee in the input list, perform an appraisal analysis based on their feedback and gaps.
    Return a strictly formatted JSON LIST of objects. 
    
    REQUIRED JSON STRUCTURE PER EMPLOYEE:
    {{
      "employee_name": "String",
      "employee_analysis": {{
        "quantitative_scores": {{
          "performance_score": Integer (1-10),
          "potential_score": Integer (1-10),
          "risk_of_attrition": "Low/Medium/High"
        }},
        "qualitative_analysis": {{
          "top_skills": [List of 3 inferred skills based on role],
          "behavioral_traits": [List of 3 traits],
          "competency_gaps": [List of gaps from input]
        }},
        "reasoning": "A short summary explaining the scores based on the input feedback."
      }}
    }}

    IMPORTANT:
    - Output ONLY valid JSON.
    - Start the output with [ and end with ].
    - Do not include markdown formatting (like ```json).
    """

    print(f"... Processing batch of {len(employee_batch)} employees...")
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1} # Low temp for consistent formatting
        )
        
        return response["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Error calling LLM: {e}")
        return None

# --- 5. MAIN EXECUTION LOOP ---
all_results = []

# Loop through employees in chunks of BATCH_SIZE
for i in range(0, len(employees), BATCH_SIZE):
    batch = employees[i : i + BATCH_SIZE]
    
    raw_response = process_batch(batch)
    
    if raw_response:
        parsed_data = safe_json_parse(raw_response)
        
        if parsed_data:
            print(f"[OK] Batch {i//BATCH_SIZE + 1} Success! Got {len(parsed_data)} records.")
            all_results.extend(parsed_data)
        else:
            print(f"[WARN] Failed to parse JSON for Batch {i//BATCH_SIZE + 1}")
            print("Raw Output snippet:", raw_response[:200]) # Debugging
    else:
        print("[WARN] No response from LLM.")

# --- 6. FINAL OUTPUT ---
print("\n" + "="*30)
print(f"[DONE] COMPLETED. Total Processed: {len(all_results)}/{len(employees)}")
print("="*30 + "\n")

# Save to file
with open("batch_analysis_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print first result as sample
if all_results:
    print("Sample Result (First Employee):")
    print(json.dumps(all_results[0], indent=2))