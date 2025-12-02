# import asyncio
# import time
# from google import genai

# API_KEY = "AIzaSyCBXa7gant3u0-wR1hEG3r_cSQ-syqmCg8"
# MODEL = "gemini-2.0-flash-lite"

# client = genai.Client(api_key=API_KEY)

# # ---- SINGLE REQUEST FUNCTION ----
# async def call_api(request_id: int):
#     """Send a single request and measure response time."""
#     loop = asyncio.get_event_loop()
    
#     start_time = time.perf_counter()  # Start timer

#     # Run sync code in thread executor
#     response = await loop.run_in_executor(
#         None,
#         lambda: client.models.generate_content(
#             model=MODEL,
#             contents=f"Explain how AI works in a few words. Request #{request_id}"
#         )
#     )

#     end_time = time.perf_counter()  # End timer
#     duration = end_time - start_time

#     print(f"Response {request_id}: {response.text[:50]}... | Time: {duration:.2f}s")
#     return {"request_id": request_id, "text": response.text, "time_sec": duration}


# # ---- RUN 5 BATCHES OF 4 PARALLEL REQUESTS ----
# async def run_batches():
#     batch_size = 4
#     total_requests = 20

#     tasks = [call_api(i + 1) for i in range(total_requests)]

#     all_results = []
#     for b in range(5):
#         start_idx = b * batch_size
#         end_idx = start_idx + batch_size
#         batch_tasks = tasks[start_idx:end_idx]

#         print(f"\n⚡ Running batch {b+1} (requests {start_idx+1} to {end_idx})...\n")

#         batch_results = await asyncio.gather(*batch_tasks)
#         batch_time = sum(res["time_sec"] for res in batch_results) / batch_size
#         print(f"Batch {b+1} average response time: {batch_time:.2f}s\n")

#         all_results.extend(batch_results)

#     total_avg_time = sum(res["time_sec"] for res in all_results) / total_requests
#     print(f"\n✅ Total average response time per request: {total_avg_time:.2f}s")

#     return all_results


# # ---- MAIN ----
# if __name__ == "__main__":
#     output = asyncio.run(run_batches())
import asyncio
import json
import random
import re
import time
import os
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
# Replace with your actual API key
GOOGLE_API_KEY = "AIzaSyB3FaqCrIsuk2rLmKX2E7Hn6Pq0eAOil4k"
MODEL_NAME = "gemini-2.0-flash-lite"

BATCH_SIZE = 2
CONCURRENT_LIMIT = 3  # Limits parallel requests to avoid 429 errors

# Initialize the Client (New SDK style)
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- 2. DATA GENERATION (Mock Data) ---
roles = ["Developer", "QA", "Data Analyst", "Project Manager", "Designer"]
skills = ["Leadership", "Communication", "Time Management", "Technical", "Problem Solving", "Teamwork", "Creativity"]

employees = []
for i in range(1, 21):
    name = f"Employee_{i}"
    role = random.choice(roles)
    gaps = random.sample(skills, k=2)
    raw_feedback = f"{name} is a decent {role} but struggles with {gaps[0]} and needs to improve {gaps[1]}."
    employees.append({"id": i, "name": name, "role": role, "gaps": gaps, "feedback": raw_feedback})

print(f"[OK] Generated {len(employees)} employee profiles.\n")

# --- 3. HELPER FUNCTION: JSON PARSER ---
def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback if the model adds markdown like ```json ... ```
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# --- 4. CORE ASYNC FUNCTION (Updated for google-genai) ---
async def process_batch_async(batch, batch_id, semaphore):
    """
    Async function using the new client.aio.models.generate_content method.
    """
    async with semaphore:
        batch_input_str = json.dumps(batch, indent=2)

        prompt = f"""
        You are an Expert HR AI. Process the following list of employees.
        
        INPUT DATA:
        {batch_input_str}

        INSTRUCTIONS:
        For EACH employee, perform an appraisal analysis.
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
              "top_skills": [List of 3 inferred skills],
              "competency_gaps": [List of gaps from input]
            }},
            "reasoning": "Short summary."
          }}
        }}
        """

        print(f"... Starting Batch {batch_id} (Async)...")
        
        try:
            # NEW SDK CALL: client.aio.models.generate_content
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return response.text
        except Exception as e:
            print(f"[ERROR] Batch {batch_id} failed: {e}")
            return None

# --- 5. MAIN ASYNC ORCHESTRATOR ---
async def main():
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    tasks = []
    
    start_time = time.perf_counter()

    # Create batches
    for i in range(0, len(employees), BATCH_SIZE):
        batch = employees[i : i + BATCH_SIZE]
        batch_id = (i // BATCH_SIZE) + 1
        
        task = asyncio.create_task(process_batch_async(batch, batch_id, semaphore))
        tasks.append(task)

    print(f"--- Firing {len(tasks)} batch requests concurrently (Limit: {CONCURRENT_LIMIT}) ---\n")
    
    # Wait for all
    results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # --- 6. PROCESS RESULTS ---
    all_results = []
    
    for idx, raw_response in enumerate(results):
        if raw_response:
            parsed_data = safe_json_parse(raw_response)
            if parsed_data:
                all_results.extend(parsed_data)
            else:
                print(f"[WARN] Failed to parse JSON for Batch {idx+1}")
        else:
            print(f"[WARN] No response for Batch {idx+1}")

    # --- 7. FINAL OUTPUT ---
    print("\n" + "="*40)
    print(f"[DONE] COMPLETED.")
    print(f"Total Processed: {len(all_results)}/{len(employees)}")
    print(f"Total Time Taken: {total_duration:.2f} seconds")
    print("="*40 + "\n")

    # Save to file
    with open("async_gemini_v1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        print("Please set your GOOGLE_API_KEY at the top of the script.")
    else:
        asyncio.run(main())