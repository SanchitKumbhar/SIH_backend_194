import uuid
import random
import os
import secrets
import bcrypt 
import pandas as pd 
import re 
from flask import Blueprint, request, jsonify
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename 
import datetime 
import time

# Create the Blueprint for employee routes
employee_routes = Blueprint('employee_routes', __name__)

# Define the upload folder path
UPLOAD_FOLDER = "uploads" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

# --- Helper Functions ---
def generate_employee_id(db_collection):
    """
    Generates a unique Employee ID (e.g., E-1001) that is guaranteed 
    not to exist in the specified MongoDB collection by checking for duplicates.
    """
    prefix = "E"
    
    # Retry loop to guarantee uniqueness
    while True:
        unique_num = random.randint(1000, 9999) 
        new_id = f"{prefix}-{unique_num}"
        
        # Check the database for the generated ID
        if db_collection.find_one({"EmployeeID": new_id}) is None:
            return new_id
        
def generate_initial_password(length=10):
    """
    Generates a secure, temporary password.
    Ensures a minimum length of 8.
    """
    if length < 8:
        length = 8
    return secrets.token_urlsafe(length)

def clean_header(col_name):
    """Removes non-alphanumeric characters and converts to lowercase for robust mapping."""
    return re.sub(r'[^a-z0-9]', '', str(col_name).lower()) 


# ----------------------------------------------------------------------
# ➡️ ROUTE 1: HR CREATES NEW EMPLOYEE (POST)
# ----------------------------------------------------------------------
@employee_routes.route('/hr/create-employee', methods=['POST'])
def create_new_employee():
    """
    Handles HR's request to create a new employee record with minimal data.
    Stores CorporateEmail and optionally PersonalEmail.
    """
    db = request.db
    data = request.json
    
    # MANDATORY check is now 'CorporateEmail' instead of generic 'Email'
    if not all(k in data for k in ['Name', 'CorporateEmail', 'Role']):
        return jsonify({"error": "Missing required fields: Name, CorporateEmail, and Role."}), 400

    try:
        emp_master_collection = db["Employee_Master"]
        
        emp_id = generate_employee_id(emp_master_collection)
        initial_password = generate_initial_password() 
        
        hashed_password = bcrypt.hashpw(
            initial_password.encode('utf-8'), 
            bcrypt.gensalt()
        ).decode('utf-8')
        
        employee_doc = {
            "EmployeeID": emp_id,
            "Name": data['Name'],
            "CorporateEmail": data['CorporateEmail'], # HR provided email
            "PersonalEmail": data.get('PersonalEmail'), # Optional, if HR provides it manually
            "Role": data['Role'],
            "Password": hashed_password,
            "IsProfileComplete": False, 
            "IsActive": True,
            "CreationDate": datetime.datetime.now(),
            "Department": data.get("Department"),
            "Location": data.get("Location"),
        }
        
        employee_doc = {k: v for k, v in employee_doc.items() if v is not None}
        
        emp_master_collection.insert_one(employee_doc)

        return jsonify({
            "message": "Employee record created successfully.",
            "EmployeeID": emp_id,
            "InitialPassword": initial_password, 
        }), 201

    except PyMongoError as e:
        return jsonify({"error": f"Database error during insertion: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# ----------------------------------------------------------------------
# ⬆️ ROUTE 2: HR BULK UPLOAD EMPLOYEES (POST)
# ----------------------------------------------------------------------
@employee_routes.route("/hr/bulk-upload-employees", methods=["POST"])
def bulk_upload_employees():
    """
    Handles bulk insertion/update, storing CorporateEmail and PersonalEmail separately.
    """
    db = request.db
    
    # 1. File Handling (omitted for brevity)
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Could not save file: {str(e)}"}), 500

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath, encoding='latin1') 
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or Excel."}), 400

        # --- Column Mapping and Standardization ---
        emp_master_collection = db["Employee_Master"]
        
        cleaned_cols_map = {original: clean_header(original) for original in df.columns}
        df.columns = df.columns.map(cleaned_cols_map)
        
        # Mapping definitions updated for CorporateEmail and PersonalEmail
        column_map = {
            'employeeidempide': 'EmployeeID', 'employeeid': 'EmployeeID', 'empid': 'EmployeeID', 
            'corporateemail': 'CorporateEmail', # MANDATORY
            'personalemail': 'PersonalEmail',   # OPTIONAL
            'jobtitle': 'Role', 'name': 'Name', 
            'departmentfunction': 'Department', 'department': 'Department',
            'location': 'Location', 'worklocation': 'Location',
        }
        
        required_field_source = {}
        for cleaned_col_name, target_key in column_map.items():
            if cleaned_col_name in df.columns:
                if target_key not in required_field_source:
                    required_field_source[target_key] = cleaned_col_name
        
        # Check for the presence of all MANDATORY fields
        missing_required_fields = []
        if 'EmployeeID' not in required_field_source: missing_required_fields.append('Employee ID (e.g., EmpID)')
        if 'Name' not in required_field_source: missing_required_fields.append('Name') 
        if 'CorporateEmail' not in required_field_source: missing_required_fields.append('Corporate Email') # Check updated
        if 'Role' not in required_field_source: missing_required_fields.append('Job Title')
             
        if missing_required_fields:
            os.remove(filepath)
            return jsonify({
                "error": "Missing mandatory columns in the sheet.",
                "note": f"Please ensure your sheet contains columns for: {', '.join(missing_required_fields)}"
            }), 400

        # 3. Process records and prepare for Upsert
        
        # Mandatory source columns
        source_id_col = required_field_source['EmployeeID']
        name_col = required_field_source['Name'] 
        corporate_email_col = required_field_source['CorporateEmail'] # New source column
        role_col = required_field_source['Role']
        
        # Optional source columns
        personal_email_col = required_field_source.get('PersonalEmail') # New source column
        dept_col = required_field_source.get('Department')
        location_col = required_field_source.get('Location')
        
        df.dropna(subset=[source_id_col], inplace=True)
        
        cols_to_convert = [c for c in [source_id_col, name_col, corporate_email_col, role_col, personal_email_col, dept_col, location_col] if c is not None]
        for col in cols_to_convert:
             df[col] = df[col].astype(str).fillna('') 
        
        processed_count = 0
        credentials_list = [] 

        for index, row in df.iterrows():
            
            emp_id = row[source_id_col]
            generated_password = generate_initial_password() 
            
            hashed_password = bcrypt.hashpw(
                generated_password.encode('utf-8'), 
                bcrypt.gensalt()
            ).decode('utf-8')

            # Prepare the data document 
            employee_data = {
                "EmployeeID": emp_id,
                "Name": row[name_col],         
                "CorporateEmail": row[corporate_email_col], # Storing Corporate Email
                "PersonalEmail": row[personal_email_col] if personal_email_col else None, # Storing Personal Email if found
                "Role": row[role_col],
                "Password": hashed_password, 
                "IsActive": True,
                "IsProfileComplete": True, 
                "CreationDate": datetime.datetime.now(),
                "Department": row[dept_col] if dept_col else None,
                "Location": row[location_col] if location_col else None,
            }
            
            employee_data = {k: v for k, v in employee_data.items() if v and pd.notna(v) and v is not None}

            # Perform Upsert
            result = emp_master_collection.update_one(
                {"EmployeeID": emp_id},
                {"$set": employee_data},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                processed_count += 1
                
                credentials_list.append({
                    "EmployeeID": emp_id,
                    "Name": row[name_col],
                    "PlaintextPassword": generated_password
                })
                
        # 4. Cleanup the uploaded file
        os.remove(filepath)

        # --- Generate and Save Credentials CSV ---
        credentials_df = pd.DataFrame(credentials_list)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        credentials_filename = f"new_credentials_{timestamp}.csv"
        credentials_filepath = os.path.join(UPLOAD_FOLDER, credentials_filename)
        
        credentials_df.to_csv(credentials_filepath, index=False)

        # 5. Return success response
        return jsonify({
            "message": f"Successfully processed and updated/inserted {processed_count} employee records.",
            "note": "A file containing the plaintext initial passwords has been created.",
            "credentials_file": credentials_filename, 
            "total_records_processed": len(df),
        }), 200

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({"error": f"An unexpected error occurred during file processing: {str(e)}"}), 500


# ----------------------------------------------------------------------
# 👤 ROUTE 3: EMPLOYEE COMPLETES PROFILE (PUT)
# ----------------------------------------------------------------------
@employee_routes.route('/employee/complete-profile/<employee_id>', methods=['PUT'])
def employee_complete_profile(employee_id):
    """
    Allows an employee to complete their profile with additional data.
    Allows updating Name, PersonalEmail, and ContactPhone.
    """
    db = request.db
    data = request.json
    
    update_fields = {}
    if 'Name' in data: 
        update_fields['Name'] = data['Name']
    if 'PersonalEmail' in data: # Already supports PersonalEmail update
        update_fields['PersonalEmail'] = data['PersonalEmail']
    if 'ContactPhone' in data:
        update_fields['ContactPhone'] = data['ContactPhone']
    
    if update_fields:
        update_fields['IsProfileComplete'] = True
    
    if not update_fields:
        return jsonify({"message": "No fields provided for update."}), 200

    try:
        result = db['Employee_Master'].update_one(
            {"EmployeeID": employee_id},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            return jsonify({"error": "EmployeeID not found."}), 404

        return jsonify({
            "message": "Employee profile successfully completed/updated.",
            "EmployeeID": employee_id,
            "updated_fields": list(update_fields.keys())
        }), 200

    except PyMongoError as e:
        return jsonify({"error": f"Database error during update: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# ----------------------------------------------------------------------
# 🔎 ROUTE 4: GET EMPLOYEE DETAIL (GET)
# ----------------------------------------------------------------------
@employee_routes.route('/employee/details/<employee_id>', methods=['GET'])
def get_employee_details(employee_id):
    """
    Retrieves employee details from the database using EmployeeID.
    Excludes the sensitive 'Password' hash from the result.
    """
    db = request.db

    try:
        employee = db['Employee_Master'].find_one(
            {"EmployeeID": employee_id},
            {'_id': 0, 'Password': 0} 
        )

        if not employee:
            return jsonify({"error": "EmployeeID not found."}), 404

        if 'CreationDate' in employee:
            employee['CreationDate'] = employee['CreationDate'].isoformat()
        
        return jsonify(employee), 200

    except PyMongoError as e:
        return jsonify({"error": f"Database error during retrieval: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500