import uuid
import random
import string
import os
import secrets
import bcrypt # Recommended for hashing passwords
import pandas as pd # For handling uploaded sheets
from flask import Blueprint, request, jsonify, current_app
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename # Good practice for filename handling

# Create the Blueprint for employee routes
employee_routes = Blueprint('employee_routes', __name__)

# Define the upload folder path (must match app.py or be consistently handled)
UPLOAD_FOLDER = "uploads" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure the directory exists

# Helper function to generate a unique Employee ID
def generate_employee_id():
    """Generates a unique Employee ID (e.g., CMEP-1001)"""
    # In a real system, you'd check against the database to ensure uniqueness
    prefix = "CMEP"
    # Generate a random 4-digit number for uniqueness in this example
    unique_num = random.randint(1000, 9999) 
    return f"{prefix}-{unique_num}"

# Helper function to generate a temporary password
def generate_initial_password(length=10):
    """Generates a secure, temporary password."""
    # Using secrets.token_urlsafe is more secure than random.choice
    return secrets.token_urlsafe(length)


# --- ROUTE 1: HR CREATES NEW EMPLOYEE (POST) --
@employee_routes.route('/hr/create-employee', methods=['POST'])
def create_new_employee():
    """
    Handles HR's request to create a new employee record with minimal data.
    System generates EmpID and Initial Password.
    """
    db = request.db
    data = request.json
    
    # 1. Input validation (minimal required fields)
    if not all(k in data for k in ['FirstName', 'LastName', 'Email', 'Role']):
        return jsonify({"error": "Missing required fields: FirstName, LastName, Email, and Role."}), 400

    try:
        # 2. Generate credentials
        emp_id = generate_employee_id()
        initial_password = generate_initial_password()
        
        # 3. Hash the password for storage
        hashed_password = bcrypt.hashpw(
            initial_password.encode('utf-8'), 
            bcrypt.gensalt()
        ).decode('utf-8')
        
        employee_doc = {
            "EmployeeID": emp_id,
            "FirstName": data['FirstName'],
            "LastName": data['LastName'],
            "Email": data['Email'],
            "Role": data['Role'],
            "Password": hashed_password,
            "IsProfileComplete": False, # Requires employee to complete profile later
            "IsActive": True,
            "CreationDate": pd.to_datetime("now"),
            # Additional fields from the request (if any)
            "Department": data.get("Department"),
        }
        
        # 4. Insert into emp_master collection
        db['emp_master'].insert_one(employee_doc)

        return jsonify({
            "message": "Employee record created successfully.",
            "EmployeeID": emp_id,
            "InitialPassword": initial_password, # NOTE: This should be handled securely in a real app
        }), 201

    except PyMongoError as e:
        return jsonify({"error": f"Database error during insertion: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# --- NEW ROUTE 2: HR BULK UPLOAD EMPLOYEES (POST) ---
@employee_routes.route("/hr/bulk-upload-employees", methods=["POST"])
def bulk_upload_employees():
    """
    Handles bulk insertion and updating of employee records from a CSV/XLSX file
    into the 'emp_master' collection using EmployeeID as the unique key.
    Automatically generates and hashes passwords for all records.
    """
    db = request.db
    
    # 1. File Handling and Validation
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Temporarily save the file
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Could not save file: {str(e)}"}), 500

    try:
        # 2. Read the file into a Pandas DataFrame
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or Excel."}), 400

        # Check for required columns
        required_cols = ["EmployeeID", "FirstName", "LastName", "Email", "Role"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns in sheet: {', '.join(missing_cols)}"}), 400

        # 3. Process records and prepare for Upsert
        emp_master_collection = db["emp_master"]
        
        # Drop rows where EmployeeID is missing
        df.dropna(subset=["EmployeeID"], inplace=True)
        # Convert all relevant string columns to string type and fill NaNs for safer processing
        for col in ["EmployeeID", "FirstName", "LastName", "Email", "Role"]:
             df[col] = df[col].astype(str)
        
        processed_count = 0
        new_employee_passwords = {} # Store plain passwords for immediate feedback (secure distribution necessary)

        for index, row in df.iterrows():
            emp_id = row["EmployeeID"]
            
            # Auto-generate and hash password only if the employee is new or we need to reset it.
            # For simplicity, we generate and $set the password for every record here.
            generated_password = generate_initial_password()
            hashed_password = bcrypt.hashpw(
                generated_password.encode('utf-8'), 
                bcrypt.gensalt()
            ).decode('utf-8')

            # Prepare the data document
            employee_data = {
                "EmployeeID": emp_id,
                "FirstName": row["FirstName"],
                "LastName": row["LastName"],
                "Email": row["Email"],
                "Role": row["Role"],
                "Password": hashed_password,
                "IsActive": True,
                "IsProfileComplete": True, # Assume basic profile is complete via bulk upload
                "Department": row.get("Department", None),
                "Location": row.get("Location", None),
                # Add other columns from the sheet dynamically/explicitly
            }
            
            # Remove keys where the value is None or pandas NaN after data cleaning
            employee_data = {k: v for k, v in employee_data.items() if pd.notna(v) and v is not None}

            # Perform Upsert: Update if EmployeeID exists, Insert if new
            result = emp_master_collection.update_one(
                {"EmployeeID": emp_id},
                {"$set": employee_data},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                new_employee_passwords[emp_id] = generated_password
                processed_count += 1
                
        # 4. Cleanup the uploaded file
        os.remove(filepath)
        
        # 5. Return success response
        return jsonify({
            "message": f"Successfully processed and updated/inserted {processed_count} employee records into emp_master.",
            "note": "Passwords were auto-generated and hashed. Please distribute them securely.",
            "total_records_in_sheet": len(df),
            "sample_passwords": {k: v for i, (k, v) in enumerate(new_employee_passwords.items()) if i < 3} # Show a sample of new/updated passwords
        }), 200

    except Exception as e:
        # Cleanup file if processing failed
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({"error": f"An error occurred during file processing: {str(e)}"}), 500


# --- ROUTE 3: EMPLOYEE COMPLETES PROFILE (PUT) ---
@employee_routes.route('/employee/complete-profile/<employee_id>', methods=['PUT'])
def employee_complete_profile(employee_id):
    """
    Allows an employee to complete their profile with additional data.
    """
    db = request.db
    data = request.json
    
    # Fields the employee is completing
    update_fields = {}
    if 'FirstName' in data:
        update_fields['FirstName'] = data['FirstName']
    if 'LastName' in data:
        update_fields['LastName'] = data['LastName']
    if 'PersonalEmail' in data:
        update_fields['PersonalEmail'] = data['PersonalEmail']
    if 'ContactPhone' in data:
        update_fields['ContactPhone'] = data['ContactPhone']
    
    # Mark profile as complete if they provide enough data
    if update_fields:
        update_fields['IsProfileComplete'] = True
    
    if not update_fields:
        return jsonify({"message": "No fields provided for update."}), 200

    try:
        # Update the Employee_Master document matching the EmployeeID
        result = db['emp_master'].update_one( # Updated collection name to 'emp_master'
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