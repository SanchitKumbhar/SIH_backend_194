import uuid
import random
import string
from flask import Blueprint, request, jsonify, current_app
from pymongo.errors import PyMongoError

# Create the Blueprint for employee routes
employee_routes = Blueprint('employee_routes', __name__)

# Helper function to generate a unique Employee ID
def generate_employee_id():
    """Generates a unique Employee ID (e.g., CMEP-1001)"""
    # In a real system, you'd check against the database to ensure uniqueness
    # For this example, we use a simple counter and prefix
    prefix = "CMEP"
    # Generate a random 4-digit number for uniqueness in this example
    unique_num = random.randint(1000, 9999) 
    return f"{prefix}-{unique_num}"

# Helper function to generate a temporary password
def generate_initial_password(length=8):
    """Generates a secure, temporary password."""
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for i in range(length))
    # In a real application, this password would be hashed (e.g., using bcrypt)
    return password

# --- ROUTE 1: HR CREATES NEW EMPLOYEE (POST) ---
@employee_routes.route('/hr/create-employee', methods=['POST'])
def create_new_employee():
    """
    Handles HR's request to create a new employee record with minimal data.
    System generates EmpID and Initial Password.
    """
    # FIX APPLIED HERE: Use request.db instead of current_app.db
    db = request.db
    data = request.json

    # 1. Input Validation for Mandatory HR fields
    required_fields = ["DOJ", "JobTitle", "Department", "EmploymentStatus", "ManagerEmpID", "CorporateEmail"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing mandatory HR fields."}), 400

    try:
        # 2. System Generation of Credentials
        new_emp_id = generate_employee_id()
        initial_password = generate_initial_password()
        
        # NOTE: We are storing the plain password here for simplicity in this example. 
        # In production, ALWAYS hash and salt the password before storing it.
        
        # 3. Construct the Initial Document (Employee_Master Table)
        initial_employee_record = {
            "EmployeeID": new_emp_id,
            "InitialPassword": initial_password, # Stored temporarily until first login
            "DateOfJoining": data["DOJ"],
            "JobTitle": data["JobTitle"],
            "Department": data["Department"],
            "EmploymentStatus": data["EmploymentStatus"],
            "ManagerEmpID": data["ManagerEmpID"],
            "CorporateEmail": data["CorporateEmail"],
            
            # Placeholder for employee-filled fields
            "FirstName": None,
            "LastName": None,
            "PersonalEmail": None,
            "ContactPhone": None
        }

        # 4. Insert into the Employee_Master collection (our database table)
        db['Employee_Master'].insert_one(initial_employee_record)

        # 5. Return the generated credentials to HR for communication
        return jsonify({
            "message": "Employee successfully created.",
            "EmployeeID": new_emp_id,
            "InitialPassword": initial_password,
            "NextStep": "Employee must log in and complete profile."
        }), 201

    except PyMongoError as e:
        return jsonify({"error": f"Database error during creation: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# --- ROUTE 2: EMPLOYEE COMPLETES PROFILE (PUT) ---
@employee_routes.route('/employee/complete-profile/<employee_id>', methods=['PUT'])
def complete_employee_profile(employee_id):
    """
    Handles employee's request to update and complete their profile details.
    """
    # FIX APPLIED HERE: Use request.db instead of current_app.db
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
    
    # CorporateEmail is set by HR, but we check if it's being updated (optional)
    if 'CorporateEmail' in data:
        update_fields['CorporateEmail'] = data['CorporateEmail']


    if not update_fields:
        return jsonify({"message": "No fields provided for update."}), 200

    try:
        # Update the Employee_Master document matching the EmployeeID
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