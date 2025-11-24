from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt as pyjwt
import datetime
from bson import ObjectId

auth_routes = Blueprint("auth_routes", __name__)

SECRET_KEY = "MY_SUPER_SECRET_KEY"   
# post request to signup new user
@auth_routes.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.json

        required_fields = ["full_name", "email", "password", "role"]

        # Validate fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        users = request.db["users"]

        # Check if email already exists
        if users.find_one({"email": data["email"]}):
            return jsonify({"error": "Email already registered"}), 409

        hashed_pw = generate_password_hash(data["password"])

        user_data = {
            "full_name": data["full_name"],
            "email": data["email"],
            "password": hashed_pw,
            "role": data["role"],
            "created_at": datetime.datetime.utcnow()
        }

        result = users.insert_one(user_data)
        user_data["_id"] = str(result.inserted_id)

        return jsonify({
            "message": "Signup successful",
            "user": user_data
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#login route
@auth_routes.route("/login", methods=["POST"])
def login():
    try:
        data = request.json

        required_fields = ["username", "password", "role"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        users = request.db["users"]

        # username = email in your UI
        user = users.find_one({"email": data["username"], "role": data["role"]})

        if not user:
            return jsonify({"error": "User not found or role mismatch"}), 404

        # Password check
        if not check_password_hash(user["password"], data["password"]):
            return jsonify({"error": "Invalid password"}), 401

        # Generate JWT Token
        token_payload = {
            "user_id": str(user["_id"]),
            "role": user["role"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=10)
        }

        token = pyjwt.encode(token_payload, SECRET_KEY, algorithm="HS256")

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "user_id": str(user["_id"]),
                "full_name": user["full_name"],
                "email": user["email"],
                "role": user["role"]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
