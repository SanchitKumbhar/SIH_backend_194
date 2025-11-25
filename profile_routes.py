from flask import Blueprint, request, jsonify
from bson import ObjectId

# Blueprint
profile_routes = Blueprint("profile_routes", __name__)

#post request to create success profile

@profile_routes.route("/create-profile", methods=["POST"])
def create_success_profile():
    try:
        profile_data = request.json

        if not profile_data:
            return jsonify({"error": "No JSON data provided"}), 400

        if "RoleTitle" not in profile_data:
            return jsonify({"error": "Missing 'RoleTitle'"}), 400
        if "MinimumExperienceYears" not in profile_data:
            return jsonify({"error": "Missing 'MinimumExperienceYears'"}), 400
        if "RequiredCompetencies" not in profile_data or not isinstance(profile_data["RequiredCompetencies"], dict):
            return jsonify({"error": "RequiredCompetencies must be a dictionary"}), 400

        profile_data.setdefault("ApplicableRoles", [])
        profile_data.setdefault("FunctionalSkills", [])
        profile_data.setdefault("GeographicalExperience", [])

        collection = request.db["success_profiles"]

        result = collection.insert_one(profile_data)
        profile_data["_id"] = str(result.inserted_id)

        return jsonify({
            "message": "Success Profile created successfully",
            "profile_id": profile_data["_id"],
            "data": profile_data
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# get request to fetch all success profiles
@profile_routes.route("/profiles", methods=["GET"])
def get_all_success_profiles():
    try:
        collection = request.db["success_profiles"]
        profiles = list(collection.find({}))

        for p in profiles:
            p["_id"] = str(p["_id"])

        return jsonify(profiles), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#update profile (PUT)
@profile_routes.route("/update-profile/<profile_id>", methods=["PUT"])
def update_profile(profile_id):
    try:
        update_data = request.json
        if not update_data:
            return jsonify({"error": "No data provided"}), 400

        collection = request.db["success_profiles"]

        try:
            obj_id = ObjectId(profile_id)
        except:
            return jsonify({"error": "Invalid profile ID"}), 400

        result = collection.update_one({"_id": obj_id}, {"$set": update_data})

        if result.matched_count == 0:
            return jsonify({"error": "Profile not found"}), 404

        return jsonify({"message": "Profile updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#delete profile (DELETE)
@profile_routes.route("/delete-profile/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id):
    try:
        collection = request.db["success_profiles"]

        try:
            obj_id = ObjectId(profile_id)
        except:
            return jsonify({"error": "Invalid profile ID"}), 400

        result = collection.delete_one({"_id": obj_id})

        if result.deleted_count == 0:
            return jsonify({"error": "Profile not found"}), 404

        return jsonify({"message": "Profile deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
