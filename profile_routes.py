from flask import Blueprint, request, jsonify
from bson import ObjectId

# Blueprint name MUST be profile_routes
profile_routes = Blueprint("profile_routes", __name__)

# -----------------------------
# UPDATE PROFILE
# -----------------------------
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


# -----------------------------
# DELETE PROFILE
# -----------------------------
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
