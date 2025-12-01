from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    prompt = data.get("prompt", "")

    # Call Ollama (sync)
    response = ollama.chat(
        model="qwen2:7b",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({"response": response["message"]["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
