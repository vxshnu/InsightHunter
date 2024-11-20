from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MODEL_PATH = "vxshnu8690/text-to-pandas"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get("query")
        if not input_data:
            return jsonify({"error": "No query provided"}), 400
        inputs = tokenizer.encode("generate pandas code: " + input_data, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded_output)
        return jsonify({"result": decoded_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
