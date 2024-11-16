from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

MODEL_PATH = "path_to_trained_model"  # Replace with your model path
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json.get("query")
        if not input_data:
            return jsonify({"error": "No query provided"}), 400
        
        # Tokenize the input
        inputs = tokenizer.encode("generate pandas code: " + input_data, return_tensors="pt")
        
        # Generate the output
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return the result
        return jsonify({"result": decoded_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
