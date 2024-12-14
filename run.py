from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow communication with React frontend

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_text = data.get("text")
    
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize input text and generate translation
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs['input_ids'])
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"translatedText": translated_text})

if __name__ == '__main__':
    app.run(debug=True)
