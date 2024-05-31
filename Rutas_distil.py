from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Cargar el modelo pre-entrenado y el tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model_distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_distilgpt2")

# Función para generar respuestas basadas en una entrada dada
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route("/generate", methods=["POST"])
def generate():
    # Obtener la solicitud del cuerpo JSON
    data = request.get_json()
    prompt = data["prompt"]
    
    # Generar la respuesta
    response = generate_response(prompt)
    
    # Devolver la respuesta como JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
