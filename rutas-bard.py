from flask import Flask, request, jsonify
from transformers import GPTJForCausalLM, GPT2Tokenizer


app = Flask(__name__)

# Cargar el modelo fine-tuneado y el tokenizador
model = GPTJForCausalLM.from_pretrained("./fine_tuned_model_bard")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_bard")

@app.route('/get_quote', methods=['POST'])
def get_quote():
    data = request.get_json()
    furniture_name = data['name']
    quantity = data['quantity']
    
    prompt = f"Generar cotización para {quantity} {furniture_name}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    quote = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'quote': quote})

if __name__ == '__main__':
    app.run(debug=True)
