from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Cargar el modelo y el tokenizador entrenados
model_name = "./fine_tuned_model_t5"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/generate_quote', methods=['POST'])
def generate_quote():
    data = request.get_json()
    furniture_name = data['name']
    material = data['material']
    quantity = data['quantity']
    
    # Asegurarse de que los datos estén presentes
    if not furniture_name or not material or not quantity:
        return jsonify({'quote': '', 'error': 'Missing required data'}), 400
    
    prompt = f"Generar precio y descripcion para {quantity} {furniture_name} de {material}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generar la cotización con el modelo T5
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    quote = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'Cotizacion:': quote})

if __name__ == '__main__':
    app.run(debug=True)
