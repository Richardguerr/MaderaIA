from flask import Flask, request, jsonify
import openai
import json

app = Flask(__name__)

# Cargar la API Key de OpenAI
openai.api_key = 'sk-proj-me1uFVplzCSoKqGGlcvdT3BlbkFJBsRr1kMUPfgLfeXjD38g'

# Cargar los datos de entrenamiento
with open('training_data.json') as f:
    training_data = json.load(f)

# Endpoint para agregar muebles
@app.route('/add_furniture', methods=['POST'])
def add_furniture():
    data = request.get_json()
    training_data.append(data)
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f)
    return jsonify({'message': 'Furniture added successfully'}), 201

# Endpoint para obtener una cotización
@app.route('/get_quote', methods=['POST'])
def get_quote():
    data = request.get_json()
    furniture_name = data['name']
    quantity = data['quantity']
    
    # Buscar el mueble en los datos de entrenamiento
    furniture = next((item for item in training_data if item['input'] == f"Generar cotización para una {furniture_name}."), None)
    
    if not furniture:
        return jsonify({'message': 'Furniture not found'}), 404

    # Generar la cotización usando GPT-3.5-turbo
    prompt = f"Generar cotización para {quantity} {furniture_name} de {furniture['output'].split()[4]}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente que genera cotizaciones de muebles."},
            {"role": "user", "content": prompt}
        ]
    )
    
    quote = response['choices'][0]['message']['content'].strip()
    total_cost = float(furniture['output'].split()[5][1:]) * quantity
    
    return jsonify({
        'furniture': furniture_name,
        'material': furniture['output'].split()[4],
        'quantity': quantity,
        'total_cost': total_cost,
        'quote': quote
    })

if __name__ == '__main__':
    app.run(debug=True)
