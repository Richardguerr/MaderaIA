from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///furniture.db'
db = SQLAlchemy(app)

# Cargar el modelo fine-tuneado y el tokenizador dentro de una función
def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
    return model, tokenizer

# Definir el modelo y el tokenizador como variables globales
model, tokenizer = load_model_and_tokenizer()

class Furniture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    material = db.Column(db.String(100), nullable=False)
    price_per_unit = db.Column(db.Float, nullable=False)

@app.route('/add_furniture', methods=['POST'])
def add_furniture():
    data = request.get_json()
    new_furniture = Furniture(
        name=data['name'],
        material=data['material'],
        price_per_unit=data['price_per_unit']
    )
    db.session.add(new_furniture)
    db.session.commit()
    return jsonify({'message': 'Furniture added successfully'}), 201

@app.route('/get_quote', methods=['POST'])
def get_quote():
    data = request.get_json()
    furniture_name = data['name']
    quantity = data['quantity']
    
    furniture = Furniture.query.filter_by(name=furniture_name).first()
    if not furniture:
        return jsonify({'message': 'Furniture not found'}), 404
    
    # Cargar el modelo y el tokenizador dentro del contexto de la aplicación Flask
    with app.app_context():
        model, tokenizer = load_model_and_tokenizer()
    
    prompt = f"Generar cotización para {quantity} {furniture_name} de {furniture.material}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    quote = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    total_cost = furniture.price_per_unit * quantity
    return jsonify({
        'furniture': furniture.name,
        'material': furniture.material,
        'quantity': quantity,
        'total_cost': total_cost,
        'quote': quote
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
