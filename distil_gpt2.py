from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

# Cargar los datos de entrenamiento
with open('training_data.json') as f:
    training_data = json.load(f)

# Preparar los datos de entrenamiento en un formato adecuado para Hugging Face
def preprocess_data(data):
    texts = [item["prompt"] + " " + item["completion"] for item in data]
    return texts

texts = preprocess_data(training_data)
dataset = Dataset.from_dict({"text": texts})

# Tokenizar los datos
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Configurar el token de padding

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Ajustar el modelo para la tarea de lenguaje
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Cargar el modelo DistilGPT-2
model = GPT2LMHeadModel.from_pretrained(model_name)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador entrenados
model.save_pretrained("./fine_tuned_model_distilgpt2")
tokenizer.save_pretrained("./fine_tuned_model_distilgpt2")
