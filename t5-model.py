from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import json

# Cargar el modelo y el tokenizador T5 base
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Cargar los datos de entrenamiento
with open('training_data.json') as f:
    training_data = json.load(f)

# Preparar los datos de entrenamiento en un formato adecuado para Hugging Face
def preprocess_data(data):
    inputs = [f"cotizacion: {item['input']}" for item in data]
    outputs = [item["output"] for item in data]
    return {"input_texts": inputs, "target_texts": outputs}

data = preprocess_data(training_data)
dataset = Dataset.from_dict(data)

# Tokenizar los datos
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_texts"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_texts"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input_texts", "target_texts"])

# Ajustar el modelo para la tarea de lenguaje
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
model.save_pretrained("./fine_tuned_model_t5")
tokenizer.save_pretrained("./fine_tuned_model_t5")
