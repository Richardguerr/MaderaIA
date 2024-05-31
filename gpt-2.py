from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import json
# Cargar el modelo y el tokenizador GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Preparar los datos de entrenamiento
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

# Tokenizar los datos de entrenamiento
tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(data):
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]
    inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    outputs = tokenizer(outputs, return_tensors='pt', padding=True, truncation=True)
    return inputs, outputs

inputs, outputs = tokenize_data(training_data)

# Ajustar el tamaño del lote de salida al del lote de entrada
outputs['input_ids'] = outputs['input_ids'][:, :inputs['input_ids'].shape[1]]
outputs['attention_mask'] = outputs['attention_mask'][:, :inputs['input_ids'].shape[1]]

class FurnitureDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels = self.outputs['input_ids'][idx]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dataset = FurnitureDataset(inputs, outputs)

# Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    warmup_steps=10,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Entrenar el modelo
trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
