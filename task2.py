

pip install transformers torch accelerate

import pandas as pd

csv_path = "/content/3A2M_EXTENDED.csv"

df = pd.read_csv(
    csv_path,
    engine='python',
    on_bad_lines='skip',
    quotechar='"'
)

print("Dataset shape:", df.shape)
df.head()

def format_recipe(row):
    try:
        ingredients = ", ".join(eval(row['Extended_NER']))
        directions = " ".join(eval(row['directions']))
    except:
        ingredients = ""
        directions = ""
    return f"Title: {row['title'].strip()}\nIngredients: {ingredients}\nInstructions: {directions}\n"

texts = df.apply(format_recipe, axis=1).tolist()

print(texts[0])

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

max_length = 512
encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

import torch
from torch.utils.data import Dataset

class RecipeDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

dataset = RecipeDataset(encodings)

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


subset_size = 1000
subset_dataset = torch.utils.data.Subset(dataset, range(subset_size))

train_loader = DataLoader(subset_dataset, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model.save_pretrained("recipe_gpt2_demo")
tokenizer.save_pretrained("recipe_gpt2_demo")

model.eval()
sample_prompts = [
    "Title: Spicy Chicken Curry\nIngredients: chicken, onions, garlic, tomatoes\nInstructions:",
    "Title: Vegan Chocolate Cake\nIngredients: flour, cocoa powder, sugar, almond milk\nInstructions:"
]

for prompt in sample_prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    recipe = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nGenerated Recipe:\n", recipe)

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from math import exp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

model = GPT2LMHeadModel.from_pretrained("recipe_gpt2_demo")
tokenizer = GPT2Tokenizer.from_pretrained("recipe_gpt2_demo")
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

eval_subset_size = 100
eval_dataset = torch.utils.data.Subset(dataset, range(eval_subset_size))
eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

perplexity = exp(total_loss / total_tokens)
print(f"Perplexity on eval subset: {perplexity:.2f}")

smooth = SmoothingFunction().method1
sample_prompts = [
    "Title: Spicy Chicken Curry\nIngredients: chicken, onions, garlic, tomatoes\nInstructions:",
    "Title: Vegan Chocolate Cake\nIngredients: flour, cocoa powder, sugar, almond milk\nInstructions:"
]

for prompt in sample_prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    generated_recipe = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    reference_recipe = prompt
    bleu = sentence_bleu([reference_recipe.split()], generated_recipe.split(), smoothing_function=smooth)
    print("\nPrompt:\n", prompt)
    print("Generated Recipe:\n", generated_recipe)
    print(f"BLEU score: {bleu:.4f}")

prompt = "Title: Chocolate Chip Cookies\nIngredients: flour, sugar, butter, eggs, chocolate chips\nInstructions:"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

output = model.generate(
    input_ids,
    max_length=200,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)

generated_recipe = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Recipe:\n", generated_recipe)
