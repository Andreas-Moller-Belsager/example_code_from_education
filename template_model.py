import pandas as pd
#import torch
#import transformers
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate




df = pd.read_csv('INTROVERT_EXTROVERT_binary_mbti.csv')

labels_mapper = dict()
for i,el in enumerate(set(df["labels"])):
    labels_mapper[el]=i

df["labels"] = df["labels"].map(labels_mapper)

the_dataset = Dataset.from_pandas(df)

the_dataset = the_dataset.class_encode_column("uid")



train_testvalid = the_dataset.train_test_split(test=0.1,stratify_by_column = "uid",seed=42)
test_valid = train_testvalid['test'].train_test_split(test=0.5,stratify_by_column = "uid",seed=42)

train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


the_model = 'Twitter/twhin-bert-large'

tokenizer = AutoTokenizer.from_pretrained(the_model)

def preprocessor(the_instance):
    return tokenizer(the_instance['text'], truncation=True)

tokenized_twisty = train_test_valid_dataset.map(preprocessor,batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(the_model, num_labels=16)

training_args = TrainingArguments(
    output_dir='./results_IE',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=0.1,
    save_steps=0.1,                       #such that we save 10 models in total
    save_total_limit=1,                   #such that we at max save one other model than the best one
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_twisty["train"],
    eval_dataset=tokenized_twisty["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


predictions = trainer.predict(tokenized_twisty["test"])

print(predictions.predictions.shape)

preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("accuracy")

print("final scores: ", metric.compute(predictions=preds, references=predictions.label_ids))
