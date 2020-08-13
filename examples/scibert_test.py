import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Example
text = "Mouse thymus was used as a source of glucocorticoid receptor from normal CS lymphocytes."

# Load model
tokenizer = AutoTokenizer.from_pretrained("fran-martinez/scibert_scivocab_cased_ner_jnlpba")
model = AutoModelForTokenClassification.from_pretrained("fran-martinez/scibert_scivocab_cased_ner_jnlpba")

#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Get input for BERT
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

# Predict
with torch.no_grad():
  outputs = model(input_ids)

# From the output let's take the first element of the tuple.
# Then, let's get rid of [CLS] and [SEP] tokens (first and last)
predictions = outputs[0].argmax(axis=-1)[0][1:-1]

# Map label class indexes to string labels.
for token, pred in zip(tokenizer.tokenize(text), predictions):
  print(token, '->', model.config.id2label[pred.numpy().item()])