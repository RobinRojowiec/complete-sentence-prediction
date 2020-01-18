"""

IDE: PyCharm
Project: complete-sentence-prediction
Author: Robin
Filename: transformer.py
Date: 16.01.2020

"""
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score

# load data
train_df = pd.read_csv('data/generated_train.csv')

print(train_df.head())

eval_df = pd.read_csv('data/generated_test.csv')

# define model
# 'roberta', 'roberta-base'
model_architecture = "bert-base-cased"
model_type = "bert"
model = ClassificationModel(model_name=model_architecture, model_type=model_type, num_labels=2,
                            args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 1,
                                  'fp16': False, 'use_multiprocessing': False})

# training
model.train_model(train_df)


# evaluate
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_multiclass, acc=accuracy_score)
print(result)

# show prediction
predictions, raw_outputs = model.predict(['Who was'])
print(predictions)

predictions, raw_outputs = model.predict(['Who was that'])
print(predictions)
