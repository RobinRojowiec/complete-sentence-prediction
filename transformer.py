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
train_df = pd.read_csv('data/generated_train.csv', header=None, names=["text", "labels"])
eval_df = pd.read_csv('data/generated_test.csv', header=None, names=["text", "labels"])

# define model
model = ClassificationModel('roberta', 'roberta-base', num_labels=4,
                            args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5,
                                  'fp16': False})

# training
model.train(train_df, "output/")


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
