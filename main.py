import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Config
random.seed(1)
plt.rcParams['figure.figsize'] = [5, 5]

dir_path = "eval_answers"
dataset_name = "1200_prompts_dataset.tsv"
eval_files = [
    "GPT2_answers.tsv",
    "Bloom7B_answers.tsv",
    "davinci_answers.tsv",
    "text-davinci-002_answers.tsv",
    "text-davinci-003_answers.tsv",
    "code-davinci-002_answers.tsv",
    "gpt4_answers.tsv"
]

def get_model_evals(file_names):
    return [pd.read_csv(f"{dir_path}/{eval_name}", sep="\t", index_col=0) for eval_name in file_names]

def get_model_names(file_names):
  return [file_name.split("_")[0] for file_name in file_names]

def get_dataset(file_name):
  with open(f"{dir_path}/{file_name}", "r") as infile:
    dataset = pd.read_csv(infile, sep="\t")
    try: # Drop unnamed columns
      unnamed_cols = list(dataset.filter(regex="^Unnamed"))
      return dataset[dataset.columns.drop(unnamed_cols)]
    except Exception as e:
      return dataset

# Filter dataset by predetermined clusterwise sampling
def filter_dataset(dataset, num_choices=4, random_seed=1):
  random.seed(random_seed)
  index = [random.randint(0,3) + i*num_choices for i in range(int(len(dataset)/num_choices))]
  return dataset.iloc[index]

def normalize_complexity(dataset):
  new_dataset = dataset.copy()
  new_dataset["dep_per_token"] = new_dataset["dependency_score"] / new_dataset["num_tokens"]
  new_dataset["con_per_token"] = new_dataset["constituency_score"] / new_dataset["num_tokens"]
  return new_dataset

def swap_columns(dataset, col1, col2):
  cols = list(dataset.columns)
  colA, colB = cols.index(col1), cols.index(col2)
  cols[colA], cols[colB] = cols[colB], cols[colA]
  return dataset[cols]

def trim_dataset(dataset, head=0, tail=0):
  return dataset.iloc[head:len(dataset)-tail]

def get_model_accuracy(dataset, model):
    return len(dataset[dataset[model]]) / len(dataset)

def append_evals(dataset, evals, model_names):
  new_dataset = dataset.copy()
  for name, df in zip(model_names, evals):
    new_dataset[name] = df["gold"] == df["pred"]
  return new_dataset

def print_model_stats(dataset, model_names):
  for model in model_names:
    accuracy = get_model_accuracy(dataset, model)
    print(f"{model} accuracy: {accuracy:.2f}\n")

class LogisticAnalysis(object):

  def __init__(self, dataset, model_names):
    self.data = dataset
    self.model_names = model_names

  def set_features(self, features):
    self.features = features.to_numpy()
    return self
    
  def fit_model(self, model):
    correct = np.array(self.data[model])
    return LogisticRegression(solver='liblinear', random_state=0).fit(self.features, correct)

  def fit_models(self):
    self.models = {model:self.fit_model(model) for model in self.model_names}
    return self

  def get_model_accuracy(self, model):
    return len(self.data[self.data[model]]) / len(self.data)

  def get_performance_matrix(self, models):
    perf = [models[model].predict_proba(self.features)[:,1] for model in models]
    return np.column_stack(perf)

  def plot_performance(self):
    X = [self.get_model_accuracy(model) for model in self.models]
    Y = self.get_performance_matrix(self.models)
    for i in range(len(self.data)):
      plt.plot(X, Y[i], alpha=.1)
    plt.show()
    return self
    

def main():
  evals = get_model_evals(eval_files)
  model_names = get_model_names(eval_files)
  dataset = get_dataset(dataset_name)
  dataset = trim_dataset(dataset, tail=1)
  dataset = filter_dataset(dataset)
  dataset = normalize_complexity(dataset)
  dataset = append_evals(dataset, evals, model_names)
  analysis = LogisticAnalysis(dataset, model_names)
  analysis.set_features(dataset[[
    "num_tokens",
    "dep_per_token",
    "con_per_token"
  ]]).fit_models()
  analysis.plot_performance()

  analysis.set_features(dataset[[
    "dep_per_token"
  ]]).fit_models()
  analysis.plot_performance()

if __name__ == "__main__":
  main()
