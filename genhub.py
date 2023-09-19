import requests
import csv
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSC0dOhvlVgYO9XcPv7qg68xGzSPphj7PrPiM0z9M3HxF5JS677dbgV44clx7NuVRG_SepAzMm4feBC/pub?output=csv"
response = requests.get(url)
if response.status_code == 200:
    csv_content = response.text
    with open("student_data.csv", "w", newline='') as csv_file:
        csv_file.write(csv_content)

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRHpYw-aN4tkA9lrG2hWe_i9PQLo1T3A8_ODqn-_co2AUrwUXSD0KJ5IqLcFyxtUdiKBk8qwbBCPOaC/pub?gid=1944232338&single=true&output=csv"
response = requests.get(url)
if response.status_code == 200:
    csv_content = response.text
    with open("prof_data.csv", "w", newline='') as csv_file:
        csv_file.write(csv_content)

df1 = pd.read_csv('/content/student_data.csv')
df2 = pd.read_csv('/content/prof_data.csv')
df_meta = df2['SKILLS'].str.split(',')
first_row_list = df_meta.at[0]        

data = []
embeddings = embedding_model.encode(df1['SKILLS'].tolist(), convert_to_tensor=True)
top_matches = {skill: [] for skill in first_row_list}

for idx, skill in enumerate(first_row_list):
    skill_embedding = embedding_model.encode(skill, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings, skill_embedding)
    threshold = 0.7
    matching_indices = np.where(scores > threshold)[0]
    matching_names = df1['Full Name'].iloc[matching_indices].tolist()
    top_matches[skill] = matching_names

for skill, names in top_matches.items():
    data.extend([(skill, name) for name in names])

data