import pandas as pd
import numpy as np
import joblib
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    return response


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a question: ")
query_embedding = create_embedding([incoming_query])[0]


similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()

top_result = 10

max_index = similarities.argsort()[::-1][:top_result]


new_df = df.loc[max_index]

prompt = f"""
There is deep learning course. In that videos subtitle chunks containing  video title, video number, start time in seconds ,
end time in seconds, the text at taht time:
{new_df[["title", "start", "end", "text" ]].to_json(orient="records")}
----------------------------------------------------------------------------------------------------------------------------
"{incoming_query}"
user asked this questins related to the video chunks, you have to answer in human way (don't mention above format, its just for you)
where and how much content is taught in which video (at what timestamp) and guide the user to go to that particular video.
If user asked unrelated questions, tell him that you can only answer questions related to the course
"""

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)