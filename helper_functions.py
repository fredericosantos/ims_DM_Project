import pandas as pd
import numpy as np
import plotly.express as px
colors = (["#505050", "#d1675b"])


def check_inconsistency(df, col1, col2):
    """
    col1 < col2
    """
    inc = len(df[df[col1] < df[col2]])
    print(f"Number of inconsistencies between {col1} and {col2}: {inc}")
    print(f"Number of inconsistencies between {col1} and {col2}: {round((inc/df.shape[0]) * 100, 2)}%")
    
    
def plotly_dist(df, col1):
    fig = px.histogram(df, x=col1, marginal="violin", template="ggplot2", color_discrete_sequence=colors)
    fig.show()
    
    
    
    
def entity_embeddings(learner, cat_names, xs, valid_xs, test_xs, full_xs, all_xs):
    for xs_emb in [xs, valid_xs, test_xs, full_xs, all_xs]:
        for n, cat in enumerate(cat_names):
            cat_embedding = to_np(learner.model.embeds[n].weight)
            for i in range(cat_embedding.shape[1]):
                xs_emb[cat+"_emb"+str(i)] = xs_emb[cat].replace({j: cat_embedding[j, i] for j in range(cat_embedding.shape[0])})
            xs_emb.drop(columns=cat, inplace=True)
            
            
            
            
            
def save_embeddings(learner, cat_names, name):
    for i, cat in enumerate(cat_names):
        w = to_np(learner.model.embeds[i].weight)
        np.save(f"embeddings/{name}_{cat}", w)