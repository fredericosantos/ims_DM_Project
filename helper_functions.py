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