import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as ss
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import plotly.io as pio

colors = (["#505050", "#d1675b"])

def save_graph(figure, name: "plots/name.plotly"):
    pio.write_json(figure, f"plots/{name}.plotly")

def check_inconsistency(df, col1, col2):
    """
    col1 < col2
    """
    inc = len(df[df[col1] < df[col2]])
    print(f"Number of inconsistencies between {col1} and {col2}: {inc}")
    print(f"Number of inconsistencies between {col1} and {col2}: {round((inc/df.shape[0]) * 100, 2)}%")
    

# Cramer's V - implemetation was taken from link - https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt. 
    
    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable
    
    sst = get_ss(df)  # get total sum of squares
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, affinity=dist, linkage=link_method)
        hclabels = cluster.fit_predict(df) #get cluster labels
        df_concat = pd.concat((df, pd.Series(hclabels, name='labels')), axis=1)  # concat df with labels
        ssw_labels = df_concat.groupby(by='labels').apply(get_ss)  # compute ssw for each cluster labels
        ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
        r2.append(ssb / sst)  # save the R2 of the given cluster solution
        
    return np.array(r2)
    
    
    
def plotly_dist(df, col1):
    fig = px.histogram(df, x=col1, marginal="violin", template="ggplot2", color_discrete_sequence=colors)
    fig.show()
    
def entity_embeddings(learner, cat_names, xs):
    for xs_emb in [xs]:
        for n, cat in enumerate(cat_names):
            cat_embedding = to_np(learner.model.embeds[n].weight)
            for i in range(cat_embedding.shape[1]):
                xs_emb[cat+"_emb"+str(i)] = xs_emb[cat].replace({j: cat_embedding[j, i] for j in range(cat_embedding.shape[0])})
            xs_emb.drop(columns=cat, inplace=True)
            
def save_embeddings(learner, cat_names, name):
    for i, cat in enumerate(cat_names):
        w = to_np(learner.model.embeds[i].weight)
        np.save(f"embeddings/{name}_{cat}", w)
        
def load_embeddings(learner, cat, cat_names, c: "target variable used to train"):
    learner.model.embeds[cat_names.index(cat)].weight = torch.nn.Parameter(torch.from_numpy(np.load(f"embeddings/{c}_{cat}.npy")))

def freeze_embedding(learner, cat, cat_names):
    learner.model.embeds[cat_names.index(cat)].weight.requires_grad = False
        
        
def get_non_census_features(dataframe) -> list:
    features_census = get_census_features()
    features_non_census = dataframe.columns.tolist()
    for f in features_census:
        features_non_census.remove(f)
    return features_non_census
        
def get_census_features():
    return [
    "POP901",
    "POP902",
    "POP903",
    "POP90C1",
    "POP90C2",
    "POP90C3",
    "POP90C4",
    "POP90C5",
    "ETH1",
    "ETH2",
    "ETH3",
    "ETH4",
    "ETH5",
    "ETH6",
    "ETH7",
    "ETH8",
    "ETH9",
    "ETH10",
    "ETH11",
    "ETH12",
    "ETH13",
    "ETH14",
    "ETH15",
    "ETH16",
    "AGE901",
    "AGE902",
    "AGE903",
    "AGE904",
    "AGE905",
    "AGE906",
    "AGE907",
    "CHIL1",
    "CHIL2",
    "CHIL3",
    "AGEC1",
    "AGEC2",
    "AGEC3",
    "AGEC4",
    "AGEC5",
    "AGEC6",
    "AGEC7",
    "CHILC1",
    "CHILC2",
    "CHILC3",
    "CHILC4",
    "CHILC5",
    "HHAGE1",
    "HHAGE2",
    "HHAGE3",
    "HHN1",
    "HHN2",
    "HHN3",
    "HHN4",
    "HHN5",
    "HHN6",
    "MARR1",
    "MARR2",
    "MARR3",
    "MARR4",
    "HHP1",
    "HHP2",
    "DW1",
    "DW2",
    "DW3",
    "DW4",
    "DW5",
    "DW6",
    "DW7",
    "DW8",
    "DW9",
    "HV1",
    "HV2",
    "HV3",
    "HV4",
    "HU1",
    "HU2",
    "HU3",
    "HU4",
    "HU5",
    "HHD1",
    "HHD2",
    "HHD3",
    "HHD4",
    "HHD5",
    "HHD6",
    "HHD7",
    "HHD8",
    "HHD9",
    "HHD10",
    "HHD11",
    "HHD12",
    "ETHC1",
    "ETHC2",
    "ETHC3",
    "ETHC4",
    "ETHC5",
    "ETHC6",
    "HVP1",
    "HVP2",
    "HVP3",
    "HVP4",
    "HVP5",
    "HVP6",
    "HUR1",
    "HUR2",
    "RHP1",
    "RHP2",
    "RHP3",
    "RHP4",
    "HUPA1",
    "HUPA2",
    "HUPA3",
    "HUPA4",
    "HUPA5",
    "HUPA6",
    "HUPA7",
    "RP1",
    "RP2",
    "RP3",
    "RP4",
    "MSA",
    "ADI",
    "DMA",
    "IC1",
    "IC2",
    "IC3",
    "IC4",
    "IC5",
    "IC6",
    "IC7",
    "IC8",
    "IC9",
    "IC10",
    "IC11",
    "IC12",
    "IC13",
    "IC14",
    "IC15",
    "IC16",
    "IC17",
    "IC18",
    "IC19",
    "IC20",
    "IC21",
    "IC22",
    "IC23",
    "HHAS1",
    "HHAS2",
    "HHAS3",
    "HHAS4",
    "MC1",
    "MC2",
    "MC3",
    "TPE1",
    "TPE2",
    "TPE3",
    "TPE4",
    "TPE5",
    "TPE6",
    "TPE7",
    "TPE8",
    "TPE9",
    "PEC1",
    "PEC2",
    "TPE10",
    "TPE11",
    "TPE12",
    "TPE13",
    "LFC1",
    "LFC2",
    "LFC3",
    "LFC4",
    "LFC5",
    "LFC6",
    "LFC7",
    "LFC8",
    "LFC9",
    "LFC10",
    "OCC1",
    "OCC2",
    "OCC3",
    "OCC4",
    "OCC5",
    "OCC6",
    "OCC7",
    "OCC8",
    "OCC9",
    "OCC10",
    "OCC11",
    "OCC12",
    "OCC13",
    "EIC1",
    "EIC2",
    "EIC3",
    "EIC4",
    "EIC5",
    "EIC6",
    "EIC7",
    "EIC8",
    "EIC9",
    "EIC10",
    "EIC11",
    "EIC12",
    "EIC13",
    "EIC14",
    "EIC15",
    "EIC16",
    "OEDC1",
    "OEDC2",
    "OEDC3",
    "OEDC4",
    "OEDC5",
    "OEDC6",
    "OEDC7",
    "EC1",
    "EC2",
    "EC3",
    "EC4",
    "EC5",
    "EC6",
    "EC7",
    "EC8",
    "SEC1",
    "SEC2",
    "SEC3",
    "SEC4",
    "SEC5",
    "AFC1",
    "AFC2",
    "AFC3",
    "AFC4",
    "AFC5",
    "AFC6",
    "VC1",
    "VC2",
    "VC3",
    "VC4",
    "ANC1",
    "ANC2",
    "ANC3",
    "ANC4",
    "ANC5",
    "ANC6",
    "ANC7",
    "ANC8",
    "ANC9",
    "ANC10",
    "ANC11",
    "ANC12",
    "ANC13",
    "ANC14",
    "ANC15",
    "POBC1",
    "POBC2",
    "LSC1",
    "LSC2",
    "LSC3",
    "LSC4",
    "VOC1",
    "VOC2",
    "VOC3",
    "HC1",
    "HC2",
    "HC3",
    "HC4",
    "HC5",
    "HC6",
    "HC7",
    "HC8",
    "HC9",
    "HC10",
    "HC11",
    "HC12",
    "HC13",
    "HC14",
    "HC15",
    "HC16",
    "HC17",
    "HC18",
    "HC19",
    "HC20",
    "HC21",
    "MHUC1",
    "MHUC2",
    "AC1",
    "AC2",
]