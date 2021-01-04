import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as ss
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import plotly.io as pio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import seaborn as sns

def rf_feat_importance(m, df):
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)


def plot_fi(fi, ax=None):
    return fi.plot(
        "cols", "imp", "barh", figsize=(10, len(fi.cols) // 3), legend=False, ax=ax
    )

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
    

    
def cluster_profiles(df, label_columns, figsize, compar_titles=None):
    """
    Pass df with labels columns of one or multiple clustering labels. 
    Then specify this label columns to perform the cluster profile according to them.
    """
    if compar_titles == None:
        compar_titles = [""]*len(label_columns)
        
    fig, axes = plt.subplots(nrows=len(label_columns), ncols=2, figsize=figsize, squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compar_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i!=label]
        dfax = df.drop(drop_cols, axis=1)
        
        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
        counts.columns = [label, "counts"]
        
        # Setting Data
        pd.plotting.parallel_coordinates(centroids, label, color=sns.color_palette(), ax=ax[0])
        sns.barplot(x=label, y="counts", data=counts, ax=ax[1])

        #Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
#         ax[0].annotate(text=titl, xy=(0.95,1.1), xycoords='axes fraction', fontsize=13, fontweight = 'heavy') 
        ax[0].legend(handles, cluster_labels) # Adaptable to number of clusters
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title(f"{titl} - {len(handles)} Clusters", fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-90, fontsize=9)
        ax[0].set_ylim(-1.0, 1.0)
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)
    
    plt.subplots_adjust(hspace=0.5, top=1.0)
    plt.show()
    
def get_ss_variables(df):
    """Get the SS for each variable
    """
    ss_vars = df.var() * (df.count() - 1)
    return ss_vars

def r2_variables(df, labels):
    """Get the R² for each variable
    """
    sst_vars = get_ss_variables(df)
    ssw_vars = np.sum(df.groupby(labels).apply(get_ss_variables))
    return 1 - ssw_vars/sst_vars    
    
    
def plot_dendrogram(df_centroids, threshold, linkage="ward"):
    hclust = AgglomerativeClustering(
        linkage=linkage, 
        affinity='euclidean', 
        distance_threshold=0, 
        n_clusters=None
    )
    hclust_labels = hclust.fit_predict(df_centroids)
    
    # create the counts of samples under each node (number of points being merged)
    counts = np.zeros(hclust.children_.shape[0])
    n_samples = len(hclust.labels_)

    # hclust.children_ contains the observation ids that are being merged together
    # At the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
    for i, merge in enumerate(hclust.children_):
        # track the number of observations in the current cluster being formed
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                # If this is True, then we are merging an observation
                current_count += 1  # leaf node
            else:
                # Otherwise, we are merging a previously formed cluster
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # the hclust.children_ is used to indicate the two points/clusters being merged (dendrogram's u-joins)
    # the hclust.distances_ indicates the distance between the two points/clusters (height of the u-joins)
    # the counts indicate the number of points being merged (dendrogram's x-axis)
    linkage_matrix = np.column_stack(
        [hclust.children_, hclust.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=(11,5))
    # The Dendrogram parameters need to be tuned
    y_threshold = threshold
    dendrogram(linkage_matrix, truncate_mode='level', p=5, color_threshold=y_threshold, above_threshold_color='k')
    plt.hlines(y_threshold, 0, 1000, colors="r", linestyles="dashed")
    plt.title(f'Hierarchical Clustering - {linkage.title()}\'s Dendrogram', fontsize=16)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel(f'Euclidean Distance', fontsize=13)
    plt.show()
    
    
    
    
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