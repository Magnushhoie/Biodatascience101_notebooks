import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

def dataset_x_y(df):
    dataset_y = df.iloc[:, 0]
    dataset_x = df.iloc[:, 1:]
    return(dataset_x, dataset_y)

def train_valid_split(df, valid_percent = 0.25):
    split_point = int(len(df) * (1-valid_percent))
    train_df, valid_df = df[0: split_point], df[split_point: ]
    return(train_df, valid_df)

def generate_train_valid_sets(df, valid_percent = 0.25):
    train_df, valid_df = train_valid_split(df, valid_percent)
    
    train_x, train_y = dataset_x_y(train_df)
    valid_x, valid_y = dataset_x_y(valid_df)
    
    return(train_x, train_y, valid_x, valid_y)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import scipy as sc
from sklearn.metrics import precision_score
from scipy.stats import spearmanr


def test_performance(m, test_x, test_y, verbose = 0):
    #Compute values, try to extract probabilities if possible
    try:
        y_score = m.predict_proba(test_x)[:, 1]
    except:
        y_score = m.predict(test_x)
    
    y_pred = np.where(y_score > 0.5, 1, 0)
    y_true = test_y
    y_true_binary = np.where(y_true > 0.5, 1, 0)
    
    if verbose == 1: print("y_score", y_score, "y_pred", y_pred)

    #Performance scores
    confusion_m =  confusion_matrix(y_true_binary, y_pred)
    try:
        tn, fp, fn, tp = confusion_m.ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, confusion_m[0]
    try: 
        neg_precision = tn / (tn + fn)
    except:
        neg_precision = 0
            
    #Note: Classifying AUC on y_pred and y_score has a large difference in AUC calculation
    # As the threshold becomes 0.5
    #fpr, tpr, thresholds = metrics.roc_curvey_true, y_pred, pos_label=1)
    #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    
    #auc = round(metrics.auc(fpr, tpr), 3)
    #fpr, tpr, thresholds = metrics.roc_curve(y_true_binary, y_pred, pos_label=1)
    mcc = round(matthews_corrcoef(y_true_binary, y_pred), 10) 
    precision = precision_score(y_true_binary, y_pred, average='binary')
    pearson = sc.stats.pearsonr(y_true, y_score)
    pearson = (np.round(pearson[0], 3), pearson[1])
    
    #print("AUC", auc)
    print("MCC", mcc)
    print("Pearson", pearson)
    print("Correct positives:", tp, "/", tp+fn, "positives")
    #if len(tpr) <= 2:
    #    print("TPR:", np.round(tpr, 2))
    #else:
    #    print("TPR", np.round(tpr, 2)[1])            
    print("False positives:", fp, "/", fp+tn, "negatives")
    print("Positive Precision", round(precision, 3))
    print("Negative precision", round(neg_precision, 3))
    print("Confusion matrix:")
    print("[[tn, fp]]")
    print(confusion_m)    
    print("[[fn tp]]")
    
    
def plot_boundary(model, valid_x, valid_y, figsize = (5,5)):
    from mlxtend.plotting import plot_decision_regions
    fig = plt.figure(figsize=figsize)
    X = np.array(valid_x)
    y = np.array(valid_y, dtype=int)

    #plot
    #ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=model, legend=2,
                               scatter_kwargs = {"s":100, "alpha":0.8},
                               zoom_factor = 7)

    plt.title("Decision boundary")

def generate_machine_learning_model(dataset, feature_columns, y_true_column = 0, train_valid_split = 0.2, model = "RandomForestClassifier"):
    
    dataset_at_features = dataset[feature_columns]
    train_x, train_y, valid_x, valid_y = generate_train_valid_sets(dataset_at_features, train_valid_split)
    print("Train x/y, valid x/y:", train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
    
    model = RandomForestClassifier(max_features = "sqrt",
                                 min_samples_split = 5, max_depth = 20)
    
    model.fit(train_x, train_y)
    test_performance(model, valid_x, valid_y)
    

from sklearn.decomposition import PCA
def PCA_analysis(df, n = 2):
    df = pd.DataFrame(df)
    
    #Try changing the number of components to include from 2 to other values
    n_components = 2

    #Define the number of components we want
    pca = PCA(n_components=n_components)

    #Extract principal component 1 and 2 values for each of our samples
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents,
                              columns = ["PC1", "PC2"])

    #Explained variance
    pca_explained_variance = round(sum(pca.explained_variance_ratio_[0:n_components])*100, 2)
    print("The top", n_components, "principal components explain", pca_explained_variance, " % of the variance")
    
    return(principalDf)
    
def PCA_analysis_plot(df_orig, labels = [], n = 2, figsize = (7,7), font_scale = 2, return_df = False):
    # Adding group label for each sample
    df = PCA_analysis(df_orig, n)
    print(df.shape)
    
    if len(labels) == len(df_orig): df["group"] = labels
    else: df["group"] = 0

    x = df.iloc[:, 0] #First column, first PC
    y = df.iloc[:, 1] #Second column, second PC

    #Set coloring for our groups by the order of our columns
    #groups = ["NI"]*4 + ["D3"]*4 + ["ECM"]*4
    #colors = ['#0D76BF', '#00cc96', "#EF553B"]
    

    # Initialize graph
    sns.set(font_scale = font_scale)
    fig, ax = plt.subplots(figsize = figsize)

    # Plot graph
    print("Plotting ...")
    sns.scatterplot(x = x, y = y, data = df, hue = "group", markers="+")

    # Title and axis labels
    plt.title("PCA-analysis:\n Scatterplot of principal components for all samples\n")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")

    #Non-overlapping text using adjustText for matplotlib
    #https://github.com/Phlya/adjustText
    #from adjustText import adjust_text

    #texts = [plt.text(x[i],
    #                  y[i],
    #                  df["group"][i],
    #                  ha='center',
    #                  va='center')
    #         for i in range(len(x))]
    #adjust_text(texts)
    
    if return_df == True:
        return(df)

def plot_scatter_labels(df, title, labels = [], figsize = (7,7), font_scale = 1.5, return_df = False):
    df = pd.DataFrame(df)
    
    if len(labels) == len(df):
        df["group"] = labels
    else:
        print("Labels not the same length as df:", len(labels), len(df))
        df["group"] = 0

    #print("Plotting first and second column only ...")
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    # Initialize graph
    sns.set(font_scale = font_scale)
    fig, ax = plt.subplots(figsize = figsize)

    # Plot graph
    print("Plotting", title)
    sns.scatterplot(x = x, y = y, data = df, hue = "group", markers="+")

    # Title and axis labels
    plt.title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if return_df == True:
        return(df)
  
# Read more: https://scikit-learn.org/stable/modules/manifold.html#manifold
def dimensionality_analysis_plot(df_orig, labels = [], method = "PCA", n_components = 2, n_neighbors = 10, figsize = (7,7), font_scale = 2, return_df = False):
    # Define valid methods
    valid_methods = ["PCA", "T-SNE", "Isomap", "MDS", "SE", "LLE", "Modified LLE", ]
    if method not in valid_methods:
        print(method, "not a valid method. Available methods:\n", valid_methods)
        
    # Load libraries
    from sklearn import manifold
    from functools import partial
    LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='auto')
    
        
    # Extract data from method
    if method == "PCA":
        Y = PCA_analysis(df_orig, n_components)
    elif method == "T-SNE":
        model = manifold.TSNE(n_components=n_components, init='pca',
                                 random_state=0)
        Y = model.fit_transform(df_orig)
    elif method == "LLE":
        model = LLE(method='standard')
        Y = model.fit_transform(df_orig)
    elif method == "Modified LLE":
        model = LLE(method='modified')
        Y = model.fit_transform(df_orig)
    elif method == "Isomap":
        model = manifold.Isomap(n_neighbors, n_components)
        Y = model.fit_transform(df_orig)
    elif method == "MDS":
        model = manifold.MDS(n_components, max_iter=100, n_init=1)
        Y = model.fit_transform(df_orig)
    elif method == "SE":
        model = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors)
        Y = model.fit_transform(df_orig)
        
    #Plot
    plot_scatter_labels(Y, method, labels)
    
    
def tsne_analysis_plot(df_orig, labels = [], n = 2, figsize = (7,7), font_scale = 2, return_df = False):
    # Adding group label for each sample
    
    
    df = PCA_analysis(df_orig, n)
    print(df.shape)
    
    if len(labels) == len(df_orig): df["group"] = labels
    else: df["group"] = 0

    x = df.iloc[:, 0] #First column, first PC
    y = df.iloc[:, 1] #Second column, second PC

    # Initialize graph
    sns.set(font_scale = font_scale)
    fig, ax = plt.subplots(figsize = figsize)

    # Plot graph
    print("Plotting ...")
    sns.scatterplot(x = x, y = y, data = df, hue = "group", markers="+")

    # Title and axis labels
    plt.title("T-SNE-analysis:\n Scatterplot of 2 T-sne components for all samples\n")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if return_df == True:
        return(df)