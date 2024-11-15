import numpy as np # For linear Algebra
import pandas as pd # For data processing
import seaborn as sns # For creating plot in seaborn
import matplotlib.pyplot as plt # For creating plot in matplotlib
from sklearn.preprocessing import StandardScaler # For Standardizing Featurers
from sklearn.metrics import silhouette_samples, silhouette_score # For
# validating Clusters
from sklearn.cluster import KMeans # For Kmeans Clusters
from scipy.cluster.hierarchy import linkage # For Hierarchical Linkage
from scipy.cluster.hierarchy import dendrogram # For Plotting Dendrogram
from scipy.cluster.hierarchy import cut_tree # For Hierarchical Clusters
from sklearn.cluster import AgglomerativeClustering # For Agglomerative
# Clustering
from sklearn.cluster import DBSCAN # For DBSCAN Clusters
from sklearn.decomposition import PCA # For Dimentionality Reductions
import plotly.graph_objs as go # For Plotting Global Map
from plotly.offline import init_notebook_mode,iplot # For Plotting Global Map
import plotly.express as px # For creating plot in plotly
import pickle # For Importing trained model
import warnings # For Ignoring Warnings
warnings.filterwarnings("ignore")

# Loading the dataset
E_com_cust = pd.read_csv("D:\Saad's files\BIA DSAI\Course Lectures\Projects\BIA Capstone Project\Capstone Project\Customer Segmentation App\data.csv", encoding='ISO-8859-1')

# Dropping Refund Orders
E_com_cust = E_com_cust[E_com_cust['UnitPrice']>0]

# Dropping Refund ORders
E_com_cust = E_com_cust[E_com_cust['Quantity']>0]

# Changing the date data type to datetime format
E_com_cust['InvoiceDate'] = pd.to_datetime(E_com_cust['InvoiceDate'])

# Changing customer id data type as per requirements
E_com_cust['CustomerID']=E_com_cust['CustomerID'].astype(str)

# Creating Monetary Attribute
E_com_cust['Total Amount'] = E_com_cust['Quantity']*E_com_cust['UnitPrice']

RFM_m = E_com_cust.groupby('CustomerID')['Total Amount'].sum()
RFM_m = RFM_m.reset_index()
RFM_m.columns = ['Customer ID', 'Monetary']

RFM_f = E_com_cust.groupby('CustomerID')['InvoiceNo'].count()
RFM_f = RFM_f.reset_index()
RFM_f.columns = ['Customer ID', 'Frequency']

# Merging newly created attribute

RFM_mf = pd.merge(RFM_m,RFM_f, on='Customer ID', how='inner')

min_date = min(E_com_cust['InvoiceDate'])
max_date = max(E_com_cust['InvoiceDate'])

# Compute the difference between max date and transaction date

E_com_cust['Diff_Days'] = max_date - E_com_cust['InvoiceDate']

E_com_cust['Diff_Days'] = E_com_cust['Diff_Days'].dt.days

RFM_r = E_com_cust.groupby('CustomerID')['Diff_Days'].min()
RFM_r = RFM_r.reset_index()
RFM_r.columns = ['Customer ID', 'Recency']

# Merge all the newly created attribute to get the final RFM dataframe

RFM = pd.merge(RFM_mf,RFM_r, on='Customer ID', how='inner')
RFM.columns = ['Customer ID', 'Monetary', 'Frequency', 'Recency']

# Removing (statistical) outliers for Amount
Q1 = RFM.Monetary.quantile(0.25)
Q3 = RFM.Monetary.quantile(0.75)
IQR = Q3 - Q1
RFM_outliers = RFM[(RFM.Monetary>= Q1 - 1.5*IQR) & (
    RFM.Monetary <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3 - Q1
RFM_outliers = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (
    RFM.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM_outliers = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (
    RFM.Frequency <= Q3 + 1.5*IQR)]

condition = RFM_outliers['Monetary'] == 168472.5
row_position = RFM_outliers[condition].index[0]

RFM_outliers.drop(index=row_position, inplace= True)

# Creating DataFrame fo scaling
RFM_normalization = RFM_outliers[['Recency','Frequency','Monetary']]

# Scaling the features for model training
scaler = StandardScaler()
scaler.fit(RFM_normalization)
#Store it separately for clustering
RFM_normalized = scaler.transform(RFM_normalization)

RFM_normalized = pd.DataFrame(RFM_normalized,
 columns=RFM_normalization.columns).round(2)

pca = PCA().fit(RFM_normalized)

# Fitting PCA for model training
pca = PCA(n_components=2)
pca.fit(RFM_normalized)
RFM_PCA = pca.transform(RFM_normalized)

# Fitting Hierarchical Clustering model
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=4,linkage='complete')
cluster.fit(RFM_PCA)

# Finding Silhouetter Score for Validating Hierarchical Clustering
silhouette_score_Herarchical = silhouette_score(
  RFM_PCA,cluster.fit_predict(RFM_PCA))
silhouette_score_Herarchical

# Define the filename for pickle file
Filename = 'Hierarchical_model_pkl'

# Open file in write mode
with open('Hierarchical_model_pkl','wb') as file:
  # Save the Hierarchical model to the file
  pickle.dump(cluster, file)

# Close the file
file.close()

pickle.dump(cluster, open('Hierarchical_model_pkl','wb'))
