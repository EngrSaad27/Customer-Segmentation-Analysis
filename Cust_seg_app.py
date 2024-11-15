import streamlit as st
from joblib import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Load the trained model
model = load('Hierarchical_model_pkl')

# Create a Streamlit App
st.title("Customer Segmentation App")

def preprocess_and_visualize(E_com_cust, model): 
    
    # Dropping Refund Orders
    E_com_cust = E_com_cust[E_com_cust['Price']>0]

    # Dropping Refund ORders
    E_com_cust = E_com_cust[E_com_cust['Quantity']>0]

    # Changing the date data type to datetime format
    E_com_cust['InvoiceDate'] = pd.to_datetime(E_com_cust['InvoiceDate'])

    # Changing customer id data type as per requirements
    E_com_cust['Customer ID']=E_com_cust['Customer ID'].astype(str)

    # Creating Monetary Attribute
    E_com_cust['Total Amount'] = E_com_cust['Quantity']*E_com_cust['Price']

    RFM_m = E_com_cust.groupby('Customer ID')['Total Amount'].sum()
    RFM_m = RFM_m.reset_index()
    RFM_m.columns = ['Customer ID', 'Monetary']

    RFM_f = E_com_cust.groupby('Customer ID')['Invoice'].count()
    RFM_f = RFM_f.reset_index()
    RFM_f.columns = ['Customer ID', 'Frequency']

    # Merging newly created attribute

    RFM_mf = pd.merge(RFM_m,RFM_f, on='Customer ID', how='inner')

    min_date = min(E_com_cust['InvoiceDate'])
    max_date = max(E_com_cust['InvoiceDate'])

    # Compute the difference between max date and transaction date

    E_com_cust['Diff_Days'] = max_date - E_com_cust['InvoiceDate']

    E_com_cust['Diff_Days'] = E_com_cust['Diff_Days'].dt.days

    RFM_r = E_com_cust.groupby('Customer ID')['Diff_Days'].min()
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

    # Fitting the Hierarchical Clustering model if DataFrame is valid
    model.fit(RFM_normalization)

    # Creating Hierarchical Cluster column
    RFM_normalization['Hierarchical Cluster'] = model.labels_

    # Title for visualization
    st.title("Customer Segmentation")

    # Visualizing Customer Segmentation on the basis of Hierarchical clusters
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

    sns.stripplot(data=RFM_normalization, x="Hierarchical Cluster", y="Recency",
        hue="Recency", legend=False, ax=ax1)
    ax1.set_ylabel('Recency',fontsize=20)
    ax1.set_xlabel('Cluster',fontsize=20)
    ax1.set_title('Recency-Based Clustering',fontsize=20)

    sns.stripplot(data=RFM_normalization, x="Hierarchical Cluster", y="Frequency",
        hue="Frequency", legend=False, ax=ax2)
    ax2.set_ylabel('Frequency',fontsize=20)
    ax2.set_xlabel('Cluster',fontsize=20)
    ax2.set_title('Frequency-Based Clustering',fontsize=20)

    sns.stripplot(data=RFM_normalization, x="Hierarchical Cluster", y="Monetary",
        hue="Monetary", legend=False, ax=ax3)
    ax3.set_ylabel('Monetary',fontsize=20)
    ax3.set_xlabel('Cluster',fontsize=20)
    ax3.set_title('Monetary-Based Clustering',fontsize=20)

    # Display the plot in Streamlit
    st.pyplot(fig)


# Inform the user about required columns before file upload
st.write("""### Note: The uploaded file must contain the following columns with same name as mentioned below:
- `Invoice`
- `Customer ID`
- `Price`
- `Quantity`
- `InvoiceDate`
""")

# Create a file upload button
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])


# Check if a file is uploaded
if uploaded_file is not None:
    # Display the file details
    st.write("File name:", uploaded_file.name)
    st.write("File size:", uploaded_file.size)

    # If the file is a CSV or Excel, you can display it as a DataFrame
    if uploaded_file.name.endswith(".csv"):
        E_com_cust = pd.read_csv(uploaded_file,encoding='ISO-8859-1')
        st.write("CSV Data:")
        if st.checkbox("Show Uploaded File"):
            st.dataframe(E_com_cust)
    elif uploaded_file.name.endswith(".xlsx"):
        E_com_cust = pd.read_excel(uploaded_file)
        st.write("Excel Data:")
        if st.checkbox("Show Uploaded File"):
            st.dataframe(E_com_cust)      
    else:
        st.write("File uploaded but not displayed (non-CSV/XLSX).")
        
     # Call the function
    preprocess_and_visualize(E_com_cust, model)
else:
    st.info("Please upload a file.")  

