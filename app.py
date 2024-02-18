import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

data = pd.read_csv('stock.csv')
stock_names = data.columns[1:] 

st.sidebar.title('Settings')
batch_size = st.sidebar.slider('Select the number of stocks in the batch', min_value=10, max_value=len(stock_names), value=30, step=1)
start_index = st.sidebar.slider('Select the starting index for the batch', min_value=1, max_value=len(stock_names)-batch_size+1, value=1, step=1)


cluster_options = list(range(3, 11))  # From 3 to 10
selected_clusters = st.sidebar.multiselect('Select the number of clusters for K-means', cluster_options, default=[6, 8, 10])

prices = data.iloc[:, start_index:start_index+batch_size]
current_stocks = stock_names[start_index-1:start_index+batch_size-1]  # Adjust for zero-indexing

returns = prices.pct_change().dropna()

pca = PCA(n_components=min(10, batch_size))  # Ensure we do not ask for more components than stocks
principal_components = pca.fit_transform(returns)

eigen_portfolios_figures = []
for i in range(len(pca.components_)):  # Use the actual number of components PCA found
    fig = px.bar(x=current_stocks, y=pca.components_[i], title=f'Eigen-portfolio {i+1}')
    eigen_portfolios_figures.append(fig)

annual_return = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

rv_df = pd.DataFrame({'Annual Return': annual_return, 'Volatility': volatility})

scaler = StandardScaler()
rv_scaled = scaler.fit_transform(rv_df)

cluster_figures = []
in_cluster_variances = {}
for K in selected_clusters:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(rv_scaled)
    rv_df[f'Cluster_{K}'] = kmeans.labels_

    fig = px.scatter(rv_df, x='Volatility', y='Annual Return', color=f'Cluster_{K}',
                     title=f'K-Means Clustering with K={K}')
    cluster_figures.append(fig)

    # Calculate in-cluster variance
    in_cluster_variance = sum(
        np.sum((rv_scaled[kmeans.labels_ == i] - kmeans.cluster_centers_[i]) ** 2)
        for i in range(K)
    )
    in_cluster_variances[K] = in_cluster_variance


st.sidebar.text_area('Current Stocks', ', '.join(current_stocks), height=100, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=True)

selected_option = option_menu(
    menu_title="Main Menu", 
    options=["Eigen Portfolios", "Clusters"],  
    icons=["kanban","diagram-3" ], 
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal", 
)

if selected_option == "Eigen Portfolios":
  st.title('Interactive Eigen-Portfolios')
  for i, fig in enumerate(eigen_portfolios_figures, 1):
      st.subheader(f'Eigen-portfolio {i}')
      fig.update_layout(title_text='')
      st.plotly_chart(fig)
elif selected_option == "Clusters":
  st.title('Interactive Clusters')
  for K, fig in zip(selected_clusters, cluster_figures):
      st.subheader(f'K-Means Clustering with K={K}')
      fig.update_layout(title_text='')
      st.plotly_chart(fig)
      st.write(f'In-cluster variance for K={K}: {in_cluster_variances[K].round(2)}')
