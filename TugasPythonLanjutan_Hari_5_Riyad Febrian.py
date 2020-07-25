# Importing required Library

import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from statsmodels.graphics.gofplots import qqplot

color_pallete = ['#34495e', '#c0392b', '#f39c12']  # Default color pallete


def pokemon_scrape(index):
    """
    This function will scraping pokemon data on pokemondb.net

    :param index: the limit pokedex that you want to scrape
    :return: dataframe of pokemon
    """
    response = requests.get('https://pokemondb.net/pokedex/all')
    df = pd.read_html(response.content, flavor='bs4')
    pokedex = df[0]['#'] <= index
    return df[0][pokedex]


def model_kmeans(data, k):
    """
    This function will create a simple KMeans model

    :param data: the data for cluster analysis with KMean
    :param k: how much the data want to be clustered
    :return: result fitting model KMeans

    """
    return KMeans(n_clusters=k).fit(data)


def display_qqplot(data):
    """
    QQPlot - Quantile-Quantile plot is used to test the normality of distribution
    with qualitative approach (Visualization)

     :param data: the data for cluster analysis with KMean
     :return: Plotly Figures
    """
    qqplot_data = qqplot(data, line='s').gca().lines

    fig = go.Figure()

    scatter_trace = go.Scatter(
        x=qqplot_data[0].get_xdata(),
        y=qqplot_data[0].get_ydata(),
        mode='markers',
        marker=go.scatter.Marker(color='rgba(255, 127, 14, 0.7)'),
        name='Data'
    )

    regression_trace = go.Scatter(
        x=qqplot_data[1].get_xdata(),
        y=qqplot_data[1].get_ydata(),
        mode='lines',
        line=go.scatter.Line(color='#636efa'),
        name='Regression'
    )

    layout = go.Layout(
        width=800,
        height=800,
        title=f'Quantile-Quantile Plot Data {data.name}',
        xaxis_title='Theoritical Quantities',
        xaxis_zeroline=False,
        yaxis_title='Sample Quantities'
    )

    fig = go.Figure(data=[scatter_trace, regression_trace], layout=layout)
    return fig

    pass


def display_hist(data, title):
    """
    This function will plotting distribution data with histogram

    :param data: the data for visualizing
    :param title: the title Figure and x-Axis
    :return: Plotly Figures
    """

    fig = go.Figure(go.Histogram(x=data))

    fig.update_layout(title='Distribution {title} Data',
                      xaxis_title=title,
                      yaxis_title='Frequency')
    return fig


def display_boxplot(data):
    """
    This function will generate Box plot for outlier visualization

    :param data: One column observed data
    :return: Plotly Figures
    """
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name=data.name,
        boxpoints="outliers",
        marker_color="rgba(219, 64, 82, 0.7)",
        line_color="rgb(8, 81, 156)"
    ))

    fig.update_layout(width=800,
                      height=800,
                      title_text=f'{data.name} Outlier Visualization')
    return fig


def elbow_method(data):
    """
    This function will compute elbow method and generate elbow visualization

    :param data: 2 columns dataframe for cluster analysis
    :return: Plotly Figures
    """
    distortions = []
    K = range(1, 10)
    for k in K:
        elbow_kmean = model_kmeans(data, k)
        distortions.append(elbow_kmean.inertia_)

    elbow = pd.DataFrame({'k': K,
                          'inertia': distortions})

    fig = go.Figure(data=go.Scatter(x=elbow['k'], y=elbow['inertia']))
    fig.update_layout(title='Elbows Methods for finding best K values in KMeans',
                      xaxis_title='K',
                      yaxis_title='Inertia')

    return fig


def silhouette_method(data):
    """
        Silhouette method is an addition secondary methods when Elbows get ambigous to interpret the k value

    :param data: 2 columns dataframe
    :return: Plotly Figures
    """

    sil = []
    kmax = 10

    for k in range(2, kmax + 1):
        kmeans = model_kmeans(data, k)
        labels = kmeans.labels_
        sil.append(silhouette_score(att_def, labels, metric='euclidean'))

    k_cluster = pd.DataFrame({'k': range(2, kmax + 1),
                              'silhouette score': sil})

    fig = go.Figure(data=go.Scatter(x=k_cluster['k'], y=k_cluster['silhouette score']))
    fig.update_layout(title='Silhouette Methods for finding best K values in KMeans',
                      xaxis_title='K',
                      yaxis_title='Silhouette Score')
    return fig


def scatter_cluster(data):
    """
    This function create scatter plot trace part for Plotly figures

    :param data: 2 columns dataframe
    :return: trace figures
    """
    id_cluster = data.iloc[0:1, 2].values[0]
    cluster = go.Scatter(
        x=data['Attack'],
        y=data['Defense'],
        mode='markers',
        marker=go.scatter.Marker(color=color_pallete[id_cluster]),
        name=f'Cluster {id_cluster + 1}'
    )
    return cluster


def scatter_title(title):
    """
    This function will generate layout part for Plotly Figures
    :param title: a tittle for figure title
    :return: Layout figures
    """

    layout = go.Layout(
        width=800,
        height=700,
        title=title,
        xaxis_title='Attack',
        yaxis_title='Defense',
        legend=dict(yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=1),
        showlegend=True
    )
    return layout


if __name__ == 'main':
    pokemon = pokemon_scrape(850)

    # Create RAW and normalization of observed dataset
    att_def = pokemon.loc[:, ['Attack', 'Defense']]
    normalization_data = MinMaxScaler().fit_transform(att_def[['Attack', 'Defense']])
    normal = pd.DataFrame(normalization_data, columns=['Attack', 'Defense'])

    # Model Fitting
    # Model As Expected Output
    kmean = model_kmeans(att_def, 3)

    kmean_data = att_def.copy()
    kmean_data['Label'] = kmean.labels_

    # Model with Normalization
    normal_kmean = model_kmeans(normal, 3)

    normal_kmean_data = normal.copy()
    normal_kmean_data['Label'] = normal_kmean.labels_

    # Model with K=2 and Normalization
    best_kmean = model_kmeans(normal, 2)

    best_kmean_data = normal.copy()
    best_kmean_data['Label'] = best_kmean.labels_

    # Visualization
    # As Expected Output (No Normalization)
    cluster_1, cluster_2, cluster_3 = scatter_cluster(kmean_data[kmean_data['Label'] == 0]), \
                                      scatter_cluster(kmean_data[kmean_data['Label'] == 1]), \
                                      scatter_cluster(kmean_data[kmean_data['Label'] == 2])
    given_visualization = go.Figure(data=[cluster_1, cluster_2, cluster_3],
                                    layout=scatter_title('K=3 KMeans Cluster Analysis'))

    # Normalization Data
    cluster_1, cluster_2, cluster_3 = scatter_cluster(normal_kmean_data[normal_kmean_data['Label'] == 0]), \
                                      scatter_cluster(normal_kmean_data[normal_kmean_data['Label'] == 1]), \
                                      scatter_cluster(normal_kmean_data[normal_kmean_data['Label'] == 2])
    normalization_visualization = go.Figure(data=[cluster_1, cluster_2, cluster_3],
                                            layout=scatter_title('K=3 KMeans Cluster Analysis with Normalization Data'))

    # Best Configuration (K=2 & Normalization)
    cluster_1, cluster_2 = scatter_cluster(best_kmean_data[best_kmean_data['Label'] == 0]),\
                           scatter_cluster(best_kmean_data[best_kmean_data['Label'] == 1])
    best_visualization = go.Figure(data=[cluster_1, cluster_2],
                                   layout=scatter_title('K=2 KMeans Cluster Analysis with Normalization Data'))

##################################################################################################################
# Note : To print every plot, add variable_names.show()
#
# Conclusion after Clustering analysis Attack-Defense in Pokemondb using KMeans
#
# The Quantile-Quantile plot shows that the distribution seems to be positively skewed (Right Skewed),
# since the value are concentrated in left side (head) and the right side has longer distribution.
# we can check this with diplay_hist() visualization above.
#
# The Attack & Defense data points in left and right are out from regression line,
# this means the data 'Attack' and 'Defense' have an outlier. we can check the outlier using display_boxplot()
#
# Fortunately, KMeans dont assume every model to be a gaussian or normal distribution,
# so we will not transform the distribution of data since KMeans can handle this
# and the attack & defense outlier is an actual pokemon data
# So, I'm deciding to keep the outlier. it's a valuable data.
#
# KMeans very sensitive to outlier and outweightened data. The large data points will impact to the measure of
# centroid distance and it will make the different cluster result. It shows in figures KMeans without Normalization and
# with Normalization. The normalization seems to be clearly separating the cluster with linear separator.
#
# Although KMeans very sensitive to outlier, but it's show a good result. If we want to make clustering
# model robust to outlier, we can use KMedians or KMedoids.
#
# Additional : I'm checking for the best K value for grouping the similar data in KMeans using Elbow Method
# and Silhouette Method. Both method shows 2 cluster is the best parameters for K.
#
# Next Update : Creating Dash for Dashboard visualization
##################################################################################################################
