import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import random
import plotly.express as px
import plotly.graph_objs as go

# Load data
df = pd.read_excel("C:\\Users\\dhava\\Downloads\\NewsLOC\\Final News DataFrame.xlsx")

# Function to create Plotly horizontal bar chart
def plot_horizontal_bar():
    category_counts = df['Category'].value_counts()
    categories_gt_10 = category_counts[category_counts > 9].index
    df_filtered = df[df['Category'].isin(categories_gt_10)]
    pivot_df = df_filtered.pivot_table(index='Category', columns='Sentiment', aggfunc='size', fill_value=0)
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    fig = px.bar(pivot_df, orientation='h', barmode='stack', color_discrete_map=colors)
    fig.update_layout(title='Sentiment Counts by Category (Count > 10)', xaxis_title='Count', yaxis_title='Category', width=800)
    st.plotly_chart(fig)

# Function to create Plotly sunburst chart
def plot_sunburst():
    category_counts = df['Category'].value_counts()
    categories_gt_10 = category_counts[category_counts > 13].index
    df_filtered = df[df['Category'].isin(categories_gt_10)]
    df_filtered['Sector'] = df_filtered['Sector'].str.split(',')
    df_filtered = df_filtered.explode('Sector')
    entity_count = df_filtered.groupby(['Category', 'Sector']).size().reset_index(name='Count')
    top5_entities = entity_count.groupby('Category').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)
    fig = px.sunburst(top5_entities, path=['Category', 'Sector'], values='Count', color='Sector', width=800, height=500)
    fig.update_traces(marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color='white', width=1)))
    fig.update_layout(title='Top 5 Entities by Category (Count > 10)', template='plotly_white')
    st.plotly_chart(fig)

# Function to create Plotly bar chart
def plot_bar_chart():
    category_count = df['Category'].value_counts().reset_index()
    category_count.columns = ['Category', 'Count']
    category_count_filtered = category_count[category_count['Count'] > 10]
    fig = px.bar(category_count_filtered, 
                 x='Count', 
                 y='Category', 
                 labels={'Count': 'Category Count', 'Category': 'Category'},
                 title='Category Counts (Count > 10)',
                 template='plotly_white',
                 width=800,
                 height=400,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(marker_line_color='white', marker_line_width=1.5)
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    st.plotly_chart(fig)

# Function to create Plotly stacked bar chart
def plot_stacked_bar():
    category_ticker_counts = df.groupby(['Category', 'Ticker']).size().reset_index(name='Count')
    category_ticker_counts['Category'] = category_ticker_counts['Category'].str.split().str[0]
    fig = px.bar(category_ticker_counts, 
                 x='Category', 
                 y='Count', 
                 color='Ticker',
                 title='Ticker Counts by Category',
                 barmode='stack',
                 labels={'Count': 'Ticker Count', 'Category': 'Category'},
                 template='plotly_white',
                 width=800,
                 height=400)
    st.plotly_chart(fig)

# Function to create Plotly 2D visualization
def create_2d_visualization():
    categories = df['Category']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(categories)
    k = len(df['Category'].unique())
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    tsne = TSNE(n_components=2, random_state=42)
    embedded_nodes = tsne.fit_transform(tfidf_matrix.toarray())
    x_coords = embedded_nodes[:, 0]
    y_coords = embedded_nodes[:, 1]
    graph_data = []
    cluster_names = df['Category'].unique()
    unique_labels = list(set(cluster_labels))
    random.shuffle(unique_labels)
    colors = ['rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(len(unique_labels))]
    for i, cluster_name in enumerate(cluster_names):
        cluster_color = colors[i]
        x_cluster = x_coords[cluster_labels == i]
        y_cluster = y_coords[cluster_labels == i]
        trace = go.Scatter(
            x=x_cluster,
            y=y_cluster,
            mode='markers',
            marker=dict(size=5, color=cluster_color),
            name=cluster_name
        )
        graph_data.append(trace)
    fig = go.Figure(data=graph_data)
    fig.update_layout(
        title="Category Clustering based on t-SNE Visualization",
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2"),
    )
    st.plotly_chart(fig)

# Navbar
st.title("NewsPlot Pro")

# Quick Navigation Buttons
nav_selection = st.sidebar.radio("Navigation", ["GraphQL", "Intuitive Dashboard", "Sentiment Impact"])

if nav_selection == "GraphQL":
    df = pd.read_excel("C:\\Users\\dhava\\Downloads\\NewsLOC\\AbsoluteFinal News DataFrame.xlsx")
    st.write("Welcome to the Graphs Page!")

    # Dropdown selection
    graph_type = st.selectbox("Select Graph Type", ["Category", "Sector", "Name"])

    if graph_type:
        st.sidebar.header(f"{graph_type} Values")
        nested_options = df[graph_type].unique()
        nested_selection = st.sidebar.selectbox(
            f"Select {graph_type}", nested_options)

        if nested_selection:
            st.sidebar.write(f"You selected {nested_selection}")

            # Submit button
            if st.sidebar.button("Submit"):
                st.sidebar.write("Graph settings submitted!")

                # Generate and display the graph
                #st.write("Generating the graph...")

                # Define functions to generate graphs
                def generate_graph(data, selected_value, node_type):
                    if selected_value == f"All {node_type}s":
                        sub_df = data
                    else:
                        sub_df = data[data[node_type] == selected_value]

                    G = nx.DiGraph()
                    for headline in sub_df['News Headline']:
                        G.add_node(headline, color='blue', size=30, text=headline)

                    for item in sub_df[node_type]:
                        G.add_node(item, color='red', size=25, text=item)

                    for _, row in sub_df.iterrows():
                        headline = row['News Headline']
                        value = row[node_type]
                        relationship = row['relationship']
                        intermediary_node = f"{headline} - {value}"
                        G.add_node(intermediary_node, color='orange', size=15, text=relationship)
                        G.add_edge(headline, intermediary_node)
                        G.add_edge(intermediary_node, value)
                    return G

                if graph_type == "Category":
                    graph = generate_graph(df, nested_selection, "Category")
                elif graph_type == "Sector":
                    graph = generate_graph(df, nested_selection, "Sector")
                elif graph_type == "Name":
                    graph = generate_graph(df, nested_selection, "Name")

                # Get positions using a layout algorithm
                pos = nx.kamada_kawai_layout(graph, scale=10, weight=3)

                # Convert NetworkX positions for Plotly
                plotly_pos = {node: np.array(pos[node]) for node in graph.nodes()}

                # Create Plotly figure
                fig = go.Figure()

                # Add nodes without default text
                fig.add_trace(go.Scatter(x=[pos[0] for pos in plotly_pos.values()],
                                         y=[pos[1] for pos in plotly_pos.values()],
                                         mode='markers',
                                         marker=dict(size=[graph.nodes[node]['size'] * 2 for node in graph.nodes()],
                                                     color=[graph.nodes[node]['color'] for node in graph.nodes()]
                                                     ),
                                         hovertext=[graph.nodes[node]['text'] for node in graph.nodes()],
                                         hoverinfo='text'
                                         ))

                # Add edges with hover text
                for edge in graph.edges:
                    x0, y0 = plotly_pos[edge[0]]
                    x1, y1 = plotly_pos[edge[1]]
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1],
                                             mode='lines',
                                             line=dict(color='white', width=2),
                                             hoverinfo='none'))

                legend_annotations = [
                    dict(
                        x=0.05, y=0.95, xref="paper", yref="paper",
                        text="Node Colors:", showarrow=False, font=dict(size=17, color="white"), align="left"
                    ),
                    dict(
                        x=0.05, y=0.9, xref="paper", yref="paper",
                        text="- Blue: News Headline", showarrow=False, font=dict(size=17, color="blue"), align="left"
                    ),
                    dict(
                        x=0.05, y=0.85, xref="paper", yref="paper",
                        text="- Red: Category/Sector/Name", showarrow=False, font=dict(size=17, color='red'), align="left"
                    ),
                    dict(
                        x=0.05, y=0.8, xref="paper", yref="paper",
                        text="- Orange: relationship", showarrow=False, font=dict(size=17, color='orange'), align="left"
                    )
                ]

                # Customize layout
                fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                                  plot_bgcolor='black', width=1000, height=900, annotations=legend_annotations)
                st.plotly_chart(fig)

elif nav_selection == "Intuitive Dashboard":
    st.title("Dashboard for Final News Data")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Display plots
    with col1:
        plot_horizontal_bar()
    with col2:
        plot_sunburst()
    with col3:
        plot_stacked_bar()
    with col4:
        create_2d_visualization()
elif nav_selection == "Sentiment Impact":
    senti_df = pd.read_csv("C:\\Users\\dhava\\Downloads\\NewsLOC\\Impact.csv")
    st.write(senti_df)

