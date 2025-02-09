import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="My Recommender System")
st.title("Recommender System")

with open('location_distance.pkl', 'rb') as file:
    location_df = pickle.load(file)
with open('cosine_sim1.pkl', 'rb') as file:
    cosine_sim1_df = pickle.load(file)
with open('cosine_sim2.pkl', 'rb') as file:
    cosine_sim2_df = pickle.load(file)
with open('cosine_sim3.pkl', 'rb') as file:
    cosine_sim3_df = pickle.load(file)

def recommend_properties_with_scores(property_name, top_n=6):
    
    cosine_sim_matrix = 30*cosine_sim1_df + 20*cosine_sim2_df + 8*cosine_sim3_df
  
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
   
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    top_properties = location_df.index[top_indices].tolist()
    
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df



st.title('Select location and distance')

location_select = st.selectbox('Location', sorted(location_df.columns.to_list()))
distance = st.number_input('Distance in K.M', min_value=0.0, step=0.1)

st.write("Selected Column:", location_select)
st.write("Selected Distance (KM):", distance)

if st.button('Search'):
    if location_select in location_df.columns:
        result_dis = location_df[location_df[location_select] < distance * 1000][location_select].sort_values()
        if result_dis.empty:
            st.write("No locations found within the specified distance.")
        else:
            for i, j in result_dis.items():
                st.text(f"{i} {round(j / 1000, 2)} K.M")
    else:
        st.write(f"⚠️ Column '{location_select}' not found in DataFrame!")

st.title("Recomended Appartment")
my_select_appart = st.selectbox('Select an appartment',sorted(location_df.index.to_list()))

if st.button('Recommend'):
    rec_df = recommend_properties_with_scores(my_select_appart)

    st.dataframe(rec_df)
