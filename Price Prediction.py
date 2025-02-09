import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="My Price Prediction System")

st.title("Price Prediction")

with open('df.pkl','rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

st.header('Enter the required input ')

property_type = st.selectbox('property type',['flat','house'])

sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

beedRoom = float(st.selectbox('Number of Beed Room',sorted(df['bedRoom'].unique().tolist())))

bathRoom = float(st.selectbox('Number ofBath Room',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property  Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servent_room = float(st.selectbox('Servent Room',[0.0,1.0]))

store_room = float(st.selectbox('Store Room',[0.0,1.0]))

furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))

luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))

floor_category  = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):
    data = [[property_type,sector,beedRoom,bathRoom,balcony,property_age,built_up_area,servent_room,store_room,furnishing_type,luxury_category,floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
                'agePossession', 'built_up_area', 'servant room', 'store room',
                'furnishing_type', 'luxury_category', 'floor_category']
    
    one_df = pd.DataFrame(data, columns=columns)
    st.write(one_df)



    st.dataframe(one_df)

    baseprice = np.expm1(pipeline.predict(one_df)[0])

    low = baseprice - 0.22
    high = baseprice + 0.22

    st.text('The Price Of The Flat Is Between {} cr and {} cr'.format(round(low,2),round(high,2)))