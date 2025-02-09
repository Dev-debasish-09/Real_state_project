import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import gdown
import os
st.set_page_config(page_title='My prediction and analysis')




option = st.sidebar.selectbox("Select an option :", ["Home","Analysis Module", "Price Prediction", "Recomender System"])
if option == "Home":
    st.title("_Welcome To My Home Page_")
    st.image("Gurgaon image.jpg")
elif option == "Analysis Module":
    st.title("Analysis Module")

    st.header('Geo Map ')

    new_df = pd.read_csv('data_viz1.xls')
    feature_text = pickle.load(open('feature_text.pkl','rb'))
    group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','longitude','latitude']].mean()
    fig = px.scatter_mapbox(group_df,lat = 'latitude',lon = 'longitude',color='price_per_sqft',size='built_up_area',
                        color_continuous_scale = px.colors.cyclical.IceFire,zoom = 10,
                        mapbox_style = 'open-street-map',width=1200,height=700,hover_name=group_df.index)
    st.plotly_chart(fig,use_container_width=True)


    plt.rcParams["font.family"] = "Arial"

    wordcloud = WordCloud(
        width=600, height=600,
        background_color='Black',
        stopwords=set(['s']),
        min_font_size=10
    ).generate(feature_text)

        
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)

    st.pyplot(fig)

    st.header('Scatter plot of property type with Built area and Price')

    property_type = st.selectbox('Select Property Type',['Flat','House'])
    if property_type == 'House':
        fig1 = px.scatter(new_df[new_df['property_type'] == 'house'],x="built_up_area",y="price",color = "bedRoom",title = "Area Vs Price")
        st.plotly_chart(fig1,use_container_width=True)
    else:
        fig2 = px.scatter(new_df[new_df['property_type'] == 'flat'],x="built_up_area",y="price",color = "bedRoom",title = "Area Vs Price")
        st.plotly_chart(fig2,use_container_width=True)
    st.header('BHK pie vizulatization chart')

    sec_opt = new_df['sector'].unique().tolist()
    sec_opt.insert(0,'Overall')
    selected_sec = st.selectbox('Select Sector',sec_opt)

    if selected_sec == 'Overall':
        pfig = px.pie(new_df,names = 'bedRoom')
        st.plotly_chart(pfig,use_container_width=True)
    else :
        pfig = px.pie(new_df[new_df['sector']==selected_sec],names = 'bedRoom')
        st.plotly_chart(pfig,use_container_width=True)

    st.header('BHK Price Comparison')

    fig3 = px.box(new_df[new_df['bedRoom']<=4],x = 'bedRoom',y='price')
    st.plotly_chart(fig3,use_container_width=True)

    st.header('Comparison of dist Property')
    fig4 = plt.figure(figsize=(10,4))
    sns.histplot(new_df[new_df['property_type'] == 'house']['price'], kde=True, label='House')
    sns.histplot(new_df[new_df['property_type'] == 'flat']['price'], kde=True, label='Flat')
    plt.legend()
    st.pyplot(fig4)

elif option == "Price Prediction":
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


        baseprice = np.expm1(pipeline.predict(one_df)[0])

        low = baseprice - 0.22
        high = baseprice + 0.22

        st.text('The Price Of The Flat Is Between {} cr and {} cr'.format(round(low,2),round(high,2)))

else :
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
