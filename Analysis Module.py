import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="My Analysis Module")

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
