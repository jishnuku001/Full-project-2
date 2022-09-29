import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
model = pickle.load(open('model.sav', 'rb'))

st.title('Milk Quality Prediction')
st.sidebar.header('Milk Data')
image = Image.open('milk.jpg')
st.image(image,'Milk')

def user_report():
  pH = st.sidebar.slider('pH', 0, 10, 1 )
  Temprature = st.sidebar.slider('Temprature', 30,100, 1 )
  Taste = st.sidebar.slider('Taste', 0,2, 1 )
  Odor = st.sidebar.slider('Odor', 0,2, 1 )
  Fat = st.sidebar.slider('Fat', 0,2, 1 )
  Turbidity = st.sidebar.slider('Turbidity', 23,260, 250)
  Colour = st.sidebar.slider('Colour ', 0,3, 1)
  


  user_report_data = {
      'pH':pH,
      'Temprature':Temprature,
      'Taste':Taste,
      'Odor':Odor,
      'Fat':Fat,
      'Turbidity':Turbidity,
      'Colour':Colour,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Milk Data')
st.write(user_data)

Quality = model.predict(user_data)
st.subheader('Milk Quality')
st.subheader(str(np.round(Quality[0], 2)))