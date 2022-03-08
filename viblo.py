import streamlit as st
import pandas as pd
from PIL import Image

st.title("--Cat Dog Image Classification App--")
st.write('\n')

image = Image.open('dataset/00000000.jpg')
show = st.image(image, use_column_width=False,width=300)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

# if st.sidebar.button("Click Here to Classify"):
#     if uploaded_file is None:
#         st.sidebar.write("Please upload an Image to Classify")
#     else:
#       # Predict