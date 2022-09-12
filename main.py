import streamlit as st
from fastai.vision.all import *
import PIL
import time
import requests
import numpy as np
from io import BytesIO

def predict():

    if url != '' and uploaded_image is not None:
        st.subheader('Please only use input mode! (upload image OR url)')

    elif url != '':
        try:
            with st.spinner('loading class...'):
                time.sleep(3)
            
            response = requests.get(url)

            st.image(response.content)
            st.subheader('Prediction result:')
            st.markdown(learn_inf.predict(response.content))
        except:
            st.subheader('Image not found!')

    elif uploaded_image is not None:
        with st.spinner('loading class...'):
            time.sleep(3)

        img = PIL.Image.open(uploaded_image)
        img_array = np.array(img)

        st.image(uploaded_image)
        st.subheader('Prediction result:')
        st.markdown(learn_inf.predict(img_array))



if __name__ == '__main__':
    learn_inf = load_learner('export.pkl')

    st.title('Indonesian Food Image Classification')
    st.markdown('This image classification process is created using Fast.Ai')

    url = st.text_input('Please input an image (jpg/jpeg/png) url (roti kukus mekar/lumpia/roti lapis)')
    st.subheader('OR')
    uploaded_image = st.file_uploader('Upload an image (roti kukus mekar/lumpia/roti lapis)', type=['jpg', 'jpeg', 'png'])
    predict_btn = st.button('Predict')
    if predict_btn:
        predict()
