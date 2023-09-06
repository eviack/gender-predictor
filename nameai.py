import streamlit as st
import pandas as pd
import pickle as pk
from xgboost import XGBClassifier


namemodel = pk.load(open('namemodel.pkl', 'rb'))
namevec = pk.load(open('namevec.pkl', 'rb'))

def fetch_prediction(name):
    vec = namevec.transform([name]).toarray()
    pred = namemodel.predict(vec)
    for i in pred:
        if i==0: 
            return 'Maybe a female name'
        else: 
            return 'Maybe a male name'

st.title('Name gender prediction AI')
st.markdown('''It predicts the gender of the given name. It is trained using [XGBoost Classifier](https://en.wikipedia.org/wiki/XGBoost)''')

nameinput = st.text_input('Enter name')
if st.button('Predict'):
    prediction = fetch_prediction(nameinput)
    if 'male' in prediction.split() :
        st.info(prediction)
    elif 'female' in prediction.split():
        st.error(prediction)


