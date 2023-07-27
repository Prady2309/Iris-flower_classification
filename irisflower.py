import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

st.write("""
# Iris Flower Prediction App
""")

st.sidebar.header("User Input")

def input_features() :
    s_length = st.sidebar.slider('Sepal length', 2.0, 7.9, 5.4)  # min-value, max-value, current value
    s_width = st.sidebar.slider('Sepal Width', 2.0, 7.9, 3.0)
    p_length = st.sidebar.slider('Petal length', 2.0, 7.9, 1.0)
    p_width = st.sidebar.slider('Petal Width', 2.0, 7.9, 0.4)
    data = {'sepal_length': s_length, 
            'sepal_width': s_width,
            'petal_length': p_length,
            'petal_width': p_width}
    df = pd.DataFrame(data, index=[0])
    return df

features = input_features()

st.subheader('User Input parameters')
st.write(features)

iris = load_iris()
x = iris.data     # splitting the data
y = iris.target

classifier = RandomForestClassifier()
classifier.fit(x, y)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)    # displaying the flower names

prediction = classifier.predict(features)
prediction_proba = classifier.predict_proba(features)     # for probability calculation for diff scenarios

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)