import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#Normally we export the desired values from the data set instead of importing the whole data
def get_cleaned_data():
    df = pd.read_csv("data/breast_cancer.csv")
    # Drop NaN column and useless columns
    df.drop(["Unnamed: 32","id"],axis=1,inplace=True)
    # Replace B & M to 0 & 1
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    return df

def add_sidebar(df):
    #Create a list of tuples consisting of the labels(names above the sliders) and the keys(for creating a dict)
    slider_labels=[
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]
    
    #st.sidebar.slider(...) adds the sliders
    #the slider is type of input, so it means that it will return values of each slider from the users
    #the values are needed to make prediction, therefore we use a dict to store the values 
    dict = {}

    for label, key in slider_labels:
        dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=df[key].max(),
            value=df[key].mean()
        )
    return dict #return back the dict containing the user input values

# *Since the ranges of values of every columns vary a lot, causing that certain columns will not looks good
#  in the chart
#  Altering the min max of the radial length is not a good solution (the range too small or too big)
#  Which is why the optimal solution is to scale every single value to 0.0 - 1.0
def get_scaled_values(input_dict):
    scaled_input_dict = {}

    for key, value in input_dict.items():
        # Suspicious: Probably should have import this as well rather than using the whole dataset
        df = get_cleaned_data()
        X = df.drop(['diagnosis'], axis=1, inplace=False)
        min_val = X[key].min()
        max_val = X[key].max()

        scaled_input_dict[key] = (value - min_val)/(max_val - min_val)
    return scaled_input_dict

def get_chart(input_data):
    input_data = get_scaled_values(input_data)
  
    categories = ['Radius','Texture','Perimter','Area','Smoothness','Compactness',
                'Concavity', 'Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=                              #r means the radial length from the centre
        [
            input_data['radius_mean'],input_data['texture_mean'],input_data['perimeter_mean'],
            input_data['area_mean'],input_data['smoothness_mean'],input_data['compactness_mean'],
            input_data['concavity_mean'],input_data['concave points_mean'],input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,               #Theta is the angular length(anti-clockwise)
        fill='toself',                  #No fill means no colour, just lines
        name='Mean Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=                              
        [
            input_data['radius_se'],input_data['texture_se'],input_data['perimeter_se'],
            input_data['area_se'],input_data['smoothness_se'],input_data['compactness_se'],
            input_data['concavity_se'],input_data['concave points_se'],input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,               
        fill='toself',                  
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=                              
        [
            input_data['radius_worst'],input_data['texture_worst'],input_data['perimeter_worst'],
            input_data['area_worst'],input_data['smoothness_worst'],input_data['compactness_worst'],
            input_data['concavity_worst'],input_data['concave points_worst'],input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,               
        fill='toself',                  
        name='Worst Value'
    ))        

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]        # Set the radial length from 0 to 1, *explained at the scaler function
        )),
    showlegend=True
    )
    return fig

def get_prediction(input_data):
    model = pickle.load(open("model/model.pkl","rb"))    #rb stands for Read Binary
    scaler = pickle.load(open("model/scaler.pkl","rb"))

    #Since the input data is a dict, model.predict() takes is a single row of values, so need to change to array
    # The code below: make a column of values only from the dict, then transpose it so that it looks like a row from the dataset
    # In fact: the 1 means 1 row, -1 means change from rows to colomns
    input_array = np.array(list(input_data.values())).reshape([1,-1])
    
    # Since the model is trained with scaled dataset, so the inputs have to be scaled using the same scaler too
    # Note: A row of 0s, because the dafault values of the slider are set to mean values
    input_array = scaler.transform(input_array) 
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)  # [0,0] is prob of 0, [0,1] is prob of 1

    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")
    
    st.write("The probability of being benign:\n", probability[0][0])
    st.write("The probability of being malignant:\n", probability[0][1])


def main():
    data = get_cleaned_data()

    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar(data)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_chart(input_data)
        st.plotly_chart(radar_chart,use_container_width=True)

    with col2:
        st.subheader("Cell Cluster Prediction")
        st.write("The cell cluster is:")
        get_prediction(input_data)
        
if __name__ == "__main__":
    main()