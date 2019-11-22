import pandas as pd 
import numpy as np 
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
import train_data
import test_data
from keras.models import load_model
from keras import backend as K

showpred = 0

try:
	model_path = './models/model.h5'
	model_weights_path = './models/weight.h5'
except:
	print("Need to train model")
test_path = "G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test"

#Loading the Pre-trained Models
model = load_model(model_path)
model.load_weights(model_weights_path)
st.sidebar.title("About")

st.sidebar.info("In this app, we are collaborating user interface to perform classification of animals.")

onlyfiles = [f for f in listdir("G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test") if isfile(join("G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test",f))]

st.sidebar.title("Let's Train the Neural Network")
if st.sidebar.button('Train CNN'):
	Training.train()

st.sidebar.title("Predict New Images")
imageselect = st.sidebar.selectbox("Pick an Image.",onlyfiles)
if st.sidebar.button("Predict this Animal"):
	showpred = 1
	prediction = Testing.predict((model),"G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test" + imageselect)


st.title("Animal Identification")
st.write("Pick an image from the dropdown list on the left to view it.")
st.write("When ready, you'll see a prediction on the left")

st.write("")
image = Image.open("G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test" + imageselect)
st.image(image,caption="Let's predict the animal!",use_column_width=True)

if showpred == 1:
	if prediction == 0:
		st.write("This is a **horse!**")
	if prediction == 1:
		st.write("This is an **elephant**")
	if prediction == 2:
		st.write("This is a **cat**")
	if prediction == 3:
		st.write("This is a **dog**")
