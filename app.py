import streamlit as st
from PIL import Image
from fastcore.all import *
from fastai.vision.all import *
import streamlit.components.v1 as components

st.set_page_config(page_title="Art or Waste")

classes = ['fine art','food waste','glass waste','metal waste','paper waste','plastic waste']

path = Path()

if path.ls(file_exts='.pkl'):
    learn_inf = load_learner(path/'model.pkl')

st.title("Art or Waste")

file_name = st.file_uploader("Upload an artwork in any image format.")

if file_name is not None:
    col1, col2 = st.columns([2, 1], gap="small")

    try:
        Image.open(file_name)
    except RuntimeError:
        st.warning("There was an error running the app. Please try again. If the error persists, try uploading another file.")
    else:
        image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = learn_inf.predict(image)

    col1.header(f"""Result: `{predictions[0].title()}`""")
    
    col2.write("### Probabilites")
    col2.markdown(f"""
    {classes[0]}: {round(float(predictions[2][0]) * 100, 2)}% \n
    {classes[1]}: {round(float(predictions[2][1]) * 100, 2)}% \n 
    {classes[2]}: {round(float(predictions[2][2]) * 100, 2)}% \n
    {classes[3]}: {round(float(predictions[2][3]) * 100, 2)}% \n
    {classes[4]}: {round(float(predictions[2][4]) * 100, 2)}% \n
    """)