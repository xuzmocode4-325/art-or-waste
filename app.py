import streamlit as st
from PIL import Image
from fastcore.all import *
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Art or Waste")

classes = ['fine art','food waste','glass waste','metal waste','paper waste','plastic waste']

path = Path()

if path.ls(file_exts='.pkl'):
    learn_inf = load_learner(path/'model.pkl')

with st.sidebar:
    st.title("""Do Not Trash Your Art Just Yet ♻♻♻""")
    st.markdown("""This fun project at present will classifies images as either `fine art` or one of five types of 
                waste:""")
    st.write("`food`")
    st.write("`glass`")
    st.write("`metal`")
    st.write("`paper`")
    st.write("`plastic`")
    st.markdown("In future more art categories will be added.")
    st.markdown("""In production, such a model can be used as part of a mechanical waste recycling 
                drive-chain that automatically sorts your trash for further processing, 
                and alerts you when you could have possibly discarded something of value by mistake.""")
    st.markdown("""This project is still in its infancy, so if your art gets missclassified as trash, 
                please attach it to an email and send it through to artorwaste@gmail.com.""")
    st.markdown("""The model will be retrained with your art so that it can better able to 
                recognise your style of ingenuity in futute.""")

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