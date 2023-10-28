import random 
import pathlib
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
from fastcore.all import *
from fastai.vision.all import *

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Art or Waste")

classes = ['plastic debri', 'people', 'marine life']

path = Path() 

if path.ls(file_exts='.pkl'):
    learn_inf = load_learner(path/'marine_learner.pkl')

with st.sidebar:
    st.title("Trash Waste, Not E(art)h!")
    st.subheader("♻♻♻♻♻♻")
    st.markdown("""
This project employs a vision learner to categorize objects in images as plastic debris, people, or marine life. 
The model can be integrated into a drone-based waste collection and recycling drivetrain, autonomously 
collecting plastic debris from beachfronts, riverbeds, and ocean floors.

As part of the "Art or Waste" initiative, aiming to combat pollution, the project focuses on researching 
plastic waste repurposing through art and developing AI waste identification and collection systems.

The model undergoes continuous retraining for improved predictions. Contributions of marine life or trash image 
submissions help enhance the algorithm's accuracy and reduce bias.

Feedback and suggestions are appreciated at artorwaste@gmail.com
    """)

    components.iframe(
       src = "https://github.com/sponsors/xuzmocode4-325/button", 
       height=32,
       width=114,
    )

st.title("Art or Waste")
st.write()
file_name = st.file_uploader()


test_pictures = Path('test_pictures')
file_list = list(test_pictures.glob("*"))
selection = random.sample(file_list, 9)

cols = cycle(st.columns(3, gap="small"))
for idx, picture in enumerate(selection):
    next(cols).image(Image.open(picture), use_column_width = True)

if file_name is not None:
    col1, col2 = st.columns([2, 1], gap="small")
    try:
        Image.open(file_name)
    except RuntimeError:
        st.warning("There seems to be an error processing that file. Please try again. If the error persists, try uploading a different image.")
    else:
        image = Image.open(file_name)
    col1.image(image, use_column_width=True)

    try:
        prediction = learn_inf.predict(image)
    except RuntimeError:
        st.warning("There seems to be an error processing that file. Please try again. If the error persists, try uploading a different image.")
    except NameError:
        st.warning("There seems to be an error processing that file. Please try again. If the error persists, try uploading a different image.")
    else:
        title = prediction[0].title()
        vector = int(prediction[1])
        col1.header(f"""Result: `{title}`""")
        
        col2.subheader("Probability")
        col2.write(f"""### {round(float(prediction[2][vector]) * 100, 2)}% """)