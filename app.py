import streamlit as st
from PIL import Image
from fastcore.all import *
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Art or Waste")

classes = ['scrap paper', 'scrap metal', 'waste plastic', 'food scraps', 'scrap wood',
               'visual art', 'pottery', 'wooden furniture', 'jewelery', 'plastic toy', 'glass waste']

path = Path()

if path.ls(file_exts='.pkl'):
    learn_inf = load_learner(path/'learner.pkl')

with st.sidebar:
    st.title("""Do Not Trash Your Art Just Yet ♻♻♻""")
    st.markdown("""This fun project classifies objects in images as either `scrap paper`, 
                `scrap metal`, `waste plastic`, `food scraps`, `scrap wood`, `glass waste`, `visual art`, 
                `pottery`, `wooden furniture`, `jewelery` or `plastic toy`.""")
    st.markdown("In future more art or waste categories may be added.")
    st.markdown("""In production, such a model can be used as part of a mechanical waste recycling 
                drive-chain that automatically sorts your trash for further processing, 
                and alerts you when you could have possibly discarded something of value by mistake.""")
    st.markdown("""This project is still in its infancy, we value your feedback and suggestions. 
                If your art gets missclassified as trash, please attach it to an email and send it through to 
                artorwaste@gmail.com.""")
    st.markdown("""We continously retrain our model to get better predictions. Any art or trash images 
                submissions will help to make our algorithm more accurate and unbiased.""")

st.title("Art or Waste")

file_name = st.file_uploader("Upload an artwork in any image format.")

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
    else:
        title = prediction[0].title()
        vector = int(prediction[1])
        col1.header(f"""Result: `{title}`""")
        
        col2.subheader("Probability")
        col2.write(f"""### {round(float(prediction[2][vector]) * 100, 2)}% """)