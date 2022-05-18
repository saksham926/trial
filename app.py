
import streamlit as st

# Custom imports 
from multipage import MultiPage
import pycaret_model,bert_model


# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("Cancer Disparity Analysis")

# Add all your applications (pages) here
app.add_page("Pycaret Model", pycaret_model.app)
app.add_page("BertTopic Model", bert_model.app)

# The main app
app.run()