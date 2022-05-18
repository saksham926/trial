import streamlit as st
import pycaret
from pycaret.nlp import *
import pandas as pd
import spacy
import streamlit.components.v1 as components
def fun():
    #english_model = spacy.load("./models/en/")
    spacy.load("en_core_web_sm")
def app():
    st.write("Pycaret Analysis")
    uploaded_file = st.file_uploader("Choose a file",type=['csv'])
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        st.dataframe(df)
        #fun()
        nlp1 = setup(df,target='abstract',session_id = 123)
        lda = create_model('lda', num_topics = 5)
        plot_model(save=True)
        p = open("Word Frequency.html")
        components.html(p.read(),width=1000,height=1000)
        plot_model(lda,plot="bigram",save=True)
        p=open("Bigram.html")
        components.html(p.read(),width=1000,height=1000)
        plot_model(lda,plot='sentiment',topic_num = 'Topic 0',save=True)
        p=open("Sentiments.html")
        components.html(p.read(),width=1000,height=1000)
        plot_model(lda,plot='tsne',save=True)
        p=open("TSNE.html")
        components.html(p.read(),width=1000,height=1000)
        plot_model(lda,plot='topic_distribution',save=True)
        p=open("Topic Distribution.html")
        components.html(p.read(),width=1000,height=1000)

    else:
        st.warning("you need to upload an excel file.")
