import streamlit as st
import bertopic
import re
import pandas as pd
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
import streamlit.components.v1 as components

def remove_stopwords(text,stop_words):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def app():
    st.write("BertTopic Analysis")
    regex = "\\b[0-9]{15}|[0-9]{12}\\b"
    uploaded_file = st.file_uploader("Choose a file",type=['csv'])
    st.write(type(uploaded_file))
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        st.dataframe(df)
        df.rename(columns = {'abstract':'text'}, inplace = True)
        df.text = df.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
        df.text = df.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
        df.text = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
        df['text'] = df['text'].apply(lambda x: re.sub('W*dw*','',x))
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        stop_words.add('subject')
        stop_words.add('http')
        df['text'] = df['text'].apply(lambda x: remove_stopwords(x,stop_words))
        titles = df.text.to_list()
        cancerhealth_disparities_model = BERTopic(verbose=True)
        topics, probabilities = cancerhealth_disparities_model.fit_transform(titles)
        fig = cancerhealth_disparities_model.visualize_topics(); fig
        fig.write_html("file.html")
        p=open("file.html")
        components.html(p.read(),width=1000,height=1000)
        fig=cancerhealth_disparities_model.visualize_hierarchy(top_n_topics=30);fig
        fig.write_html("hier.html")
        p=open("hier.html")
        components.html(p.read(),width=1000,height=1000)
        fig=cancerhealth_disparities_model.visualize_heatmap();fig
        fig.write_html("report.html")
        p=open("report.html")
        components.html(p.read(),width=1000,height=1000)
        fig=cancerhealth_disparities_model.visualize_barchart();fig
        fig.write_html("topic.html")
        p=open("topic.html")
        components.html(p.read(),width=1000,height=1000)
        similar_topics, similarity = cancerhealth_disparities_model.find_topics("colorectal", top_n = 3)
        most_similar = similar_topics[0]
        st.write("Most Similar Topic Info: ",cancerhealth_disparities_model.get_topic(most_similar))
        st.write()
        st.write("Similarity Score: ",similarity[0])
        
        
    else:
        st.warning("You need to upload a csv file with column header as abstract.")
