#LIBRARIES
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re


#LOAD PICKLE FILES
model = pickle.load(open('best_model.pkl','rb')) 
vectorizer = pickle.load(open('count_vectorizer.pkl','rb')) 

#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

#TEXT CLASSIFICATION
def text_classification(text):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)
        
            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fraudulent.")

#PAGE FORMATTING AND APPLICATION
def main():
    st.title("Fake Reviews Detection Application")
    
    
    # --EXPANDERS--    
    abstract = st.expander("Project Summary")
    if abstract:
        abstract.write("The digital age has revolutionised consumer decision-making through the pervasive influence of online reviews on e-commerce platforms. However, this democratisation of information has been marred by the proliferation of fake reviews, threatening the trust and satisfaction of online consumers. This research seeks to address this critical issue by employing advanced Linguistic Features and Sentiment Analysis techniques to identify and combat fake reviews on e-commerce sites. Through a systematic exploration of linguistic characteristics and emotional nuances, this study aims to develop an automated model that not only distinguishes genuine from deceptive content but also evaluates the real-world impact of fake reviews on customer trust and satisfaction. By providing practical recommendations, the research endeavours to contribute to the creation of a more reliable and trustworthy online shopping environment.")
        abstract.write("University: University of Essex")
        abstract.write("Supervised By: Prof. Stefania Paladini")
        abstract.write("Submited By: Rimpa Ghosh")
        #st.write(abstract)
    
    links = st.expander("Related Links")
    if links:
        links.write("[Dataset](https://www.kaggle.com/akudnaver/amazon-reviews-dataset)")
        links.write("[Github](https://github.com/rimpaghosh96/University_of_Esssex_MSc)")
        


    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Check authenticity of review")
    review = st.text_area("Enter Review: ")
    if st.button("Check"):
        text_classification(review)

#RUN MAIN        
main()
