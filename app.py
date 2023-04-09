import streamlit as st
from transformers import pipeline


st.title('Toxic Tweets')

models = [
    'distilbert-base-uncased-finetuned-sst-2-english',
    'cardiffnlp/twitter-roberta-base-sentiment',
    'Seethal/sentiment_analysis_generic_dataset',
]

default_tweet = """🐰🌸🐣 Happy Easter 🌸🐰🐣! It's time to crack open some eggs 🥚 and celebrate with the Easter Bunny 🐰🐇. Hop 🐇 on over to church ⛪️ and get down on your knees 🧎‍♂️🙏 for some Easter blessings 🐰✝️🌷. Did you know that Jesus 🙏💒 died and rose again 💀🙌🌅? It's a time for rejoicing 🎉 and enjoying the company of loved ones 👨‍👩‍👧‍👦. So put on your Sunday best 👗 and get ready to hunt 🕵️‍♀️ for some Easter treats 🍫🥚🍭. Happy Easter, bunnies 🐰👯‍♀️! Don't forget to spread the love ❤️ and send this message to your favorite bunnies 💌🐇.
"""

st.image('https://www.gannett-cdn.com/presto/2022/04/12/USAT/3a93e183-d87d-493a-97a9-cf75fb7b9d18-AP_Pennsylvania_Easter.jpg')
tweet = st.text_area('Enter a tweet', value=default_tweet)
model = st.selectbox('Select a model', models)
button = st.button('Predict')

labels = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}

def predict(tweet, model):
    with st.spinner('Predicting...'):
        classifier = pipeline(model=model)
        result = classifier(tweet)

        label = result[0]["label"]
        score = result[0]["score"]
        
        # Cleanup label
        if label != 'POSITIVE' and label != 'NEGATIVE':
            label = labels[label]
    
    if label == 'POSITIVE':
        st.balloons()
    
    st.info(f"Label: {label} \n\n Score: {score}")



if button:
    if not tweet:
        st.warning('Please enter a tweet')
    else:
        predict(tweet, model)

elif tweet:
    predict(tweet, model)