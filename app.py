import streamlit as st
from transformers import pipeline, DistilBertTokenizerFast

st.title("Toxic Tweets")

# Available models for prediction
models = [
    "notbhu/toxic-tweet-classifier",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "Seethal/sentiment_analysis_generic_dataset",
]


# Default tweet for users to test on
default_tweet = """🐰🌸🐣 Happy Easter 🌸🐰🐣! It's time to crack open some eggs 🥚 and celebrate with the Easter Bunny 🐰🐇. Hop 🐇 on over to church ⛪️ and get down on your knees 🧎‍♂️🙏 for some Easter blessings 🐰✝️🌷. Did you know that Jesus 🙏💒 died and rose again 💀🙌🌅? It's a time for rejoicing 🎉 and enjoying the company of loved ones 👨‍👩‍👧‍👦. So put on your Sunday best 👗 and get ready to hunt 🕵️‍♀️ for some Easter treats 🍫🥚🍭. Happy Easter, bunnies 🐰👯‍♀️! Don't forget to spread the love ❤️ and send this message to your favorite bunnies 💌🐇.
"""


# Loads background image for Streamlit
st.image(
    "https://www.gannett-cdn.com/presto/2022/04/12/USAT/3a93e183-d87d-493a-97a9-cf75fb7b9d18-AP_Pennsylvania_Easter.jpg"
)


# User input for tweet
tweet = st.text_area("Enter a tweet", value=default_tweet)

# User selection for model
model = st.selectbox("Select a model", models)

# Checks if predict button is clicked
button = st.button("Predict")


# Function used to get the label for the model
def getLabel(label, model):
    labels = {
        "notbhu/toxic-tweet-classifier": {
            "LABEL_0": "toxic",
            "LABEL_1": "severe_toxic",
            "LABEL_2": "obscene",
            "LABEL_3": "threat",
            "LABEL_4": "insult",
            "LABEL_5": "identity_hate",
        },
        "distilbert-base-uncased-finetuned-sst-2-english": {
            "POSITIVE": "POSITIVE",
            "NEGATIVE": "NEGATIVE",
        },
        "cardiffnlp/twitter-roberta-base-sentiment": {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE",
        },
        "Seethal/sentiment_analysis_generic_dataset": {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "POSITIVE",
        },
    }
    return labels[model][label]


# Function used to predict the tweet based on selected model
def predict(tweet, model):
    with st.spinner("Predicting..."):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        classifier = pipeline(model=model, tokenizer=tokenizer)

        try:
            result = classifier(tweet)
            label = result[0]["label"]
            score = result[0]["score"]

            label = getLabel(label, model)
            
            if label == "POSITIVE":
                st.balloons()

            st.info(f"Label: {label} \n\n Score: {score}")
        except Exception as e:
            st.error("Something went wrong")
            st.error(e)


# Main Streamlit logic is controlled here
if button:
    if not tweet:
        st.warning("Please enter a tweet")
    else:
        predict(tweet, model)

elif tweet:
    predict(tweet, model)