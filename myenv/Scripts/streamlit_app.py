import streamlit as st
import pickle
import pandas as pd

# # Load the classifier and vectorizer
# with open('G:\police hackathon\deploy\myenv\Scripts\spam_classifier.pkl', 'rb') as f:
#     classifier = pickle.load(f)
# with open('G:\police hackathon\deploy\myenv\Scripts\vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)
with open('G:/police hackathon/deploy/myenv/Scripts/spam_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('G:/police hackathon/deploy/myenv/Scripts/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Define the predict function
def predict_spam(text):
    # Vectorize the text
    text_vectorized = vectorizer.transform([text])

    # Make a prediction
    prediction = classifier.predict(text_vectorized)[0]

    # Return the prediction
    return prediction

# Define the Streamlit app
def app():
    # Set the app title
    st.title('ShieldTrack')
    st.title('Spam Detector')


    # Add a text input for the user to enter a message
    message = st.text_input('Enter a message')

    # When the user clicks the "Predict" button, make a prediction
    if st.button('Predict'):
        prediction = predict_spam(message)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    app()
