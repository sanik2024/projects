import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define function to analyze sentiment
def get_sentiment(text):
    # Analyze sentiment of the text
    scores = sia.polarity_scores(text)
    # Determine sentiment based on compound score
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit UI
def main():
    # Set page title
    st.title('Sentiment Analyzer')
    # Text input for user to provide text
    user_text = st.text_area('Enter your text here:')
    # Button to analyze sentiment
    if st.button('Analyze'):
        # Perform sentiment analysis
        sentiment = get_sentiment(user_text)
        # Display the result
        st.write('Sentiment:', sentiment)

if __name__ == '__main__':
    main()
