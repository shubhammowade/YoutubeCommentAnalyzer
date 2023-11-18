# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import googleapiclient.discovery
import nltk
import string

# Function to fetch YouTube comments using the provided video ID
def get_youtube_comments(video_id, developer_key):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=developer_key)

    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['textDisplay']
            ])

        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
        else:
            break

    return comments

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    ps = nltk.stem.PorterStemmer()
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Function to perform sentiment analysis on comments
def perform_sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()

    df = df.dropna(subset=["preprocessed_comment"])

    sentiment_scores = []
    sentiment_labels = []

    for preprocessed_comment in df["preprocessed_comment"]:
        scores = sia.polarity_scores(preprocessed_comment)
        sentiment_scores.append(scores["compound"])
        if scores["compound"] >= 0.05:
            sentiment_labels.append("Positive")
        elif scores["compound"] <= -0.05:
            sentiment_labels.append("Negative")
        else:
            sentiment_labels.append("Neutral")

    df["Sentiment_Score"] = sentiment_scores
    df["Sentiment_Label"] = sentiment_labels
    df["Sentiment"] = df["Sentiment_Label"].map({"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"})

    return df

# Function to perform spam classification on comments
def perform_spam_classification(df):
    model_filename = "spam_model.pkl"
    vectorizer_filename = "tfidf_vectorizer.pkl"

    clf = joblib.load(model_filename)
    tfidf_vectorizer = joblib.load(vectorizer_filename)

    X_tfidf = tfidf_vectorizer.transform(df["preprocessed_comment"])
    predicted_labels = clf.predict(X_tfidf)

    df["Classification"] = predicted_labels

    return df

# Function to create and display pie chart
def display_pie_chart(df, label_column, title, pie_size=6):
    counts = df[label_column].value_counts()
    labels = counts.index
    sizes = counts.values
    explode = [0.1] * len(counts)

    fig, ax = plt.subplots(figsize=(pie_size, pie_size))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    ax.set_title(title)

    return fig

# Streamlit app
def main():
    st.title("YouTube Comments Analysis")

    # Input for YouTube video ID
    video_id = st.text_input("Enter YouTube Video ID:")
    developer_key = st.text_input("Enter Your YouTube API Key:")

    if st.button("Analyze Comments"):
        # Fetch YouTube comments
        comments = get_youtube_comments(video_id, developer_key)

        # Preprocess comments
        df = pd.DataFrame(comments, columns=['comment'])
        df["preprocessed_comment"] = df["comment"].apply(transform_text)

        # Perform Sentiment Analysis
        df_sentiment = perform_sentiment_analysis(df)

        # Perform Spam Classification
        df_spam = perform_spam_classification(df)

        # Display results
        st.subheader("Sentiment Analysis Results:")
        st.write(df_sentiment.head(10))  # Display the first 10 rows of sentiment analysis results

        st.subheader("Spam Classification Results:")
        st.write(df_spam.head(10))  # Display the first 10 rows of spam classification results

        # Display pie chart for sentiment distribution
        st.subheader("Sentiment Distribution:")
        sentiment_chart = display_pie_chart(df_sentiment, "Sentiment", "Sentiment Distribution", pie_size=6)
        st.pyplot(sentiment_chart)

        # Display pie chart for spam distribution
        st.subheader("Spam Distribution:")
        spam_chart = display_pie_chart(df_spam, "Classification", "Spam Distribution", pie_size=6)
        st.pyplot(spam_chart)

if __name__ == "__main__":
    main()
