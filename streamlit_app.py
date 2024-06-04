import streamlit as st
import pandas as pd
import os
import functions  # Assuming functions.py contains your processing logic

st.title("Welcome to the sentiment analyzer, Plushy!")
st.write("Please upload your CSV file below:")
file = st.file_uploader("Select")

if file is not None:
    data = pd.read_csv(file)

    st.write("Pick your configuration options.")

    review_type = st.radio("Which review types do you want to include?",
                           ["Private Only", "Public Only", "All Reviews"])

    metric = st.radio("Do you want to see sentiment distribution, NLP metric, or both?",
                      ["Sentiment Distribution", "NLP Metric", "Both metrics"])

    # Process file based on config options using functions.py
    data = functions.state(data)
    data = functions.process_files(data, review_type)
    data = functions.master_nlp(data)

    # Temporary CSV (optional)
    processed_filepath = os.path.join('uploads', 'processed_data.csv')
    os.makedirs('uploads', exist_ok=True)
    data.to_csv(processed_filepath, index=False)

    # Show results
    if metric == "Sentiment Distribution":
        functions.plot_sentiment_distribution(data)
        st.image('static/sentiment_distribution.png')  # Assuming plot saved to this file
    elif metric == "NLP Metric":
        sentiment_trend_plots = functions.plot_sentiment_trend_by_category(data)
        for category, plot_html in sentiment_trend_plots.items():
            st.write(f"Sentiment Trend for {category}")
            st.components.v1.html(plot_html, height=600, scrolling=True)  # Use st.components.v1.html to display HTML content
    elif metric == "Both metrics":
        functions.plot_sentiment_distribution(data)
        st.image('static/sentiment_distribution.png')
        sentiment_trend_plots = functions.plot_sentiment_trend_by_category(data)
        for category, plot_html in sentiment_trend_plots.items():
            st.write(f"Sentiment Trend for {category}")
            st.components.v1.html(plot_html, height=600, scrolling=True)  # Use st.components.v1.html to display HTML content
