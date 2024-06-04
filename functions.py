import pandas as pd
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# Download necessary NLTK resources
nltk_resources = [
    'punkt',
    'stopwords',
    'vader_lexicon'
]

for resource in nltk_resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

# Define functions
def state(data):
    """Add a state column if not present"""
    data['state'] = data['property_name'].str.extract(r'(\w\w)$').str.upper().fillna('N/A')
    return data

def process_files(data, review_type):
    """Process the data based on review type."""
    essential_columns = ['checkout_date', 'property_name', 'state']  # Essential columns to retain

    if review_type == 'Public':
        data = data[essential_columns + ['public_review']].rename(columns={'public_review': 'review'})
    elif review_type == 'Private':
        data = data[essential_columns + ['private_feedback']].rename(columns={'private_feedback': 'review'})
    else:  # 'All'
        public = data[essential_columns + ['public_review']].rename(columns={'public_review': 'review'})
        private = data[essential_columns + ['private_feedback']].rename(columns={'private_feedback': 'review'})
        data = pd.concat([public, private], ignore_index=True)

    return data

def cleaning(data):
    data['review'] = data['review'].replace('', pd.NA)
    data = data.dropna(subset=['review'])
    data = data.drop_duplicates(subset=['review', 'checkout_date', 'property_name'], keep='first')
    return data

def expand_reviews_by_sentence(data):
    data['segment'] = data['review'].str.split(r'\.\s+')
    data = data.explode('segment')
    data['segment'] = data['segment'].str.rstrip('.')
    data = data.dropna(subset=['segment'])
    return data

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")

# Custom stopwords setup, removing typical negations that are significant in sentiment analysis
custom_stopwords = set(stopwords.words('english')) - {"no", "not"}

# Contraction mapping
contraction_mapping = {
    "didn't": "did not",
    "don't": "do not",
    "aren't": "are not",
    "couldn't": "could not",
    # Add more contractions as necessary
}

def preprocess_segments(data):
    def clean_text(text):
        if pd.isna(text):
            return ""

        for contraction, expanded in contraction_mapping.items():
            text = re.sub(r'\b' + contraction + r'\b', expanded, text)

        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in custom_stopwords]

        doc = nlp(' '.join(filtered_tokens))
        lemmatized = [token.lemma_ for token in doc]

        return ' '.join(lemmatized)

    data['cleaned_segment'] = data['segment'].apply(lambda x: clean_text(x))
    return data

categories = {
    "communication": ["communicat", "contact", "responsive", "response", "reply", "communication", "host", "hospitality", "professional", "responsive", "responded", "request", "getting back", "respond", "available", "kind", "customer service", "accommodating", "helpful"],
    "location": ["accessible", "location", "area", "place", "stay" , "accommodation", "spacious", "setting", "walk", "walkability", "parking", "driving", "drive", "biking", "located", "close", "near", "far", "cozy", "local", "neighbourhood", "neighborhood", "across the street", "view", "next to", "market", "cafe"],
    "cleanliness": ["well kept", "clean", "tidy", "hygiene", "dirty", "cleanliness", "gross", "housekeeping", "stain", "spotless", "beach", "crumb", "dirt", "dust", "bug", "wash", "ant", "maintainence", "infest", "spotless", "sanitized", "sanitary"],
    "accuracy": ["accurate", "exact", "description", "expectation", "picture", "catfish", "described", "space", "money", "expensive", "room", "commensurate", "price", "temperature", "comfortable", "uncomfortable", "photo", "mislead", "luxury", "decorated", "decoration", "advertised", "outside", "interiors", "size", "backyard", "yard", "photos", "incorrect", "inform", "wrong", "instructions"],
    "check-in": ["check-in", "arrival", "welcome", "attention", "checkin", "check in", "checked", "entry", "keypad", "code", "key"],
    "amenities": ["towels", "blinds", "coffee machine", "kitchenware", "oven", "utensil", "accommodations", "alarms", "tv", "t.v", "tools", "coffee", "tea", "toilet paper", "laundry", "linen", "toiletries", "facilities", "stocked", "equipped", "heater", "refrigerator", "couch", "fridge", "amenities", "water", "thermostat", "broken", "dish soap", "dishwasher", "dishes", "furnishing", "furnished", "interiors", "ac", "a/c", "a.c.", "air conditioning", "ice maker", "gear", "garbage", "trash", "internet", "blanket", "towel", "shower", "washer", "dryer", "appliances", "decor", "wi-fi", "wifi", "knives", "container", "bowl", "washing machine", "heat", "napkin", "plate", "cup", "pillow", "bed", "sheet", "table", "chair", "glasses", "pool", "pans"]
}

def categorize_segments(data, categories):
    def match_categories(cleaned_segment):
        matched_categories = []
        for category, keywords in categories.items():
            if any(keyword in cleaned_segment for keyword in keywords):
                matched_categories.append(category)
        return matched_categories if matched_categories else ['other']

    expanded_rows = []
    for _, row in data.iterrows():
        cleaned_segment = row['cleaned_segment'] if pd.notna(row['cleaned_segment']) else ''
        matched_cats = match_categories(cleaned_segment)
        for category in matched_cats:
            new_row = row.copy()
            new_row['category'] = category
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

def add_sentiment_scores(data):
    sia = SentimentIntensityAnalyzer()
    data = data[data['cleaned_segment'].notna() & data['cleaned_segment'].str.strip().ne('')]
    data['compound_score'] = data['cleaned_segment'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data

def master_nlp(data):
    data = cleaning(data)
    data = expand_reviews_by_sentence(data)
    data = preprocess_segments(data)
    data = categorize_segments(data, categories)
    data = add_sentiment_scores(data)
    return data

def plot_sentiment_distribution(data):
    sentiment_counts = data['sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('static/sentiment_distribution.png')
    plt.close()

def plot_sentiment_trend_by_category(data):
    data['checkout_date'] = pd.to_datetime(data['checkout_date'])
    data['Year-Month'] = data['checkout_date'].dt.to_period('M')

    category_names = ['amenities', 'location', 'check-in', 'cleanliness', 'accuracy', 'communication']
    plots = {}

    for category in category_names:
        category_data = data[data['category'] == category]

        if category_data['Year-Month'].dtype == object:
            category_data['Year-Month'] = category_data['Year-Month'].astype('period[M]')

        year_month_labels = category_data['Year-Month'].astype(str).tolist()

        category_data['classification'] = pd.cut(category_data['compound_score'], bins=[-float('inf'), -0.05, 0.05, float('inf')], labels=['Negative', 'Neutral', 'Positive'])
        sentiment_counts = category_data.groupby(['Year-Month', 'classification']).size().unstack(fill_value=0)

        all_months = pd.period_range(start=sentiment_counts.index.min(), end=sentiment_counts.index.max(), freq='M')
        sentiment_counts = sentiment_counts.reindex(all_months, fill_value=0)

        fig = go.Figure()

        for sentiment in ['Positive', 'Neutral', 'Negative']:
            fig.add_trace(go.Bar(
                x=sentiment_counts.index.astype(str),
                y=sentiment_counts[sentiment],
                name=sentiment,
                marker_color={'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}[sentiment]
            ))

        fig.update_layout(
            barmode='stack',
            title=f'Sentiment Trend for {category.capitalize()} Over Time',
            xaxis=dict(title='Year-Month', type='category', tickangle=-45),
            yaxis=dict(title='Count of Segments'),
            autosize=True,
            width=1000,
            height=600,
            legend_title_text='Sentiment'
        )

        plots[category] = pio.to_html(fig, full_html=False)

    return plots

def calculate_metrics(data):
    data['checkout_date'] = pd.to_datetime(data['checkout_date'])
    data['Year-Month'] = data['checkout_date'].dt.to_period('M')
    data['checkout_date'] = data['checkout_date'].dt.strftime('%Y-%m-%d')
    
    monthly_metrics = data.groupby(['category', 'Year-Month']).apply(lambda group: pd.Series({
      'Promoters': len(group[group['compound_score'] >= 0.61]),
      'Detractors': len(group[group['compound_score'] <= 0.20]),
      'All Segments': len(group),
      'Neutrals': len(group) - len(group[group['compound_score'] >= 0.61]) - len(group[group['compound_score'] <= 0.20]),
      'NPS-like Metric': (len(group[group['compound_score'] >= 0.61]) - len(group[group['compound_score'] <= 0.20])) / len(group)
      })).reset_index()

    return monthly_metrics

def plot_trend(data):
    category_names = ['amenities', 'location', 'check-in', 'cleanliness', 'accuracy', 'communication']
    plots = {}
    for category in category_names:
        category_data = data[data['category'] == category]

        if category_data['Year-Month'].dtype == object:
            category_data['Year-Month'] = pd.to_datetime(category_data['Year-Month'], format='%Y-%m')

        year_month_labels = category_data['Year-Month'].dt.strftime('%Y-%m').tolist()
        nps_values = category_data['NPS-like Metric']

        fig = go.Figure()
        fig.add_trace(go.Bar(x=year_month_labels, y=nps_values, name='NPS Metric'))
        fig.add_trace(go.Scatter(x=year_month_labels, y=nps_values, mode='lines+markers', name='Trend Line', line=dict(color='red')))

        fig.update_layout(
            title='{} NPS Metric Trend'.format(category.capitalize()),
            xaxis_title='Year-Month',
            yaxis_title='NPS-like Metric',
            xaxis=dict(type='category', tickangle=-45),
            autosize=False,
            width=1800,
            height=600,
            margin=dict(l=50, r=50, b=100, t=100, pad=4))
        
        plots[category] = pio.to_html(fig, full_html=False)

    return plots
