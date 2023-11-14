import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Current Working Directory of a process (i.e., cwd)
DIR_PATH = os.getcwd()
FILE_PATH = os.path.join(DIR_PATH, "india_news_headlines.csv")

df = pd.read_csv(FILE_PATH, parse_dates=[0])

# Data preprocessing
def data_cleaning(df):
    df1 = df.replace(to_replace=['unknown', 'entertainment.hindi.bollywood', 'entertainment.english.hollywood'], value=np.nan)
    df1 = df1.dropna().reset_index(drop=True)
    df1 = df1.assign(year=df['publish_date'].dt.year)
    df1 = df1.reindex(columns=['year', 'publish_date', 'headline_category', 'headline_text', 'sentiment', 'score'])
    df1 = df1.groupby('year').head(25).reset_index(drop=True)
    return df1

df1 = data_cleaning(df)

# NLP prep
def sentiment_score(df):
    sentiment = []
    score = []

    sent_tok = df['headline_text'].apply(lambda x: sent_tokenize(x))
    sia = SentimentIntensityAnalyzer()

    for sentences in sent_tok:
        scores = [sia.polarity_scores(sentence) for sentence in sentences]
        score.append(scores)
        sentiment.append(['Positive' if s['pos'] > s['neg'] else 'Neutral' if s['neu'] == 1.0 else 'Negative' for s in scores])

    data = df.assign(sentiment=sentiment, score=score)
    return data

data = sentiment_score(df1)

# Data visualization
def mat_data(df, sentiment):
    df2 = df.drop(columns=['publish_date', 'headline_category', 'headline_text'], axis=1)
    df2 = df2.set_index('year')
    
    years = list(df2.index.unique())
    categories = ['positive', 'negative', 'neutral']
 
    positive = []
    negative = []
    neutral = []
    
    for index, row in df2.iterrows():
        year = index
        sentiment = row['sentiment'][0]  # Assuming sentiment is a list with one element

        if sentiment == 'Positive':
            positive.append((year, 1))
        elif sentiment == 'Negative':
            negative.append((year, 1))
        elif sentiment == 'Neutral':
            neutral.append((year, 1))
        
    return categories, positive, negative, neutral
    
categories, positive, negative, neutral = mat_data(data, data['sentiment'])

def get_tot(a, b, c):
    res = []
    
    for data, label in [(a, 'positive'), (b, 'neutral'), (c, 'negative')]:
        d = [x for (x, y) in data]
        for item in set(d):
            count = d.count(item)
            res.append((item, count, label))

    tot_lst = pd.DataFrame(res, columns=['year', 'count', 'label'])
    tot_df = tot_lst.pivot(index='year', columns='label', values='count').reset_index()
    tot_df.columns.name = None
    tot_df.columns = ['year', 'Positive', 'Neutral', 'Negative']
    #tot_df = tot_df.set_index('year')
    return tot_df

tot_df = get_tot(positive, neutral, negative)

stacked = tot_df.plot.barh('year', stacked=True, color=['#457b9d','#ced4da', '#f02d3a'], width=0.7, figsize=(10, 10))
stacked.axes.invert_yaxis()
stacked.axes.legend(ncols=len(categories), bbox_to_anchor=(0.5, 1.075), loc='upper center', fancybox=True, shadow=True)
plt.ylabel('Year')
plt.xlabel('Sentiment Score')


plt.show()