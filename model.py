import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Load Netflix dataset
netflix = pd.read_csv('netflix_titles/netflix_titles.csv')

# Display dataset info
netflix.head()
netflix.shape
netflix.count()

# Split data into Movies and TV Shows
netflix_shows = netflix[netflix['type'] == 'TV Show']
netflix_movies = netflix[netflix['type'] == 'Movie']

# Plot distribution of types
sns.set(style='darkgrid')
ax = sns.countplot(x='type', data=netflix, palette='Set2')

plt.figure(figsize=(12, 10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix_movies, palette="Set2",
                   order=netflix_movies['rating'].value_counts().index[0:15])

# Load IMDb data
imdb_ratings = pd.read_csv('IMDb ratings/IMDb ratings.csv', usecols=['weighted_average_vote'])
imdb_titles = pd.read_csv('IMDb movies/IMDb movies.csv', usecols=['title', 'year', 'genre'])

# Process IMDb data
ratings = pd.DataFrame({'Title': imdb_titles.title,
                        'Release Year': imdb_titles.year,
                        'Rating': imdb_ratings.weighted_average_vote,
                        'Genre': imdb_titles.genre})

ratings.drop_duplicates(subset=['Title', 'Release Year', 'Rating'], inplace=True)
ratings.shape
ratings.dropna()

# Merge IMDb and Netflix datasets
joint_data = ratings.merge(netflix, left_on='Title', right_on='title', how='inner')
joint_data = joint_data.sort_values(by='Rating', ascending=False)

# Visualize top-rated shows
import plotly.express as px

top_rated = joint_data[0:10]
fig = px.sunburst(top_rated, path=['title', 'country'], values='Rating', color='Rating')
fig.show()

plt.figure(figsize=(12, 10))
sns.set(style='darkgrid')
ax = sns.countplot(y='release_year', data=netflix_movies, palette='Set2',
                   order=netflix_movies['release_year'].value_counts().index[0:15])

# Process country data
countries = {}
netflix_movies['country'] = netflix_movies['country'].fillna('Unknown')
cou = list(netflix_movies['country'])

for i in cou:
    i = list(i.split(','))
    if len(i) == 1:
        if i[0] in countries:
            countries[i[0]] += 1
        else:
            countries[i[0]] = 1
    else:
        for j in i:
            if j in countries:
                countries[j] += 1
            else:
                countries[j] = 1

countries_fin = {k.strip(): v for k, v in sorted(countries.items(), key=lambda item: item[1], reverse=True)}

# Plot top 10 countries
plt.figure(figsize=(8, 8))
ax = sns.barplot(x=list(countries_fin.keys())[0:10], y=list(countries_fin.values())[0:10])
ax.set_xticklabels(list(countries_fin.keys())[0:10], rotation=90)

# Process movie durations
netflix_movies['duration'].isnull().sum()
netflix_movies['duration'] = netflix_movies['duration'].str.replace(' min', '')
netflix_movies['duration'] = netflix_movies['duration'].astype(float)

sns.set(style='darkgrid')
sns.kdeplot(netflix_movies['duration'], shade=True)

# Process genres
from collections import Counter

genres = list(netflix_movies['listed_in'])
gen = []

for i in genres:
    i = list(i.split(','))
    for j in i:
        gen.append(j.strip())

g = Counter(gen)

# Generate WordCloud for genres
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

text = list(set(gen))
plt.rcParams['figure.figsize'] = (13, 13)

mask = np.array(Image.open('star.png'))
wordcloud = WordCloud(max_words=1000000, background_color="white", mask=mask).generate(str(text))

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Plot genre distribution
g = {k: v for k, v in sorted(g.items(), key=lambda item: item[1], reverse=True)}

fig, ax = plt.subplots()
fig = plt.figure(figsize=(14, 10))

x = list(g.keys())
y = list(g.values())

ax.vlines(x, ymin=0, ymax=y, color='green')
ax.plot(x, y, "o", color='maroon')
ax.set_xticklabels(x, rotation=90)
ax.set_ylabel("Count of movies")
ax.set_title("Genres")

# TF-IDF Vectorization for content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
netflix['description'] = netflix['description'].fillna('')

tfidf_matrix = tfidf.fit_transform(netflix['description'])
tfidf_matrix.shape

# Compute cosine similarity
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
indices = pd.Series(netflix.index, index=netflix['title']).drop_duplicates()

def get_recommendation(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix['title'].iloc[movie_indices]

get_recommendation('Peaky Blinders')
get_recommendation('Narcos')

# Content-based filtering on multiple metrics
filledna = netflix.fillna('')

def clean_data(x):
    return str.lower(x.replace(" ", ""))

features = ['title', 'director', 'cast', 'listed_in', 'description']
filledna = filledna[features]

for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)

filledna['soup'] = filledna.apply(lambda x: x['title'] + ' ' + x['director'] + ' ' + x['cast'] + ' ' + x['listed_in'] + ' ' + x['description'], axis=1)

# Count Vectorizer and Cosine Similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

filledna = filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])

def get_recommendation_new(title, cosine_sim=cosine_sim2):
    title = title.replace(' ', '').lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix['title'].iloc[movie_indices]

get_recommendation_new('3 Idiots', cosine_sim2)

# Netflix shows/movies based on books
books = pd.read_csv('books/books.csv')
books['original_title'] = books['original_title'].str.lower()

x = netflix
x['title'] = x['title'].str.lower()
netflix_books = x.merge(books, left_on='title', right_on='original_title', how='inner')

# Plot Netflix shows from books
import plotly.graph_objects as go

labels = ['Shows from books', 'Shows not from books']
values = [248, 6234]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
