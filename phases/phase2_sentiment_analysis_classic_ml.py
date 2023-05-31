import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class TextPreprocessor:

    def __init__(self, source_file, target_file, cluster_count=3):
        self.source_file = source_file
        self.target_file = target_file
        self.cluster_count = cluster_count

    @staticmethod
    def sanitize_text(raw_text):
        raw_text = raw_text.lower()
        raw_text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r'\@\w+|\#', '', raw_text)
        raw_text = re.sub(r'[^a-zA-Z\s]', '', raw_text)
        raw_text = ' '.join([token for token in raw_text.split() if token not in stopwords.words('english')])
        return raw_text

    @staticmethod
    def mean_vector(tokens, vector_map):
        vectors = [vector_map[token] for token in tokens if token in vector_map]
        if not vectors:
            return None
        return normalize(np.sum(vectors).reshape(1, -1))

    def run_preprocessing(self):
        # Loading dataset
        dataset = pd.read_csv(self.source_file, sep='\t')

        # Clean and preprocess data
        dataset['processed_title'] = dataset['title'].apply(lambda x: self.sanitize_text(x))

        # Tokenization
        dataset['tokenized_title'] = dataset['processed_title'].apply(lambda x: x.split())

        # Word embeddings via pre-trained model (Word2Vec)
        word2vec_model = Word2Vec(dataset['tokenized_title'].tolist(), vector_size=100, window=5, min_count=1, workers=4)

        # Fetch vector representation for each word in the title
        vectors = word2vec_model.wv

        # Compute average vector for each title
        dataset['avg_vector'] = dataset['tokenized_title'].apply(
            lambda x: self.mean_vector(x, vectors)
            .tolist() if self.mean_vector(x, vectors) is not None else None)
        dataset = dataset.dropna(subset=['avg_vector'])

        # K-Means clustering on the embeddings
        cluster_input = np.vstack(dataset['avg_vector'].values)
        k_means = KMeans(n_clusters=self.cluster_count, random_state=0).fit(cluster_input)

        # Assigning labels to clusters
        dataset['cluster'] = k_means.labels_

        # Write preprocessed data to file
        dataset.to_csv(self.target_file, index=False, lineterminator='\n', float_format='%.8f', header=True, sep='\t')
