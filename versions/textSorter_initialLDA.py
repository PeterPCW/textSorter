import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from fuzzywuzzy import fuzz

def preprocess_text(text):
    # Tokenize the text into lines
    lines = text.split('\n')

    # Remove stopwords and punctuation, and apply stemming for each line
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    preprocessed_lines = []
    for line in lines:
        words = word_tokenize(line)
        preprocessed_words = [stemmer.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        preprocessed_lines.append(' '.join(preprocessed_words))

    return preprocessed_lines

def word_similarity(word1, word2):
    similarity = fuzz.ratio(word1, word2) / 100.0
    if word1 == word2:
        return similarity * 5
    else:
        return similarity

def sort_text(input_text):
    # Preprocess the input text
    preprocessed_lines = preprocess_text(input_text)

    # Convert the preprocessed lines into numerical representations using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_lines)

    # Apply Latent Dirichlet Allocation (LDA) to identify topics
    num_topics = min(10, len(preprocessed_lines))  # Adjust the number of topics based on the number of lines/documents
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    # Extract the most probable topic for each line
    topic_assignments = lda.transform(tfidf_matrix).argmax(axis=1)

    # Calculate pairwise similarities between words in each line
    similarity_matrix = np.zeros((len(preprocessed_lines), len(preprocessed_lines)))
    for i, line1 in enumerate(preprocessed_lines):
        for j, line2 in enumerate(preprocessed_lines):
            similarity = 0.0
            words1 = line1.split()
            words2 = line2.split()
            for word1 in words1:
                for word2 in words2:
                    similarity += word_similarity(word1, word2)
            similarity_matrix[i, j] = similarity

    # Sort the text items based on their topics, word similarity, and line length
    sorted_indices = []
    for topic_idx in range(num_topics):
        topic_indices = np.where(topic_assignments == topic_idx)[0]
        topic_lines = [preprocessed_lines[i] for i in topic_indices]

        topic_sorted_indices = sorted(range(len(topic_lines)), key=lambda x: (similarity_matrix[topic_indices[x]].sum(), -len(topic_lines[x].split()), x), reverse=True)
        sorted_indices.extend(topic_indices[topic_sorted_indices])

    # Retrieve the items from each cluster and concatenate them into a sorted output
    sorted_output = ''
    for idx in sorted_indices:
        sorted_output += input_text.split('\n')[idx] + '\n'

    return sorted_output, num_topics
