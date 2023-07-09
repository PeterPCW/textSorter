import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def preprocess_text(text):
    # Replace hyphens with spaces
    text = re.sub(r'-', ' ', text)
    
    # Tokenize the text into lines
    lines = text.split('\n')

    # Remove stopwords and punctuation, and apply stemming for each line
    stop_words = set(stopwords.words('english'))
    preprocessed_lines = []
    for line in lines:
        words = word_tokenize(line)
        preprocessed_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        preprocessed_lines.append(' '.join(preprocessed_words))

    return preprocessed_lines

def word_similarity(word1, word2):
    similarity = fuzz.ratio(word1, word2) / 25.0
    if word1 == word2:
        return similarity * 5
    else:
        return similarity

def line_similarity(line1, line2):
    similarity = 0.0

    # Calculate similarity for the tokenized line as a whole
    line_similarity = fuzz.ratio(line1, line2) / 100.0
    similarity += line_similarity
    #print(f"  Line Similarity: {line_similarity}")

    # Tokenize individual non-stop words
    words1 = [word for word in word_tokenize(line1) if word.lower() not in stopwords.words('english')]
    words2 = [word for word in word_tokenize(line2) if word.lower() not in stopwords.words('english')]

    # Calculate similarity for individual words
    word_similarities = []
    for word1 in words1:
        word_similarities_line = []
        for word2 in words2:
            each_similarity = word_similarity(word1, word2)
            word_similarities_line.append(each_similarity)
            #print(f"    Word Similarity ({word1}, {word2}): {each_similarity}")
        word_similarities.append(word_similarities_line)

    # Calculate the combined similarity score
    num_comparisons = (len(words1) + len(words2)) / 2
    if num_comparisons > 0 and any(word_similarities):
        #similarity /= num_comparisons
        #print(f"    Line Similarity {similarity}")
        word_similarity_sum = sum([max(word_similarities[i]) for i in range(len(word_similarities))])
        #word_similarity_sum /= num_comparisons
        #print(f"    Total Word Similarity {word_similarity_sum}")
        similarity += word_similarity_sum
        #print(f" Similarity: {similarity}")

    return similarity

def determine_elbow_index(inertia_values):
    # Calculate the second differences
    differences = np.diff(inertia_values, 2)

    # Find the index of the maximum second difference
    elbow_index = np.argmax(differences) + 1

    return elbow_index

def sort_text(input_text):
    # Preprocess the input text
    preprocessed_lines = preprocess_text(input_text)

    # Calculate pairwise line similarities
    similarity_matrix = np.zeros((len(preprocessed_lines), len(preprocessed_lines)))
    for i, line1 in enumerate(preprocessed_lines):
        for j, line2 in enumerate(preprocessed_lines):
            similarity_matrix[i, j] = line_similarity(line1, line2)

    # Determine the optimal number of clusters using the elbow method
    inertias = []
    max_clusters = min(len(preprocessed_lines), len(preprocessed_lines))  # Set the maximum number of clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(similarity_matrix)
        inertias.append(kmeans.inertia_)

    # Find the number of clusters based on the elbow method
    elbow_index = determine_elbow_index(inertias)
    num_clusters = elbow_index + 1

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = clustering.fit_predict(similarity_matrix)
    print(f"Cluster Labels: {cluster_labels}")

    # Sort the cluster labels
    sorted_clusters = np.argsort(cluster_labels)

    # Retrieve the items from each cluster and concatenate them into a sorted output
    sorted_output = ''
    for cluster_id in sorted_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        for idx in cluster_indices:
            sorted_output += input_text.split('\n')[idx] + '\n'

    return sorted_output, num_clusters
