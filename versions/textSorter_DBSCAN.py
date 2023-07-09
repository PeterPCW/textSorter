import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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
        return similarity * 10
    else:
        return similarity

def line_similarity(line1, line2):
    similarity = 0.0

    # Calculate similarity for the tokenized line as a whole
    line_similarity = fuzz.ratio(line1, line2) / 100.0
    similarity += line_similarity

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
        word_similarities.append(word_similarities_line)

    # Calculate the combined similarity score
    if any(word_similarities):
        word_similarity_sum = sum([max(word_similarities[i]) for i in range(len(word_similarities))])
        similarity += word_similarity_sum
    
    return similarity

def sort_text(input_text):
    # Preprocess the input text
    preprocessed_lines = preprocess_text(input_text)

    samples = len(preprocessed_lines)**2
    progress_bar = tqdm(total=samples, unit='sample')

    # Calculate pairwise line similarities
    similarity_matrix = np.zeros((len(preprocessed_lines), len(preprocessed_lines)))
    for i, line1 in enumerate(preprocessed_lines):
        for j, line2 in enumerate(preprocessed_lines):
            similarity_matrix[i, j] = line_similarity(line1, line2)
            progress_bar.update(1)

    progress_bar.close()

    # Filter similarity matrix
    similarity_matrix = np.clip(similarity_matrix, None, 40)

    while True:
        # Prompt the user for the epsilon (eps) value
        eps_input = input("Enter the epsilon (eps) value (or 'q' to quit). This will determine the number of clusters (higher means fewer clusters): ")
        if eps_input == 'q':
            break

        eps = float(eps_input)

        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_labels = dbscan.fit_predict(similarity_matrix)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Number of Clusters: {num_clusters}")
        
        user_choice = input("Does that seem right? If not then you will be prompted for a new eps value. (y/n)")
        
        if user_choice.lower() in ['y', 'yes']:
            break

    # Sort the cluster labels
    sorted_indices = np.argsort(cluster_labels)

    # Retrieve the items from each cluster and concatenate them into a sorted output
    sorted_output = ''
    for idx in sorted_indices:
        sorted_output += input_text.split('\n')[idx] + '\n'

    return sorted_output, num_clusters
