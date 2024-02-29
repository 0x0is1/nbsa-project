from .data_utils import load_data
from nltk.tokenize import word_tokenize
from collections import Counter

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to filter data by label
def filter_data_by_label(data, label):
    """
    Filter dataset based on a specific label.

    Args:
    data (DataFrame): Input dataset.
    label (str): Label to filter the dataset.

    Returns:
    DataFrame: Filtered dataset.
    """
    return data.loc[data['label'] == label]

# Training logic

def calculate_feature_likelihoods(data):
    """
    Calculate likelihoods of features given each class.

    Args:
    data (DataFrame): Training dataset.

    Returns:
    dict: Dictionary containing likelihoods of features for each class.
    """
    likelihoods = {}
    for class_label in data['label'].unique():
        class_data = filter_data_by_label(data, class_label)
        class_text = ' '.join(class_data['text'])
        tokens = word_tokenize(class_text)
        likelihoods[class_label] = dict(Counter(tokens))
    return likelihoods

def calculate_prior_probabilities(data):
    """
    Calculate prior probabilities of each class.

    Args:
    data (DataFrame): Training dataset.

    Returns:
    dict: Dictionary containing prior probabilities of each class.
    """
    total_samples = len(data)
    priors = {}
    for class_label in data['label'].unique():
        class_data = filter_data_by_label(data, class_label)
        priors[class_label] = len(class_data) / total_samples
    return priors

def train_naive_bayes_model(data_file):
    """
    Train Naive Bayes model.

    Args:
    data_file (str): Path to the training data file.

    Returns:
    tuple: Tuple containing likelihoods of features and prior probabilities.
    """
    data = load_data(data_file)
    likelihoods = calculate_feature_likelihoods(data)  # Features calculated as P(feature|class) using counts
    priors = calculate_prior_probabilities(data)  # Prior probabilities P(class)
    return likelihoods, priors

# Prediction logic

def calculate_class_probability(predict_statement, likelihoods, priors):
    """
    Calculate the probability of a statement belonging to each class.

    Args:
    predict_statement (str): Statement to predict the class for.
    likelihoods (dict): Dictionary containing likelihoods of features for each class.
    priors (dict): Dictionary containing prior probabilities of each class.

    Returns:
    str: Predicted class label.
    """
    tokens = word_tokenize(predict_statement)
    statement_probability = {}
    for class_label, class_likelihoods in likelihoods.items():
        probability = priors[class_label]  # Initialize with prior probability P(class)
        for token in tokens:
            # Update probability using Naive Bayes formula: 
            ## P(class|features) = P(class) * Π P(feature|class)
            probability *= class_likelihoods.get(token, 0) + 1  # Laplace smoothing applied
        statement_probability[class_label] = probability
    return max(statement_probability, key=statement_probability.get)  # Predict label with maximum probability
