from .train import calculate_class_probability
from .data_utils import load_data

# Test logic

def test_naive_bayes_model(test_data_file, label_features, prior_probabilities):
    """
    Test the Naive Bayes model on a test dataset.

    Args:
    test_data_file (str): Path to the test data file.
    label_features (dict): Dictionary containing likelihoods of features for each class.
    prior_probabilities (dict): Dictionary containing prior probabilities of each class.

    Returns:
    float: Accuracy of the model on the test dataset.
    """
    correct_predictions = 0
    test_data = load_data(test_data_file)
    for _, row in test_data.iterrows():
        predict_statement = row['text']
        predicted_label = calculate_class_probability(predict_statement, label_features, prior_probabilities)
        if predicted_label == row['label']:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy
