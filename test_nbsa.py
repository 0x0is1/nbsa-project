from nbsa import train_naive_bayes_model, test_naive_bayes_model

# Training
train_data_file = "datasets/training.csv"
label_features, prior_probabilities = train_naive_bayes_model(train_data_file)

# Testing
test_data_file = "datasets/test.csv"
accuracy = test_naive_bayes_model(test_data_file, label_features, prior_probabilities)
print("Accuracy:", accuracy)
