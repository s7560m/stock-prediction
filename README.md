# STOCK PREDICTOR
This is a stock prediction neural network built with python, tensorflow (keras), pandas, and numpy, which predicts a stock price based on an inputted date (day, month, year) to roughly a 90-95% accuracy.
The neural network's feature layer has 3 inputs (for the dates), two hidden layers with 64 units each, along with some regularization to prevent overfitting our data, and a final output layer which is a single node.
The data is standardized using typical Z-score standardization for both the featuers and labels, and is currently trained on a dataset of AAPL stock's prices.
