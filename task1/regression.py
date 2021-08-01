################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
import pickle

import pandas as pd
import test_data_processing
import numpy as np

our_features = ['budget', 'vote_count', 'runtime', 'Year', 'Friday', 'Monday', 'Saturday',
                'Sunday', 'Thursday', 'Tuesday', 'Wednesday', '20th Century Fox',
                'Columbia Pictures', 'Metro', 'New Line Cinema', 'Paramount',
                'Universal Pictures', 'Warner Bros', 'Action', 'Adventure', 'Animation',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller',
                'War', 'en', 'holiday_month', 'based on novel or book', 'murder', 'woman director']


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    test_data = pd.read_csv(csv_file)
    processed_test_data = test_data_processing.run_process(test_data)
    processed_test_data = processed_test_data[our_features]
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    rev = pickle_model.predict(processed_test_data)
    vote = np.zeros(len(processed_test_data)) + 6.3  # this is an unbised estimator
    return list(rev), list(vote)


if __name__ == "__main__":
    a, b = predict("test_f.csv")
