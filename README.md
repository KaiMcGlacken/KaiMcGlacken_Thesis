The functions used for gathering and analysing data are in the file methods.py. These were run and accessed through the program main.py.

The game data used in these programs could not be uploaded because of the file sizes. However, the details of how the data was collected can be read in the thesis pdf file under the Methodology section.

To collect the data, a valid Lichess API token is required.

To calculate the average error and coincidence values, a version of Stockfish needs to be downloaded and used. The version used in this project was Stockfish version 15.1.

The following functions were used for collecting the non-cheat game data from the Lichess website: 
retrieve_games() retrieve_usernames() retrieve_single_game_from_users()

The following function was used to gather cheat games from the Lichess website: 
retrieve_bot_games()

The following function was used to convert ECO sequences to FEN format: 
eco_to_fen()

The following functions were used to convert pgn games to matrices: 
format_board_position() position_to_index() pgn_to_list() convert_game_to_matrix()

The following functions were used for traininf and testing the convolutional neural network: 
format_and_label_games() train_cnn_with_stats() train_cnn_no_stats()

The following function was used for calculating the average error and coincidence values: 
calculate_ae_and_cv()

The following function was used to train and test the logistic regression model:
train_test_lr
