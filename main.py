import time
from methods import (pgn_to_list, 
                     format_board_position, 
                     retrieve_bot_games,
                     convert_game_to_matrix,
                     calculate_ae_and_cv,
                     format_and_label_games,
                     train_cnn_with_stats,
                     train_cnn_no_stats)
import pprint as pp
from io import StringIO
import pandas as pd
from stockfish import Stockfish
# also download stockfish 15.1 
from lichess.format import SINGLE_PGN
import numpy as np




def main():
    
    start = time.time()
    
    print('Running Program...')
    
    randomised_games, randomised_game_data, randomised_labels = format_and_label_games()
    
    train_cnn_no_stats(randomised_games, randomised_labels)
            
    
    print('##########')
    print(time.time() - start)
    print('##########')

    print('Program Ended')
    

if __name__=='__main__':
    main()








