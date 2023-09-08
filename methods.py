import time
from key import token # personal key for lichess API
import tqdm
import pickle
import numpy
import lichess.api
from lichess.format import SINGLE_PGN
import chess
import io
import chess.pgn
import tensorflow as tf
import pandas as pd
from stockfish import Stockfish
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np


# function for calculating the ae and cv of each move in each game
def calculate_ae_and_cv(filename):
    game_list = pgn_to_list(filename)

    with open('fen_openings.pkl', 'rb') as f:
            openings_list = pickle.load(f)

    stockfish = Stockfish(path="stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe", 
                            depth=20, 
                            parameters={"Threads": 7, 
                                        "Hash": 8192, 
                                        "Minimum Thinking Time": 20})  

    df = pd.DataFrame(columns =  ['Game_Index',
                                'white_user',
                                'white_elo', 
                                'white_cv', 
                                'white_ae',
                                'black_user',
                                'black_elo', 
                                'black_cv', 
                                'black_ae',
                                'game_moves'])

    game_index = 0

    #iterate over each game in game_list
    for game in tqdm.tqdm(game_list):
        
        game_moves = str(game.mainline_moves())
        
        if game_index > -1:   
            print(game_index)
            
            str_game = str(game)

            # retrieve white rating
            white_elo = str_game.split("[WhiteElo \"",1)[1]
            try:
                white_elo = int(white_elo[:4])
            except ValueError:
                try:
                    white_elo = int(white_elo[:3])
                except ValueError:
                    white_elo = white_elo[:1]

            # retrieve black rating
            black_elo = str_game.split("[BlackElo \"",1)[1]
            try:
                black_elo = int(black_elo[:4])
            except ValueError:
                try:
                    black_elo = int(black_elo[:3])
                except ValueError:
                    black_elo = black_elo[:1]

            # retrieve white username
            white_username = str_game.split("[White \"",1)[1]
            white_username, sep, tail = white_username.partition('\"')

            # retrieve black username
            black_username = str_game.split("[Black \"",1)[1]
            black_username, sep, tail = black_username.partition('\"')
            
            board = chess.Board()
            white_top_moves = []
            black_top_moves = []
            top_moves = []
            count = 1
            opening_flag = True
            white_tm_diff = []
            black_tm_diff = []
            
            # iterate over moves in the game
            for move in game.mainline_moves():
                str_move = str(move)
                
                # check if new board position has been reached
                if board.fen() not in openings_list:
                        opening_flag = False
                            
                if opening_flag == False:
                    stockfish.set_fen_position(board.fen())
                    top_moves = stockfish.get_top_moves(3)
                    top_move_list = []

                    # CHECK IF PLAYER HAS CHOSEN TOP THREE ENGINE MOVE
                    for top_move in top_moves:
                            top_move_list.append(top_move['Move'])
                            
                    # if whites move
                    if count % 2 != 0:
                        if str_move in top_move_list:
                            white_top_moves.append(True) 
                        else:
                            white_top_moves.append(False)
                            
                    #if blacks move     
                    else:
                        if str_move in top_move_list:
                            black_top_moves.append(True)  
                        else:
                            black_top_moves.append(False)
                            
                    
                    # CALCULATE AVERAGE ERROR
                    # if the played move is the same as the top move, the tm_diff is 0
                    if str_move == top_moves[0]['Move']:
                        tm_diff = 0
                        # if whites move
                        if count % 2 != 0:
                            white_tm_diff.append(tm_diff)
                        #if blacks move 
                        else:
                            black_tm_diff.append(tm_diff)
                            
                    # otherwise calculate the difference in eval       
                    else:
                        #play best move and evaluate position
                        stockfish.make_moves_from_current_position([top_moves[0]['Move']])
                        top_move_eval = stockfish.get_evaluation()
                        
                        #reset to previous move, play the played move and evaluate position
                        stockfish.set_fen_position(board.fen())
                        stockfish.make_moves_from_current_position([str_move])
                        played_move_evaluation = stockfish.get_evaluation()
                        
                        
                        # if top move is not a mating sequence, calculate centipawn difference 
                        if top_move_eval['type'] != 'mate' and played_move_evaluation['type'] != 'mate':
                            
                            # if whites move
                            if count % 2 != 0:
                                # if top move evaluation is less than the played move
                                if top_move_eval['value'] <= played_move_evaluation['value']:
                                    tm_diff = 0
                                    white_tm_diff.append(tm_diff)
                                else:
                                    # calculate difference between the two
                                    tm_diff = top_move_eval['value'] - played_move_evaluation['value']
                                    # if tm_diff is negative, make positive
                                    if tm_diff < 0:
                                        tm_diff = tm_diff * -1
                                    white_tm_diff.append(tm_diff)
                            
                            # if blacks move
                            else:
                                # if top move evaluation is greater than the played move
                                if top_move_eval['value'] >= played_move_evaluation['value']:
                                    tm_diff = 0
                                    black_tm_diff.append(tm_diff)
                                else:
                                    # calculate difference between the two
                                    tm_diff = top_move_eval['value'] - played_move_evaluation['value']
                                    # if tm_diff is negative, make positive
                                    if tm_diff < 0:
                                        tm_diff = tm_diff * -1
                                    black_tm_diff.append(tm_diff)
                    
                    count += 1              
                            
                #ADD ONLY GAMES WITH MIN 60 MOVES
                if count == 61:
                    white_cv = (white_top_moves.count(True) / len(white_top_moves)) * 100
                    black_cv = (black_top_moves.count(True) / len(black_top_moves)) * 100
                    
                    white_ae =  sum(white_tm_diff) / len(white_tm_diff)
                    black_ae =  sum(black_tm_diff) / len(black_tm_diff)
                    
                    print(len(white_top_moves))
                    print(len(black_top_moves))
                    
                    print(len(white_tm_diff))
                    print(len(black_tm_diff))
                    
                    df = df.append({'Game_Index': game_index,
                                    'white_user': white_username,
                                    'white_elo': white_elo, 
                                    'white_cv': white_cv, 
                                    'white_ae': white_ae,
                                    'black_user': black_username,
                                    'black_elo': black_elo, 
                                    'black_cv': black_cv, 
                                    'black_ae': black_ae,
                                    'game_moves': game_moves}, ignore_index=True)
                    
                    print(df) 
                
                    df.to_pickle('E:/Thesis_Programs/testing_df.pkl')
                    
                    break
                
                board.push(move) 
    
        game_index += 1


# retrieve cheat games from Lichess website
def retrieve_bot_games():
    bot_names = ['ResoluteBot', 
                 'Mate-AI', 
                 'ToromBot', 
                 'ArasanX', 
                 'ReinforcementTrial', 
                 'Shineshou90_BOT',
                 'BRAINLEARN11_NNUE',
                 'blundar_bot',
                 'LazyBot',
                 'TheMatrix2029',
                 'faith_bot',
                 'ShashChess_MsP',
                 'EmanPureNN',
                 'NikitosikVariantsbot',
                 'TorBot',
                 'chessbot2880',
                 'chessfyBOT',
                 'EmptikBot',
                 'ProteusSF-Open',
                 'OldStockfish',
                 'EmptikVariantsBot',
                 'auto8',
                 'MustafaYilmazBot',
                 'MedipolUniversity',
                 'Weiawaga', 
                 'JuroJ', 
                 'Nikitosikbot', 
                 'YoBot_v2', 
                 'RaspFish', 
                 'expositor',
                 'AKS-Mantissa',
                 'FlamingDragon_9000',
                 'fornax-engine',
                 'TuksuBot',
                 'odonata-bot',
                 'misteick_bot',
                 'SaxtonEngine',
                 'pawn_git',
                 'admete_bot',
                 'bot_adario',
                 'ProteusSF-lite',
                 'bekbot',
                 'auto8',
                 'StockDory'
                 ]
    
    for bot in tqdm.tqdm(bot_names):
        print(bot)
        try:
            bot_white_games = lichess.api.user_games(bot, 
                                                    color='white',
                                                    perfType= "ultraBullet,bullet,blitz,rapid,classical",
                                                    auth=token, 
                                                    format=SINGLE_PGN)
            
            with open("bot_white_games.pgn", "a+", encoding="utf-8") as f:
                f.write(bot_white_games)
    
        except:
            time.sleep(60)
            pass 
        
        try:
            bot_black_games = lichess.api.user_games(bot, 
                                                    color='black',
                                                    perfType= "ultraBullet,bullet,blitz,rapid,classical",
                                                    auth=token, 
                                                    format=SINGLE_PGN)
            
            with open("bot_black_games.pgn", "a+", encoding="utf-8") as f:
                f.write(bot_black_games)
    
        except:
            time.sleep(60)
            pass  
        
        time.sleep(60)


#retrieve games from the following 6 users
def retrieve_games():
    usernames = ['kmcglacken', 'ronancorless', 'Luke51278', 'chiere', 'MrKeanu', 'german11']
        
    for user in tqdm.tqdm(usernames):
        print(user)
        try:
            game = lichess.api.user_games(user, auth=token, format=SINGLE_PGN)
            
            with open("games.pgn", "a+", encoding="utf-8") as f:
                f.write(game)
    
        except:
            time.sleep(1)
            pass  


#retrieve all usernames from games.pgn file
def retrieve_usernames(content_list):

    usernames = []

    for i in content_list:
        if 'White' in i and 'WhiteElo' not in i and 'WhiteRatingDiff' not in i:
            usernames.append(i[8:-3])
            
        elif 'Black' in i and 'BlackElo' not in i and 'BlackRatingDiff' not in i:
            usernames.append(i[8:-3])

    filtered_usernames = [*set(usernames)]
    
    return filtered_usernames


#retrieve one game from each user in usernames_list.pkl and write to user_games.pgn
def retrieve_single_game_from_users():
    with open('usernames_list.pkl', 'rb') as f:
        usernames_list = pickle.load(f)
    
    count = 0
    no_games = 0
    
    for user in tqdm.tqdm(usernames_list):
        print(user)
        print(no_games)
        try:
            game = lichess.api.user_games(user, 
                                          max=1, 
                                          auth=token, 
                                          perfType= "ultraBullet,bullet,blitz,rapid,classical", 
                                          format=SINGLE_PGN)
            
            with open("user_games.pgn", "a+", encoding="utf-8") as f:
                f.write(game)
                
            count += 1
            no_games += 1
            
            time.sleep(1)
            
            if count == 29:
                time.sleep(60)
                count = 0
        except:
            time.sleep(1)
            pass


#convert eco openings to fen positions
def eco_to_fen():
    with open('output.txt', 'r') as f:
        move_sequences = [line.strip() for line in f]
    
    fen_list = []
    board = chess.Board()
    fen = board.fen()
    fen_list.append(fen)

    for sequence in move_sequences:
        board = chess.Board()
        pgn = io.StringIO(sequence)
        game = chess.pgn.read_game(pgn)

        for move in game.mainline_moves():
            board.push(move)
            fen = board.fen()
            if fen not in fen_list:
                fen_list.append(fen)

    with open('fen_openings.pkl', 'wb') as f:
        pickle.dump(fen_list, f)


# realnihal (2021) https://github.com/realnihal/Chess-AI-with-TensorFlow/blob/main/Chess_AI.ipynb
# convert each position in a game to a matrix
def format_board_position(board):
    
    #initialise an array of zeros
    board_matrix = numpy.zeros((14, 8, 8), dtype=numpy.int8)
    
    # filling the matrix depending on the position of each piece on the board
    for piece in chess.PIECE_TYPES:
        #filling the matrix for each white piece
        for position in board.pieces(piece, chess.WHITE):
            i = numpy.unravel_index(position, (8, 8))
            board_matrix[piece - 1][7 - i[0]][i[1]] = 1
        #filling the matrix for each black piece
        for position in board.pieces(piece, chess.BLACK):
            i = numpy.unravel_index(position, (8, 8))
            board_matrix[piece + 5][7 - i[0]][i[1]] = 1

    # filling the matrix for the position of all pieces on each side
    for piece in chess.PIECE_TYPES:
        # for the white pieces
        for position in board.pieces(piece, chess.WHITE):
            a, b = position_to_index(position)
            board_matrix[12][a][b] = 1
        # for the black pieces
        for position in board.pieces(piece, chess.BLACK):
            a, b = position_to_index(position)
            board_matrix[13][a][b] = 1
        
    return board_matrix


# realnihal (2021) https://github.com/realnihal/Chess-AI-with-TensorFlow/blob/main/Chess_AI.ipynb
# converts the position on the board to the corresponding index in the numpy array
def position_to_index(square):
    
    squares_index = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6,
        'h': 7
        }
    
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


# converts games in pgn file to a list of games
def pgn_to_list(filename):
    pgn = open(filename, encoding="utf-8")

    game_list = []
    for game in pgn:
        game = chess.pgn.read_game(pgn)
        game_list.append(game)
        
    return game_list
     

#convert the moves in a game into a 14 x 60 x 8 x 8 matrix
def convert_game_to_matrix(game_moves):
    with open('fen_openings.pkl', 'rb') as f:
        openings_list = pickle.load(f)
        
    print(game_moves)
        
    game_moves = io.StringIO(game_moves)
    
    game_moves = chess.pgn.read_game(game_moves)
    
    board = chess.Board()
    position_matrix = []
    opening_flag = True
    
    for move in game_moves.mainline_moves():
        board.push(move)
            
        if board.fen() not in openings_list:
            opening_flag = False
            
        if opening_flag == False:
            position_matrix.append(format_board_position(board))
            if len(position_matrix) == 60: #stop once 60 moves have been converted
                return position_matrix
                
    if len(position_matrix) < 60:
        return ""
        
# prepare and label games for neural network
def format_and_label_games():
    with open('final_50_move_games.pkl', 'rb') as f:
        df_no_cheats_games = pickle.load(f)
        
    with open('final_white_cheats.pkl', 'rb') as f:
        df_white_cheats_games = pickle.load(f)
        
    with open('final_black_cheats.pkl', 'rb') as f:
        df_black_cheats_games = pickle.load(f)
        
    print(len(df_white_cheats_games))
    
    
    # CHOOSE WHAT STATS TO INCLUDE HERE BUT CHANGE SHAPE IN NEURAL NET
    # convert non-cheat game stats to numpy arrays
    no_cheat_white_cv = df_no_cheats_games['white_cv'].to_numpy() 
    no_cheat_white_ae = df_no_cheats_games['white_ae'].to_numpy() 
    no_cheat_black_cv = df_no_cheats_games['black_cv'].to_numpy() 
    no_cheat_black_ae = df_no_cheats_games['black_ae'].to_numpy() 
    no_cheat_games = df_no_cheats_games['game_matrices'].to_list() 
    no_cheat_games = np.array(no_cheat_games)
    
    # convert white-cheat game stats to numpy arrays
    white_cheats_white_cv = df_white_cheats_games['white_cv'].to_numpy() 
    white_cheats_white_ae = df_white_cheats_games['white_ae'].to_numpy() 
    white_cheats_black_cv = df_white_cheats_games['black_cv'].to_numpy() 
    white_cheats_black_ae = df_white_cheats_games['black_ae'].to_numpy() 
    white_cheat_games = df_white_cheats_games['game_matrices'].to_list() 
    white_cheat_games = np.array(white_cheat_games)
    
    # convert black-cheat game stats to numpy arrays 
    black_cheats_white_cv = df_black_cheats_games['white_cv'].to_numpy() 
    black_cheats_white_ae = df_black_cheats_games['white_ae'].to_numpy() 
    black_cheats_black_cv = df_black_cheats_games['black_cv'].to_numpy() 
    black_cheats_black_ae = df_black_cheats_games['black_ae'].to_numpy() 
    black_cheat_games = df_black_cheats_games['game_matrices'].to_list() 
    black_cheat_games = np.array(black_cheat_games)
    
    # create a 4D array of the no-cheat game engine evaluations
    no_cheat_game_data = np.column_stack((
                                          no_cheat_white_cv,
                                          no_cheat_black_cv,
                                          no_cheat_white_ae,
                                          no_cheat_black_ae))
    
    # create a 4D array of the white cheats games engine evaluations
    white_cheats_game_data = np.column_stack((
                                          white_cheats_white_cv,
                                          white_cheats_black_cv,
                                          white_cheats_white_ae,
                                          white_cheats_black_ae))
    
    # create a 4D array of the black cheats games engine evaluations
    black_cheats_game_data = np.column_stack((
                                          black_cheats_white_cv,
                                          black_cheats_black_cv,
                                          black_cheats_white_ae,
                                          black_cheats_black_ae))
    
    print(len(no_cheat_games))
    print(len(white_cheat_games))
    print(len(black_cheat_games))
    
    # join all of the games 
    games = np.concatenate((no_cheat_games, white_cheat_games, black_cheat_games))
    
    print(games.shape)
    
    # join all of the game data
    game_data = np.concatenate((no_cheat_game_data, white_cheats_game_data, black_cheats_game_data))
    
    # provide label for each class, 0 = no_cheats, 1 = white_cheats, 2 = black_cheats
    no_cheats_label = np.full(10028, 0)
    white_cheats_label = np.full(1275, 1)
    black_cheats_label = np.full(1311, 2)
    
    labels = np.concatenate((no_cheats_label, white_cheats_label, black_cheats_label))
    
    
    # randomize order of data and labels but have them match each other
    random_order = np.random.permutation(len(games))
    randomised_games = games[random_order]
    randomised_game_data = game_data[random_order]
    randomised_labels = labels[random_order]
    
    # set the correct type
    randomised_games = randomised_games.astype(np.float32)
    randomised_game_data = randomised_game_data.astype(np.float32)
    randomised_labels = randomised_labels.astype(np.float32)
    
    
    return randomised_games, randomised_game_data, randomised_labels


def train_cnn_with_stats(games, game_data, labels): 
    
    # board positions layers
    positions_input = tf.keras.layers.Input(shape=(60, 14, 8, 8) )
    X = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(positions_input)
    X = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(X)
    X = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(X)
    X = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(X)
    X = tf.keras.layers.Flatten()(X)
    
    # engine evaluations layers
    # change this line depending on how many engine evaluations being used, eg. shape=(4,) or shape=(2,)
    stats_input = tf.keras.layers.Input(shape=(4,))
    Z = tf.keras.layers.Dense(64, activation='relu')(stats_input)
    Z = tf.keras.layers.Dense(128, activation='relu')(Z)
    
    # combine pathways
    combined = tf.keras.layers.concatenate([X, Z])
    combined = tf.keras.layers.Dropout(0.5)(combined)
    
    # classify games into 1 of 3 groups
    outputs = tf.keras.layers.Dense(3, activation='softmax')(combined)

    # create model
    model = tf.keras.models.Model(inputs=[positions_input, stats_input], outputs=outputs)
    model.compile(optimizer='adam', 
                  metrics=['accuracy'], 
                  loss='categorical_crossentropy')

    
    # stratified 5 Fold cross-validation 
    stratified_5_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # save predictions and true labels from all folds
    y_test_list = []
    y_pred_classes_list = []


    
    # iterate over the folds selecting index points for data split (80:20)
    for i, j in stratified_5_fold.split(games, labels):
        
        # split data into training and testing sets
        X_positions_training_data, X_positions_testing_data = games[i], games[j]
        X_stats_training_data, X_stats_testing_data = game_data[i], game_data[j]
        y_training_labels, y_testing_labels = labels[i], labels[j]
        
        # format data
        X_positions_training_data = X_positions_training_data.reshape((-1,) + (60, 14, 8, 8))

        # convert labels to one-hot encoding
        y_training_labels = tf.keras.utils.to_categorical(y_training_labels, 3)

        early_stopping = EarlyStopping(monitor='val_loss', 
                                       restore_best_weights=True, 
                                       patience=5)

        # train model
        model.fit([X_positions_training_data, X_stats_training_data], 
                  y_training_labels, 
                  batch_size=32, 
                  epochs=100, 
                  validation_split=0.2, 
                  callbacks=[early_stopping])

        # evaluate model
        X_positions_testing_data_reshaped = X_positions_testing_data.reshape((-1,) + (60, 14, 8, 8) )
        y_pred = model.predict([X_positions_testing_data_reshaped, X_stats_testing_data])
        y_pred_classes = np.argmax(y_pred, axis=1)

        precision = precision_score(y_testing_labels, y_pred_classes, average='weighted')
        accuracy = accuracy_score(y_testing_labels, y_pred_classes)
        recall = recall_score(y_testing_labels, y_pred_classes, average='weighted')
        f1 = f1_score(y_testing_labels, y_pred_classes, average='weighted')

        f1_list.append(f1)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        # print evaluation metrics for each fold
        print('metrics for fold:')
        print('accuracy: ' + str(accuracy))
        print('precision: '+ str(precision))
        print('recall: '+ str(recall))
        print('f1-Score: '+ str(f1))
        print()
        conf_matrix = confusion_matrix(y_testing_labels, y_pred_classes)

        # print confusion matrix for each fold
        print('confusion matrix:')
        print(conf_matrix)
        
        # save predictions for cm
        y_test_list.extend(y_testing_labels)
        y_pred_classes_list.extend(y_pred_classes)
        
    # find average evaluation metrics across all folds
    precision_mean = sum(precision_list) / len(precision_list)
    accuracy_mean = sum(accuracy_list) / len(accuracy_list)
    recall_mean = sum(recall_list) / len(recall_list)
    f1_mean = sum(f1_list) / len(f1_list)

    # print average evaluation metrics
    print('average evaluation metrics:')
    print('average accuracy: ' + str(accuracy_mean))
    print('average recall: '+ str(recall_mean))
    print('average precision: '+ str(precision_mean))
    print('average f1-score: '+ str(f1_mean))

    # print confusion matrix
    conf_matrix = confusion_matrix(y_test_list, y_pred_classes_list)
    print('confusion matrix across all folds):')
    print(conf_matrix)


def train_cnn_no_stats(games, labels):
    
    # board positions layers
    positions_input = tf.keras.layers.Input(shape=(60, 14, 8, 8) )
    X = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(positions_input)
    X = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(X)
    X = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(X)
    X = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    # classify games into 1 of 3 groups
    positions_output = tf.keras.layers.Dense(3, activation='softmax')(X)
    
    # create model
    model = tf.keras.models.Model(inputs=[positions_input], outputs=positions_output)
    model.compile(optimizer='adam', 
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    
    # stratified 5 Fold cross-validation
    stratified_5_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # save predictions and true labels from all folds
    y_test_list = []
    y_pred_classes_list = []
    
    # iterate over folds
    for i, j in stratified_5_fold.split(games, labels):
        
        # split data into training and testing sets
        X_positions_training_data, X_positions_testing_data = games[i], games[j]
        y_training_labels, y_testing_labels = labels[i], labels[j]

        X_positions_training_data = X_positions_training_data.reshape((-1,) + (60, 14, 8, 8))

        # convert labels to one-hot encoding
        y_training_labels = tf.keras.utils.to_categorical(y_training_labels, 3)

        early_stopping = EarlyStopping(monitor='val_loss', 
                                       restore_best_weights=True,
                                       patience=5)

        model.fit(X_positions_training_data, 
                  y_training_labels, 
                  batch_size=32, 
                  epochs=100, 
                  validation_split=0.2, 
                  callbacks=[early_stopping])

        # evaluate the model 
        X_positions_testing_data_reshaped = X_positions_testing_data.reshape((-1,) + (60, 14, 8, 8))
        y_pred = model.predict(X_positions_testing_data_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)

        precision = precision_score(y_testing_labels, y_pred_classes, average='weighted')
        accuracy = accuracy_score(y_testing_labels, y_pred_classes)
        recall = recall_score(y_testing_labels, y_pred_classes, average='weighted')
        f1 = f1_score(y_testing_labels, y_pred_classes, average='weighted')

        precision_list.append(precision)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        f1_list.append(f1)

        # print evaluation metrics for each fold
        print('metrics for fold:')
        print('accuracy: ' + str(accuracy))
        print('precision: '+ str(precision))
        print('recall: '+ str(recall))
        print('f1-Score: '+ str(f1))
        print()
        conf_matrix = confusion_matrix(y_testing_labels, y_pred_classes)

        # print confusion matrix for each fold
        print('confusion matrix:')
        print(conf_matrix)
        
        # save predictions for cm
        y_test_list.extend(y_testing_labels)
        y_pred_classes_list.extend(y_pred_classes)
        
    # find average evaluation metrics across all folds
    accuracy_mean = sum(accuracy_list) / len(accuracy_list)
    precision_mean = sum(precision_list) / len(precision_list)
    recall_mean = sum(recall_list) / len(recall_list)
    f1_mean = sum(f1_list) / len(f1_list)

    # print average evaluation metrics
    print('average evaluation metrics:')
    print('average accuracy: ' + str(accuracy_mean))
    print('average precision: '+ str(precision_mean))
    print('average recall: '+ str(recall_mean))
    print('average f1-score: '+ str(f1_mean))
    
    # calculate cm for all folds
    conf_matrix = confusion_matrix(y_test_list, y_pred_classes_list)

    # print confusion matrix for all folds
    print("confusion matrix for all folds:")
    print(conf_matrix)
    
    
    

# pass games with no cheating and games where either white cheats or black cheats
def train_test_lr(df_no_cheats_games, df_black_cheats_games):
    
    df_no_cheats_games['cheated'] = 0
    df_black_cheats_games['cheated'] = 1

    df_all_games = pd.concat([df_no_cheats_games, 
                              df_black_cheats_games], #change if passing games where white cheats
                             axis=0) 
    df_all_games = df_all_games.reset_index(drop=True)
    
    # get only the cv scores
    X_games = df_all_games[['black_cv']] # change 'black_cv' depending on what engine evaluation you are using
    y_labels = df_all_games.cheated
    
    model = LogisticRegression()
    
    model.fit(X_games, y_labels)
    
    probabilty_threshold= 0.5  
    
    # find where logistic regression curve and y = 0.5 intercept
    cv_threshold = (np.log(1 / probabilty_threshold - 1) - model.intercept_) / model.coef_
    print(cv_threshold)

    # stratified 5 Fold cross-validation
    stratified_5_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []

        
    # iterate over folds
    for i, j in stratified_5_fold.split(X_games, y_labels):
        
        # split data into training and testing sets
        X_training_data, X_testing_data = X_games[i], y_labels[j]
        y_training_labels, y_testing_labels = X_games[i], y_labels[j]

        # create model
        model = LogisticRegression()

        model.fit(X_training_data, y_training_labels)
        
        y_pred = model.predict(X_testing_data)

        precision = precision_score(y_testing_labels, y_pred)
        accuracy = accuracy_score(y_testing_labels, y_pred)
        recall = recall_score(y_testing_labels, y_pred)
        f1 = f1_score(y_testing_labels, y_pred)

        precision_list.append(precision)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        f1_list.append(f1)

        # print evaluation metrics for each fold
        print('metrics for fold:')
        print('accuracy: ' + str(accuracy))
        print('precision: '+ str(precision))
        print('recall: '+ str(recall))
        print('f1-Score: '+ str(f1))
    


    
    
    