from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import pickle
import numpy as np
import pandas as pd
import copy
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict

def check_latest_n_id(N_id_list, df):
	previous_4_n_id = df[["win_n_id","lose_n_id"]].iloc[-2:].values.reshape(-1)
	if len(set(previous_4_n_id).intersection(N_id_list)) >= 0:
		raise ValueError('Data already exists. Please check your N_id.')

	else:
		print("Updating new entries to the existing game data.")


def check_latest_url_id(N_id_list,nID_chef_dict):
	for i in N_id_list:
		if i in nID_chef_dict.keys():
			raise ValueError('Data already exists. Please check your N_id.')
		else:
			print("Updating new entries to the existing game data.")


########################## ELO RATING SYSTEM ##############################
def expected(A,B):
    """
    Calculate expected score of A in a match against B
    """
    return 1 / (1+10 ** ((B - A ) / 400))

def elo(old, exp, score, k=32):
    """
    Calculate the new Elo rating for a player
    """
    return old + k * (score - exp)

def update_elo(winner_elo, loser_elo, k_factor, elo_width):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo, elo_width)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b, elo_width):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

#############################################################################

########################## INGREDIENTS ##############################

def remove_no_ingredients(df):
    mask = df["nID"] != -1
    df_ = df.loc[mask]
    if len(df_) == 0:
        raise ValueError('No record after filtering out no nID records.')
    return df_

def select_chef_df(df,chef_name):
    mask = df["chef_name"] == chef_name
    df_1 = df.loc[mask]
    df_2 = remove_no_ingredients(df_1)
    return df_2

def shuffle_df(df):
    df_ = df.sample(frac=1).reset_index(drop=True)
    return df_

def get_unique_ingredients_list(df,nID_ingredients_new_dict ):
    # Build matrix with row as competition number, and columns as unique ingredients
    # Unique ingredients
    all_nID = df["nID"].tolist()
    all_ingredients = []
    for n in all_nID:
        lemme_ingredient = nID_ingredients_new_dict.get(n)
        lemme_ingredients = [x for x in lemme_ingredient if x != '']
        all_ingredients.append(lemme_ingredients)
    unique_ingredients = list(set([x for y in all_ingredients for x in y]))
    return unique_ingredients

def prepare_ingredients_matrix(df,nID_ingredients_new_dict):
    # row_num = num_nID
    row_num = len(df["nID"].tolist())
    
    # col_num = unqiue ingredients count + 1 (last 1 for Y)
    uni_ingred = get_unique_ingredients_list(df,nID_ingredients_new_dict)
    col_num = len(uni_ingred) + 1
    
    # prepare ing_matrixID_dict
    ing_matrixID_dict = dict(zip(uni_ingred, list(range(0,len(uni_ingred)))))
    matrixID_ing_dict = {k:v for v,k in ing_matrixID_dict.items()}
    
    # Make matrix
    chef_matrix = np.zeros([row_num, col_num])
    
    return chef_matrix, ing_matrixID_dict, matrixID_ing_dict
    
def update_matrix(df,chef_matrix,ing_matrixID_dict,nID_ingredients_new_dict):
    # Prepare Y values
    result_list = df["result"].tolist()
    
    # For all recipe
    all_nID = df["nID"].tolist()
    
    # Mark ingredients +1 <-- X
    for idx, n in enumerate(all_nID):
        lemme_ingredient = nID_ingredients_new_dict.get(n)
        lemme_ingredients = [x for x in lemme_ingredient if x != '']
        
        for m in lemme_ingredients:
            mat_ID = ing_matrixID_dict.get(m)
            chef_matrix[idx,mat_ID] += 1
            
        # And mark result (win or lose) <-- Y
        chef_matrix[idx,-1] = result_list[idx]
    return chef_matrix, ing_matrixID_dict

def prepare_XY(chef_name, total_df,nID_ingredients_new_dict):
    # Select chef's record only
    chef_df_square = select_chef_df(total_df, chef_name)
    
    # If less than X num, not enough for calculation
    if len(chef_df_square) < 5:
        pass
    else:
        
        # Shuffle df
        chef_df = shuffle_df(chef_df_square)
        
        # Prepare matrix
        chef_matrix, ing_matrixID_dict, matrixID_ing_dict = prepare_ingredients_matrix(chef_df,nID_ingredients_new_dict)
        # Update matrix
        updated_matrix, ing_matrixID_dict = update_matrix(chef_df, chef_matrix, ing_matrixID_dict,nID_ingredients_new_dict)
        
        X = updated_matrix[:, 0:-1]
        Y = updated_matrix[:, -1]
        return updated_matrix, matrixID_ing_dict, X, Y
    
def create_model(X):
    model = Sequential()
    model.add(Dense(60, input_dim = X[0].shape[0], kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def categorical_accuracy(y_true, y_pred):
    accu = 0
    for idx, y in enumerate(y_true):
        y_t = 0
        if y[0] == 1:
            pass
        else:
            y_t += 1
            
        if y_t == y_pred[idx]:
            accu += 1
    accuracy = accu / len(y_true)
    #print("Accuracy: ", accuracy)
    return accuracy

def train_model(X,Y):
    ratio = 0.8
    y_categorical = np_utils.to_categorical(Y, 2)
    
    # Prepare train and test data
    x_train = np.array(X[:int((len(X)*ratio))])
    x_test = np.array(X[len(x_train):])
    y_train = np.array(y_categorical[:int((len(y_categorical)*ratio))])
    y_test = np.array(y_categorical[len(y_train):])

    # Create model
    model = create_model(X)
    
    # Compile model
    model.fit(x_train, y_train,batch_size=32, epochs=4,verbose=0,validation_data=(x_test, y_test))

    prediction = model.predict(x_test)
    predicted_classes = model.predict_classes(x_test)
    accuracy_ = categorical_accuracy(y_test,predicted_classes)
    
    return accuracy_

def lucky_ingredients(updated_matrix, matrixID_ing_dict):
    Y = updated_matrix[:, -1]
    X = updated_matrix[:, 0:-1]
    #print(updated_matrix.shape)
    lucky_food = np.zeros([len(matrixID_ing_dict),])
    all_food = np.zeros([len(matrixID_ing_dict),])

    winning_rows = np.where(Y == 1)[0]
    for r in winning_rows:
        win_row = X[r]
        for j, f in enumerate(win_row):
            lucky_food[j] += f
            
#     # Get best performing ingredients
#     luck_idx = (-lucky_food).argsort()[:5]
#     for m in luck_idx:
#         print(matrixID_ing_dict.get(m))
#         print(lucky_food[m])
    print("----------------")
    for w, f in enumerate(Y):
        every_row = X[w]
        for s, v in enumerate(every_row):
            all_food[s] += v
            
#     fam_idx = (-all_food).argsort()[:5]
#     for m in fam_idx:
#         print(matrixID_ing_dict.get(m))
#         print(all_food[m])
        
    get_lucky = lucky_food/all_food
    idx = (-get_lucky).argsort()[:50]
    
    print_num = 0
    for g in idx:
        if all_food[g] < 2:
            pass
        else:
            if print_num < 10:
                print("Lucky food: ", matrixID_ing_dict.get(g))
                print(" Used: ", all_food[g], " times.")
                print(" Won: ", lucky_food[g], "times.")
                print_num += 1
            else:
                break

    return lucky_food, all_food