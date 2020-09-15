### Imports
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
### Definition functions
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Function for loading data
def load_data(nb_train=40000, nb_test=10000, full=False):
    """
    Function to load data in the keras way.
    
    Parameters
    ----------
    nb_train (int): Number of training examples
    nb_test (int): Number of testing examples
    full (bool): If True, whole csv will be loaded, we dont have nb_test
                    all of the data using for training
    
    Returns
    -------
    Xtrain, ytrain (np.array, np.array),
        shapes (nb_train, 9, 9), (nb_train, 9, 9): Training samples
    Xtest, ytest (np.array, np.array), 
        shapes (nb_test, 9, 9), (nb_test, 9, 9): Testing samples
    """
    # if full is true, load the whole dataset
    if full:
        sudokus = pd.read_csv('Enter Location!/sudoku.csv').values
        # Transpose of sudokus matrix to quizzes and solutions
        quizzes, solutions = sudokus.T
        # Create sudoku shape from quizzes
        Xs = np.array([np.reshape([int(element) for element in sud], (9, 9))
                                   for sud in quizzes])
        # Create sudoku shape from solutions
        Ys = np.array([np.reshape([int(element) for element in sud], (9, 9))
                                   for sud in solutions])
        return (Xs[:nb_train], Ys[:nb_train])
    else:
        sudokus = next(
            pd.read_csv('Enter Location!/sudoku.csv', chunksize=(nb_train + nb_test))
        ).values
        # Transpose of sudokus matrix to quizzes and solutions
        quizzes, solutions = sudokus.T
        # Create sudoku shape from quizzes
        Xs = np.array([np.reshape([int(element) for element in sud], (9, 9))
                                   for sud in quizzes])
        # Create sudoku shape from solutions
        Ys = np.array([np.reshape([int(element) for element in sud], (9, 9))
                                   for sud in solutions])
        return (Xs[:nb_train], Ys[:nb_train]), (Xs[nb_train:], Ys[nb_train:])
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Function for deleting random digits from sudoku 
def delete_digits(X, n_delet=1):
    """
    This function is used to create sudoku quizzes from solutions
    
    Parameters
    ----------
    X (np.array), shape (?, 9, 9, 9|10): input solutions grids.
    n_delet (integer): max number of digit to suppress from original solutions
    
    Returns
    -------
    grids: np.array of grids to guess in one-hot way. Shape: (?, 9, 9, 10)
    """
    # argmax function can convert categorical shape to sudoku shape
    grids = X.argmax(3)  # get the grid in a (9, 9) integer shape
    # this 'for' generates blanks with n_delet size
    # this works randomly and give sudoku shape
    for grid in grids:
        grid.flat[np.random.randint(0, 81, n_delet)] = 0  # generate blanks (replace = True)
    # and the end return the categorical shape of generated sudoku
    return to_categorical(grids)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Function for solving smartly
def batch_smart_solver(grids, solver):
    """
    NOTE : This function is ugly, feel free to optimize the code ...
    
    This function solves quizzes in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank.
    
    Parameters
    ----------
    grids (np.array), shape (?, 9, 9): Batch of quizzes to solve (smartly ;))
    solver (tensorflow.keras.model): The neural net solver
    
    Returns
    -------
    grids (np.array), shape (?, 9, 9): Smartly solved quizzes.
    """
    # getting work with copy of grids
    grids = grids.copy()
    # (grids == 0).sum((1, 2))--> number of zeros
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids

def smart_solver(grid, solver):
    """
    NOTE : This function is ugly, feel free to optimize the code ...
    
    This function solves quiz in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank.
    
    Parameters
    ----------
    grid (np.array), shape (1, 9, 9)
    solver (tensorflow.keras.model): The neural net solver
    
    Returns
    -------
    grid (np.array), shape (9, 9): Smartly solved quiz.
    """
    # getting work with copy of grids
    grid = grid.copy()
    # (grids == 0).sum((1, 2))--> number of zeros
    for _ in range((grid == 0).sum().max()):
        preds = np.array(solver.predict(to_categorical(np.reshape(grid, (1, 9, 9))))) # get predictions
        probs = preds.max(2).T # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1 # get corresponding values
        zeros = (grid == 0).reshape((1, 81)) # get blank positions
        for prob, value, zero in zip(probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return np.reshape(grid, (9,9))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Function for checking sudoku
def batch_checker(predicted_quizzes):
    """
    This function checks the batch of quizzes for that those are right or wrong.
    
    Parameter
    ---------
    predicted_quizzes (np.array), shape (?, 9, 9)
    
    Return
    ------
    checked_quizzes : list of True or False for each quiz in predicted_quizzes
    """
    predicted_quizzes.copy()
    checked_quizzes = []
    for quiz in predicted_quizzes:
        right = True
        for i in range(9):
            for j in range(9):
                if (list(quiz[i])).count(j+1) == 2 : 
                    right = False
                    break
                else:
                    quiz = quiz.T
                    if (list(quiz[i])).count(j+1) == 2 : 
                        right = False
                        break
            if right == False: break
        checked_quizzes.append(right)
    return checked_quizzes

def checker(predicted_quiz):
    """
    This function checks the quiz for that it is right or wrong.
    
    Parameter
    ---------
    predicted_quiz (np.array), shape (1, 9, 9)
    
    Return
    ------
    True : sudoku solved rightly. 
    False : sudoku solved wrongly.
    """
    predicted_quiz.copy()
    for i in range(9):
        for j in range(9):
            if (list(predicted_quiz[0][i])).count(j+1) == 2 : return False
            else:
                predicted_quiz[0] = predicted_quiz[0].T
                if (list(predicted_quiz[0][i])).count(j+1) == 2 : return False

    return True
