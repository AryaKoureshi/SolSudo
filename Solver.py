# Imports
import numpy as np
from tensorflow.keras.models import load_model
from Functions import smart_solver, checker
from FindSud import finder_from_image, create_solved_sudoku, finder_from_text
#:::::::::::::::::::::::::::::::::::::::::::::::
# Finding Sudoku
print("Finding Sudoku...")
key = int(input("What you want?\n 1.with Text\n 2.with Image (need Tesseract)\n Enter the number : "))
while True:
    if key == 1:
        entered_sudoku = input("Please enter the sudoku with 0 for blank digits : ")
        entered_sudoku = str(entered_sudoku)
        finded_sudoku = finder_from_text(entered_sudoku)
        path_of_sudoku = False
        break
    elif key == 2:
        path_of_sudoku = input("Please enter the path of sudoku (with */* and file name): ")
        path_of_sudoku = str(path_of_sudoku)
        finded_sudoku = finder_from_image(path_of_sudoku)
        print("Find the Sudoku was successful.")
        break
    else:
        print("Try Again!")
        key = int(input("What you want? 1 or 2 ??"))
#:::::::::::::::::::::::::::::::::::::::::::::::
# Load Model and Predict the Sudoku
print("Loading Model...")
path_of_model = input("Please enter the path of Model (with */* and model name): ")
path_of_model = str(path_of_model)
solver = load_model(path_of_model)
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Loading Model was successful.")
#predicting
print("Solving Sudoku...")
solved = smart_solver(np.reshape(finded_sudoku, (1, 9, 9)), solver)
print("Sudoku Solved. \nChecking...")
checked = checker(np.reshape(solved, (1, 9, 9)))
print("Cheked successful.")
if checked == True :
    print("Solved the Sudoku was Right!")
    print("Show solved Sudoku rightly...")
    if path_of_sudoku == False : print(solved)
    else : create_solved_sudoku(path_of_sudoku, solved.T)
else :
    print("I'm sorry, I couldn't solve this sudoku rightly! You can create your Model with 'CreatingSudokuSolverModels.py' better than this Model.")
    print("Show solved Sudoku wrongly...")
    if path_of_sudoku == False : print(solved)
    else : create_solved_sudoku(path_of_sudoku, solved.T)

    