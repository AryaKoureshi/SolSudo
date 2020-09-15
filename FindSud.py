# Imports
import cv2
import numpy as np
import pytesseract
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def finder_from_image(file_path):
    """
    this function used for finding the sudoku from image 
    
    Parameter
    ---------
    file_path : wiht */* and file name

    Returns
    -------
    findedSudoku : shape(9, 9)
    
    """
# Image Initializing
    image = cv2.imread(file_path)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image2 = cv2.threshold(image2, 200, 255, cv2.THRESH_BINARY) 
    edge = cv2.Canny(image2, 50, 100)
    y1,x1 = np.argwhere(edge).min(axis=0)
    y2,x2 = np.argwhere(edge).max(axis=0)
    cropped = image2[y1:y2, x1:x2]
    ycell = int(np.shape(cropped)[0]/9)
    xcell = int(np.shape(cropped)[1]/9)
    # Finding numbers and writing them into a string
    findedNumbers = ""
    for i in range(9):
        for k in range(9):
            selected = cropped[(i*ycell)+((y2-y1)//57):((i+1)*ycell)-((y2-y1)//57),
                               (k*xcell)+((x2-x1)//57):((k+1)*xcell)-((x2-x1)//57)]
            copyfindedNumbers = findedNumbers
            findedNumbers += pytesseract.image_to_string(selected, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
            findedNumbers = findedNumbers.replace("\n", "")
            findedNumbers = findedNumbers.replace("\x0c", "")
            if findedNumbers == copyfindedNumbers : findedNumbers += "0"
    # Show detected sudoku and finded numbers
    tagged = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
    #cv2.imshow('Detected Sudoku',tagged)
    findedSudoku = np.array(np.reshape(list(int(i) for i in findedNumbers), (9, 9)))
    cv2.imwrite('detected.png', tagged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return findedSudoku
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def create_solved_sudoku(unsolved_sudoku_path, solved_sudoku):
    """
    this function used for showing the solved sudoku on image 
    
    Parameter
    ---------
    unsolved_sudoku_path : the path of unsolved sudoku with */* and file name
    solved_sudoku : shape (9, 9) --> received from predicted sudoku by model

    Returns
    -------
    show_solved_sudoku : shape(9, 9)

    """
    # Image Initializing
    show_solved_sudoku = cv2.imread(unsolved_sudoku_path)
    show_solved_sudoku2 = cv2.cvtColor(show_solved_sudoku, cv2.COLOR_BGR2GRAY)
    _, image2 = cv2.threshold(show_solved_sudoku2, 200, 255, cv2.THRESH_BINARY) 
    edge = cv2.Canny(show_solved_sudoku2, 50, 100)
    y1,x1 = np.argwhere(edge).min(axis=0)
    y2,x2 = np.argwhere(edge).max(axis=0)
    cropped = show_solved_sudoku2[y1:y2, x1:x2]
    ycell = int(np.shape(cropped)[0]/9)
    xcell = int(np.shape(cropped)[1]/9)
    for i in range(9):
        for k in range(9):
            selected = cropped[(i*ycell)+((y2-y1)//55):((i+1)*ycell)-((y2-y1)//55), 
                               (k*xcell)+((x2-x1)//55):((k+1)*xcell)-((x2-x1)//55)]
            xx = int(np.shape(selected)[0])
            yy = int(np.shape(selected)[1])
            if ((list(np.reshape(selected, (xx * yy)))).count(255)) >= 0.95 * (xx * yy) :
                cv2.putText(show_solved_sudoku[y1 + (i*ycell):y1 + ((i+1)*ycell), 
                                               x1 + (k*xcell):x1 + ((k+1)*xcell)],
                            '{}'.format(solved_sudoku[i][k]),
                            (int(xcell/3.333), int(ycell/1.333)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            xx/32, 
                            (0,0,255), 
                            int(xx/16))
    # Show solved sudoku
    show_solved_sudoku = cv2.rectangle(show_solved_sudoku, 
                                       (x1,y1), (x2,y2), 
                                       (0,255,0), 
                                       3, 
                                       cv2.LINE_AA)
    cv2.imwrite('solved.png', show_solved_sudoku)
    cv2.imshow('Solved Sudoku!',show_solved_sudoku)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return print(":)")
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# From Text
def finder_from_text(str_sudoku):
    """
    this function used for find sudoku from inputed string 
    
    Parameter
    ---------
    str_sudoku: str

    Returns
    -------
    sudoku : shape(9, 9)

    """
    sudoku = np.array(np.reshape(list(int(i) for i in str_sudoku), (9, 9)))
    return sudoku

