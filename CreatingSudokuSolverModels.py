# imports
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from Functions import load_data, delete_digits, batch_smart_solver
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Prepare data
# we creating our Xtrain data from Ytrain data with delete_digits function
(_, Ytrain), (Xtest, Ytest) = load_data(nb_train=60000, nb_test=1)  # We won't use _. We will work directly with Ytrain
# one-hot-encoding --> shapes become :
# (?, 9, 9, 10) for Xs ____ because in Xtrain data we have 0 for blanks digits
# (?, 9, 9, 9) for Ys  ____ but in Ytrain data we don't have 0
Xtrain = to_categorical(Ytrain).astype('float32')  # from Ytrain cause we will creates quizzes from solusions
Xtest = to_categorical(Xtest).astype('float32')
Ytrain = to_categorical(Ytrain-1).astype('float32') # (y-1) because we 
Ytest = to_categorical(Ytest-1).astype('float32')   # don't want to predict zeros
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Creating model
# input_shape is (9,9,10) because we have sudokus with shape (9,9) that were the categorical
input_shape = (9, 9, 10)
# first layer
grid = Input(shape=input_shape)  # inputs
# creating costomized model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
# grid is a layer and in this code, we attaching the created model to the grid
features = model(grid)  # commons features
# define one Dense layer for each of the digit we want to predict
# 81 layers atached to the model
digit_placeholders = [
    Dense(9, activation='softmax')(features)
    for i in range(81)
]
# creating final model
solver = Model(grid, digit_placeholders)  # build the whole model
# compiling created model
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# grid ---> model ---> degit_placeholders
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Train Model

# First train
# in the firs training we don't delete any digit
solver.fit(
    delete_digits(Xtrain, 0),  # we don't delete any digit for now
    [Ytrain[:, i, j, :] for i in range(9) for j in range(9)],  # each digit of solution
    batch_size=128,
    epochs=1,  # 1 epoch should be enough for the task
    verbose=1,
)

# Second train
early_stop = EarlyStopping(patience=2, verbose=1)
i = 1
for nb_epochs, nb_delete in zip(
        [5, 10, 10],#[1, 2, 3, 4, 6, 8, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 20, 25, 30],  # epochs for each round
        [20, 55, 58] #[1, 2, 3, 4, 6, 8, 10, 12, 14, 17, 20, 23, 25, 30, 35, 40, 45, 50, 55, 60]  # digit to pull off
):
    print('Pass n° {} ...'.format(i))
    i += 1
    
    solver.fit(
        delete_digits(Xtrain, nb_delete),  # delete digits from training sample
        [Ytrain[:, i, j, :] for i in range(9) for j in range(9)],
        validation_batch_size=0.01,
        shuffle=True,
        batch_size=128,
        epochs=nb_epochs,
        verbose=1,
        callbacks=[early_stop]
    )
print("Saving trained model..")
solver.save('C:/Users/Arya/Desktop/solver.h5')
print("Saved.")
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Predicting
quizzes = Xtest.argmax(3)  # quizzes in the (?, 9, 9) shape. From the test set
true_grids = Ytest.argmax(3) + 1  # true solutions dont forget to add 1 
smart_guesses = batch_smart_solver(quizzes, solver)  # make smart guesses !
print("Saving trained model..")
solver.save('C:/Users/Arya/Desktop/solver.h5')
print("Saved.")
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Plot model
print("Ploting Model...")
plot_model(solver, to_file='C:/Users/Arya/Desktop/solver.png', show_shapes=True, expand_nested=True, dpi=300)
print("Plotted Successful.")
print("All Process Passed Successful!")