import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard # visualize how model doing
import numpy as np
import os
import random
# build model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3), # why is 176, 200, 3?????
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))
# set learning rate and optimizer
learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6) # use Adam (google it), decay: float >= 0. Learning rate decay over each update.

model.compile(loss='categorical_crossentropy', # loss function (google it)
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/stage1") # This callback writes a log for TensorBoard, 
                                                # which allows you to visualize dynamic graphs of your training and test metrics, 
                                                # as well as activation histograms for the different layers in your model.

def check_data(): # just check the length of all the list is correct 
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:",total_data)
    return lengths


train_data_dir = "train_data"
hm_epochs = 10 # number of epochs

for i in range(hm_epochs):
    current = 0
    increment = 200
    not_maxinum = True
    all_files = os.listdir(train_data_dir) # all train data
    maximum = len(all_files) 
    random.shuffle(all_files) # reshuffle it
    while not_maxinum:
        print("Currently doing {}:{}".format(current,current+increment))
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []
        for file in all_files[current:current+increment]: # everytime only train # increment data set
            full_path = os.path.join(train_data_dir,file)
            data = np.load(full_path)
            data = list(data)
            for d in data:
                choice = np.argmax(d[0]) # Returns the indices of the maximum values along an axis.
                if choice == 0:
                    no_attacks.append(d) # or just write .append(d)
                elif choice == 1:
                    attack_closest_to_nexus.append(d)
                elif choice == 2:
                    attack_enemy_structures.append(d)
                elif choice == 3:
                    attack_enemy_start.append(d)
        lengths = check_data()
        lowest_data = min(lengths)

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)
        # balance the number of different choices to make the result of neural network better
        no_attacks = no_attacks[:lowest_data]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        check_data()
        # put train set together
        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
        random.shuffle(train_data)
        print(len(train_data))
        
        test_size = 100
        batch_size = 128
        
        x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3) # opencv map info
        y_train = np.array([i[0] for i in train_data[:-test_size]]) # attack choice
        # not out of sample. Should separate a test set instead.
        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in train_data[-test_size:]])
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=1, # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. For tracking info
                  callbacks=[tensorboard])
        
        model.save("./model_trained/BasicCNN-{}-epochs-{}-LR-STAGE1".format(i, learning_rate))
        current += increment
        if current > maximum:
            not_maximum = False