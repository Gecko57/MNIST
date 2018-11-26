from tkinter import *
import numpy as np
from nolearn.dbn import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

t1 = time.time()

mnist_data = pd.read_csv(os.getcwd()+'\\train.csv')
os.getcwd()

# Full data
X = mnist_data[mnist_data.columns[1:]].values
y = mnist_data.label.values

n_train = int(2*X.shape[0]/3)

# Trainning data
X_train = X[:n_train, :]
y_train = y[:n_train]

# Testing Data
X_test = X[n_train:, :]
y_test = y[n_train:]

# Neural Network
clf_nn = DBN([X_train.shape[1], 300, 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=15)
clf_nn.fit(X_train, y_train)
# acc_nn = clf_nn.score(X_test,y_test)
# print("neural network accuracy: ",acc_nn)

# Test
t2 = time.time()
print('Step 1 execution time: ' + str(t2-t1) + 's')

Matrix = np.zeros((280, 280))


class Paint(object):

    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.display = Text(self.root)
        self.text = '?'
        self.display.grid(row=0, column=1)

        self.guess_button = Button(self.root, text='Guess', command=self.guess)
        self.guess_button.grid(row=0, column=2)

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 10
        self.active_button = self.pen_button
        self.color = 'black'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)
        print(self.c.find_withtag('line'))

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        global Matrix
        self.line_width = 10
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width, fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
            if self.line_width <= event.x <= 280-self.line_width and self.line_width <= event.x <= 280-self.line_width:
                for i in range(int(self.line_width)):
                    for j in range(int(self.line_width)):
                        Matrix[int(event.y+(i-self.line_width/2)), int(event.x+(j-self.line_width/2))] = 1

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete('all')
        global Matrix
        Matrix = np.zeros((280, 280))

    def guess(self):
        global Matrix, clf_nn
        M_bis = np.zeros((28, 28))
        for i in range(280):
            for j in range(280):
                M_bis[int(i/10), int(j/10)] += Matrix[i, j]
        for i in range(28):
            for j in range(28):
                M_bis[i, j] = min(M_bis[i, j]*255/25, 255)
        y_predict = clf_nn.predict(M_bis.reshape(1, -1))
        print(y_predict)
        s = '{}'.format(y_predict)
        self.display.insert(END, s)
        self.display.see(END)

        plt.clf()
        image = M_bis.reshape(28, 28)
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    Paint()
