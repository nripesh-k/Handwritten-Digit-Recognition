import neuralnetwork as nn
try:
    import sys, os
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    sys.stdout = open(os.devnull,'w')
    sys.stderr = open(os.devnull,'w')
    import pygame
    import numpy as np
    from tkinter import *
    from tkinter import messagebox
    sys.stdout = stdout
    sys.stderr = stderr
except:
    import install_requirements
    import sys, os
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    sys.stdout = open(os.devnull,'w')
    sys.stderr = open(os.devnull,'w')
    import pygame
    import numpy as np
    from tkinter import *
    from tkinter import messagebox
    sys.stdout = stdout
    sys.stderr = stderr

class pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255,255,255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):
        # Get the neighbours of each pixel in the grid, this is used for drawing thicker lines
        j = self.x // 20 # the var i is responsible for denoting the current col value in the grid
        i = self.y // 20 # the var j is responsible for denoting thr current row value in the grid
        rows = 28
        cols = 28

        # Horizontal and vertical neighbors
        if i < cols - 1:  # Right
            self.neighbors.append(g.pixels[i + 1][j])
        if i > 0:  # Left
            self.neighbors.append(g.pixels[i - 1][j])
        if j < rows - 1:  # Up
            self.neighbors.append(g.pixels[i][j + 1])
        if j > 0:  # Down
            self.neighbors.append(g.pixels[i][j - 1])

        # Diagonal neighbors
        if j > 0 and i > 0:  # Top Left
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j + 1 < rows and i > -1 and i - 1 > 0:  # Bottom Left
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j - 1 < rows and i < cols - 1 and j - 1 > 0:  # Top Right
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:  # Bottom Right
            self.neighbors.append(g.pixels[i + 1][j + 1])


class grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generatePixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generatePixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbors(self)

    def clicked(self, pos): #Return the position in the grid that user clicked on
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        li = self.pixels

        newMatrix = [[] for x in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li[i])):
                if li[i][j].color == (255,255,255):
                    newMatrix[i].append(0)
                else:
                    newMatrix[i].append(1)

        x_test=np.empty([784,1],dtype=int)
        i=0
        for row in range(28):
            for x in range(28):
                x_test[i,0] = newMatrix[row][x]
                i=i+1
        return x_test


def guess(li):
    layer_sizes=(784,32,10)

    net = nn.NeuralNetwork(layer_sizes)
    net.weights=np.load("w_32.npy",allow_pickle=True)
    net.biases=np.load("b_32.npy",allow_pickle=True)
    predictions = net.predict(li)
    t = (np.argmax(predictions))
    print("I predict this number is a:", t)
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediction", "I predict this number is a: " + str(t))
    window.destroy()
    #plt.imshow(li[0], cmap=plt.cm.binary)
    #plt.show()


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                guess(li)
                g.generatePixels()
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0,0,0)
                for n in clicked.neighbors:
                    n.color = (0,0,0)

            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255,255,255)
                except:
                    pass

        g.draw(win)
        pygame.display.update()

pygame.init()
width = height = 560
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Number Guesser")
g = grid(28, 28, width, height)
main()


pygame.quit()
quit()
