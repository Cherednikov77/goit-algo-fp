import turtle
import math
def draw_tree(t, l, depth):
    if depth == 0:
        return
    for  _ in range(4):
        t.forward(1)
        t.left(90)
    t.forward(1)
    t.left(45)
new_1 = 1 * math.sqrt(2) / 2
t.right(90)
t.forward(new_1)
draw_tree(t, new_1, depth - 1)
t.backward(new_1)
t.left(45)
t.backward(1)
screen = turtle.Screen()
screen.tracer(0)

draw_tree(t, 100, 10)


    