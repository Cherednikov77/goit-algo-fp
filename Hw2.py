import turtle
import math
def draw_tree(t, l, depth):
    if depth == 0:
        return
    for  _ in range(4):
        t.forward(1)
        t.left(90)
    new_l = l * math.sqrt(2) / 2
    t.forward(new_l)
    draw_tree(t, new_l, depth - 1)
    t.backward(l)
    t.left(45)
    t.backward(l)
def main():
    screen = turtle.Screen()
    screen.tracer(0)
    t = turtle.Turtle()
    t.speed(0)
    t.left(90)

    draw_tree(t, 100, 10)
    screen.update()
    screen.mainloop()

if __name__ == "__main__":
    main()


    #####



