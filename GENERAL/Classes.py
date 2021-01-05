import turtle 

draw_area = turtle.Screen()
draw_area.setup(height = 600, width = 650,
                 startx = 750, starty = 150)
bob = turtle.Turtle()
bob.pensize(3)
bob.speed(0)

def moveit(which_bob, x = 0, y = 0, theta = 0):
    which_bob.penup()
    which_bob.goto(x,y)
    which_bob.setheading(theta)
    which_bob.pendown()

moveit(bob, -75, -150)

# Defining a class
class Polygon :
    def __init__(self, sides, name):
        self.sides = sides
        self.name = name
        self.anglesum = (sides - 2) * 180
        # Methods of the class, what they return from object properties
        
    def draw(self, size = 100) :
        # 100 is default if nothing specified
        for i in range(self.sides):
            bob.forward(size)
            bob.left(180 - (self.anglesum / self.sides))
        turtle.done()
            
# Adding objects to class
square = Polygon(4, 'Square')
pentagon = Polygon(5, 'Pentagon')
hexagon = Polygon(6, 'Hexagon')
octagon = Polygon(8, 'Octagon')

print(square.name)
print(square.sides)
print(square.anglesum)

octagon.draw()
