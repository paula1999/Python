from math import *

print("Para resolver la ecuacion de segundo grado ax^2 + bx + c = 0")

a = float(input("Introduce el valor de a: "))
b = float(input("Introduce el valor de b: "))
c = float(input("Introduce el valor de c: "))

if (a != 0):
    d = b**2 - 4*a*c # Discriminante

    if (d >= 0):
        x1 = (-b + sqrt(d)) / 2*a
        x2 = (-b - sqrt(d)) / 2*a

        print("Las soluciones son x1 = %3.2f" %x1, " y x2 = %3.2f" %x2)

    else:
        x1Real = -b/2*a
        x1Im = sqrt(-d)/2*a
        x2Real = -b/2*a
        x2Im = -sqrt(-d)/2*a
        x1 = str(x1Real) + " + i" + str(x1Im)
        x2 = str(x2Real) + " + i" + str(x2Im)

        print("Las soluciones son x1 = %s" %x1, "y x2 = %s" %x2)
