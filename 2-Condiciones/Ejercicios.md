# Ejercicios
## 1. Calcula las soluciones de una ecuación de segundo grado
```py
print("Para resolver la ecuación de segundo grado ax^2 + bx + c = 0")

a = float(input("Introduce el valor de a: "))
b = float(input("Introduce el valor de b: "))
c = float(input("Introduce el valor de c: "))

if (a != 0):
    raiz = math.sqrt(b**2 -4*a*c)
    x1 = (-b + raiz) / 2*a
    x2 = (-b - raiz) / 2*a

    print("Las soluciones son x1 = %3.2f" %x1, " y x2 = %3.2f" %x2)

```
