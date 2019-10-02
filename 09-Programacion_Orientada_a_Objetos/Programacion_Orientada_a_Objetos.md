# 9. Programación orientada a objetos
## 9.1 Clases
El formato es el siguiente:

```py
class Clase: # Suele empezar por mayúscula
    # Atributos de la clase
    atributos

    # Métodos de la clase
    # Constructor de la clase
    def __init__ (self, atributos):
        # Referencia a un atributo de la clase: self.atributos
        # ...

    def metodo (self):
        # ...
```

Para acceder a un método haríamos:

```py
objeto = Clase(abtributos)  # Lo creamos
objeto.metodo()
```

## 9.2 Declaración y uso de Setters y Getters
Deberíamos definir los siguientes métodos dentro de la clase:

```py
# Getter
def getAtributos (self):
    return self.atributos

# Setter
def setAtributos (self, atributos2):
    self.atributos = atributos2
```

## 9.3 Sobreescritura de operadores
### 9.3.1 Sobrecarga del operador +
Deberíamos definir el siguiente método dentro de la clase:

```py
def __add__ (self, objeto):
    self.atributos = self.atributos + objeto.atributos
```

## 9.3.2 Sobrecarga del operador -
Deberíamos definir el siguiente método dentro de la clase:

```py
def __sub__ (self, objeto):
    self.atributos = self.atributos - objeto.atributos
```

## 9.4 Herencia
El formato es el siguiente:

```py
class ClaseHija(ClasePadre):
    def __init__ (self, atributos):
        super().__init__(atributos) # Llamamos al constructor de ClasePadre

    # Se pueden agregar nuevos métodos que no están en ClasePadre
```

## 9.5 Sobreescritura de métodos
Cuando hay herencia, también se pueden sobreescribir los métodos del padre. Solo hay que definir el método en la clase hija (con el mismo nombre y número de parámetros).
