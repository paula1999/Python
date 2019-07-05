# Calcular el numero de vocales de una cadena o string
frase = "El perro corre despacio"
base = "aeiouAEIOU"
contador = 0

for i in frase: # Para cada caracter en palabra
    if i in base: # Comprueba si el caracter esta en la base
        contador += 1

print ("El numero de vocales es ", contador)
