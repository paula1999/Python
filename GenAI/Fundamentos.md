# Gen AI
## Técnicas avanzadas
### Retrieval Augmented Generation (RAG)

RAG es una técnica que está permitiendo a muchos LLM que tengan el contexto o la información más allá de lo que pudieran haber aprendido en Internet.

Por ejemplo: chat con archivos PDF, responder preguntas basadas en articulos de un sitio web, formulario en búsqueda web...

De esta manera, dando contexto relevante, le podemos pedir a un LLM que lea una parte de texto, y luego procesarlo para conseguir una respuesta. Es decir, lo usamos como un motor razonable para procesar la información, más que como una fuente de información.

### Fine-tuning

Es una técnica para darle más información a los LLM, en concreto, cuando el texto de entrada es demasiado grande. Sirve para que el modelo pueda hacer gran variedad de tareas.

Se usa cuando una tarea a realizar no es fácil de definir en la entrada, como por ejemplo para especificar el tipo de estilo de un resumen, hablar o escribir con el estilo de una persona.

También se puede usar para ayudar a los LLM a ganar conocimiento específico, como notas médicas, documentos legales...

También se puede usar para conseguir un modelo más pequeño que haga una tarea.

Básicamente, se encarga de mostrarle ejemplos de cómo debería responder a una instrucción expecífica

### Preentrenar un LLM

Se puede preentrenar un LLM para que aprenda de lo que hay en internet. Esto puede servir para hacer una aplicación específica


## Elegir modelo
### Tamaño del modelo

| 1B parámetros    | Coincidencia de patrones y conocimiento básico del mundo       | Reseñas de sentimientos de un restaurante |
| 10B parámetros   | Más conocimiento del mundo. Puede seguir instrucciones básicas | Chatbot para pedir comida |
| 100B+ parámetros | Muy buen conocimiento del mundo. Razonamiento complejo         | Compaero de lluvia de ideas |

##  Cómo LLM sigue instrucciones: instruction tuning y RLHF

Para que un LLM siga instrucciones y no prediga la siguiente palabra, existe una técnica llamada "instruction tuning", que consiste en coger un LLM preentrenado y aplicarle fine tune sobre ejemplos de buenas respuestas a preguntas o buenos ejemplos de LLM siguiendo las instrucciones.

Además, existe otra ténica **Reinforcement learning from human feecback (RLHF)** que es capaz de mejorar la calidad de las respuestas. Se basa en: ser de ayuda, honesto e inofensivo.

- Paso 1: entrenar un modelo con una respuesta de calidad (puntuación). Básicamente, se entrena un algoritmo de aprendizaje supervisado con varias respuestas seguidas de una puntuación, más alta cuanto más calidad tenga.
- Paso 2: hacer que LLM genere muchas respuestas. Entrenarlo para que genere más respuestas con puntuaciones altas.



## Uso de herramientas y agentes

El modelo LLM puede detectar llamadas a otras funciones del sistema

Con los agentes, los LLM pueden elegir y llevas a cabo una serie de acciones complejas

## Transformadores

- **Encoder only models**: funciona como un modelo secuencia a secuencia, de esta forma la entrada y la salida tienen la misma longitud. Ejemplos: clasificacion de análisis de sentimientos.
- **Encoder-decoder models**: realiza tareas secuencia a secuencia, de tal manera que la entrada y la salida tienen diferente longitud. Ejemplos: traducir, generación de texto...
- **Decoder only models**: pueden realizar casi cualquier tarea.

## Prompt engineering
### Zero shot inference
Se le pide al LLM que haga una cosa.

### In-context learning (ICL) - one shot inference
En el texto de entrada se muestra un ejemplo de lo que debe hacer el modelo, y luego se le pide que haga algo parecido.

### In-context learning (ICL) - few shot inference
Igual que el anterior, pero se le pasan varios ejemplos.

## Configuración - parámetros de inferencia

- Máximo del número de tokens: limitar el número de tokens que el modelo genera.
- Modo de elección de la salida:
    - Greedy: se elige la palabra/token con la mayor probabilidad.
    - Random sampling: se elige aleatoriamente una palabra basándose en los pesos de cada probabilidad.
- Sampling top k: solo seleccionar entre los `k` tokens que tengan mayor probabilidad.
- Sampling top p: solo seleccionar entre los que la suma de probabilidades sea menor o igual a `p`
- Temperature: tiene influencia sobre la forma de la distribución de la probabilidad que el modelo calcula para el siguiente token. Cuanta más alta (>1) sea la temperatura, más aleatorio será; es decir, cuanto más baja sea (<1), más tendrá en cuenta la probabilidad de cada palabra.

## Parámetros eficientees fine-tuning (PEFT)

- **Selección**: seleccionar un subconjunto de los parámetros de LLM para fine-tune
- **Reparametrización**: reparametrizar los pesos delmodelo usando una representación de bajo rango. **LoRA**
- **Aditivo**: agregar capas o parámetros entrenables al modelo. Adaptadores o indicaciones suaves (*soft prompts*, *Prompt Tuning*).

### Técnica 1: LoRA
Reduce el número de parámetros al entrenar

- Congela los parámetros del modelo original
- Inyecta dos matrices de descomposición de rangos
- Entrena los pesos de las matrices pequeñas

### Técnica 2: indicaciones suaves (soft prompts)

Con prompt tuning, se añaden tokens entrenables adicionables al prompt. El conjunto de tokens entrenables se llama **soft prompt** y suelen ser 20-100 tokens.

En fine tuning, el conjunto de entrenamiento consiste en prompts de entrada y etiquetas o *completions* de salida. Los pesos del LLM se actualizan durante el aprendizaje supervisado.

Por otra parte, con prompt tuning, los pesos del LLM se congelan y el modelo no se actualiza. En cambio, los vectores del soft prompt se actualizan con el tiempo para optimizar el *completion* del modelo del prompt


## Evaluación del modelo: métricas

### LLM

$$accuracy = \frac{\text{predicciones correctas}}{\text{total predicciones}}$$

- ROUGE
  - Se usa para resúmenes de texto.
  - Compara un resumen con otro o más resumenes.
  - $recall = \frac{\text{coincidencias de unigramas}}{\text{unigramas en referencia}}$
  - $precision = \frac{\text{coincidencias de unigramas}}{\text{unigramas en la salida}}$
  - $F1 = 2 \frac{precision \cdot recall}{precision + recall}$
- BLEU SCORE
  - Se usa para traducciones.
  - Compara las traducciones hechas por personas.
  - $\text{bleu metric} = avg(\text{precision en todo el rango de tamaños de n gramos})$



## Aprendizaje reforzado a partir de la retroalimentación humana (RLHF)

Se aplica fine-tuning con la retroalimentación de las respuestas humanas.

El aprendizaje reforzado (RL) es un tipo de aprendizaje automático enel cual un agente aprende a tomar decisiones relacionadas con un objetivo específico tomando acciones en un ambiente. El objetivo es maximizar la recompensa recibida por las acciones.

En este caso, el agente es el LLM y su objetivo es generar texto alineado.

El primer paso es seleccionar el modelo y usarlo para preparar el conjunto de datos para la retroalimentación humana. Se usa este LLM junto con un dataset de entrada para generar un número diferente de respuestas para cada entrada. El dataset de entrada está compuesto de múltiples entradas, cada uno se procesa por el LLM para producir un conjunto de terminaciones. 

El siguiente paso es conseguir retroalimentación humana sobre las terminaciones generadas. Para ello, se define el criterio que debe seguir el modelo y para cada respuesta que se ha generado, se obtiene la retroalimentación humana a través de etiquetas.

```
Prompt: My house is too hot

# Generation
Completion 1: There is nothing you can do about hot houses.
Completion 2: You can cool your house with air conditioning.
Completion 3: It is not too hot.

# Rank the completions with criterion (helpfulness)
Completion 1: 2
Completion 2: 1
Completion 3: 3
```

Una vez asignado el ranking, dependiendo del número N de terminaciones alternativas por prompt, se tendrá N combinaciones de dos terminaciones. Para cada par, se asignará una recompensa de 1 para la respuesta preferida y 0 para la respuesta menos preferida.

```
# Convert rankings 
Completion 1 y 2: [0, 1]
Completion 1 y 3: [1, 0]
Completion 2 y 3: [1, 0]
```

Luego se reordenan los prompts para que la opción preferida esté primero.

```
# Reordenar rankings 
Completion 2 y 1: [1, 0]
Completion 1 y 3: [1, 0]
Completion 2 y 3: [1, 0]
```


