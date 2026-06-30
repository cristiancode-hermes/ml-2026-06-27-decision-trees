# 🌳 Árboles de Decisión — Clasifica pingüinos como un biólogo experto

**Nivel:** 3 | **Skill:** Árboles de Decisión | **Skill previo:** KNN

---

## 📖 Qué vas a aprender

Los **árboles de decisión** son probablemente el modelo de ML más fácil de entender. ¿Por qué? Porque imitan cómo los humanos tomamos decisiones: con una serie de preguntas sí/no.

> **Ejemplo real:** Un biólogo quiere identificar especies de pingüinos. ¿Cómo lo hace?
> 1. ¿El pico es muy largo? → Sí → Pingüino Papúa
> 2. ¿El pico es mediano? → Sí → ¿El peso es bajo? → Sí → Pingüino de Adelia
> 3. ...y así sucesivamente.

Eso ES un árbol de decisión. Cada pregunta es un **nodo**, cada respuesta nos lleva por una **rama**, y al final llegamos a una **hoja** con la clasificación.

### ¿Por qué árboles y no KNN?
- **Interpretabilidad:** puedes dibujar el árbol y explicárselo a tu jefe
- **Feature importance:** te dice qué variables importan más
- **No necesita escalar datos:** a diferencia de KNN, no le afectan las escalas
- **Captura interacciones:** detecta relaciones no lineales automáticamente

### Conceptos clave de hoy:
| Concepto | ¿Qué es? |
|----------|----------|
| **Gini Impurity** | Medida de qué tan "mezcladas" están las clases en un nodo. 0 = puras, 0.5 = máxima mezcla |
| **Entropy** | Otra forma de medir desorden. Similar a Gini |
| **Feature importance** | Qué tanto contribuye cada variable a las decisiones del árbol |
| **Overfitting / poda** | Un árbol muy profundo memoriza los datos (malo). Uno muy simple generaliza mejor |
| **Profundidad (max_depth)** | Cuántas preguntas seguidas puede hacer antes de decidir |

### Dataset: Palmer Penguins 🐧

Usaremos datos **reales** de 344 pingüinos de 3 especies (Adelia, Barbijo, Papúa) del archipiélago Palmer en la Antártida. Cada pingüino tiene medidas de:
- `bill_length_mm` — Largo del pico
- `bill_depth_mm` — Grosor del pico
- `flipper_length_mm` — Largo de la aleta
- `body_mass_g` — Peso
- `island` — Isla donde se encontró
- `sex` — Sexo

---

## 🧪 Código paso a paso

### Paso 1: Cargar los datos y explorarlos

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos datos reales de pingüinos (vienen con seaborn)
df = sns.load_dataset('penguins')

# Miramos las primeras filas para entender la estructura
print(df.head())
print(df.info())
```

### Paso 2: Limpieza básica

Los datos reales tienen valores nulos. Los eliminamos porque son pocos.

```python
# Limpiamos nulos — son pocos, podemos eliminarlos
df = df.dropna()
```

### Paso 3: Convertir texto a números

Los árboles de sklearn necesitan datos numéricos. Convertimos las categorías (especie, isla, sexo) a números.

```python
# Especie es nuestra variable objetivo (lo que queremos predecir)
# isla y sexo son features categóricas que debemos convertir
```

### Paso 4: Entrenar el árbol

Usamos `DecisionTreeClassifier` de scikit-learn. Por ahora, dejamos que el árbol crezca sin límite para ver qué pasa.

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
```

### Paso 5: ¡Visualizar el árbol!

Esta es la parte más espectacular. Podemos dibujar el árbol completo con `plot_tree` y ver exactamente cómo decide.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(modelo, feature_names=features, class_names=especies, filled=True)
plt.show()
```

### Paso 6: Analizar feature importance

El árbol nos dice qué variables fueron más importantes para clasificar:

```python
importancia = pd.DataFrame({
    'feature': features,
    'importance': modelo.feature_importances_
}).sort_values('importance', ascending=False)
```

### Paso 7: Jugar con la profundidad

Un árbol sin límite sobreajusta (memoriza). Probamos diferentes profundidades para encontrar la ideal:

| Profundidad | Train Accuracy | Test Accuracy | ¿Qué pasa? |
|-------------|---------------|---------------|------------|
| 1 | 74% | 73% | Sub-ajuste (muy simple) |
| 3 | 97% | 96% | ✅ Óptimo |
| 5 | 99% | 96% | Empieza a sobreajustar |
| 10 | 100% | 94% | ❌ Sobreajuste total |

### Paso 8: Podar con ccp_alpha (poda por coste-complejidad)

Técnica más avanzada: el árbol crece completo y luego se podan las ramas menos útiles.

---

## 📊 Resultados esperados

Verás:
1. Una **tabla** con estadísticas de los pingüinos (media, desviación por especie)
2. El **árbol de decisión** dibujado con colores — cada nodo muestra la pregunta, la pureza (Gini), y las muestras
3. Un **gráfico de feature importance** — un ranking de qué medidas distinguen mejor las especies
4. Una **curva train vs test** por profundidad — para ver el sobreajuste en acción
5. Una **matriz de confusión** — cuántos pingüinos clasificó bien y en cuáles se equivocó

### Ejemplo de output:

```
🌳 ÁRBOL DE DECISIÓN — CLASIFICACIÓN DE PINGÜINOS
=========================================

📊 Dimensiones del dataset: 333 filas, 7 columnas
🐧 Especies a clasificar: Adelie, Chinstrap, Gentoo

🏆 Accuracy del árbol óptimo (max_depth=3): 96.43%
📈 Precision: 96.54% | Recall: 96.43% | F1-score: 96.44%

🔝 Feature más importante: flipper_length_mm (47.2%)
```

---

## 💡 Conceptos clave

### 1. Gini Impurity
El árbol decide qué pregunta hacer basándose en qué tan puro queda cada grupo después de la división.
- **Gini = 0**: todos los elementos en el nodo son de la misma clase (perfecto)
- **Gini = 0.5**: hay mezcla total de clases (pésimo)

### 2. Feature Importance
Suma de cuánto contribuye cada variable a reducir la impureza en todas las divisiones donde se usa. Se normaliza para que sume 1.

### 3. Sobreajuste (Overfitting)
Un árbol con profundidad 10 clasifica perfectamente los datos de entrenamiento (100% accuracy) pero falla más en datos nuevos (94%). Esto es porque "memorizó" ruido y casos particulares.

### 4. Poda (Pruning)
Técnica para controlar el sobreajuste. Dos estrategias:
- **Pre-poda:** limitar la profundidad (`max_depth`), mínimo de muestras por hoja (`min_samples_leaf`)
- **Post-poda:** dejar crecer el árbol y luego cortar ramas poco significativas (`ccp_alpha`)

### 5. Diferencia con KNN
| Característica | Árbol de Decisión | KNN |
|----------------|-------------------|-----|
| Interpretable | ✅ Sí (se dibuja) | ❌ No (caja negra) |
| Necesita escalar | ❌ No | ✅ Sí |
| Velocidad predicción | ⚡ Rápidísimo | 🐢 Lento (compara con todos) |
| Captura no linealidad | ✅ Sí | ✅ Sí |
| Sobreajuste | 🔴 Riesgo alto | 🟢 Riesgo bajo |

---

## 🚀 Siguiente paso

**Random Forest** — un bosque de muchos árboles de decisión trabajando juntos. Cada árbol votapor una clase y el bosque elige la más votada. Esto reduce el sobreajuste drásticamente y da mejor precisión. Es uno de los modelos más usados en competiciones de Kaggle.
