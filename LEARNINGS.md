# Learnings — 2026-06-27

## Qué se hizo
Proyecto de Árboles de Decisión para clasificación de 3 especies de pingüinos (Palmer Penguins). El proyecto cubre desde la carga y exploración de datos reales hasta la optimización del modelo con poda y validación cruzada.

## Conceptos nuevos
- [x] Árboles de Decisión: cómo funcionan las preguntas secuenciales sí/no
- [x] Gini Impurity: medida de pureza de un nodo (0 = puro, 0.5 = mezcla máxima)
- [x] Feature Importance: ranking automático de qué variables importan más
- [x] Sobreajuste (Overfitting) en árboles: relación profundidad vs rendimiento
- [x] max_depth: control de profundidad para evitar sobreajuste
- [x] Poda por coste-complejidad (ccp_alpha): poda post-entrenamiento
- [x] Validación cruzada (cross_val_score) para estimación robusta
- [x] Visualización del árbol con plot_tree
- [x] Matriz de confusión: qué clases se confunden entre sí
- [x] Diferencia entre árboles y KNN (interpretabilidad, escalado, velocidad)

## Dificultades encontradas
- El árbol sin límite (profundidad 15+) memoriza los datos: 100% train, ~94% test
- Con profundidad 3 se obtiene ~96% test — el punto óptimo está entre 3 y 5
- ccp_alpha produce árboles más sencillos pero el accuracy pico es similar a max_depth simple
- La variable island (isla) resultó ser la segunda más importante, lo cual tiene sentido biológico (especies distintas viven en islas distintas)

## Notas para el próximo proyecto
- Random Forest es el siguiente paso natural
- Añadir más datasets de clasificación (ej: Titanic, Iris, o datos de salud)
- Entender cómo Random Forest mejora el sobreajuste de árboles individuales
- Feature importance será aún más útil en Random Forest (más robusto)
