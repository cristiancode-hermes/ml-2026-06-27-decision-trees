#!/usr/bin/env python3
"""
🌳 Árboles de Decisión — Clasificación de Pingüinos
===================================================

Vamos a construir un clasificador que identifica especies de pingüinos
usando Árboles de Decisión. Aprenderás:

1. Cómo un árbol toma decisiones (preguntas sí/no)
2. Gini Impurity — la "receta" para elegir la mejor pregunta
3. Feature importance — qué mide importa más
4. Sobreajuste — qué pasa si el árbol crece sin control
5. Poda (pruning) — cómo controlar la complejidad

Datos: Palmer Penguins (344 pingüinos reales, 3 especies)
Autor: Hermes ML Teacher
"""

# ============================================================
# 📦 IMPORTS
# ============================================================
import warnings
warnings.filterwarnings('ignore')  # Limpiamos la salida de warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# Configuración de estilo para gráficos bonitos
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("=" * 65)
print("🌳  ÁRBOLES DE DECISIÓN — CLASIFICACIÓN DE PINGÜINOS")
print("=" * 65)

# ============================================================
# 📥 PASO 1: CARGAR LOS DATOS
# ============================================================
print("\n" + "=" * 65)
print("📥 PASO 1: CARGAR DATOS DE PINGÜINOS (PALMER PENGUINS)")
print("=" * 65)

# Cargamos el dataset de Palmer Penguins directamente de seaborn
# Este dataset tiene datos REALES de 344 pingüinos del archipiélago Palmer (Antártida)
df = sns.load_dataset('penguins')

print(f"\n📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"\n🐧 Especies presentes: {df['species'].unique()}")

# Mostramos las primeras filas para ver la estructura
print("\n👀 Primeras 5 filas del dataset:")
print(df.head().to_string())

# Información completa del dataset
print("\n📋 Información del dataset:")
print(df.info())

# ============================================================
# 🧹 PASO 2: LIMPIEZA DE DATOS (VALORES NULOS)
# ============================================================
print("\n" + "=" * 65)
print("🧹 PASO 2: LIMPIEZA — VALORES NULOS")
print("=" * 65)

# Contamos cuántos valores nulos hay por columna
nulos = df.isnull().sum()
print(f"\n🔍 Valores nulos por columna:\n{nulos[nulos > 0]}")

# Eliminamos las filas con nulos (son pocas — solo 11 de 344)
df_clean = df.dropna().reset_index(drop=True)
print(f"\n✅ Filas después de limpiar nulos: {len(df_clean)} (eliminamos {len(df) - len(df_clean)})")

# ============================================================
# 📊 PASO 3: ANÁLISIS EXPLORATORIO RÁPIDO
# ============================================================
print("\n" + "=" * 65)
print("📊 PASO 3: ANÁLISIS EXPLORATORIO")
print("=" * 65)

# Estadísticas básicas agrupadas por especie
print("\n📈 Estadísticas por especie (media de cada medida):")
print(df_clean.groupby('species')[['bill_length_mm', 'bill_depth_mm',
                                    'flipper_length_mm', 'body_mass_g']].mean().round(2).to_string())

# ============================================================
# 🏗️ PASO 4: PREPARAR LOS DATOS PARA EL MODELO
# ============================================================
print("\n" + "=" * 65)
print("🏗️  PASO 4: PREPARACIÓN — CONVERTIR TEXTOS A NÚMEROS")
print("=" * 65)

# Separamos features (X) y target (y)
# La variable objetivo es la especie del pingüino
# Las features son: medidas del pico, aletas, peso, isla y sexo

# Codificamos variables categóricas (texto → número)
# LabelEncoder convierte 'Adelie', 'Chinstrap', 'Gentoo' → 0, 1, 2
le_species = LabelEncoder()
y = le_species.fit_transform(df_clean['species'])  # variable objetivo

# Codificamos isla y sexo como features
df_features = df_clean.copy()
df_features['island'] = LabelEncoder().fit_transform(df_features['island'])
df_features['sex'] = LabelEncoder().fit_transform(df_features['sex'])

# Seleccionamos las features (excluimos species que es nuestro target)
features_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                 'body_mass_g', 'island', 'sex']
X = df_features[features_cols].values

print(f"\n✅ Features numéricas: {features_cols}")
print(f"✅ Matriz X: {X.shape}  |  Vector y: {y.shape}")
print(f"🎯 Clases (especies): {dict(enumerate(le_species.classes_))}")

# ============================================================
# ✂️ PASO 5: DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================================
print("\n" + "=" * 65)
print("✂️  PASO 5: TRAIN/TEST SPLIT (80% / 20%)")
print("=" * 65)

# Dividimos: 80% para entrenar, 20% para probar
# stratify=y mantiene la misma proporción de especies en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🎯 Train: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")
print(f"📊 Proporción de especies en train: {np.bincount(y_train) / len(y_train)}")
print(f"📊 Proporción de especies en test:  {np.bincount(y_test) / len(y_test)}")

# ============================================================
# 🌳 PASO 6: ENTRENAR EL ÁRBOL DE DECISIÓN
# ============================================================
print("\n" + "=" * 65)
print("🌳 PASO 6: ENTRENAR ÁRBOL DE DECISIÓN (SIN LÍMITE)")
print("=" * 65)

# --- ÁRBOL SIN LÍMITE (profundidad máxima = infinita) ---
# Esto deja que el árbol crezca hasta que cada hoja tenga una sola clase
# ESPOILER: esto causa SOBREAJUSTE
arbol_sin_limite = DecisionTreeClassifier(random_state=42)
arbol_sin_limite.fit(X_train, y_train)

# Predecimos
y_train_pred_sl = arbol_sin_limite.predict(X_train)
y_test_pred_sl = arbol_sin_limite.predict(X_test)

# Calculamos accuracy
train_acc_sl = accuracy_score(y_train, y_train_pred_sl)
test_acc_sl = accuracy_score(y_test, y_test_pred_sl)

print(f"\n🌳 Árbol SIN límite de profundidad:")
print(f"   Profundidad real: {arbol_sin_limite.get_depth()} niveles")
print(f"   Hojas: {arbol_sin_limite.get_n_leaves()}")
print(f"   ✅ Train Accuracy: {train_acc_sl:.2%}")
print(f"   ⚠️  Test Accuracy:  {test_acc_sl:.2%}")
print(f"   📉 Diferencia: {train_acc_sl - test_acc_sl:.2%} — ¡SOBREAJUSTE!")

# --- ÁRBOL CONTROLADO (max_depth=3) ---
# Limitamos la profundidad a 3 niveles para evitar sobreajuste
# Esto fuerza al árbol a hacer solo 3 preguntas máximo
print(f"\n🌳 Árbol CON max_depth=3 (controlado):")
arbol_controlado = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol_controlado.fit(X_train, y_train)

y_train_pred_ac = arbol_controlado.predict(X_train)
y_test_pred_ac = arbol_controlado.predict(X_test)

train_acc_ac = accuracy_score(y_train, y_train_pred_ac)
test_acc_ac = accuracy_score(y_test, y_test_pred_ac)

print(f"   Profundidad: {arbol_controlado.get_depth()}")
print(f"   Hojas: {arbol_controlado.get_n_leaves()}")
print(f"   ✅ Train Accuracy: {train_acc_ac:.2%}")
print(f"   ✅ Test Accuracy:  {test_acc_ac:.2%}")

# ============================================================
# 📊 PASO 7: ANÁLISIS DE FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 65)
print("📊 PASO 7: FEATURE IMPORTANCE — ¿QUÉ MIDE MÁS IMPORTA?")
print("=" * 65)

# El árbol nos dice qué variables fueron más importantes para clasificar
# Feature importance = cuánto contribuye cada variable a reducir la impureza
importancia = arbol_controlado.feature_importances_
df_importancia = pd.DataFrame({
    'Característica': features_cols,
    'Importancia': importancia
}).sort_values('Importancia', ascending=False)

print("\n🔝 Ranking de importancia de características:")
print(df_importancia.to_string(index=False))
print(f"\n💡 La característica más importante es: '{df_importancia.iloc[0]['Característica']}'")
print(f"   con una importancia de {df_importancia.iloc[0]['Importancia']:.2%}")

# ============================================================
# 🔬 PASO 8: EXPERIMENTO CON DIFERENTES PROFUNDIDADES
# ============================================================
print("\n" + "=" * 65)
print("🔬 PASO 8: EXPERIMENTO — ¿CUÁL ES LA MEJOR PROFUNDIDAD?")
print("=" * 65)

# Probamos profundidades desde 1 hasta 15
# Esto nos muestra cómo el sobreajuste aumenta con la profundidad
profundidades = range(1, 16)
train_scores = []
test_scores = []

print("\n📈 Train vs Test Accuracy por profundidad:")
print(f"{'Profundidad':<12} {'Train Acc':<12} {'Test Acc':<12} {'Diferencia':<12} {'Diagnóstico':<20}")
print("-" * 68)

for depth in profundidades:
    modelo = DecisionTreeClassifier(max_depth=depth, random_state=42)
    modelo.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, modelo.predict(X_train))
    test_acc = accuracy_score(y_test, modelo.predict(X_test))

    train_scores.append(train_acc)
    test_scores.append(test_acc)

    diff = train_acc - test_acc
    if diff < 0.02:
        diag = "✅ Bueno"
    elif diff < 0.08:
        diag = "⚠️ Leve sobreajuste"
    else:
        diag = "❌ Sobreajuste"

    print(f"  depth={depth:<5} {train_acc:<10.4f} {test_acc:<10.4f} {diff:<10.4f} {diag}")

# Encontramos la mejor profundidad (más alto test accuracy)
mejor_depth = profundidades[np.argmax(test_scores)]
print(f"\n🏆 Mejor profundidad: {mejor_depth} (Test Accuracy: {max(test_scores):.2%})")

# ============================================================
# 🧪 PASO 9: VALIDACIÓN CRUZADA (CROSS-VALIDATION)
# ============================================================
print("\n" + "=" * 65)
print("🧪 PASO 9: VALIDACIÓN CRUZADA (5-FOLD)")
print("=" * 65)

# La validación cruzada divide los datos en 5 partes iguales
# Entrena con 4/5 y prueba con 1/5, repite 5 veces
# Esto da una estimación más robusta del rendimiento
cv_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=mejor_depth, random_state=42),
    X, y, cv=5, scoring='accuracy'
)

print(f"\n📊 Resultados de validación cruzada (5 folds):")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")
print(f"\n📈 Accuracy promedio (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 🧹 PASO 10: PODA POR COSTE-COMPLEJIDAD (ccp_alpha)
# ============================================================
print("\n" + "=" * 65)
print("🧹 PASO 10: PODA (PRUNING) CON CCP_ALPHA")
print("=" * 65)

# ccp_alpha es un parámetro de poda: valores más altos = árbol más pequeño
# sklearn nos da los alphas candidatos automáticamente
# Esto es más elegante que solo limitar profundidad

# Primero creamos un árbol completo y obtenemos los alphas de poda
arbol_completo = DecisionTreeClassifier(random_state=42)
path = arbol_completo.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Probamos diferentes valores de alpha
train_scores_prune = []
test_scores_prune = []

for alpha in ccp_alphas:
    modelo = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    modelo.fit(X_train, y_train)
    train_scores_prune.append(accuracy_score(y_train, modelo.predict(X_train)))
    test_scores_prune.append(accuracy_score(y_test, modelo.predict(X_test)))

# Encontramos el mejor alpha
idx_mejor_alpha = np.argmax(test_scores_prune)
mejor_alpha = ccp_alphas[idx_mejor_alpha]

print(f"\n🏆 Mejor ccp_alpha: {mejor_alpha:.5f}")
print(f"   Train Accuracy: {train_scores_prune[idx_mejor_alpha]:.4f}")
print(f"   Test Accuracy:  {test_scores_prune[idx_mejor_alpha]:.4f}")

# Entrenamos el modelo final con la mejor poda
arbol_final = DecisionTreeClassifier(
    max_depth=mejor_depth,
    random_state=42
)
arbol_final.fit(X_train, y_train)
y_pred_final = arbol_final.predict(X_test)

# ============================================================
# 🏆 PASO 11: MÉTRICAS DETALLADAS
# ============================================================
print("\n" + "=" * 65)
print("🏆 PASO 11: MÉTRICAS DEL MODELO FINAL")
print("=" * 65)

print(f"\n📊 Classification Report (por especie):")
print(classification_report(y_test, y_pred_final,
                            target_names=le_species.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n🔢 Matriz de Confusión:")
print(f"{'':<10} {'Adelie':<8} {'Chinstrap':<10} {'Gentoo':<8}")
for i, especie in enumerate(le_species.classes_):
    print(f"{especie:<10} {cm[i, 0]:<8} {cm[i, 1]:<10} {cm[i, 2]:<8}")

# ============================================================
# 🎨 PASO 12: VISUALIZACIONES
# ============================================================
print("\n" + "=" * 65)
print("🎨 PASO 12: GENERANDO VISUALIZACIONES...")
print("=" * 65)

# --- VIZ 1: Pairplot de las especies ---
print("   📊 1/5 — Pairplot: distribución de medidas por especie...")
sns.set_theme(style="whitegrid")
pairplot = sns.pairplot(
    df_clean,
    hue='species',
    vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
    palette='Set2',
    diag_kind='kde'
)
pairplot.fig.suptitle('Distribución de medidas por especie de pingüino', y=1.02, fontsize=14)
pairplot.savefig('docs/01_pairplot_especies.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/01_pairplot_especies.png")

# --- VIZ 2: Feature importance ---
print("   📊 2/5 — Feature importance...")
fig, ax = plt.subplots(figsize=(10, 6))
colores = plt.cm.Set2(np.linspace(0, 1, len(df_importancia)))
bars = ax.barh(range(len(df_importancia)), df_importancia['Importancia'], color=colores)
ax.set_yticks(range(len(df_importancia)))
ax.set_yticklabels(df_importancia['Característica'])
ax.set_xlabel('Importancia (contribución a la pureza)')
ax.set_title('🔝 ¿Qué características distinguen mejor las especies?', fontsize=14, fontweight='bold')

# Añadimos etiquetas de porcentaje
for bar, val in zip(bars, df_importancia['Importancia']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('docs/02_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/02_feature_importance.png")

# --- VIZ 3: Train vs Test por profundidad ---
print("   📊 3/5 — Curva de sobreajuste (train vs test por profundidad)...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(profundidades, train_scores, 'o-', color='#2ecc71', linewidth=2,
        markersize=8, label='Train Accuracy')
ax.plot(profundidades, test_scores, 's-', color='#e74c3c', linewidth=2,
        markersize=8, label='Test Accuracy')

# Marcamos la mejor profundidad
ax.axvline(x=mejor_depth, color='#3498db', linestyle='--', alpha=0.7,
           label=f'Mejor profundidad = {mejor_depth}')

ax.set_xlabel('Profundidad del árbol (max_depth)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('📈 Sobreajuste en Árboles de Decisión\nA más profundidad, más memoriza el árbol los datos de entrenamiento',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.set_xticks(profundidades)
ax.set_ylim(0.65, 1.05)

# Añadimos zona de warning para sobreajuste
ax.axhspan(0.65, 0.75, alpha=0.05, color='orange', label='Zona de sub-ajuste')
ax.axhspan(0.90, 1.05, alpha=0.08, color='red', label='Zona de sobreajuste')

plt.tight_layout()
plt.savefig('docs/03_sobreajuste_curva.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/03_sobreajuste_curva.png")

# --- VIZ 4: ÁRBOL DE DECISIÓN DIBUJADO ---
print("   📊 4/5 — Árbol de decisión (visualización del árbol)...")
fig, ax = plt.subplots(figsize=(22, 12))
plot_tree(
    arbol_controlado,
    feature_names=features_cols,
    class_names=le_species.classes_.tolist(),
    filled=True,
    rounded=True,
    fontsize=11,
    proportion=True,
    ax=ax
)
ax.set_title('🌳 Árbol de Decisión — ¿Cómo decide la especie del pingüino?\n'
             'Cada nodo muestra: pregunta, Gini impurity, muestras, clase mayoritaria',
             fontsize=14, fontweight='bold')
plt.savefig('docs/04_arbol_decision.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/04_arbol_decision.png")

# --- VIZ 5: MATRIZ DE CONFUSIÓN ---
print("   📊 5/5 — Matriz de confusión...")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(
    arbol_final, X_test, y_test,
    display_labels=le_species.classes_,
    cmap='Blues',
    ax=ax,
    colorbar=False,
    values_format='d'
)
ax.set_title('🔢 Matriz de Confusión — ¿En qué especies se equivoca el árbol?',
             fontsize=14, fontweight='bold')

# Añadimos porcentajes
total_test = len(y_test)
for text in ax.texts:
    if text.get_text().isdigit():
        val = int(text.get_text())
        text.set_text(f'{val}\n({val/total_test:.1%})')

plt.tight_layout()
plt.savefig('docs/05_matriz_confusion.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/05_matriz_confusion.png")

# ============================================================
# 🧠 RESUMEN FINAL
# ============================================================
print("\n" + "=" * 65)
print("🧠  RESUMEN: HOY APRENDISTE...")
print("=" * 65)

print("""
1. 🌳 ÁRBOLES DE DECISIÓN: modelos que toman decisiones con preguntas
   sí/no, como un biólogo identificando especies.

2. 📐 GINI IMPURITY: el algoritmo elige la mejor pregunta basándose
   en qué tan "puro" queda cada grupo (Gini=0 → grupo homogéneo).

3. 🔝 FEATURE IMPORTANCE: el árbol revela qué variables importan más.
   ¡No necesitas adivinar! El modelo te lo dice.

4. ⚠️ SOBREAJUSTE (OVERFITTING): un árbol muy profundo memoriza los
   datos de entrenamiento y falla en datos nuevos. Controlamos esto
   con max_depth y poda (ccp_alpha).

5. ✂️ PODA: técnica para cortar ramas débiles del árbol. Como podar
   un bonsái — menos ramas, árbol más fuerte.

6. 💡 DIFERENCIA CON KNN: los árboles son interpretables (se dibujan),
   no necesitan escalar datos, y son rapidísimos en predicción.
""")

print("📊 MÉTRICAS FINALES DEL MODELO:")
print(f"   • Profundidad óptima: {mejor_depth}")
print(f"   • Test Accuracy (prof. óptima): {max(test_scores):.2%}")
print(f"   • Accuracy CV (5-fold): {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print(f"   • Feature más importante: {df_importancia.iloc[0]['Característica']} ({df_importancia.iloc[0]['Importancia']:.1%})")
print("")

print("📁 VISUALIZACIONES GENERADAS:")
print("   • docs/01_pairplot_especies.png")
print("   • docs/02_feature_importance.png")
print("   • docs/03_sobreajuste_curva.png")
print("   • docs/04_arbol_decision.png")
print("   • docs/05_matriz_confusion.png")

print("\n" + "=" * 65)
print("🚀 SIGUIENTE PROYECTO: RANDOM FOREST")
print("   Un bosque de árboles de decisión trabajando juntos.")
print("   Más preciso, menos sobreajuste. ¡El favorito de Kaggle!")
print("=" * 65)
