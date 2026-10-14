# TP2 – Clasificación supervisada: Bank Marketing

72.75 Aprendizaje Automático (Machine Learning) · ITBA · 2026 Q2

**Integrantes:** _completar_

Predicción de si un cliente contratará un depósito a plazo fijo (`y` = yes/no) a partir de información disponible **antes** de la llamada, para priorizar a quién llamar en una campaña telefónica.
Dataset: [Bank Marketing (Kaggle)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) – 41,188 llamadas, 20 variables, 11.3% de `yes`.

## Estructura

```
├── data/
│   ├── bank-additional-full.csv        dataset original
│   └── ds_{train,test}_set{1,2,3}_*.csv subconjuntos generados por el notebook 1
├── notebooks/                           ejecutar en orden
├── figures/                             gráficas para la presentación (01–16)
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Consigna | Contenido |
|---|---|---|
| `01_preprocesamiento_eda.ipynb` | 1 | Limpieza, split 80/20 estratificado, exclusión de `duration` (data leakage), EDA, codificación y selección de variables (información mutua, VIF) → 3 subconjuntos candidatos |
| `02_clasificacion.ipynb` | 2 | Comparación de los 3 sets y de Naive Bayes, SVM, KNN y Random Forest con validación cruzada estratificada (k = 5) sobre train; métricas ROC-AUC y F1 |
| `03_ajuste_hiperparametros.ipynb` | 3 | Curvas de validación: SVM (`C`), KNN (`n_neighbors`), Random Forest (`max_depth`) |
| `04_modelo_final.ipynb` | 4 | Random Forest final entrenado con todo train y evaluado una única vez en test |
| `05_conclusiones.ipynb` | 5 | Comparación de modelos, supuestos, errores relevantes, limitaciones y mejoras |

Sets de variables: el notebook 1 propone 3 (14, 7 y 5 variables); en el notebook 2 el **set 1 (14 variables)** obtiene igual o mejor ROC-AUC en los 4 modelos y se usa para todos.

## Resultados

| Modelo (ajustado) | ROC-AUC validación | F1 validación |
|---|---|---|
| **Random Forest** (`max_depth = 8`) | **0.798** | **0.465** |
| KNN (`k = 401`) | 0.786 | 0.317 |
| Naive Bayes | 0.777 | 0.421 |
| SVM (`C = 0.01`) | 0.776 | 0.404 |

**Modelo final en test:** ROC-AUC **0.813** · F1 **0.489** · recall de `yes` 64% · precisión 40%.

## Cómo ejecutar

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Abrir los notebooks de `notebooks/` con el kernel `.venv` y ejecutarlos en orden (el 1 genera los archivos de `data/` que usan los demás). `random_state = 42` en todo el proyecto. El notebook 3 tarda ≈ 20 min (SVM).
