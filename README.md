# Predicción de Actividad Comercial en Madrid

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production-green.svg)](https://github.com/alexandrobg/madrid-commercial-prediction)

## Resumen Ejecutivo

Este proyecto implementa un sistema de Machine Learning para predecir la inactividad de locales comerciales en Madrid, utilizando datos administrativos del Ayuntamiento de Madrid (2020-2024) enriquecidos con indicadores socioeconómicos. El modelo principal (MLP) alcanza una precisión del 88.48% y un AUC de 0.854, proporcionando una herramienta robusta para la toma de decisiones en planificación urbana y política comercial.

## Descripción del Proyecto

### Contexto y Motivación

Madrid, como capital económica de España, concentra más de 60,000 establecimientos comerciales que emplean directamente a más de 305,000 personas. Sin embargo, eventos como la pandemia COVID-19 y cambios en los patrones de consumo han generado cierres comerciales desiguales según ubicación y sector. Este proyecto aborda la necesidad de anticipar estos cierres para optimizar políticas públicas y decisiones de inversión privada.

### Características Técnicas

- **Dataset**: 8.3+ millones de registros históricos (2020-2024)
- **Variables**: Geolocalización, clasificación CNAE, tipo de acceso, renta per cápita, población
- **Metodología**: SEMMA (SAS Enterprise Miner) con validación temporal
- **Modelos**: 9 algoritmos diferentes con optimización de hiperparámetros
- **Interpretabilidad**: Análisis SHAP para explicabilidad del modelo

## Resultados Principales

### Rendimiento de Modelos (Conjunto de Test 2024)

| Modelo | AUC | F1-Score | Accuracy | Características |
|--------|-----|----------|----------|----------------|
| **MLP** | **0.8540** | **0.9253** | **0.8848** | Mejor equilibrio global; seleccionado por parsimonia |
| VotingClassifier | 0.8541 | 0.9242 | 0.8828 | Ensamble robusto (RF + XGB + MLP) |
| Random Forest | 0.8489 | 0.9250 | 0.8841 | Alto recall; interpretable |
| XGBoost | 0.8491 | 0.9151 | 0.8673 | Buen AUC; conservador en especificidad |
| Regresión Logística | 0.8475 | 0.9253 | 0.8848 | Baseline sólido y estable |

### Variables Más Influyentes (Análisis SHAP)

1. **desc_seccion_sin_actividad** — Indicador de falta de actividad (impacto negativo)
2. **desc_tipo_acceso_local_puerta_calle** — Acceso directo desde calle (impacto positivo)
3. **Renta_Media** — Renta media del distrito (mayor renta → mayor actividad)
4. **Total_Poblacion** — Población del barrio (impacto positivo moderado)
5. **desc_seccion_hosteleria** — Presencia del sector hostelería (impacto positivo)

## Arquitectura del Proyecto

```
madrid-commercial-prediction/
├── Makefile
├── README.md
├── TFM-Alexandro Bazan Guardia.pdf
├── VERIFICATION.md
├── requirements.txt
├── setup.py
├── variables_boruta.txt
├── variables_rfecv.txt
├── variables_sbf.txt
├── variables_shap.txt
├── variables_stepwise.txt
├── data/
│   ├── raw/                    # Datos originales
│   │   ├── Actividades Economicas de Madrid.csv
│   │   ├── MadridActividades.csv
│   │   ├── RentaPOB.xlsx
│   │   └── actividadeconomicamadrid.csv
│   ├── processed/              # Datos procesados
│   │   └── df_limpio.pkl
│   └── external/               # Fuentes externas
├── src/
│   ├── __init__.py
│   ├── main.py                 # Pipeline principal
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Carga de datos
│   │   ├── data_cleaner.py     # Limpieza y normalización
│   │   └── data_preprocessor.py # Preprocesamiento
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_selector.py  # Selección de variables (Boruta, RFECV, etc.)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_models.py     # Entrenamiento de 9 modelos
│   │   └── model_evaluation.py # Evaluación y comparación
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuración
│       └── helpers.py          # Utilidades
├── results/
│   ├── figures/                # Curvas ROC, matrices de confusión, SHAP
│   ├── logs/                   # Archivos de log
│   │   └── madrid_prediction.log
│   └── reports/                # Reportes CSV y HTML
│       ├── feature_selection_consensus.csv
│       ├── model_comparison.csv
│       └── shap_importances.csv
└── models/                     # Modelos entrenados y artefactos
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── logistic_regression_model.pkl
    ├── mlp_model.pkl
    ├── random_forest_model.pkl
    ├── stacking_classifier_model.pkl
    ├── svm_model.pkl
    ├── voting_classifier_model.pkl
    ├── xgboost_model.pkl
    ├── scaler.joblib
    ├── selected_features_*.pkl  # Features seleccionadas
    ├── X_train.joblib
    ├── X_train_scaled.joblib
    ├── X_test.joblib
    ├── X_test_scaled.joblib
    ├── y_train.joblib
    └── y_test.joblib
```

## Instalación y Uso

### Prerrequisitos

- Python 3.9+
- 16GB+ RAM (recomendado para dataset completo)
- Espacio en disco: 5GB+

### Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/alexandrobg/madrid-commercial-prediction.git
cd madrid-commercial-prediction

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con rutas personalizadas
```

### Datos Requeridos

Coloque los siguientes archivos en `data/raw/`:

```
data/raw/
├── Actividades Economicas de Madrid.csv  # Datos del Ayuntamiento
└── RentaPOB.xlsx                        # Datos socioeconómicos
```

**Enlace a dataset principal**: [Google Drive](https://drive.google.com/file/d/17HAOjSxzSkesvHLXmwoyr__u_nAywIL9/view?usp=drive_link)

### Ejecución

#### Pipeline Completo (Recomendado)
```bash
python -m src.main
```

#### Ejecución Modular
```bash
# Solo procesamiento de datos
python -m src.data.data_preprocessor

# Solo entrenamiento de modelos
python -m src.models.train_models

# Solo evaluación
python -m src.models.model_evaluation
```

### Preprocesamiento de Datos

1. **Limpieza**: Imputación de coordenadas geográficas, normalización de texto, corrección de inconsistencias
2. **Ingeniería de Variables**: One-hot encoding, estandarización Z-score, creación de variable objetivo binaria
3. **Selección de Variables**: 5 métodos comparados (Boruta seleccionado como óptimo)

### Modelos Implementados

- **Algoritmos Base**: Regresión Logística, Árboles de Decisión, Random Forest, SVM, KNN, XGBoost, MLP
- **Ensambles**: VotingClassifier (soft voting), StackingClassifier
- **Optimización**: GridSearchCV y RandomizedSearchCV con validación cruzada estratificada
- **Validación**: División temporal (entrenamiento: 2020-2023, test: 2024)

### Métricas de Evaluación

- **Accuracy**: Exactitud global
- **Precision/Recall**: Balance falsos positivos/negativos
- **F1-Score**: Media armónica precision-recall
- **AUC-ROC**: Capacidad discriminativa
- **Especificidad**: Detección de verdaderos negativos

## Aplicaciones Prácticas

### Casos de Uso

- **Planificación Urbana**: Identificación de zonas comerciales en riesgo
- **Política Pública**: Focalización de ayudas y subvenciones
- **Inversión Privada**: Evaluación de ubicaciones comerciales
- **Transparencia**: Open data para ciudadanía y investigadores

### Implementación Recomendada

1. **API Interna**: Despliegue como servicio REST para departamentos municipales
2. **Dashboard Interactivo**: Visualización en tiempo real de predicciones
3. **Alertas Automáticas**: Sistema de notificaciones para zonas de alto riesgo
4. **Explicabilidad**: Reportes SHAP para justificar decisiones

## Testing y Calidad

```bash
# Ejecutar todas las pruebas
python -m pytest tests/ -v

# Pruebas específicas por módulo
python -m pytest tests/test_models/ -v
python -m pytest tests/test_data/ -v
python -m pytest tests/test_features/ -v
```

### Cobertura de Pruebas

- Validación de carga de datos
- Pruebas de transformación y limpieza
- Validación de modelos entrenados
- Tests de integración del pipeline completo

## Contribuciones

### Proceso de Contribución

1. Fork del proyecto
2. Crear rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'Descripción clara'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abrir Pull Request con descripción detallada

### Estándares de Código

- PEP 8 para estilo de Python
- Docstrings en todas las funciones públicas
- Type hints cuando sea apropiado
- Tests unitarios para nueva funcionalidad

## Información del Proyecto

### Autoría

**Alexandro Bazán Guardia**
Máster en Ciencia de Datos e Inteligencia de Negocios
Universidad Complutense de Madrid

- **Email**: alexandro.bazan9712@gmail.com
- **LinkedIn**: [alexandrobg](https://www.linkedin.com/in/alexandrobg/)
- **GitHub**: [alexandrobg](https://github.com/alexandrobg)

### Supervisión Académica

**Tutora**: Belén Rodríguez-Cánovas
Facultad de Estudios Estadísticos - Universidad Complutense de Madrid

### Agradecimientos

- Facultad de Estudios Estadísticos - UCM por el soporte académico
- Ayuntamiento de Madrid por el acceso a datos públicos
- Comunidad open source por las librerías utilizadas

## Licencia y Citación

### Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo `LICENSE` para detalles completos.

### Cita Académica

Si utiliza este trabajo en investigación académica, por favor cite:

```bibtex
@mastersthesis{bazan2025madrid,
  title={Predicción de la inactividad de locales comerciales en Madrid mediante técnicas de Machine Learning},
  author={Bazán Guardia, Alexandro},
  year={2025},
  school={Universidad Complutense de Madrid},
  type={Trabajo de Fin de Máster},
  url={https://github.com/alexandrobg/madrid-commercial-prediction}
}
```

### Referencias Principales

- Ayuntamiento de Madrid. (2025). *Actividades Económicas de Madrid* [Dataset]
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32
- Lundberg, S., & Lee, S. (2017). A unified approach to interpreting model predictions. *NIPS*

## Roadmap y Desarrollo Futuro

### Versión Actual (v1.0)
- ✅ Pipeline completo de ML
- ✅ 9 modelos comparados
- ✅ Análisis SHAP
- ✅ Documentación completa

### Próximas Versiones
- 🔄 API REST para predicciones en tiempo real
- 🔄 Dashboard web interactivo
- 🔄 Integración con datos en streaming
- 🔄 Modelos de deep learning (LSTM, Transformer)
- 🔄 Análisis de series temporales avanzado

---

**Status**: Producción | **Última actualización**: Septiembre 2025 | **Versión**: 1.0.0
