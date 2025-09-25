# Commercial Activity Prediction in Madrid | Predicción de Actividad Comercial en Madrid

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production-green.svg)](https://github.com/alexandrobg/madrid-commercial-prediction)

---

## English Version

### Executive Summary

This project implements a Machine Learning system to predict commercial establishment inactivity in Madrid, utilizing administrative data from Madrid City Council (2020-2024) enriched with socioeconomic indicators. The main model (MLP) achieves 88.48% accuracy and 0.854 AUC, providing a robust tool for decision-making in urban planning and commercial policy.

### Project Description

#### Context and Motivation

Madrid, as Spain's economic capital, concentrates over 60,000 commercial establishments that directly employ more than 305,000 people. However, events such as the COVID-19 pandemic and changes in consumption patterns have generated unequal commercial closures based on location and sector. This project addresses the need to anticipate these closures to optimize public policies and private investment decisions.

#### Technical Features

- **Dataset**: 8.3+ million historical records (2020-2024)
- **Variables**: Geolocation, CNAE classification, access type, per capita income, population
- **Methodology**: SEMMA (SAS Enterprise Miner) with temporal validation
- **Models**: 9 different algorithms with hyperparameter optimization
- **Interpretability**: SHAP analysis for model explainability

### Main Results

#### Model Performance (2024 Test Set)

| Model | AUC | F1-Score | Accuracy | Features |
|--------|-----|----------|----------|----------|
| **MLP** | **0.8540** | **0.9253** | **0.8848** | Best overall balance; selected for parsimony |
| VotingClassifier | 0.8541 | 0.9242 | 0.8828 | Robust ensemble (RF + XGB + MLP) |
| Random Forest | 0.8489 | 0.9250 | 0.8841 | High recall; interpretable |
| XGBoost | 0.8491 | 0.9151 | 0.8673 | Good AUC; conservative in specificity |
| Logistic Regression | 0.8475 | 0.9253 | 0.8848 | Solid and stable baseline |

#### Most Influential Variables (SHAP Analysis)

1. **desc_seccion_sin_actividad** — No activity indicator (negative impact)
2. **desc_tipo_acceso_local_puerta_calle** — Direct street access (positive impact)
3. **Renta_Media** — District median income (higher income → greater activity)
4. **Total_Poblacion** — Neighborhood population (moderate positive impact)
5. **desc_seccion_hosteleria** — Hospitality sector presence (positive impact)

### Project Architecture

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
│   ├── raw/                    # Original data
│   │   ├── Actividades Economicas de Madrid.csv
│   │   ├── MadridActividades.csv
│   │   ├── RentaPOB.xlsx
│   │   └── actividadeconomicamadrid.csv
│   ├── processed/              # Processed data
│   │   └── df_limpio.pkl
│   └── external/               # External sources
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading
│   │   ├── data_cleaner.py     # Cleaning and normalization
│   │   └── data_preprocessor.py # Preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_selector.py  # Variable selection (Boruta, RFECV, etc.)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_models.py     # Training of 9 models
│   │   └── model_evaluation.py # Evaluation and comparison
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration
│       └── helpers.py          # Utilities
├── results/
│   ├── figures/                # ROC curves, confusion matrices, SHAP
│   ├── logs/                   # Log files
│   │   └── madrid_prediction.log
│   └── reports/                # CSV and HTML reports
│       ├── feature_selection_consensus.csv
│       ├── model_comparison.csv
│       └── shap_importances.csv
└── models/                     # Trained models and artifacts
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
    ├── selected_features_*.pkl  # Selected features
    ├── X_train.joblib
    ├── X_train_scaled.joblib
    ├── X_test.joblib
    ├── X_test_scaled.joblib
    ├── y_train.joblib
    └── y_test.joblib
```

### Installation and Usage

#### Prerequisites

- Python 3.9+
- 16GB+ RAM (recommended for full dataset)
- Disk space: 5GB+

#### Installation

```bash
# 1. Clone repository
git clone https://github.com/alexandrobg/madrid-commercial-prediction.git
cd madrid-commercial-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with custom paths
```

#### Required Data

The data required for this project consists of two main sources:

**1. Madrid Economic Activities Data**
**Link**: [Economic Activities of Madrid](https://drive.google.com/file/d/18VwmTOxIbtYH0jCHbaAlBx5lHU-ywuYt/view?usp=drive_link)

**Origin**: Monthly CSV files from Madrid City Council's open data portal (January 2020 - December 2024)

**2. Socioeconomic Data**
**Link**: [RentaPOB - Income and Population by Neighborhood](https://docs.google.com/spreadsheets/d/1iNgb0F5JdqcnIyjNU3ZsyBFVH1yP51PS/edit?usp=drive_link&ouid=106414683644608503271&rtpof=true&sd=true)

**Origin**: Madrid City Council Database with information disaggregated by neighborhood and year

#### Execution

**Complete Pipeline (Recommended)**
```bash
python -m src.main
```

**Modular Execution**
```bash
# Data processing only
python -m src.data.data_preprocessor

# Model training only
python -m src.models.train_models

# Evaluation only
python -m src.models.model_evaluation
```

### Practical Applications

#### Use Cases

- **Urban Planning**: Identification of at-risk commercial areas
- **Public Policy**: Targeting of aid and subsidies
- **Private Investment**: Evaluation of commercial locations
- **Transparency**: Open data for citizens and researchers

#### Recommended Implementation

1. **Internal API**: Deployment as REST service for municipal departments
2. **Interactive Dashboard**: Real-time prediction visualization
3. **Automatic Alerts**: Notification system for high-risk areas
4. **Explainability**: SHAP reports to justify decisions

### Testing and Quality

```bash
# Run all tests
python -m pytest tests/ -v

# Module-specific tests
python -m pytest tests/test_models/ -v
python -m pytest tests/test_data/ -v
python -m pytest tests/test_features/ -v
```

### Project Information

#### Authorship

**Alexandro Bazán Guardia**
Master in Data Science and Business Intelligence
Complutense University of Madrid

- **Email**: alexandro.bazan9712@gmail.com
- **LinkedIn**: [alexandrobg](https://www.linkedin.com/in/alexandrobg/)
- **GitHub**: [alexandrobg](https://github.com/alexandrobg)

#### Academic Supervision

**Supervisor**: Belén Rodríguez-Cánovas
Faculty of Statistical Studies - Complutense University of Madrid

### License and Citation

#### License

This project is licensed under the **MIT License**. See `LICENSE` file for complete details.

#### Academic Citation

If you use this work in academic research, please cite:

```bibtex
@mastersthesis{bazan2025madrid,
  title={Prediction of commercial establishment inactivity in Madrid using Machine Learning techniques},
  author={Bazán Guardia, Alexandro},
  year={2025},
  school={Complutense University of Madrid},
  type={Master's Thesis},
  url={https://github.com/alexandrobg/madrid-commercial-prediction}
}
```

---

## Versión en Español

### Resumen Ejecutivo

Este proyecto implementa un sistema de Machine Learning para predecir la inactividad de locales comerciales en Madrid, utilizando datos administrativos del Ayuntamiento de Madrid (2020-2024) enriquecidos con indicadores socioeconómicos. El modelo principal (MLP) alcanza una precisión del 88.48% y un AUC de 0.854, proporcionando una herramienta robusta para la toma de decisiones en planificación urbana y política comercial.

### Descripción del Proyecto

#### Contexto y Motivación

Madrid, como capital económica de España, concentra más de 60,000 establecimientos comerciales que emplean directamente a más de 305,000 personas. Sin embargo, eventos como la pandemia COVID-19 y cambios en los patrones de consumo han generado cierres comerciales desiguales según ubicación y sector. Este proyecto aborda la necesidad de anticipar estos cierres para optimizar políticas públicas y decisiones de inversión privada.

#### Características Técnicas

- **Dataset**: 8.3+ millones de registros históricos (2020-2024)
- **Variables**: Geolocalización, clasificación CNAE, tipo de acceso, renta per cápita, población
- **Metodología**: SEMMA (SAS Enterprise Miner) con validación temporal
- **Modelos**: 9 algoritmos diferentes con optimización de hiperparámetros
- **Interpretabilidad**: Análisis SHAP para explicabilidad del modelo

### Resultados Principales

#### Rendimiento de Modelos (Conjunto de Test 2024)

| Modelo | AUC | F1-Score | Accuracy | Características |
|--------|-----|----------|----------|----------------|
| **MLP** | **0.8540** | **0.9253** | **0.8848** | Mejor equilibrio global; seleccionado por parsimonia |
| VotingClassifier | 0.8541 | 0.9242 | 0.8828 | Ensamble robusto (RF + XGB + MLP) |
| Random Forest | 0.8489 | 0.9250 | 0.8841 | Alto recall; interpretable |
| XGBoost | 0.8491 | 0.9151 | 0.8673 | Buen AUC; conservador en especificidad |
| Regresión Logística | 0.8475 | 0.9253 | 0.8848 | Baseline sólido y estable |

#### Variables Más Influyentes (Análisis SHAP)

1. **desc_seccion_sin_actividad** — Indicador de falta de actividad (impacto negativo)
2. **desc_tipo_acceso_local_puerta_calle** — Acceso directo desde calle (impacto positivo)
3. **Renta_Media** — Renta media del distrito (mayor renta → mayor actividad)
4. **Total_Poblacion** — Población del barrio (impacto positivo moderado)
5. **desc_seccion_hosteleria** — Presencia del sector hostelería (impacto positivo)

### Arquitectura del Proyecto

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

### Instalación y Uso

#### Prerrequisitos

- Python 3.9+
- 16GB+ RAM (recomendado para dataset completo)
- Espacio en disco: 5GB+

#### Instalación

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

#### Datos Requeridos

Los datos necesarios para este proyecto se componen de dos fuentes principales:

**1. Datos de Actividades Económicas de Madrid**
**Enlace**: [Actividades Económicas de Madrid](https://drive.google.com/file/d/18VwmTOxIbtYH0jCHbaAlBx5lHU-ywuYt/view?usp=drive_link)

**Origen**: Archivos CSV mensuales del portal de datos abiertos del Ayuntamiento de Madrid (enero 2020 - diciembre 2024)

**2. Datos Socioeconómicos**
**Enlace**: [RentaPOB - Renta y Población por Barrio](https://docs.google.com/spreadsheets/d/1iNgb0F5JdqcnIyjNU3ZsyBFVH1yP51PS/edit?usp=drive_link&ouid=106414683644608503271&rtpof=true&sd=true)

**Origen**: Banco de Datos del Ayuntamiento de Madrid con información desagregada por barrio y año

#### Ejecución

**Pipeline Completo (Recomendado)**
```bash
python -m src.main
```

**Ejecución Modular**
```bash
# Solo procesamiento de datos
python -m src.data.data_preprocessor

# Solo entrenamiento de modelos
python -m src.models.train_models

# Solo evaluación
python -m src.models.model_evaluation
```

### Aplicaciones Prácticas

#### Casos de Uso

- **Planificación Urbana**: Identificación de zonas comerciales en riesgo
- **Política Pública**: Focalización de ayudas y subvenciones
- **Inversión Privada**: Evaluación de ubicaciones comerciales
- **Transparencia**: Open data para ciudadanía e investigadores

#### Implementación Recomendada

1. **API Interna**: Despliegue como servicio REST para departamentos municipales
2. **Dashboard Interactivo**: Visualización en tiempo real de predicciones
3. **Alertas Automáticas**: Sistema de notificaciones para zonas de alto riesgo
4. **Explicabilidad**: Reportes SHAP para justificar decisiones

### Testing y Calidad

```bash
# Ejecutar todas las pruebas
python -m pytest tests/ -v

# Pruebas específicas por módulo
python -m pytest tests/test_models/ -v
python -m pytest tests/test_data/ -v
python -m pytest tests/test_features/ -v
```

### Información del Proyecto

#### Autoría

**Alexandro Bazán Guardia**
Máster en Ciencia de Datos e Inteligencia de Negocios
Universidad Complutense de Madrid

- **Email**: alexandro.bazan9712@gmail.com
- **LinkedIn**: [alexandrobg](https://www.linkedin.com/in/alexandrobg/)
- **GitHub**: [alexandrobg](https://github.com/alexandrobg)

#### Supervisión Académica

**Tutora**: Belén Rodríguez-Cánovas
Facultad de Estudios Estadísticos - Universidad Complutense de Madrid

### Licencia y Citación

#### Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo `LICENSE` para detalles completos.

#### Cita Académica

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

### Agradecimientos

- Facultad de Estudios Estadísticos - UCM por el soporte académico
- Ayuntamiento de Madrid por el acceso a datos públicos
- Comunidad open source por las librerías utilizadas

### Roadmap y Desarrollo Futuro

#### Versión Actual (v1.0)
- ✅ Pipeline completo de ML
- ✅ 9 modelos comparados
- ✅ Análisis SHAP
- ✅ Documentación completa

#### Próximas Versiones
- 🔄 API REST para predicciones en tiempo real
- 🔄 Dashboard web interactivo
- 🔄 Integración con datos en streaming
- 🔄 Modelos de deep learning (LSTM, Transformer)
- 🔄 Análisis de series temporales avanzado

---

**Status**: Producción | **Última actualización**: Septiembre 2025 | **Versión**: 1.0.0
