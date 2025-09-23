# Predicción de Actividad Comercial en Madrid

## Descripción

Este proyecto utiliza técnicas de Machine Learning para predecir la inactividad de locales comerciales en Madrid mediante el análisis de datos administrativos del Ayuntamiento de Madrid (2020-2024), enriquecidos con datos socioeconómicos.

## Características del Proyecto

- **Dataset**: +8 millones de registros de locales comerciales de Madrid
- **Variables**: Geolocalización, tipo de acceso, clasificación CNAE, renta per cápita, población
- **Modelos**: Regresión Logística, Random Forest, XGBoost, SVM, MLP, Ensambles
- **Resultados (test 2024)**:
  - **Accuracy**: 0.88–0.89 (mejores modelos: MLP, Regresión Logística, Random Forest, SVM)
  - **F1-score**: ~0.925 (MLP, RF, LR, SVM)
  - **AUC ROC**: ~0.85 (máx. 0.854 con Voting/MLP)
- **Interpretabilidad**: Análisis SHAP para explicabilidad del modelo

## Estructura del Proyecto

```
madrid-commercial-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── data_preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_selector.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── train_models.py
│   │   └── model_evaluation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py
│   │   └── maps.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
├── models/
│   └── saved_models/
├── results/
│   ├── figures/
│   ├── reports/
│   └── predictions/
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_features/
│   └── test_models/
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    └── api_documentation.md
```

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/madrid-commercial-prediction.git
cd madrid-commercial-prediction
```

### 2. Crear entorno virtual
```bash
# Con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Con conda (alternativo)
conda create -n madrid_prediction python=3.9
conda activate madrid_prediction
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus rutas y configuraciones
```

## Uso

### Ejecución completa del pipeline
```bash
python -m src.main
```

### Ejecución por módulos
```bash
# Solo procesamiento de datos
python -m src.data.data_preprocessor

# Solo entrenamiento de modelos
python -m src.models.train_models

# Solo evaluación
python -m src.models.model_evaluation
```

### Usando notebooks

Los notebooks están organizados secuencialmente:

1. `01_exploratory_data_analysis.ipynb` - Análisis exploratorio
2. `02_feature_engineering.ipynb` - Ingeniería de características
3. `03_model_training.ipynb` - Entrenamiento de modelos
4. `04_model_interpretation.ipynb` - Interpretabilidad con SHAP

## Datos Requeridos

### Archivo principal
- `Actividades Economicas de Madrid.csv` - Datos del Ayuntamiento de Madrid

### Archivo complementario
- `RentaPOB.xlsx` - Datos de renta per cápita y población

### Estructura de datos esperada

Coloque los archivos de datos en la carpeta `data/raw/`:

```
data/raw/
├── Actividades Economicas de Madrid.csv
└── RentaPOB.xlsx
```

## Modelos Implementados

> **Notas:**
> - Métricas en test (enero–diciembre 2024).
> - Modelo seleccionado por equilibrio rendimiento/parsimonia: **MLP**.

| Modelo | AUC | F1-Score | Accuracy | Características |
|--------|-----|----------|----------|----------------|
| **MLP** | **0.8540** | **0.9253** | **0.8848** | Mejor equilibrio global; alta sensibilidad y buena especificidad |
| VotingClassifier | 0.8541 | 0.9242 | 0.8828 | Ensamble robusto (RF + XGB + MLP, soft voting) |
| Random Forest | 0.8489 | 0.9250 | 0.8841 | Alto recall en clase positiva; interpretable por importancia de variables |
| XGBoost | 0.8491 | 0.9151 | 0.8673 | Buen AUC; algo más conservador en especificidad |
| Regresión Logística | 0.8475 | 0.9253 | 0.8848 | Baseline sólido y estable |
| SVM | 0.8179 | 0.9254 | 0.8848 | F1 máximo; AUC inferior; buen balance precisión/recall |
| KNN | 0.8189 | 0.9028 | 0.8498 | Correcto en datasets reducidos; menor discriminación |
| Decision Tree | 0.7418 | 0.8820 | 0.8098 | Simple, interpretable; rendimiento más bajo |
| StackingClassifier | 0.8316 | 0.8936 | 0.8409 | Mejora limitada frente a modelos individuales |

## Variables Más Importantes

Según análisis SHAP (modelo MLP):

1. **desc_seccion_sin_actividad** — Indicador de falta de actividad (impacto negativo fuerte)
2. **desc_tipo_acceso_local_puerta_calle** — Acceso directo desde la calle (impacto positivo)
3. **Renta_Media** — Renta media del distrito (mayor renta, mayor probabilidad de actividad)
4. **Total_Poblacion** — Población del barrio (impacto positivo moderado)
5. **desc_seccion_hosteleria** — Presencia del sector hostelería (impacto positivo)

Otras variables con efecto menor o dependiente del contexto: desc_seccion_comercio al por mayor y al por menor; reparación de vehículos..., desc_seccion_construccion, desc_seccion_informacion y comunicaciones, desc_seccion_educacion, entre otras.

## Resultados y Casos de Uso

### Aplicaciones
- **Planificación urbana**: Identificar zonas de riesgo comercial
- **Política pública**: Focalizar ayudas y subvenciones
- **Inversión privada**: Evaluar ubicaciones comerciales
- **Transparencia**: Open data para ciudadanía

### Visualizaciones
- Mapas interactivos de actividad comercial
- Curvas ROC comparativas
- Gráficos SHAP de interpretabilidad
- Análisis temporal de evolución

## Testing

Ejecutar las pruebas:

```bash
# Todas las pruebas
python -m pytest tests/

# Pruebas específicas
python -m pytest tests/test_models/
python -m pytest tests/test_data/
```

## Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## Autor

**Alexandro Bazán Guardia**

- Email: alexandro.bazan9712@gmail.com
- LinkedIn: https://www.linkedin.com/in/alexandrobg/
- Universidad Complutense de Madrid - Máster en Ciencia de Datos e Inteligencia de Negocios

## Agradecimientos

- Facultad de Estudios Estadísticos - Universidad Complutense de Madrid
- Ayuntamiento de Madrid por los datos públicos
- Tutora: Belén Rodríguez-Cánovas

## Citas

Si usas este trabajo en tu investigación, por favor cita:

```bibtex
@mastersthesis{bazan2025madrid,
  title={Predicción de la inactividad de locales comerciales en Madrid mediante técnicas de Machine Learning},
  author={Bazán Guardia, Alexandro},
  year={2025},
  school={Universidad Complutense de Madrid},
  type={Trabajo de Fin de Máster}
}
```
