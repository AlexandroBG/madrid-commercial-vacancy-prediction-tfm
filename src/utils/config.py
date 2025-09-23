"""
Configuración centralizada del proyecto.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas principales del proyecto
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Archivos de datos
DATA_FILES = {
    'actividades': RAW_DATA_DIR / "MadridActividades.csv",
    'renta_poblacion': RAW_DATA_DIR / "RentaPOB.xlsx",
}

# Configuración de modelos
MODEL_CONFIG = {
    'random_state': 12345,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,
}

# Parámetros de muestreo para diferentes procesos
SAMPLING_CONFIG = {
    'boruta_sample_size': 100000,
    'rfecv_sample_size': 50000,
    'stepwise_sample_size': 500000,
    'sbf_sample_size': 50000,
    'shap_sample_size': 10000,
}

# Configuración de hiperparámetros para GridSearch
HYPERPARAMETERS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'decision_tree': {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'xgboost': {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    },
    'knn': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [1e-4, 1e-3],
        'learning_rate': ['constant', 'adaptive']
    }
}

# Configuración de selección de características
FEATURE_SELECTION_CONFIG = {
    'boruta': {
        'n_estimators': 'auto',
        'max_iter': 100,
        'alpha': 0.05
    },
    'rfecv': {
        'step': 1,
        'min_features_to_select': 1
    },
    'stepwise': {
        'threshold_in': 0.001,
        'threshold_out': 0.01
    }
}

# Configuración de visualización
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'Set2',
    'font_size': 12
}

# Configuración de mapas
MAP_CONFIG = {
    'madrid_center': [40.4168, -3.7038],
    'default_zoom': 12,
    'tiles': 'CartoDB positron',
    'marker_cluster_max_zoom': 15
}

# Configuración de procesamiento de texto
TEXT_PROCESSING_CONFIG = {
    'encoding': 'latin1',
    'remove_accents': True,
    'lowercase': True,
    'remove_extra_spaces': True
}

# Variables categóricas a procesar (ELIMINAMOS desc_distrito_local)
CATEGORICAL_VARIABLES = [
    'desc_barrio_local',
    'desc_tipo_acceso_local',
    'desc_seccion'
]

# Variables a eliminar durante el preprocesamiento
COLUMNS_TO_DROP = [
    'actividad', 'rotulo', 'desc_vial_acceso', 'Fecha_Reporte', 'Mes', 'Año',
    'num_acceso', 'cal_acceso', 'latitud_local', 'longitud_local',
    'id_local', 'id_distrito_local', 'cod_barrio_local',
    'id_tipo_acceso_local', 'id_seccion', 'id_epigrafe', 'desc_epigrafe',
    'id_division', 'desc_division', 'desc_situacion_local',
    'clase_vial_acceso', 'nom_acceso', 'cal_acceso'
]

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': RESULTS_DIR / 'logs' / 'madrid_prediction.log'
}

# Crear directorio de logs
(RESULTS_DIR / 'logs').mkdir(parents=True, exist_ok=True)

def get_config() -> Dict[str, Any]:
    """Retorna la configuración completa del proyecto."""
    return {
        'paths': {
            'base_dir': BASE_DIR,
            'data_dir': DATA_DIR,
            'raw_data_dir': RAW_DATA_DIR,
            'processed_data_dir': PROCESSED_DATA_DIR,
            'models_dir': MODELS_DIR,
            'results_dir': RESULTS_DIR,
            'figures_dir': FIGURES_DIR,
            'reports_dir': REPORTS_DIR,
        },
        'data_files': DATA_FILES,
        'model_config': MODEL_CONFIG,
        'sampling_config': SAMPLING_CONFIG,
        'hyperparameters': HYPERPARAMETERS,
        'feature_selection': FEATURE_SELECTION_CONFIG,
        'plot_config': PLOT_CONFIG,
        'map_config': MAP_CONFIG,
        'text_processing': TEXT_PROCESSING_CONFIG,
        'categorical_variables': CATEGORICAL_VARIABLES,
        'columns_to_drop': COLUMNS_TO_DROP,
        'logging': LOGGING_CONFIG,
    }

def get_data_file_path(file_key: str) -> Path:
    """Retorna la ruta de un archivo de datos específico."""
    return DATA_FILES.get(file_key)

def get_model_save_path(model_name: str) -> Path:
    """Retorna la ruta para guardar un modelo específico."""
    return MODELS_DIR / f"{model_name}.joblib"

def get_results_path(filename: str, subfolder: str = None) -> Path:
    """Retorna la ruta para guardar resultados."""
    if subfolder:
        path = RESULTS_DIR / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
    return RESULTS_DIR / filename
