# src/__init__.py
"""
Madrid Commercial Prediction - Predicción de Actividad Comercial en Madrid
"""

__version__ = "1.0.0"
__author__ = "Alexandro Bazán Guardia"
__email__ = "alexandro.bazan@ejemplo.com"

# src/data/__init__.py
"""
Módulo de datos - Carga, limpieza y preprocesamiento
"""

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .data_preprocessor import DataPreprocessor

__all__ = ['DataLoader', 'DataCleaner', 'DataPreprocessor']

# src/features/__init__.py
"""
Módulo de características - Selección e ingeniería de variables
"""

from .feature_selector import FeatureSelector

__all__ = ['FeatureSelector']

# src/models/__init__.py
"""
Módulo de modelos - Entrenamiento y evaluación
"""

from .train_models import ModelTrainer
from .model_evaluation import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator']

# src/utils/__init__.py
"""
Módulo de utilidades - Configuración y funciones auxiliares
"""

from .config import get_config, get_data_file_path, get_model_save_path
from .helpers import setup_logging, create_project_directories

__all__ = ['get_config', 'get_data_file_path', 'get_model_save_path',
           'setup_logging', 'create_project_directories']

# src/visualization/__init__.py
"""
Módulo de visualización - Gráficos y mapas
"""

# Este módulo puede expandirse en el futuro
__all__ = []
