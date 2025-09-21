"""
Script principal para ejecutar el pipeline completo de predicción comercial Madrid.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import argparse
from typing import Optional

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader, save_processed_data, load_processed_data
from data.data_cleaner import DataCleaner
from data.data_preprocessor import DataPreprocessor
from features.feature_selector import FeatureSelector
from models.train_models import ModelTrainer
from models.model_evaluation import ModelEvaluator
from utils.config import get_config
from utils.helpers import setup_logging, create_project_directories

def setup_project():
    """Configuración inicial del proyecto."""
    config = get_config()
    setup_logging(config['logging'])
    create_project_directories(config)

    logger = logging.getLogger(__name__)
    logger.info("Proyecto Madrid Commercial Prediction iniciado")
    logger.info("="*80)

    return config, logger

def load_and_clean_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Carga y limpia los datos, usando cache si está disponible.

    Args:
        force_reload: Si True, fuerza la recarga y limpieza completa

    Returns:
        DataFrame limpio y listo para procesamiento
    """
    logger = logging.getLogger(__name__)

    # Intentar cargar datos ya procesados
    if not force_reload:
        df_clean = load_processed_data('df_limpio.pkl')
        if df_clean is not None:
            logger.info(f"Datos limpios cargados desde cache: {df_clean.shape}")
            return df_clean

    logger.info("Cargando y limpiando datos desde archivos originales...")

    # Cargar datos raw
    data_loader = DataLoader()
    if not data_loader.validate_data_files():
        raise FileNotFoundError("Faltan archivos de datos requeridos")

    df_actividades, df_renta = data_loader.load_all_data()

    # Limpiar datos de actividades
    cleaner = DataCleaner()
    df_clean = cleaner.clean_full_dataset(df_actividades)

    # Merge con datos de renta
    preprocessor = DataPreprocessor()
    df_final = preprocessor.merge_with_renta_data(df_clean, df_renta)

    # Guardar datos limpios
    save_processed_data(df_final, 'df_limpio', format='pickle')
    logger.info(f"Datos limpios guardados. Shape final: {df_final.shape}")

    return df_final

def run_feature_engineering(df: pd.DataFrame, force_reload: bool = False) -> dict:
    """
    Ejecuta ingeniería de características y selección de variables.

    Args:
        df: DataFrame limpio
        force_reload: Si True, fuerza el reprocesamiento

    Returns:
        Diccionario con datasets preparados y variables seleccionadas
    """
    logger = logging.getLogger(__name__)

    # Intentar cargar datasets ya procesados
    if not force_reload:
        try:
            import joblib
            config = get_config()
            models_dir = config['paths']['models_dir']

            datasets = {
                'X_train': joblib.load(models_dir / 'X_train_scaled.joblib'),
                'X_test': joblib.load(models_dir / 'X_test_scaled.joblib'),
                'y_train': joblib.load(models_dir / 'y_train.joblib'),
                'y_test': joblib.load(models_dir / 'y_test.joblib'),
                'scaler': joblib.load(models_dir / 'scaler.joblib')
            }

            import pickle
            with open(models_dir / 'selected_features_boruta.pkl', 'rb') as f:
                selected_features = pickle.load(f)

            logger.info("Datasets y características cargados desde cache")
            return {'datasets': datasets, 'selected_features': selected_features}

        except Exception as e:
            logger.info(f"No se pudieron cargar datos en cache: {e}")

    logger.info("Ejecutando ingeniería de características...")

    # Preprocesamiento
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_for_modeling(df)

    # Selección de características
    feature_selector = FeatureSelector()
    selected_features = feature_selector.run_boruta_selection(
        datasets['X_train_scaled'],
        datasets['y_train']
    )

    logger.info(f"Características seleccionadas: {len(selected_features)}")

    return {'datasets': datasets, 'selected_features': selected_features}

def train_and_evaluate_models(datasets: dict, selected_features: list,
                            force_retrain: bool = False) -> dict:
    """
    Entrena y evalúa todos los modelos.

    Args:
        datasets: Diccionario con datasets preparados
        selected_features: Lista de características seleccionadas
        force_retrain: Si True, fuerza el reentrenamiento

    Returns:
        Diccionario con resultados y mejor modelo
    """
    logger = logging.getLogger(__name__)

    # Filtrar datasets con características seleccionadas
    X_train = datasets['X_train_scaled'][selected_features]
    X_test = datasets['X_test_scaled'][selected_features]
    y_train = datasets['y_train']
    y_test = datasets['y_test']

    # Entrenar modelos
    trainer = ModelTrainer()

    if not force_retrain:
        # Intentar cargar modelos ya entrenados
        trained_models = trainer.load_trained_models()
        if trained_models:
            logger.info("Modelos cargados desde cache")
        else:
            logger.info("Entrenando modelos desde cero...")
            trained_models = trainer.train_all_models(X_train, y_train)
    else:
        logger.info("Reentrenando todos los modelos...")
        trained_models = trainer.train_all_models(X_train, y_train)

    # Evaluar modelos
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(trained_models, X_test, y_test)

    # Encontrar mejor modelo
    best_model_info = evaluator.get_best_model(results)

    logger.info(f"Mejor modelo: {best_model_info['name']} (AUC: {best_model_info['auc']:.4f})")

    return {
        'trained_models': trained_models,
        'evaluation_results': results,
        'best_model': best_model_info
    }

def generate_reports_and_visualizations(df: pd.DataFrame, results: dict,
                                      selected_features: list):
    """
    Genera reportes y visualizaciones finales.

    Args:
        df: DataFrame original
        results: Resultados de evaluación de modelos
        selected_features: Características seleccionadas
    """
    logger = logging.getLogger(__name__)
    logger.info("Generando reportes y visualizaciones...")

    evaluator = ModelEvaluator()

    # Generar visualizaciones comparativas
    evaluator.plot_roc_curves_comparison(results['evaluation_results'])
    evaluator.plot_confusion_matrices(results['evaluation_results'])
    evaluator.create_metrics_comparison_table(results['evaluation_results'])

    # Análisis de interpretabilidad
    if results['best_model']:
        evaluator.generate_shap_analysis(
            results['best_model']['model'],
            results['datasets']['X_test'][selected_features]
        )

    # Casos de uso específicos
    evaluator.analyze_specific_cases(df, results['best_model'])

    logger.info("Reportes generados exitosamente")

def main():
    """Función principal que orquesta todo el pipeline."""
    parser = argparse.ArgumentParser(description='Madrid Commercial Prediction Pipeline')
    parser.add_argument('--force-reload', action='store_true',
                       help='Fuerza la recarga completa de datos')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Fuerza el reentrenamiento de modelos')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Omite generación de visualizaciones')
    parser.add_argument('--model-only', action='store_true',
                       help='Solo entrena modelos, asume datos ya procesados')

    args = parser.parse_args()

    try:
        # Configuración inicial
        config, logger = setup_project()

        if not args.model_only:
            # 1. Cargar y limpiar datos
            logger.info("PASO 1: Carga y limpieza de datos")
            logger.info("="*50)
            df_clean = load_and_clean_data(force_reload=args.force_reload)

            # 2. Ingeniería de características
            logger.info("\nPASO 2: Ingeniería de características")
            logger.info("="*50)
            feature_results = run_feature_engineering(df_clean, force_reload=args.force_reload)
        else:
            # Cargar datos ya procesados
            logger.info("Modo solo-modelos: Cargando datos procesados...")
            df_clean = load_processed_data('df_limpio.pkl')
            if df_clean is None:
                raise FileNotFoundError("No se encontraron datos procesados")
            feature_results = run_feature_engineering(df_clean, force_reload=False)

        # 3. Entrenamiento y evaluación
        logger.info("\nPASO 3: Entrenamiento y evaluación de modelos")
        logger.info("="*50)
        model_results = train_and_evaluate_models(
            feature_results['datasets'],
            feature_results['selected_features'],
            force_retrain=args.force_retrain
        )

        # 4. Reportes y visualizaciones
        if not args.skip_viz:
            logger.info("\nPASO 4: Generación de reportes")
            logger.info("="*50)
            generate_reports_and_visualizations(
                df_clean,
                model_results,
                feature_results['selected_features']
            )

        # Resumen ejecutivo
        logger.info("\n" + "="*80)
        logger.info("RESUMEN EJECUTIVO")
        logger.info("="*80)
        logger.info(f"Dataset procesado: {df_clean.shape[0]:,} registros")
        logger.info(f"Características seleccionadas: {len(feature_results['selected_features'])}")
        logger.info(f"Mejor modelo: {model_results['best_model']['name']}")
        logger.info(f"AUC del mejor modelo: {model_results['best_model']['auc']:.4f}")
        logger.info(f"F1-Score: {model_results['best_model']['f1']:.4f}")
        logger.info("="*80)
        logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        logger.exception("Detalles del error:")
        sys.exit(1)

if __name__ == "__main__":
    main()
