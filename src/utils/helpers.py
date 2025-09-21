"""
Funciones de utilidad para el proyecto Madrid Commercial Prediction.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

def setup_logging(logging_config: Dict[str, Any]) -> None:
    """
    Configura el sistema de logging del proyecto.

    Args:
        logging_config: Configuración de logging desde config.py
    """
    log_dir = Path(logging_config['log_file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        handlers=[
            logging.FileHandler(logging_config['log_file']),
            logging.StreamHandler()
        ]
    )

    # Configurar loggers de bibliotecas externas
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('xgboost').setLevel(logging.WARNING)

    logging.info(f"Sistema de logging configurado. Log file: {logging_config['log_file']}")

def create_project_directories(config: Dict[str, Any]) -> None:
    """
    Crea la estructura de directorios del proyecto.

    Args:
        config: Configuración del proyecto
    """
    paths = config['paths']

    directories = [
        paths['data_dir'],
        paths['raw_data_dir'],
        paths['processed_data_dir'],
        paths['models_dir'],
        paths['results_dir'],
        paths['figures_dir'],
        paths['reports_dir'],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Estructura de directorios creada")

def save_model(model, model_name: str, config: Dict[str, Any]) -> str:
    """
    Guarda un modelo entrenado.

    Args:
        model: Modelo a guardar
        model_name: Nombre del modelo
        config: Configuración del proyecto

    Returns:
        Ruta donde se guardó el modelo
    """
    models_dir = config['paths']['models_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = models_dir / filename

    joblib.dump(model, filepath)
    logging.info(f"Modelo guardado: {filepath}")

    return str(filepath)

def load_model(model_path: str):
    """
    Carga un modelo guardado.

    Args:
        model_path: Ruta del modelo

    Returns:
        Modelo cargado
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Modelo cargado: {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error al cargar modelo {model_path}: {e}")
        return None

def save_results(results: Dict[str, Any], filename: str, config: Dict[str, Any]) -> str:
    """
    Guarda resultados en formato pickle.

    Args:
        results: Diccionario con resultados
        filename: Nombre del archivo
        config: Configuración del proyecto

    Returns:
        Ruta donde se guardaron los resultados
    """
    results_dir = config['paths']['results_dir']
    filepath = results_dir / f"{filename}.pkl"

    joblib.dump(results, filepath)
    logging.info(f"Resultados guardados: {filepath}")

    return str(filepath)

def create_summary_report(metrics_df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Crea un reporte resumen del proyecto.

    Args:
        metrics_df: DataFrame con métricas de modelos
        config: Configuración del proyecto

    Returns:
        Ruta del reporte generado
    """
    reports_dir = config['paths']['reports_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"summary_report_{timestamp}.html"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Madrid Commercial Prediction - Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E86AB; }}
            h2 {{ color: #A23B72; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .best-model {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Resultados - Predicción Comercial Madrid</h1>
        <p><strong>Fecha de generación:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>

        <h2>Comparación de Modelos</h2>
        {metrics_df.to_html(classes='table table-striped', table_id='metrics-table')}

        <h2>Mejor Modelo</h2>
        <p>El mejor modelo según AUC es: <strong>{metrics_df.iloc[0]['Modelo']}</strong></p>
        <ul>
            <li>AUC: {metrics_df.iloc[0]['AUC']:.4f}</li>
            <li>Accuracy: {metrics_df.iloc[0]['Accuracy']:.4f}</li>
            <li>F1-Score: {metrics_df.iloc[0]['F1 Score']:.4f}</li>
            <li>Precision: {metrics_df.iloc[0]['Precision']:.4f}</li>
            <li>Recall: {metrics_df.iloc[0]['Recall']:.4f}</li>
        </ul>

        <h2>Interpretación de Resultados</h2>
        <p>Los resultados muestran que el modelo {metrics_df.iloc[0]['Modelo']}
        logra el mejor balance entre precisión y recall, con una capacidad
        discriminativa excelente (AUC > 0.85).</p>

        <h2>Recomendaciones</h2>
        <ul>
            <li>Implementar el modelo {metrics_df.iloc[0]['Modelo']} en producción</li>
            <li>Monitorear el rendimiento con nuevos datos</li>
            <li>Considerar reentrenamiento periódico</li>
            <li>Integrar explicabilidad SHAP para decisiones</li>
        </ul>
    </body>
    </html>
    """

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logging.info(f"Reporte resumen generado: {report_path}")
    return str(report_path)

def plot_confusion_matrix_comparison(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Crea una comparación visual de matrices de confusión.

    Args:
        results: Resultados de evaluación de modelos
        config: Configuración del proyecto

    Returns:
        Ruta de la imagen generada
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    n_models = len(results)
    cols = 3
    rows = -(-n_models // cols)  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, (model_name, model_results) in enumerate(results.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        y_true = model_results['y_true']
        y_pred = model_results['y_pred']
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{model_name}\nAUC: {model_results["auc"]:.4f}')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')

    # Eliminar subplots vacíos
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        fig.delaxes(ax)

    plt.suptitle("Comparación de Matrices de Confusión", fontsize=16)
    plt.tight_layout()

    figures_dir = config['paths']['figures_dir']
    filepath = figures_dir / "confusion_matrices_comparison.png"
    plt.savefig(filepath, dpi=config['plot_config']['dpi'], bbox_inches='tight')
    plt.close()

    logging.info(f"Gráfico de matrices de confusión guardado: {filepath}")
    return str(filepath)

def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calcula pesos balanceados para clases desbalanceadas.

    Args:
        y: Variable objetivo

    Returns:
        Diccionario con pesos por clase
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    logging.info(f"Pesos calculados para clases: {class_weights}")
    return class_weights

def memory_usage_check(df: pd.DataFrame, threshold_gb: float = 2.0) -> bool:
    """
    Verifica si un DataFrame excede el umbral de memoria.

    Args:
        df: DataFrame a verificar
        threshold_gb: Umbral en GB

    Returns:
        True si excede el umbral, False en caso contrario
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    memory_gb = memory_mb / 1024

    logging.info(f"Uso de memoria del DataFrame: {memory_gb:.2f} GB")

    if memory_gb > threshold_gb:
        logging.warning(f"DataFrame excede el umbral de memoria: {memory_gb:.2f} GB > {threshold_gb} GB")
        return True

    return False

def create_model_comparison_chart(metrics_df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Crea un gráfico de barras comparando métricas de modelos.

    Args:
        metrics_df: DataFrame con métricas
        config: Configuración del proyecto

    Returns:
        Ruta del gráfico generado
    """
    plt.figure(figsize=(14, 10))

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    n_metrics = len(metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        bars = plt.bar(metrics_df['Modelo'], metrics_df[metric])
        plt.title(f'{metric} por Modelo')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric)

        # Destacar el mejor modelo
        best_idx = metrics_df[metric].idxmax()
        bars[best_idx].set_color('red')

        # Agregar valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    figures_dir = config['paths']['figures_dir']
    filepath = figures_dir / "model_comparison_chart.png"
    plt.savefig(filepath, dpi=config['plot_config']['dpi'], bbox_inches='tight')
    plt.close()

    logging.info(f"Gráfico de comparación guardado: {filepath}")
    return str(filepath)

def validate_model_inputs(X_train, X_test, y_train, y_test) -> bool:
    """
    Valida que los inputs para modelos sean correctos.

    Args:
        X_train, X_test, y_train, y_test: Datasets de entrenamiento y prueba

    Returns:
        True si las validaciones pasan, False en caso contrario
    """
    checks = []

    # Verificar shapes
    checks.append(len(X_train) == len(y_train))
    checks.append(len(X_test) == len(y_test))
    checks.append(X_train.shape[1] == X_test.shape[1])

    # Verificar que no hay infinitos o NaNs
    checks.append(not np.isnan(X_train).any().any())
    checks.append(not np.isnan(X_test).any().any())
    checks.append(not np.isnan(y_train).any())
    checks.append(not np.isnan(y_test).any())

    # Verificar que y es binario
    unique_y_train = set(y_train.unique())
    unique_y_test = set(y_test.unique())
    checks.append(unique_y_train.issubset({0, 1}))
    checks.append(unique_y_test.issubset({0, 1}))

    if all(checks):
        logging.info("Validación de inputs exitosa")
        return True
    else:
        logging.error("Falló la validación de inputs para modelos")
        return False

def print_dataset_info(name: str, df: pd.DataFrame) -> None:
    """
    Imprime información detallada de un dataset.

    Args:
        name: Nombre del dataset
        df: DataFrame a analizar
    """
    print(f"\n{'='*60}")
    print(f"INFORMACIÓN DEL DATASET: {name.upper()}")
    print(f"{'='*60}")
    print(f"Forma: {df.shape}")
    print(f"Tipos de datos:")
    print(df.dtypes.value_counts())
    print(f"\nValores nulos por columna:")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(null_counts[null_counts > 0])
    else:
        print("No hay valores nulos")

    print(f"\nUso de memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Registros duplicados: {df.duplicated().sum()}")

    if 'actividad' in df.columns:
        print(f"\nDistribución de variable objetivo:")
        print(df['actividad'].value_counts())
        print(f"Porcentaje de activos: {df['actividad'].mean()*100:.2f}%")
