"""
Módulo para evaluación y comparación de modelos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import shap
import joblib
from src.utils.config import get_config
from src.utils.helpers import create_summary_report, create_model_comparison_chart

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Clase para evaluación completa de modelos."""

    def __init__(self):
        self.config = get_config()
        self.plot_config = self.config['plot_config']

    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str = "Model") -> Dict[str, Any]:
        """
        Evalúa un modelo individual en el conjunto de prueba.

        Args:
            model: Modelo entrenado
            X_test: Variables independientes de prueba
            y_test: Variable objetivo de prueba
            model_name: Nombre del modelo

        Returns:
            Diccionario con métricas y predicciones
        """
        logger.info(f"Evaluando modelo: {model_name}")

        # Predicciones
        y_pred = model.predict(X_test)

        # Probabilidades (si están disponibles)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = None
        except:
            y_proba = None

        # Calcular métricas
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Métricas adicionales
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics.update({
            'specificity': specificity,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'confusion_matrix': cm
        })

        # Guardar predicciones
        results = {
            'metrics': metrics,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            },
            'model': model
        }

        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")

        return results

    def evaluate_all_models(self, trained_models: Dict[str, Any],
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evalúa todos los modelos entrenados.

        Args:
            trained_models: Diccionario con modelos entrenados
            X_test: Variables independientes de prueba
            y_test: Variable objetivo de prueba

        Returns:
            Diccionario con resultados de evaluación de todos los modelos
        """
        logger.info("="*60)
        logger.info("EVALUANDO TODOS LOS MODELOS")
        logger.info("="*60)

        evaluation_results = {}

        for model_name, model_info in trained_models.items():
            try:
                model = model_info['model']

                # Evaluar modelo
                results = self.evaluate_single_model(model, X_test, y_test, model_name)
                evaluation_results[model_name] = results

            except Exception as e:
                logger.error(f"Error evaluando {model_name}: {e}")
                continue

        # Crear tabla comparativa
        comparison_df = self.create_metrics_comparison_table(evaluation_results)

        logger.info("="*60)
        logger.info("EVALUACIÓN COMPLETADA")
        logger.info("="*60)

        return {
            'individual_results': evaluation_results,
            'comparison_table': comparison_df
        }

    def create_metrics_comparison_table(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Crea tabla comparativa de métricas.

        Args:
            evaluation_results: Resultados de evaluación

        Returns:
            DataFrame con métricas comparativas
        """
        comparison_data = []

        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Modelo': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'AUC': metrics['auc'] if metrics['auc'] is not None else 0,
                'Especificidad': metrics['specificity'],
                'Verdaderos Negativos': metrics['true_negatives']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)

        # Guardar tabla
        reports_dir = self.config['paths']['reports_dir']
        comparison_df.to_csv(reports_dir / 'model_comparison.csv', index=False)

        logger.info("\nTabla comparativa de modelos:")
        logger.info(comparison_df.round(4).to_string(index=False))

        return comparison_df

    def plot_roc_curves_comparison(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Crea gráfico comparativo de curvas ROC.

        Args:
            evaluation_results: Resultados de evaluación

        Returns:
            Ruta del gráfico guardado
        """
        plt.figure(figsize=self.plot_config['figure_size'])

        for model_name, results in evaluation_results.items():
            predictions = results['predictions']
            y_true = predictions['y_true']
            y_proba = predictions['y_proba']

            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = results['metrics']['auc']

                plt.plot(fpr, tpr,
                        label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Comparación de Curvas ROC')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        # Guardar gráfico
        figures_dir = self.config['paths']['figures_dir']
        filepath = figures_dir / "roc_curves_comparison.png"
        plt.savefig(filepath, dpi=self.plot_config['dpi'], bbox_inches='tight')
        plt.close()

        logger.info(f"Curvas ROC guardadas en: {filepath}")
        return str(filepath)

    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Crea gráfico con matrices de confusión de todos los modelos.

        Args:
            evaluation_results: Resultados de evaluación

        Returns:
            Ruta del gráfico guardado
        """
        n_models = len(evaluation_results)
        cols = 3
        rows = -(-n_models // cols)  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        # Manejar caso de un solo modelo
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, (model_name, results) in enumerate(evaluation_results.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = results['metrics']['confusion_matrix']
            auc_score = results['metrics']['auc']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{model_name.replace("_", " ").title()}\nAUC: {auc_score:.4f}')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')

        # Eliminar subplots vacíos
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            fig.delaxes(ax)

        plt.suptitle("Matrices de Confusión por Modelo", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Guardar gráfico
        figures_dir = self.config['paths']['figures_dir']
        filepath = figures_dir / "confusion_matrices_comparison.png"
        plt.savefig(filepath, dpi=self.plot_config['dpi'], bbox_inches='tight')
        plt.close()

        logger.info(f"Matrices de confusión guardadas en: {filepath}")
        return str(filepath)

    def get_best_model(self, evaluation_results: Dict[str, Any],
                      metric: str = 'auc') -> Dict[str, Any]:
        """
        Identifica el mejor modelo según una métrica específica.

        Args:
            evaluation_results: Resultados de evaluación
            metric: Métrica para selección ('auc', 'f1_score', 'accuracy')

        Returns:
            Información del mejor modelo
        """
        best_score = -1
        best_model_info = None

        for model_name, results in evaluation_results.items():
            score = results['metrics'][metric]
            if score is not None and score > best_score:
                best_score = score
                best_model_info = {
                    'name': model_name,
                    'model': results['model'],
                    'metrics': results['metrics'],
                    'score': score
                }

        if best_model_info:
            logger.info(f"Mejor modelo según {metric}: {best_model_info['name']} "
                       f"(score: {best_score:.4f})")

        return best_model_info

    def generate_shap_analysis(self, model, X_test: pd.DataFrame,
                             sample_size: int = 1000) -> str:
        """
        Genera análisis de interpretabilidad con SHAP.

        Args:
            model: Modelo a analizar
            X_test: Dataset de prueba
            sample_size: Tamaño de muestra para SHAP

        Returns:
            Ruta donde se guardaron los gráficos SHAP
        """
        logger.info("Generando análisis SHAP...")

        # Tomar muestra
        sample_size = min(sample_size, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=self.config['model_config']['random_state'])

        try:
            # Crear explainer
            if hasattr(model, 'predict_proba'):
                def predict_fn(X):
                    return model.predict_proba(pd.DataFrame(X, columns=X_test.columns))[:, 1]
            else:
                def predict_fn(X):
                    return model.predict(pd.DataFrame(X, columns=X_test.columns))

            # Usar una muestra más pequeña para el background
            background_sample = X_test.sample(100, random_state=42)
            explainer = shap.KernelExplainer(predict_fn, background_sample.values)

            # Calcular SHAP values en lotes
            batch_size = 100
            all_shap_values = []

            for i in range(0, len(X_sample), batch_size):
                batch = X_sample.iloc[i:i+batch_size]
                shap_vals = explainer.shap_values(batch.values, nsamples=100)
                all_shap_values.append(shap_vals)

            shap_values = np.vstack(all_shap_values)

            # Gráfico resumen
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="dot",
                            max_display=20, show=False)

            figures_dir = self.config['paths']['figures_dir']
            summary_path = figures_dir / "shap_summary_plot.png"
            plt.tight_layout()
            plt.savefig(summary_path, dpi=self.plot_config['dpi'], bbox_inches='tight')
            plt.close()

            # Ranking de importancia
            importancia_media = np.abs(shap_values).mean(axis=0)
            ranking = pd.DataFrame({
                'Variable': X_sample.columns,
                'Impacto_SHAP_Medio': importancia_media
            }).sort_values(by='Impacto_SHAP_Medio', ascending=False)

            # Guardar ranking
            reports_dir = self.config['paths']['reports_dir']
            ranking.to_csv(reports_dir / "ranking_variables_shap.csv", index=False)

            logger.info("Top 10 variables más influyentes según SHAP:")
            for _, row in ranking.head(10).iterrows():
                logger.info(f"  {row['Variable']}: {row['Impacto_SHAP_Medio']:.4f}")

            logger.info(f"Análisis SHAP completado. Gráficos guardados en: {figures_dir}")
            return str(summary_path)

        except Exception as e:
            logger.error(f"Error en análisis SHAP: {e}")
            return None

    def analyze_specific_cases(self, df: pd.DataFrame, best_model_info: Dict[str, Any]) -> None:
        """
        Analiza casos específicos con el mejor modelo.

        Args:
            df: DataFrame original
            best_model_info: Información del mejor modelo
        """
        if not best_model_info:
            logger.warning("No hay mejor modelo para análisis de casos")
            return

        logger.info("Analizando casos específicos...")

        # Aquí puedes agregar análisis específicos basados en tu TFM
        # Por ejemplo, casos por distrito, tipo de actividad, etc.

        logger.info("Análisis de casos específicos completado")

    def create_final_report(self, evaluation_results: Dict[str, Any],
                          comparison_df: pd.DataFrame) -> str:
        """
        Crea reporte final del proyecto.

        Args:
            evaluation_results: Resultados de evaluación
            comparison_df: Tabla comparativa de modelos

        Returns:
            Ruta del reporte generado
        """
        logger.info("Generando reporte final...")

        # Crear reporte HTML
        report_path = create_summary_report(comparison_df, self.config)

        # Crear gráfico comparativo
        create_model_comparison_chart(comparison_df, self.config)

        logger.info(f"Reporte final generado: {report_path}")
        return report_path
