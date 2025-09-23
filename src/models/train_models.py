# -*- coding: utf-8 -*-
"""
Módulo para entrenamiento de modelos siguiendo la estructura específica indicada.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_validate,
    StratifiedKFold, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
import time
import psutil
from src.utils.config import get_config
from src.utils.helpers import save_model, validate_model_inputs

def cross_validate_model(model, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    """Ejecuta validación cruzada y devuelve resultados."""
    from sklearn.model_selection import cross_validate, StratifiedKFold

    cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=12345)

    cv_results = cross_validate(
        model, X, y,
        cv=cv_stratified,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    return cv_results

def mostrar_metricas_cv(cv_results, scoring):
    """Imprime métricas de validación cruzada."""
    print("\n=== MÉTRICAS VALIDACIÓN CRUZADA ===")
    for metric in scoring:
        media = cv_results[f'test_{metric}'].mean()
        std   = cv_results[f'test_{metric}'].std()
        print(f"{metric.upper():<10}: {media:.4f} ± {std:.4f}")

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para entrenar múltiples modelos siguiendo la estructura específica."""

    def __init__(self):
        self.config = get_config()
        self.model_config = self.config['model_config']
        self.trained_models = {}

    def train_logistic_regression(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                                 y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena Regresión Logística con las variables seleccionadas por Boruta.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO REGRESIÓN LOGÍSTICA")
        logger.info("="*50)

        # ----------------------------------------
        # 1. ¿Qué es LogisticRegression?
        # ----------------------------------------
        logger.info("LogisticRegression es un modelo de clasificación lineal")
        logger.info("que estima la probabilidad usando una función logística (sigmoide).")

        # ----------------------------------------
        # 2. Selección de variables con Boruta ya escaladas
        # ----------------------------------------
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta = X_test_boruta[selected_features_boruta]

        # ----------------------------------------
        # 3. Búsqueda eficiente de hiperparámetros con RandomizedSearchCV (3-fold)
        # ----------------------------------------
        modelo_base = LogisticRegression(random_state=12345, max_iter=1000)

        param_distributions = {
            'C': loguniform(1e-3, 1e2),          # Regularización (escala logarítmica)
            'penalty': ['l2'],                   # Solo 'l2' para menor carga computacional
            'solver': ['liblinear', 'saga']      # Ambos compatibles con 'l2'
        }

        random_search = RandomizedSearchCV(
            modelo_base,
            param_distributions=param_distributions,
            n_iter=10,         # Número de combinaciones aleatorias
            cv=3,              # 3-fold CV para menor carga
            scoring='f1',      # Métrica objetivo
            random_state=123,
            n_jobs=1,          # Ejecuta en serie para evitar MemoryError
            verbose=1
        )

        # Ejecutar la búsqueda de hiperparámetros
        random_search.fit(X_train_boruta, y_train)
        modelo_logit = random_search.best_estimator_

        logger.info("\n=== MEJORES HIPERPARÁMETROS ENCONTRADOS ===")
        logger.info(random_search.best_params_)

        # ----------------------------------------
        # 4. Validación cruzada (3-fold) sobre el conjunto de entrenamiento
        # ----------------------------------------
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        resultados_cv = cross_validate(modelo_logit, X_train_boruta, y_train, cv=3, scoring=scoring)

        logger.info("\n=== VALIDACIÓN CRUZADA (3-FOLD) ===")
        for metric in scoring:
            media = resultados_cv[f'test_{metric}'].mean()
            std = resultados_cv[f'test_{metric}'].std()
            logger.info(f"{metric.upper():<10}: {media:.4f} ± {std:.4f}")

        # ----------------------------------------
        # 5. Entrenamiento final sobre todo el conjunto de entrenamiento
        # ----------------------------------------
        modelo_logit.fit(X_train_boruta, y_train)

        # ----------------------------------------
        # 6. Evaluación final sobre el conjunto de test
        # ----------------------------------------
        y_pred_log = modelo_logit.predict(X_test_boruta)
        y_proba_log = modelo_logit.predict_proba(X_test_boruta)[:, 1]

        logger.info("\n=== MATRIZ DE CONFUSIÓN - TEST ===")
        logger.info(confusion_matrix(y_test, y_pred_log))

        logger.info("\n=== REPORTE DE CLASIFICACIÓN - TEST ===")
        logger.info(classification_report(y_test, y_pred_log))

        # Visualizar matriz de confusión
        try:
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusión - Test Logistic Regression')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudo mostrar matriz de confusión: {e}")

        # ----------------------------------------
        # 7. Curva ROC
        # ----------------------------------------
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba_log)
            roc_auc = roc_auc_score(y_test, y_proba_log)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (área = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Curva ROC - Test Logistic Regression')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudo mostrar curva ROC: {e}")

        return {
            'model': modelo_logit,
            'best_params': random_search.best_params_,
            'cv_results': resultados_cv,
            'y_pred': y_pred_log,
            'y_proba': y_proba_log,
            'search_object': random_search
        }

    def train_decision_tree(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena Árbol de Decisión con optimización de hiperparámetros.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO ÁRBOL DE DECISIÓN")
        logger.info("="*50)

        # Usar variables seleccionadas por Boruta
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta = X_test_boruta[selected_features_boruta]

        # ----------------------------------------
        # Parámetros para explorar en Grid Search
        # ----------------------------------------
        param_grid = {
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Crear modelo base
        dt = DecisionTreeClassifier(random_state=12345)

        # Configurar GridSearchCV
        grid_search = GridSearchCV(estimator=dt,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='f1',
                                   n_jobs=-1)

        # Ajustar con los datos de entrenamiento seleccionados por Boruta
        grid_search.fit(X_train_boruta, y_train)

        # Mostrar mejores hiperparámetros
        logger.info("Mejores parámetros:", grid_search.best_params_)

        # Usar el mejor modelo encontrado
        modelo_dt_tuned = grid_search.best_estimator_

        # Predicciones del modelo ajustado con las variables Boruta
        y_pred_tuned = modelo_dt_tuned.predict(X_test_boruta)
        y_proba_tuned = modelo_dt_tuned.predict_proba(X_test_boruta)[:, 1]

        logger.info("\n=== MATRIZ DE CONFUSIÓN ===")
        logger.info(confusion_matrix(y_test, y_pred_tuned))

        logger.info("\n=== REPORTE DE CLASIFICACIÓN ===")
        logger.info(classification_report(y_test, y_pred_tuned))

        # Visualización del árbol (limitado)
        try:
            plt.figure(figsize=(40, 20))
            from sklearn.tree import plot_tree
            plot_tree(
                modelo_dt_tuned,
                max_depth=3,  # Mantener profundidad deseada
                feature_names=selected_features_boruta,  # Nombres de variables
                class_names=['Clase 0', 'Clase 1'],  # Nombres de clases
                filled=True,  # Colorear nodos
                rounded=True,  # Bordes redondeados
                fontsize=16,  # Fuente más grande para mejor lectura
                impurity=True,  # Mostrar índice Gini
                proportion=False  # Mostrar conteos reales
            )
            plt.title('Visualización del Árbol de Decisión (profundidad 4)', fontsize=24)
            plt.tight_layout(pad=5.0)
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudo mostrar visualización del árbol: {e}")

        # Curva ROC
        try:
            auc_dt = roc_auc_score(y_test, y_proba_tuned)
            logger.info(f"AUC en test: {auc_dt:.4f}")

            fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_tuned)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr_dt, tpr_dt, label=f'Árbol de Decisión (AUC = {auc_dt:.4f})', color='blue')
            plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - Árbol de Decisión')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudo mostrar curva ROC: {e}")

        return {
            'model': modelo_dt_tuned,
            'best_params': grid_search.best_params_,
            'y_pred': y_pred_tuned,
            'y_proba': y_proba_tuned,
            'search_object': grid_search
        }

    def train_xgboost(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena XGBoost con optimización de hiperparámetros.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO XGBOOST")
        logger.info("="*50)

        # ----------------------------------------
        # 1. Selección de variables Boruta ya escaladas
        # ----------------------------------------
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta  = X_test_boruta[selected_features_boruta]

        # ----------------------------------------
        # 2. Malla de hiperparámetros
        # ----------------------------------------
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8]
        }

        # ----------------------------------------
        # 3. Inicializar modelo base
        # ----------------------------------------
        xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=12345)

        # ----------------------------------------
        # 4. GridSearchCV con validación cruzada estratificada
        # ----------------------------------------
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=12345)
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=1,
            verbose=1
        )
        grid_search.fit(X_train_boruta, y_train)

        # ----------------------------------------
        # 5. Mejores hiperparámetros encontrados
        # ----------------------------------------
        logger.info("\n=== MEJORES HIPERPARÁMETROS ===")
        logger.info(grid_search.best_params_)

        # ----------------------------------------
        # 6. Validación cruzada con el mejor modelo
        # ----------------------------------------
        xgb_mejorado = grid_search.best_estimator_
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_resultados = cross_validate(xgb_mejorado, X_train_boruta, y_train, cv=cv, scoring=scoring)

        logger.info("\n=== VALIDACIÓN CRUZADA CON MEJOR MODELO ===")
        for metric in scoring:
            media = cv_resultados[f'test_{metric}'].mean()
            std   = cv_resultados[f'test_{metric}'].std()
            logger.info(f"{metric.upper():<10}: {media:.4f} ± {std:.4f}")

        # ----------------------------------------
        # 7. Evaluación en conjunto de test
        # ----------------------------------------
        xgb_mejorado.fit(X_train_boruta, y_train)
        y_pred_xgb  = xgb_mejorado.predict(X_test_boruta)
        y_proba_xgb = xgb_mejorado.predict_proba(X_test_boruta)[:, 1]

        logger.info("\n=== MATRIZ DE CONFUSIÓN - TEST ===")
        logger.info(confusion_matrix(y_test, y_pred_xgb))

        logger.info("\n=== REPORTE DE CLASIFICACIÓN - TEST ===")
        logger.info(classification_report(y_test, y_pred_xgb))

        # ----------------------------------------
        # AUC y curva ROC
        # ----------------------------------------
        try:
            auc = roc_auc_score(y_test, y_proba_xgb)
            logger.info(f"\nAUC en test: {auc:.4f}")

            fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - XGBoost')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

            # Mapa de calor de matriz de confusión
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Oranges')
            plt.title('Matriz de Confusión - Test (XGBoost)')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos: {e}")

        return {
            'model': xgb_mejorado,
            'best_params': grid_search.best_params_,
            'cv_results': cv_resultados,
            'y_pred': y_pred_xgb,
            'y_proba': y_proba_xgb,
            'search_object': grid_search
        }

    def train_knn(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena KNN con pipeline de escalado.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO KNN")
        logger.info("="*50)

        # Usar variables seleccionadas por Boruta
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta = X_test_boruta[selected_features_boruta]

        # --- 0. Muestreo para acelerar (100k filas, ajustar según memoria y tiempo) ---
        X_sample, _, y_sample, _ = train_test_split(
            X_train_boruta, y_train,
            train_size=100000,  # o menos si quieres más rápido
            stratify=y_train,
            random_state=12345
        )

        # --- 1. Definir pipeline ---
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        # --- 2. Rejilla de hiperparámetros reducida ---
        param_grid_knn = {
            'knn__n_neighbors': [3, 5],
            'knn__weights': ['distance'],
            'knn__metric': ['euclidean']
        }

        # --- 3. Validación cruzada con menos folds ---
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=12345)

        # --- 4. GridSearchCV ---
        grid_search_knn = GridSearchCV(
            estimator=knn_pipe,
            param_grid=param_grid_knn,
            cv=cv,
            scoring='accuracy',
            n_jobs=2,  # puedes probar con 1 también
            verbose=1
        )

        # --- 5. Ajustar el modelo con la muestra ---
        grid_search_knn.fit(X_sample, y_sample)

        # --- 6. Resultados de validación cruzada ---
        logger.info("\n=== MEJORES HIPERPARÁMETROS KNN ===")
        logger.info(grid_search_knn.best_params_)
        logger.info(f"\nMejor score de validación cruzada: {grid_search_knn.best_score_:.4f}")

        # --- 7. Evaluación en conjunto de test completo ---
        y_pred_knn = grid_search_knn.predict(X_test_boruta)
        y_prob_knn = grid_search_knn.predict_proba(X_test_boruta)[:, 1]

        # Reporte de clasificación
        logger.info("\n=== CLASIFICACIÓN EN TEST ===")
        logger.info(classification_report(y_test, y_pred_knn))

        # Visualizaciones
        try:
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_knn)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Matriz de Confusión - KNN")
            plt.show()

            # AUC y Curva ROC
            auc_roc_knn = roc_auc_score(y_test, y_prob_knn)
            logger.info(f"AUC ROC en test: {auc_roc_knn:.4f}")

            fpr, tpr, thresholds = roc_curve(y_test, y_prob_knn)
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_roc_knn)
            roc_display.plot()
            plt.title("Curva ROC - KNN")
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos: {e}")

        return {
            'model': grid_search_knn.best_estimator_,
            'best_params': grid_search_knn.best_params_,
            'y_pred': y_pred_knn,
            'y_proba': y_prob_knn,
            'search_object': grid_search_knn
        }

    def train_random_forest(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena Random Forest con muestreo para eficiencia computacional.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO RANDOM FOREST")
        logger.info("="*50)

        # ----------------------------------------
        # Datos con variables seleccionadas por Boruta
        # ----------------------------------------
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta  = X_test_boruta[selected_features_boruta]

        # ----------------------------------------
        # Submuestreo para evitar MemoryError
        # ----------------------------------------
        X_train_sample = X_train_boruta.sample(n=100_000, random_state=12345)
        y_train_sample = y_train.loc[X_train_sample.index]

        # ----------------------------------------
        # Malla de hiperparámetros reducida
        # ----------------------------------------
        param_grid_rf = {
            'n_estimators': [100, 150],
            'max_depth': [5, 10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

        # Modelo base
        rf = RandomForestClassifier(random_state=12345)

        # Validación cruzada estratificada
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=12345)

        # GridSearchCV con backend threading para menor uso de memoria
        from sklearn.utils import parallel_backend
        grid_search_rf = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_rf,
            cv=cv,
            scoring='f1',
            n_jobs=1,         # ← Reduce uso de RAM
            verbose=1
        )

        # Ajustar con datos de entrenamiento Boruta (submuestreados)
        with parallel_backend('threading'):
            grid_search_rf.fit(X_train_sample, y_train_sample)

        # ----------------------------------------
        # Modelo optimizado y mejores parámetros
        # ----------------------------------------
        logger.info("\n=== MEJORES HIPERPARÁMETROS - RANDOM FOREST ===")
        logger.info(grid_search_rf.best_params_)

        modelo_rf_tuned = grid_search_rf.best_estimator_

        # ----------------------------------------
        # Validación cruzada con múltiples métricas
        # ----------------------------------------
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_resultados = cross_validate(modelo_rf_tuned, X_train_sample, y_train_sample, cv=cv, scoring=scoring)

        logger.info("\n=== VALIDACIÓN CRUZADA CON MEJOR MODELO RF ===")
        for metric in scoring:
            media = cv_resultados[f'test_{metric}'].mean()
            std = cv_resultados[f'test_{metric}'].std()
            logger.info(f"{metric.upper():<10}: {media:.4f} ± {std:.4f}")

        # ----------------------------------------
        # Evaluación en conjunto de test
        # ----------------------------------------
        modelo_rf_tuned.fit(X_train_sample, y_train_sample)
        y_pred_rf = modelo_rf_tuned.predict(X_test_boruta)
        y_proba_rf = modelo_rf_tuned.predict_proba(X_test_boruta)[:, 1]

        # Matriz de confusión y reporte
        logger.info("\n=== MATRIZ DE CONFUSIÓN - TEST RF ===")
        logger.info(confusion_matrix(y_test, y_pred_rf))

        logger.info("\n=== REPORTE DE CLASIFICACIÓN - TEST RF ===")
        logger.info(classification_report(y_test, y_pred_rf, digits=2))

        # Visualizaciones
        try:
            # Heatmap de la matriz de confusión
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
            plt.title('Matriz de Confusión - Test Random Forest')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            plt.show()

            # Curva ROC
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
            auc_rf = roc_auc_score(y_test, y_proba_rf)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', color='darkorange')
            plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.5)')
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Curva ROC - Random Forest')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos: {e}")

        return {
            'model': modelo_rf_tuned,
            'best_params': grid_search_rf.best_params_,
            'cv_results': cv_resultados,
            'y_pred': y_pred_rf,
            'y_proba': y_proba_rf,
            'search_object': grid_search_rf
        }

    def train_svm(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena SVM con muestreo para eficiencia.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO SVM")
        logger.info("="*50)

        # Usar variables seleccionadas por Boruta
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta = X_test_boruta[selected_features_boruta]

        # ----------------------------------------
        # 1. Submuestreo de entrenamiento para acelerar el ajuste
        # ----------------------------------------
        X_train_small = X_train_boruta.sample(n=100_000, random_state=12345)
        y_train_small = y_train.loc[X_train_small.index]

        # ----------------------------------------
        # 2. Modelo base SVM (sin probas, más rápido)
        # ----------------------------------------
        svm_base = SVC(random_state=12345)

        # ----------------------------------------
        # 3. Grid de hiperparámetros reducido
        # ----------------------------------------
        param_grid = {
            'C': [0.1, 1],
            'kernel': ['linear']
        }

        # ----------------------------------------
        # 4. Búsqueda de hiperparámetros con GridSearchCV
        # ----------------------------------------
        grid_svm = GridSearchCV(
            estimator=svm_base,
            param_grid=param_grid,
            cv=2,                    # Solo 2 folds para acelerar
            scoring='f1',
            n_jobs=2,
            verbose=1
        )

        # ----------------------------------------
        # 5. Entrenamiento
        # ----------------------------------------
        grid_svm.fit(X_train_small, y_train_small)

        # ----------------------------------------
        # 6. Mejor modelo
        # ----------------------------------------
        best_svm = grid_svm.best_estimator_
        logger.info("\n=== MEJORES PARÁMETROS SVM ===")
        logger.info(grid_svm.best_params_)

        # ----------------------------------------
        # 7. Evaluación en conjunto de prueba completo
        # ----------------------------------------
        y_pred_svm = best_svm.predict(X_test_boruta)

        logger.info("\n=== MÉTRICAS SVM EN TEST ===")
        logger.info("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_svm))
        logger.info(f"Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
        logger.info(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
        logger.info(f"Recall   : {recall_score(y_test, y_pred_svm):.4f}")
        logger.info(f"F1 Score : {f1_score(y_test, y_pred_svm):.4f}")

        # ----------------------------------------
        # Calcular scores con decision_function sobre el test con Boruta
        # ----------------------------------------
        try:
            scores_svm = best_svm.decision_function(X_test_boruta)
            auc_svm = roc_auc_score(y_test, scores_svm)
            logger.info(f"AUC en test (con Boruta): {auc_svm:.4f}")

            # Curva ROC
            fpr_svm, tpr_svm, _ = roc_curve(y_test, scores_svm)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.4f})', color='purple')
            plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - SVM (Boruta)')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudo mostrar curva ROC: {e}")
            scores_svm = y_pred_svm  # Fallback para compatibilidad

        return {
            'model': best_svm,
            'best_params': grid_svm.best_params_,
            'y_pred': y_pred_svm,
            'y_proba': scores_svm,
            'search_object': grid_svm
        }

    def train_mlp(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str]) -> Dict[str, Any]:
        """
        Entrena MLP (Red Neuronal) con optimización de hiperparámetros.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO MLP (RED NEURONAL)")
        logger.info("="*50)

        # Verificar memoria disponible
        mem_disp = psutil.virtual_memory().available / (1024**2)
        logger.info(f"💾 Memoria disponible: {mem_disp:.2f} MB")

        # Preparar los datos filtrados por Boruta
        logger.info("🔹 Preparando los datos filtrados por Boruta...")
        X_train_boruta = X_train_boruta[selected_features_boruta]
        X_test_boruta  = X_test_boruta[selected_features_boruta]

        # ⚠️ Limitar número de features si son demasiadas
        if X_train_boruta.shape[1] > 100:
            logger.info("⚠️ Limitando a las 100 variables más importantes...")
            selected_features_boruta = selected_features_boruta[:100]
            X_train_boruta = X_train_boruta[selected_features_boruta]
            X_test_boruta = X_test_boruta[selected_features_boruta]

        # OPCIONAL: reducir tamaño del conjunto para tuning
        logger.info("🔹 Dividiendo conjunto de entrenamiento para tuning rápido...")
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train_boruta, y_train, train_size=0.5, random_state=42
        )

        # Definir espacio de hiperparámetros (reducido y útil)
        logger.info("🔹 Definiendo el espacio de hiperparámetros...")
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [1e-4, 1e-3],
            'learning_rate': ['constant', 'adaptive']
        }

        # Inicializar el modelo base
        logger.info("🔹 Inicializando MLPClassifier...")
        mlp_base = MLPClassifier(
            solver='adam',
            max_iter=200,
            early_stopping=True,
            random_state=12345
        )

        # Ejecutar GridSearchCV (cv=2 para mayor velocidad)
        logger.info("🔹 Ejecutando GridSearchCV (rápido y eficiente)...")
        grid_search = GridSearchCV(
            estimator=mlp_base,
            param_grid=param_grid,
            scoring='f1',
            cv=2,
            verbose=2,
            n_jobs=1
        )

        grid_search.fit(X_train_sub, y_train_sub)

        # Evaluar el mejor modelo con los datos completos
        logger.info("\n✅ GridSearch finalizado. Mejores parámetros:")
        logger.info(grid_search.best_params_)

        logger.info("\n🔹 Reentrenando el mejor modelo con todos los datos...")
        best_mlp = grid_search.best_estimator_
        best_mlp.fit(X_train_boruta, y_train)

        y_pred_best = best_mlp.predict(X_test_boruta)

        logger.info("\n📊 Métricas del modelo final:")
        logger.info("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_best))
        logger.info(f"Accuracy : {accuracy_score(y_test, y_pred_best):.4f}")
        logger.info(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
        logger.info(f"Recall   : {recall_score(y_test, y_pred_best):.4f}")
        logger.info(f"F1 Score : {f1_score(y_test, y_pred_best):.4f}")

        # Curva ROC
        try:
            logger.info("\n🔹 Generando la curva ROC...")
            y_proba_best = best_mlp.predict_proba(X_test_boruta)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba_best)
            auc = roc_auc_score(y_test, y_proba_best)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'MLP Óptimo (AUC = {auc:.4f})', color='green')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - MLP optimizado con GridSearchCV')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            logger.info("\n✅ Proceso completado con éxito.")
        except Exception as e:
            logger.warning(f"No se pudo mostrar curva ROC: {e}")
            y_proba_best = y_pred_best  # Fallback

        return {
            'model': best_mlp,
            'best_params': grid_search.best_params_,
            'y_pred': y_pred_best,
            'y_proba': y_proba_best,
            'search_object': grid_search
        }

    def train_voting_classifier(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str],
                               trained_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entrena VotingClassifier usando modelos ya entrenados.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO VOTING CLASSIFIER")
        logger.info("="*50)

        # Preparar datos con variables Boruta
        X_train_boruta_np = X_train_boruta[selected_features_boruta].to_numpy(copy=True)
        X_test_boruta_np  = X_test_boruta[selected_features_boruta].to_numpy(copy=True)
        y_train_np = y_train.to_numpy(copy=True)
        y_test_np  = y_test.to_numpy(copy=True)

        # Definir modelo MLP (mismo usado antes)
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            alpha=0.0001,
            max_iter=200,
            early_stopping=True,
            random_state=12345
        )

        # Ensamblador VotingClassifier (voto suave)
        # Necesitamos los modelos entrenados de RF y XGB
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', trained_models.get('random_forest', {}).get('model')),
                ('xgb', trained_models.get('xgboost', {}).get('model')),
                ('mlp', mlp)
            ],
            voting='soft',         # combina probabilidades
            weights=[3, 2, 1],      # peso personalizado
            n_jobs=-1
        )

        # Entrenamiento del ensamblado
        voting_clf.fit(X_train_boruta_np, y_train_np)

        # Predicción en test
        y_pred_vot = voting_clf.predict(X_test_boruta_np)
        y_proba_vot = voting_clf.predict_proba(X_test_boruta_np)[:, 1]

        # Matriz de confusión y métricas
        cm = confusion_matrix(y_test_np, y_pred_vot)
        logger.info("=== MATRIZ DE CONFUSIÓN ===")
        logger.info(cm)

        logger.info("\n=== REPORTE DE CLASIFICACIÓN ===")
        logger.info(classification_report(y_test_np, y_pred_vot))

        # Visualizaciones
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Matriz de Confusión - VotingClassifier (Boruta)")
            plt.show()

            # Curva ROC y AUC
            auc = roc_auc_score(y_test_np, y_proba_vot)
            logger.info(f"\nAUC en test: {auc:.4f}")

            fpr, tpr, _ = roc_curve(y_test_np, y_proba_vot)
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            roc_disp.plot()
            plt.title("Curva ROC - VotingClassifier (Boruta)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos: {e}")

        return {
            'model': voting_clf,
            'best_params': None,
            'y_pred': y_pred_vot,
            'y_proba': y_proba_vot,
            'search_object': None
        }

    def train_stacking_classifier(self, X_train_boruta: pd.DataFrame, X_test_boruta: pd.DataFrame,
                                 y_train: pd.Series, y_test: pd.Series, selected_features_boruta: List[str],
                                 trained_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entrena StackingClassifier usando modelos ya entrenados.
        """
        logger.info("="*50)
        logger.info("ENTRENANDO STACKING CLASSIFIER")
        logger.info("="*50)

        # Preparar los datos filtrados por Boruta
        X_train_boruta_arr = X_train_boruta[selected_features_boruta].to_numpy(copy=True)
        y_train_boruta_arr = y_train.to_numpy(copy=True)

        # Definir el modelo MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            alpha=0.0001,
            max_iter=200,
            early_stopping=True,
            random_state=12345
        )

        # Crear StackingClassifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', trained_models.get('random_forest', {}).get('model')),
                ('xgb', trained_models.get('xgboost', {}).get('model')),
                ('mlp', mlp)
            ],
            final_estimator=LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=12345
            ),
            cv=3,
            n_jobs=1,
            passthrough=False
        )

        # Entrenar el modelo ensamblado
        stacking_clf.fit(X_train_boruta_arr, y_train_boruta_arr)

        # Convertir test con Boruta a arrays writeable
        X_test_boruta_arr = X_test_boruta[selected_features_boruta].to_numpy(copy=True)
        y_test_boruta_arr = y_test.to_numpy(copy=True)

        # Predicción y probabilidades
        y_pred_stack  = stacking_clf.predict(X_test_boruta_arr)
        y_proba_stack = stacking_clf.predict_proba(X_test_boruta_arr)[:, 1]

        # Matriz de confusión y reporte
        cm = confusion_matrix(y_test_boruta_arr, y_pred_stack)
        logger.info("=== MATRIZ DE CONFUSIÓN ===")
        logger.info(cm)

        logger.info("\n=== REPORTE DE CLASIFICACIÓN ===")
        logger.info(classification_report(y_test_boruta_arr, y_pred_stack))

        # Visualizaciones
        try:
            # Plot de matriz
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Oranges)
            plt.title("Matriz de Confusión - StackingClassifier (Boruta)")
            plt.show()

            # Curva ROC y AUC
            auc = roc_auc_score(y_test_boruta_arr, y_proba_stack)
            logger.info(f"\nAUC en test: {auc:.4f}")

            fpr, tpr, _ = roc_curve(y_test_boruta_arr, y_proba_stack)
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            roc_disp.plot()
            plt.title("Curva ROC - StackingClassifier (Boruta)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos: {e}")

        return {
            'model': stacking_clf,
            'best_params': None,
            'y_pred': y_pred_stack,
            'y_proba': y_proba_stack,
            'search_object': None
        }

    def compare_all_models(self, y_test: pd.Series, trained_models: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compara todos los modelos entrenados y genera visualizaciones.
        """
        logger.info("="*50)
        logger.info("COMPARANDO TODOS LOS MODELOS")
        logger.info("="*50)

        # Diccionario de resultados para comparación
        modelos_resultados = {}
        colors = ['blue', 'green', 'darkorange', 'red', 'teal', 'purple', 'deeppink', 'black', 'gray']

        for i, (nombre, datos) in enumerate(trained_models.items()):
            if 'y_pred' in datos and 'y_proba' in datos:
                modelos_resultados[nombre] = {
                    "y_pred": datos["y_pred"],
                    "y_proba": datos["y_proba"],
                    "color": colors[i % len(colors)]
                }

        # Comparar métricas en DataFrame
        metricas = []
        for nombre, datos in modelos_resultados.items():
            y_pred = datos["y_pred"]
            y_proba = datos["y_proba"]
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)

            metricas.append({
                "Modelo": nombre,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "AUC": auc_score
            })

        df_metricas = pd.DataFrame(metricas).sort_values("AUC", ascending=False)
        logger.info("\n=== COMPARACIÓN DE MÉTRICAS ===")
        logger.info(df_metricas.round(4))

        # Graficar todas las curvas ROC
        try:
            plt.figure(figsize=(10, 8))

            for nombre, datos in modelos_resultados.items():
                fpr, tpr, _ = roc_curve(y_test, datos["y_proba"])
                roc_auc = roc_auc_score(y_test, datos["y_proba"])
                plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.4f})', color=datos["color"])

            plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Comparación de Curvas ROC')
            plt.legend(loc='lower right', fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Gráfico de barras agrupadas
            metricas_a_graficar = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
            df_melt = df_metricas.melt(id_vars="Modelo", value_vars=metricas_a_graficar,
                                     var_name="Métrica", value_name="Valor")

            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melt, x="Métrica", y="Valor", hue="Modelo")
            plt.title("Comparación de Métricas por Modelo")
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.warning(f"No se pudieron mostrar gráficos de comparación: {e}")

        return df_metricas

    def save_all_models(self, trained_models: Dict[str, Dict[str, Any]], output_path: str = "models/"):
        """
        Guarda todos los modelos entrenados.
        """
        logger.info("="*50)
        logger.info("GUARDANDO MODELOS")
        logger.info("="*50)

        import os
        os.makedirs(output_path, exist_ok=True)

        for nombre, datos in trained_models.items():
            if 'model' in datos:
                model_path = os.path.join(output_path, f"{nombre}_model.pkl")
                try:
                    joblib.dump(datos['model'], model_path)
                    logger.info(f"✅ Modelo {nombre} guardado en {model_path}")
                except Exception as e:
                    logger.error(f"❌ Error guardando {nombre}: {e}")

        logger.info("🎉 Proceso de guardado completado.")

    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        selected_features: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Entrena todos los modelos en secuencia.
        """
        logger.info("="*60)
        logger.info("INICIANDO ENTRENAMIENTO DE TODOS LOS MODELOS")
        logger.info("="*60)

        trained_models = {}

        try:
            # 1. Logistic Regression
            trained_models['logistic_regression'] = self.train_logistic_regression(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 2. Decision Tree
            trained_models['decision_tree'] = self.train_decision_tree(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 3. XGBoost
            trained_models['xgboost'] = self.train_xgboost(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 4. KNN
            trained_models['knn'] = self.train_knn(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 5. Random Forest
            trained_models['random_forest'] = self.train_random_forest(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 6. SVM
            trained_models['svm'] = self.train_svm(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 7. MLP
            trained_models['mlp'] = self.train_mlp(
                X_train, X_test, y_train, y_test, selected_features
            )

            # 8. Voting Classifier
            trained_models['voting_classifier'] = self.train_voting_classifier(
                X_train, X_test, y_train, y_test, selected_features, trained_models
            )

            # 9. Stacking Classifier
            trained_models['stacking_classifier'] = self.train_stacking_classifier(
                X_train, X_test, y_train, y_test, selected_features, trained_models
            )

            # 10. Comparar todos los modelos
            comparison_df = self.compare_all_models(y_test, trained_models)

            # 11. Guardar modelos
            self.save_all_models(trained_models)

            logger.info("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")

        except Exception as e:
            logger.error(f"❌ Error durante el entrenamiento: {e}")
            raise

        return trained_models
