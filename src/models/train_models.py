"""
Módulo para entrenamiento de modelos de machine learning.
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
import xgboost as xgb
import joblib
from src.utils.config import get_config
from src.utils.helpers import save_model, validate_model_inputs

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para entrenar múltiples modelos de machine learning."""

    def __init__(self):
        self.config = get_config()
        self.model_config = self.config['model_config']
        self.hyperparameters = self.config['hyperparameters']
        self.trained_models = {}

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena modelo de regresión logística con búsqueda de hiperparámetros.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando Regresión Logística...")

        # Configurar modelo base
        lr_base = LogisticRegression(
            random_state=self.model_config['random_state'],
            max_iter=1000
        )

        # Búsqueda de hiperparámetros
        search = RandomizedSearchCV(
            lr_base,
            self.hyperparameters['logistic_regression'],
            n_iter=10,
            cv=3,
            scoring='f1',
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs']
        )

        search.fit(X_train, y_train)

        # Validación cruzada con mejor modelo
        cv_results = self._cross_validate_model(search.best_estimator_, X_train, y_train)

        logger.info(f"Logistic Regression - Mejores parámetros: {search.best_params_}")
        logger.info(f"Logistic Regression - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena árbol de decisión con optimización de hiperparámetros.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando Árbol de Decisión...")

        dt = DecisionTreeClassifier(random_state=self.model_config['random_state'])

        search = GridSearchCV(
            dt,
            self.hyperparameters['decision_tree'],
            cv=5,
            scoring='f1',
            n_jobs=self.model_config['n_jobs']
        )

        search.fit(X_train, y_train)
        cv_results = self._cross_validate_model(search.best_estimator_, X_train, y_train)

        logger.info(f"Decision Tree - Mejores parámetros: {search.best_params_}")
        logger.info(f"Decision Tree - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena Random Forest con muestreo para eficiencia computacional.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando Random Forest...")

        # Muestreo para GridSearch
        sample_size = min(100000, len(X_train))
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=self.model_config['random_state']
        )

        rf = RandomForestClassifier(random_state=self.model_config['random_state'])

        # Búsqueda con parámetros reducidos para eficiencia
        reduced_params = {
            'n_estimators': [100, 150],
            'max_depth': [5, 10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

        search = GridSearchCV(
            rf,
            reduced_params,
            cv=3,
            scoring='f1',
            n_jobs=1,  # Evitar sobrecarga
            verbose=1
        )

        search.fit(X_sample, y_sample)

        # Reentrenar con dataset completo
        best_rf = search.best_estimator_
        best_rf.fit(X_train, y_train)

        cv_results = self._cross_validate_model(best_rf, X_train, y_train)

        logger.info(f"Random Forest - Mejores parámetros: {search.best_params_}")
        logger.info(f"Random Forest - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': best_rf,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena XGBoost con optimización de hiperparámetros.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando XGBoost...")

        xgb_base = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.model_config['random_state']
        )

        search = GridSearchCV(
            xgb_base,
            self.hyperparameters['xgboost'],
            scoring='roc_auc',
            cv=3,
            n_jobs=1,
            verbose=1
        )

        search.fit(X_train, y_train)
        cv_results = self._cross_validate_model(search.best_estimator_, X_train, y_train)

        logger.info(f"XGBoost - Mejores parámetros: {search.best_params_}")
        logger.info(f"XGBoost - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena KNN con pipeline de escalado.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando KNN...")

        # Muestreo para eficiencia
        sample_size = min(100000, len(X_train))
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=self.model_config['random_state']
        )

        # Pipeline con escalado
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        # Parámetros reducidos para eficiencia
        reduced_params = {
            'knn__n_neighbors': [3, 5],
            'knn__weights': ['distance'],
            'knn__metric': ['euclidean']
        }

        search = GridSearchCV(
            knn_pipe,
            reduced_params,
            cv=3,
            scoring='accuracy',
            n_jobs=2,
            verbose=1
        )

        search.fit(X_sample, y_sample)

        # Validación en muestra (KNN puede ser costoso con dataset completo)
        cv_results = self._cross_validate_model(search.best_estimator_, X_sample, y_sample)

        logger.info(f"KNN - Mejores parámetros: {search.best_params_}")
        logger.info(f"KNN - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena SVM con muestreo para eficiencia.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando SVM...")

        # Muestreo significativo para SVM
        sample_size = min(100000, len(X_train))
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=self.model_config['random_state']
        )

        svm_base = SVC(random_state=self.model_config['random_state'])

        # Parámetros reducidos para eficiencia
        reduced_params = {
            'C': [0.1, 1],
            'kernel': ['linear']  # Solo linear para eficiencia
        }

        search = GridSearchCV(
            svm_base,
            reduced_params,
            cv=2,
            scoring='f1',
            n_jobs=2,
            verbose=1
        )

        search.fit(X_sample, y_sample)
        cv_results = self._cross_validate_model(search.best_estimator_, X_sample, y_sample)

        logger.info(f"SVM - Mejores parámetros: {search.best_params_}")
        logger.info(f"SVM - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_mlp(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena Red Neuronal MLP con early stopping.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con modelo entrenado y métricas
        """
        logger.info("Entrenando MLP (Red Neuronal)...")

        # Muestreo para GridSearch
        sample_size = min(len(X_train) // 2, 200000)
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=self.model_config['random_state']
        )

        mlp_base = MLPClassifier(
            solver='adam',
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.model_config['random_state']
        )

        # Parámetros reducidos para eficiencia
        reduced_params = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [1e-4, 1e-3],
            'learning_rate': ['adaptive']
        }

        search = GridSearchCV(
            mlp_base,
            reduced_params,
            scoring='f1',
            cv=3,
            verbose=2,
            n_jobs=1
        )

        search.fit(X_sample, y_sample)

        # Reentrenar con dataset completo
        best_mlp = search.best_estimator_
        best_mlp.fit(X_train, y_train)

        cv_results = self._cross_validate_model(best_mlp, X_train, y_train)

        logger.info(f"MLP - Mejores parámetros: {search.best_params_}")
        logger.info(f"MLP - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': best_mlp,
            'best_params': search.best_params_,
            'cv_results': cv_results,
            'search_object': search
        }

    def train_ensemble_voting(self, base_models: Dict[str, Any],
                            X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena ensemble de votación con modelos base.

        Args:
            base_models: Diccionario con modelos base entrenados
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con ensemble entrenado
        """
        logger.info("Entrenando Voting Classifier...")

        # Seleccionar mejores modelos base
        selected_models = []
        if 'random_forest' in base_models:
            selected_models.append(('rf', base_models['random_forest']['model']))
        if 'xgboost' in base_models:
            selected_models.append(('xgb', base_models['xgboost']['model']))
        if 'mlp' in base_models:
            selected_models.append(('mlp', base_models['mlp']['model']))

        if len(selected_models) < 2:
            logger.warning("Insuficientes modelos base para ensemble")
            return None

        voting_clf = VotingClassifier(
            estimators=selected_models,
            voting='soft',
            weights=[3, 2, 1],  # Pesos personalizados
            n_jobs=self.model_config['n_jobs']
        )

        # Entrenar con arrays numpy para compatibilidad
        X_train_np = X_train.to_numpy(copy=True)
        y_train_np = y_train.to_numpy(copy=True)

        voting_clf.fit(X_train_np, y_train_np)

        # Validación cruzada
        cv_results = self._cross_validate_model(voting_clf, X_train_np, y_train_np)

        logger.info(f"Voting Classifier - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': voting_clf,
            'base_models': selected_models,
            'cv_results': cv_results
        }

    def train_ensemble_stacking(self, base_models: Dict[str, Any],
                              X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena ensemble de stacking con meta-learner.

        Args:
            base_models: Diccionario con modelos base entrenados
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con ensemble entrenado
        """
        logger.info("Entrenando Stacking Classifier...")

        # Seleccionar modelos base
        selected_models = []
        if 'random_forest' in base_models:
            selected_models.append(('rf', base_models['random_forest']['model']))
        if 'xgboost' in base_models:
            selected_models.append(('xgb', base_models['xgboost']['model']))
        if 'mlp' in base_models:
            selected_models.append(('mlp', base_models['mlp']['model']))

        if len(selected_models) < 2:
            logger.warning("Insuficientes modelos base para stacking")
            return None

        # Meta-learner
        meta_learner = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.model_config['random_state']
        )

        stacking_clf = StackingClassifier(
            estimators=selected_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1,
            passthrough=False
        )

        # Entrenar con arrays numpy
        X_train_np = X_train.to_numpy(copy=True)
        y_train_np = y_train.to_numpy(copy=True)

        stacking_clf.fit(X_train_np, y_train_np)

        stacking_clf.fit(X_train_np, y_train_np)

        cv_results = self._cross_validate_model(stacking_clf, X_train_np, y_train_np)

        logger.info(f"Stacking Classifier - CV F1: {cv_results['test_f1'].mean():.4f}")

        return {
            'model': stacking_clf,
            'base_models': selected_models,
            'meta_learner': meta_learner,
            'cv_results': cv_results
        }

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Entrena todos los modelos del pipeline.

        Args:
            X_train: Variables independientes de entrenamiento
            y_train: Variable objetivo de entrenamiento

        Returns:
            Diccionario con todos los modelos entrenados
        """
        logger.info("="*60)
        logger.info("INICIANDO ENTRENAMIENTO DE TODOS LOS MODELOS")
        logger.info("="*60)

        # Validar inputs
        if not validate_model_inputs(X_train, X_train, y_train, y_train):
            raise ValueError("Validación de inputs falló")

        trained_models = {}

        # Lista de métodos de entrenamiento
        training_methods = [
            ('logistic_regression', self.train_logistic_regression),
            ('decision_tree', self.train_decision_tree),
            ('random_forest', self.train_random_forest),
            ('xgboost', self.train_xgboost),
            ('knn', self.train_knn),
            ('svm', self.train_svm),
            ('mlp', self.train_mlp)
        ]

        # Entrenar modelos individuales
        for model_name, training_method in training_methods:
            try:
                logger.info(f"\n--- Entrenando {model_name.upper()} ---")
                model_result = training_method(X_train, y_train)
                trained_models[model_name] = model_result

                # Guardar modelo
                self.save_trained_model(model_result['model'], model_name)

            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {e}")
                continue

        # Entrenar ensambles si hay modelos base disponibles
        if len(trained_models) >= 3:
            try:
                logger.info(f"\n--- Entrenando VOTING CLASSIFIER ---")
                voting_result = self.train_ensemble_voting(trained_models, X_train, y_train)
                if voting_result:
                    trained_models['voting_classifier'] = voting_result
                    self.save_trained_model(voting_result['model'], 'voting_classifier')
            except Exception as e:
                logger.error(f"Error entrenando Voting Classifier: {e}")

            try:
                logger.info(f"\n--- Entrenando STACKING CLASSIFIER ---")
                stacking_result = self.train_ensemble_stacking(trained_models, X_train, y_train)
                if stacking_result:
                    trained_models['stacking_classifier'] = stacking_result
                    self.save_trained_model(stacking_result['model'], 'stacking_classifier')
            except Exception as e:
                logger.error(f"Error entrenando Stacking Classifier: {e}")

        logger.info("="*60)
        logger.info(f"ENTRENAMIENTO COMPLETADO: {len(trained_models)} modelos")
        logger.info("="*60)

        self.trained_models = trained_models
        return trained_models

    def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """
        Ejecuta validación cruzada en un modelo.

        Args:
            model: Modelo a evaluar
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Diccionario con resultados de validación cruzada
        """
        cv = StratifiedKFold(
            n_splits=self.model_config['cv_folds'],
            shuffle=True,
            random_state=self.model_config['random_state']
        )

        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=self.model_config['n_jobs'],
            return_train_score=False
        )

        return cv_results

    def save_trained_model(self, model, model_name: str) -> str:
        """
        Guarda un modelo entrenado.

        Args:
            model: Modelo a guardar
            model_name: Nombre del modelo

        Returns:
            Ruta donde se guardó el modelo
        """
        return save_model(model, model_name, self.config)

    def load_trained_models(self) -> Optional[Dict[str, Any]]:
        """
        Intenta cargar modelos ya entrenados.

        Returns:
            Diccionario con modelos cargados o None si no existen
        """
        try:
            models_dir = self.config['paths']['models_dir']
            model_files = [
                'logistic_regression.joblib',
                'random_forest.joblib',
                'xgboost.joblib',
                'mlp.joblib'
            ]

            loaded_models = {}

            for model_file in model_files:
                model_path = models_dir / model_file
                if model_path.exists():
                    model_name = model_file.replace('.joblib', '')
                    model = joblib.load(model_path)
                    loaded_models[model_name] = {'model': model}

            if loaded_models:
                logger.info(f"Modelos cargados desde cache: {list(loaded_models.keys())}")
                return loaded_models

        except Exception as e:
            logger.warning(f"No se pudieron cargar modelos desde cache: {e}")

        return None

    def get_model_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de todos los modelos entrenados.

        Returns:
            DataFrame con resumen de modelos
        """
        if not self.trained_models:
            logger.warning("No hay modelos entrenados")
            return pd.DataFrame()

        summary_data = []

        for model_name, model_info in self.trained_models.items():
            if 'cv_results' in model_info:
                cv_results = model_info['cv_results']

                summary_data.append({
                    'Model': model_name,
                    'CV_Accuracy': cv_results['test_accuracy'].mean(),
                    'CV_Precision': cv_results['test_precision'].mean(),
                    'CV_Recall': cv_results['test_recall'].mean(),
                    'CV_F1': cv_results['test_f1'].mean(),
                    'CV_AUC': cv_results['test_roc_auc'].mean(),
                    'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
                    'CV_F1_Std': cv_results['test_f1'].std()
                })

        summary_df = pd.DataFrame(summary_data)
        return summary_df.sort_values('CV_F1', ascending=False)

    def get_best_model_by_metric(self, metric: str = 'f1') -> Tuple[str, Any]:
        """
        Obtiene el mejor modelo según una métrica específica.

        Args:
            metric: Métrica a usar para selección ('f1', 'accuracy', 'auc')

        Returns:
            Tupla con (nombre_modelo, modelo)
        """
        if not self.trained_models:
            raise ValueError("No hay modelos entrenados")

        best_score = -1
        best_model_name = None
        best_model = None

        metric_key = f'test_{metric}' if metric != 'auc' else 'test_roc_auc'

        for model_name, model_info in self.trained_models.items():
            if 'cv_results' in model_info:
                score = model_info['cv_results'][metric_key].mean()
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model_info['model']

        logger.info(f"Mejor modelo según {metric}: {best_model_name} (score: {best_score:.4f})")
        return best_model_name, best_model
