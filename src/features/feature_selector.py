"""
Módulo para selección de características usando múltiples métodos.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from boruta import BorutaPy
import statsmodels.api as sm
import shap
import xgboost as xgb
import joblib
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Clase para manejar la selección de características."""

    def __init__(self):
        self.config = get_config()
        self.sampling_config = self.config['sampling_config']
        self.model_config = self.config['model_config']

    def run_boruta_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ejecuta selección de características usando Boruta.

        Args:
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Lista de características seleccionadas
        """
        logger.info("Iniciando selección Boruta...")

        # Muestreo para reducir carga computacional
        sample_size = min(self.sampling_config['boruta_sample_size'], len(X))
        X_sample = X.sample(n=sample_size, random_state=self.model_config['random_state'])
        y_sample = y.loc[X_sample.index]

        # Configurar Random Forest para Boruta
        rf = RandomForestClassifier(
            n_jobs=self.model_config['n_jobs'],
            class_weight='balanced',
            max_depth=5,
            random_state=self.model_config['random_state']
        )

        # Ejecutar Boruta
        boruta_config = self.config['feature_selection']['boruta']
        boruta_selector = BorutaPy(
            estimator=rf,
            n_estimators=boruta_config['n_estimators'],
            max_iter=boruta_config['max_iter'],
            alpha=boruta_config['alpha'],
            random_state=self.model_config['random_state'],
            verbose=1
        )

        boruta_selector.fit(X_sample.values, y_sample.values)

        # Obtener características seleccionadas
        selected_features = X.columns[boruta_selector.support_].tolist()

        logger.info(f"Boruta completado: {len(selected_features)} características seleccionadas")
        logger.info(f"Características: {selected_features[:10]}...")

        # Guardar resultado
        self._save_selected_features(selected_features, 'boruta')

        return selected_features

    def run_rfecv_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ejecuta selección usando Recursive Feature Elimination with CV.

        Args:
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Lista de características seleccionadas
        """
        logger.info("Iniciando selección RFECV...")

        # Muestreo
        sample_size = min(self.sampling_config['rfecv_sample_size'], len(X))
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=self.model_config['random_state']
        )

        # Configurar modelo base
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.model_config['random_state']
        )

        # Configurar RFECV
        cv = StratifiedKFold(
            n_splits=self.model_config['cv_folds'],
            shuffle=True,
            random_state=self.model_config['random_state']
        )

        rfecv_config = self.config['feature_selection']['rfecv']
        selector = RFECV(
            estimator=model,
            step=rfecv_config['step'],
            cv=cv,
            scoring='accuracy',
            n_jobs=self.model_config['n_jobs'],
            min_features_to_select=rfecv_config['min_features_to_select']
        )

        selector.fit(X_sample, y_sample)

        # Obtener características seleccionadas
        selected_features = X_sample.columns[selector.support_].tolist()

        logger.info(f"RFECV completado: {len(selected_features)} características seleccionadas")

        # Guardar resultado
        self._save_selected_features(selected_features, 'rfecv')

        return selected_features

    def run_stepwise_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ejecuta selección stepwise usando regresión logística.

        Args:
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Lista de características seleccionadas
        """
        logger.info("Iniciando selección Stepwise...")

        # Muestreo más grande para stepwise
        sample_size = min(self.sampling_config['stepwise_sample_size'], len(X))
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=self.model_config['random_state']
        )

        # Filtrar variables con varianza baja
        selector_var = VarianceThreshold(threshold=1e-5)
        X_filtered = pd.DataFrame(
            selector_var.fit_transform(X_sample),
            columns=X_sample.columns[selector_var.get_support()],
            index=X_sample.index
        )

        # Eliminar variables altamente correlacionadas
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_filtered = X_filtered.drop(columns=to_drop)

        # Ejecutar stepwise
        stepwise_config = self.config['feature_selection']['stepwise']
        selected_features = self._stepwise_selection_verbose(
            X_filtered,
            y_sample,
            threshold_in=stepwise_config['threshold_in'],
            threshold_out=stepwise_config['threshold_out']
        )

        logger.info(f"Stepwise completado: {len(selected_features)} características seleccionadas")

        # Guardar resultado
        self._save_selected_features(selected_features, 'stepwise')

        return selected_features

    def run_sbf_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ejecuta Sequential Backward Feature Selection.

        Args:
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Lista de características seleccionadas
        """
        logger.info("Iniciando selección SBF...")

        # Muestreo
        sample_size = min(self.sampling_config['sbf_sample_size'], len(X))
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=self.model_config['random_state']
        )

        # Configurar modelo base
        model_sbf = LogisticRegression(
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            random_state=self.model_config['random_state']
        )

        # Configurar SBF
        sbf = SequentialFeatureSelector(
            estimator=model_sbf,
            direction='backward',
            n_features_to_select='auto',
            scoring='roc_auc',
            cv=3,
            n_jobs=self.model_config['n_jobs']
        )

        sbf.fit(X_sample, y_sample)

        # Obtener características seleccionadas
        selected_features = X_sample.columns[sbf.get_support()].tolist()

        logger.info(f"SBF completado: {len(selected_features)} características seleccionadas")

        # Guardar resultado
        self._save_selected_features(selected_features, 'sbf')

        return selected_features

    def run_shap_selection(self, X: pd.DataFrame, y: pd.Series, top_n: int = 25) -> List[str]:
        """
        Ejecuta selección de características usando SHAP values.

        Args:
            X: Variables independientes
            y: Variable objetivo
            top_n: Número de características principales a seleccionar

        Returns:
            Lista de características seleccionadas
        """
        logger.info("Iniciando selección SHAP...")

        # Entrenar modelo XGBoost
        model_shap = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs'],
            use_label_encoder=False,
            eval_metric='logloss'
        )

        model_shap.fit(X, y)

        # Calcular SHAP values en una muestra
        sample_size = min(self.sampling_config['shap_sample_size'], len(X))
        X_sample = X.sample(n=sample_size, random_state=self.model_config['random_state'])

        explainer = shap.Explainer(model_shap, X)
        shap_values = explainer(X_sample)

        # Calcular importancia media absoluta
        shap_importances = pd.DataFrame({
            'variable': X.columns,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by='mean_abs_shap', ascending=False)

        # Seleccionar top_n características
        selected_features = shap_importances['variable'].head(top_n).tolist()

        logger.info(f"SHAP completado: {len(selected_features)} características seleccionadas")

        # Guardar resultado e importancias
        self._save_selected_features(selected_features, 'shap')
        self._save_shap_importances(shap_importances)

        return selected_features

    def compare_selection_methods(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[str]]:
        """
        Compara múltiples métodos de selección de características.

        Args:
            X: Variables independientes
            y: Variable objetivo

        Returns:
            Diccionario con características seleccionadas por cada método
        """
        logger.info("Comparando métodos de selección de características...")

        methods_results = {}

        # Ejecutar todos los métodos
        methods = [
            ('boruta', self.run_boruta_selection),
            ('rfecv', self.run_rfecv_selection),
            ('stepwise', self.run_stepwise_selection),
            ('sbf', self.run_sbf_selection),
            ('shap', self.run_shap_selection)
        ]

        for method_name, method_func in methods:
            try:
                logger.info(f"Ejecutando método: {method_name}")
                selected_vars = method_func(X, y)
                methods_results[method_name] = selected_vars
                logger.info(f"{method_name}: {len(selected_vars)} variables seleccionadas")
            except Exception as e:
                logger.error(f"Error en método {method_name}: {e}")
                methods_results[method_name] = []

        # Análisis de consenso
        self._analyze_consensus(methods_results)

        return methods_results

    def _stepwise_selection_verbose(self, X: pd.DataFrame, y: pd.Series,
                                  initial_list: List[str] = [],
                                  threshold_in: float = 0.001,
                                  threshold_out: float = 0.01,
                                  verbose: bool = False) -> List[str]:
        """
        Implementa selección stepwise con regresión logística.

        Args:
            X: Variables independientes
            y: Variable objetivo
            initial_list: Lista inicial de variables
            threshold_in: Umbral para incluir variable
            threshold_out: Umbral para excluir variable
            verbose: Si mostrar proceso paso a paso

        Returns:
            Lista de variables seleccionadas
        """
        included = list(initial_list)

        while True:
            changed = False
            excluded = list(set(X.columns) - set(included))

            # Forward step
            if excluded:
                new_pval = pd.Series(index=excluded, dtype=float)
                for new_column in excluded:
                    try:
                        model = sm.Logit(
                            y,
                            sm.add_constant(pd.DataFrame(X[included + [new_column]]))
                        ).fit(disp=0)

                        if new_column in model.pvalues.index:
                            new_pval[new_column] = model.pvalues[new_column]
                        else:
                            new_pval[new_column] = 1
                    except:
                        new_pval[new_column] = 1

                best_pval = new_pval.min()
                if best_pval < threshold_in:
                    best_feature = new_pval.idxmin()
                    included.append(best_feature)
                    changed = True
                    if verbose:
                        logger.info(f'Add {best_feature} with p-value {best_pval:.6f}')

            # Backward step
            if included:
                try:
                    model = sm.Logit(
                        y,
                        sm.add_constant(pd.DataFrame(X[included]))
                    ).fit(disp=0)

                    pvalues = model.pvalues.iloc[1:]  # Excluir constante
                    worst_pval = pvalues.max()

                    if worst_pval > threshold_out:
                        worst_feature = pvalues.idxmax()
                        included.remove(worst_feature)
                        changed = True
                        if verbose:
                            logger.info(f'Drop {worst_feature} with p-value {worst_pval:.6f}')
                except:
                    pass

            if not changed:
                break

        return included

    def _save_selected_features(self, features: List[str], method_name: str) -> None:
        """
        Guarda características seleccionadas.

        Args:
            features: Lista de características
            method_name: Nombre del método
        """
        import pickle

        models_dir = self.config['paths']['models_dir']
        filepath = models_dir / f'selected_features_{method_name}.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump(features, f)

        logger.info(f"Características {method_name} guardadas en: {filepath}")

    def _save_shap_importances(self, importances: pd.DataFrame) -> None:
        """
        Guarda importancias SHAP.

        Args:
            importances: DataFrame con importancias
        """
        reports_dir = self.config['paths']['reports_dir']
        filepath = reports_dir / 'shap_importances.csv'
        importances.to_csv(filepath, index=False)
        logger.info(f"Importancias SHAP guardadas en: {filepath}")

    def _analyze_consensus(self, methods_results: Dict[str, List[str]]) -> None:
        """
        Analiza el consenso entre métodos de selección.

        Args:
            methods_results: Resultados de cada método
        """
        logger.info("\n" + "="*60)
        logger.info("ANÁLISIS DE CONSENSO ENTRE MÉTODOS")
        logger.info("="*60)

        # Crear DataFrame para análisis
        all_features = set()
        for features in methods_results.values():
            all_features.update(features)

        consensus_data = []
        for feature in all_features:
            methods_count = sum(1 for features in methods_results.values()
                              if feature in features)
            methods_list = [method for method, features in methods_results.items()
                           if feature in features]

            consensus_data.append({
                'feature': feature,
                'methods_count': methods_count,
                'methods': ', '.join(methods_list)
            })

        consensus_df = pd.DataFrame(consensus_data).sort_values('methods_count', ascending=False)

        # Mostrar características con mayor consenso
        logger.info("\nTop 15 características con mayor consenso:")
        for _, row in consensus_df.head(15).iterrows():
            logger.info(f"{row['feature']}: {row['methods_count']} métodos ({row['methods']})")

        # Guardar análisis completo
        reports_dir = self.config['paths']['reports_dir']
        consensus_df.to_csv(reports_dir / 'feature_selection_consensus.csv', index=False)

        # Estadísticas por método
        logger.info(f"\nEstadísticas por método:")
        for method, features in methods_results.items():
            logger.info(f"{method}: {len(features)} características")

        logger.info("="*60)

    def load_selected_features(self, method_name: str) -> List[str]:
        """
        Carga características seleccionadas previamente guardadas.

        Args:
            method_name: Nombre del método

        Returns:
            Lista de características seleccionadas
        """
        import pickle

        models_dir = self.config['paths']['models_dir']
        filepath = models_dir / f'selected_features_{method_name}.pkl'

        try:
            with open(filepath, 'rb') as f:
                features = pickle.load(f)
            logger.info(f"Características {method_name} cargadas: {len(features)} variables")
            return features
        except FileNotFoundError:
            logger.warning(f"No se encontraron características guardadas para {method_name}")
            return []
