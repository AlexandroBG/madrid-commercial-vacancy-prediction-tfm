# -*- coding: utf-8 -*-
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
from collections import Counter
from itertools import chain
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Clase para manejar la selección de características."""

    def __init__(self):
        self.config = get_config()
        self.model_config = self.config['model_config']

    def run_boruta_selection(self, X_train_scaled: pd.DataFrame, y_train: pd.Series) -> List[str]:
        """
        Ejecuta selección de características usando Boruta.
        """
        logger.info("Iniciando selección Boruta...")

        # Reducir muestra a 100,000 filas (solo para selección)
        X_boruta_sample = X_train_scaled.sample(n=100000, random_state=12345)
        y_boruta_sample = y_train.loc[X_boruta_sample.index]

        # RandomForest base para Boruta
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=12345)

        # Boruta
        boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=1, random_state=12345)

        # Ajustar Boruta a la muestra pequeña
        boruta_selector.fit(X_boruta_sample.values, y_boruta_sample.values)

        # Variables seleccionadas
        selected_features_boruta = X_train_scaled.columns[boruta_selector.support_].tolist()

        logger.info(f"Variables seleccionadas por Boruta: {selected_features_boruta}")

        # Guardar resultado
        self._save_selected_features(selected_features_boruta, 'boruta')

        return selected_features_boruta

    def run_rfecv_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
        """
        Ejecuta selección usando Recursive Feature Elimination with CV.
        """
        logger.info("Iniciando selección RFECV...")

        # 1. Usar X_train y y_train definidos previamente en tu pipeline
        X_sample = X_train.sample(n=50000, random_state=42)
        y_sample = y_train.loc[X_sample.index]

        # 2. Definir modelo
        model = RandomForestClassifier(n_estimators=100, random_state=123)

        # 3. Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

        # 4. RFECV
        selector = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy', n_jobs=-1)
        selector.fit(X_sample, y_sample)

        # 5. Variables seleccionadas
        selected_features = X_sample.columns[selector.support_].tolist()
        logger.info("✅ Variables seleccionadas por RFECV:")
        logger.info(selected_features)

        # Guardar resultado
        self._save_selected_features(selected_features, 'rfecv')

        return selected_features

    def run_stepwise_selection(self, X_train_scaled: pd.DataFrame, y_train: pd.Series) -> List[str]:
        """
        Ejecuta selección stepwise usando regresión logística.
        """
        logger.info("Iniciando selección Stepwise...")

        # Crear muestra con semilla fija
        X_step_sample, _, y_step_sample, _ = train_test_split(
            X_train_scaled, y_train, train_size=500000, stratify=y_train, random_state=12345
        )

        # Filtrar variables con varianza muy baja
        selector = VarianceThreshold(threshold=1e-5)
        X_step_sample_filtered = pd.DataFrame(selector.fit_transform(X_step_sample),
                                              columns=X_step_sample.columns[selector.get_support()])
        X_step_sample_filtered.index = X_step_sample.index

        # Eliminar variables altamente correlacionadas
        corr_matrix = X_step_sample_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_step_sample_filtered.drop(columns=to_drop, inplace=True)

        # Ejecutar stepwise
        vars_stepwise = self._stepwise_selection_verbose(X_step_sample_filtered, y_step_sample)

        logger.info("\nVariables seleccionadas por Stepwise:")
        logger.info(vars_stepwise)

        # Guardar resultado
        self._save_selected_features(vars_stepwise, 'stepwise')

        return vars_stepwise

    def run_sbf_selection(self, X_train_scaled: pd.DataFrame, y_train: pd.Series) -> List[str]:
        """
        Ejecuta Sequential Backward Feature Selection.
        """
        logger.info("Iniciando selección SBF...")

        # 1) Reducir filas para SBF (ajustada a 50,000 para mejor rendimiento)
        X_sbf, _, y_sbf, _ = train_test_split(
            X_train_scaled,
            y_train,
            train_size=50000,    # Más rápido y suficiente para SBF
            stratify=y_train,
            random_state=12345
        )

        # 2) Definir modelo base
        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=12345)

        # 3) Crear selector secuencial backward (SBF) con CV reducido
        sbf = SequentialFeatureSelector(
            estimator=model,
            direction='backward',
            n_features_to_select='auto',     # o por ejemplo: 30
            scoring='roc_auc',
            cv=3,                            # Menor carga computacional
            n_jobs=-1
        )

        # 4) Medir tiempo de ejecución
        import time
        start = time.time()
        sbf.fit(X_sbf, y_sbf)
        end = time.time()

        # 5) Mostrar variables seleccionadas y tiempo
        vars_sbf = X_sbf.columns[sbf.get_support()].tolist()

        logger.info("Variables seleccionadas por SBF:")
        for v in vars_sbf:
            logger.info(f"- {v}")

        logger.info(f"⏱️ Tiempo de ejecución: {round(end - start, 2)} segundos.")

        # Guardar resultado
        self._save_selected_features(vars_sbf, 'sbf')

        return vars_sbf

    def run_shap_selection(self, X_train_scaled: pd.DataFrame, y_train: pd.Series,
                          X_test_scaled: pd.DataFrame, top_n: int = 30) -> List[str]:
        """
        Ejecuta selección de características usando SHAP values.
        """
        logger.info("Iniciando selección SHAP...")
        logger.info("🔎 Generando gráficos de importancia SHAP...")

        # BLOQUE 2: Entrenar modelo XGBoost
        model_shap = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=12345,
            n_jobs=-1
        )
        model_shap.fit(X_train_scaled, y_train)

        # BLOQUE 3: Crear el explainer SHAP
        explainer = shap.Explainer(model_shap, X_train_scaled)

        # BLOQUE 4: Calcular valores SHAP para el conjunto de test
        shap_values = explainer(X_test_scaled)

        # BLOQUE 5: Gráficos de importancia de variables
        try:
            import matplotlib.pyplot as plt
            # Summary plot (tipo swarm)
            shap.summary_plot(shap_values, X_test_scaled, max_display=30)
            # Gráfico de barras de importancia media absoluta
            shap.plots.bar(shap_values, max_display=30)
        except Exception as e:
            logger.warning(f"No se pudieron generar gráficos SHAP: {e}")

        # BLOQUE 6: Exportar importancia promedio de cada variable
        shap_importances = pd.DataFrame({
            'variable': X_test_scaled.columns,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by='mean_abs_shap', ascending=False)

        # Mostrar top 30
        logger.info(f"\n🔝 Top {top_n} variables más importantes según SHAP:")
        logger.info(shap_importances.head(top_n))

        # BLOQUE 7: Guardar nombres de top variables
        top_vars_shap = shap_importances['variable'].head(top_n).tolist()
        logger.info(f"\n📦 Lista de top {top_n} variables seleccionadas por SHAP:")
        logger.info(top_vars_shap)

        # BLOQUE 8: Guardar importancias
        self._save_shap_importances(shap_importances)

        # Guardar resultado
        self._save_selected_features(top_vars_shap, 'shap')

        return top_vars_shap

    def compare_selection_methods(self, selected_features_boruta: List[str],
                                 selected_features_rfe: List[str],
                                 vars_stepwise: List[str],
                                 vars_sbf: List[str],
                                 top_vars_shap: List[str] = None) -> Dict[str, List[str]]:
        """
        Compara múltiples métodos de selección de características.
        """
        logger.info("Comparando métodos de selección de características...")

        # Diccionario con todos los métodos
        all_vars = {
            "Boruta": set(selected_features_boruta),
            "RFE": set(selected_features_rfe),
            "Stepwise": set(vars_stepwise),
            "SBF": set(vars_sbf),
        }

        if top_vars_shap:
            all_vars["SHAP"] = set(top_vars_shap)

        # Ver en cuántos métodos aparece cada variable
        all_selected = list(chain.from_iterable(all_vars.values()))
        counter = Counter(all_selected)
        common_vars = [var for var, count in counter.items() if count > 1]

        logger.info(f"Variables comunes en 2+ métodos: {common_vars}")

        # Guardar variables de cada método en archivos
        self._save_method_variables(selected_features_boruta, 'boruta')
        self._save_method_variables(selected_features_rfe, 'rfe')
        self._save_method_variables(vars_stepwise, 'stepwise')
        self._save_method_variables(vars_sbf, 'sbf')
        if top_vars_shap:
            self._save_method_variables(top_vars_shap, 'shap')

        # Análisis de consenso
        self._analyze_consensus(all_vars)

        return {
            "Boruta": selected_features_boruta,
            "RFE": selected_features_rfe,
            "Stepwise": vars_stepwise,
            "SBF": vars_sbf,
            "SHAP": top_vars_shap if top_vars_shap else [],
            "Common": common_vars
        }

    def evaluate_variable_sets(self, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series,
                              variable_sets: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Evalúa cada conjunto de variables con XGBoost.
        """
        logger.info("Evaluando conjuntos de variables...")

        from sklearn.metrics import accuracy_score, roc_auc_score

        resultados = []

        for nombre, variables in variable_sets.items():
            if not variables:  # Skip empty lists
                continue

            logger.info(f"Evaluando método: {nombre}")

            # Filtrar columnas seleccionadas
            X_train_sel = X_train_scaled[variables]
            X_test_sel = X_test_scaled[variables]

            # Modelo XGBoost
            modelo = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=12345,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            modelo.fit(X_train_sel, y_train)

            # Predicciones
            y_pred = modelo.predict(X_test_sel)
            y_proba = modelo.predict_proba(X_test_sel)[:, 1]

            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            num_vars = len(variables)

            logger.info(f"▶ {nombre}")
            logger.info(f" - Accuracy: {accuracy:.4f}")
            logger.info(f" - AUC:      {auc:.4f}")
            logger.info(f" - Nº Variables: {num_vars}")
            logger.info("---------------------------")

            resultados.append({
                "nombre": nombre,
                "accuracy": accuracy,
                "auc": auc,
                "num_variables": num_vars
            })

        # Crear DataFrame con resultados
        df_resultados = pd.DataFrame(resultados).sort_values(by='auc', ascending=False)

        logger.info("\n📊 Comparación de métodos por AUC:")
        logger.info(df_resultados)

        return df_resultados

    def _stepwise_selection_verbose(self, X: pd.DataFrame, y: pd.Series,
                                   initial_list: List[str] = [],
                                   threshold_in: float = 0.001,
                                   threshold_out: float = 0.01,
                                   verbose: bool = True) -> List[str]:
        """
        Implementa selección stepwise con regresión logística.
        """
        included = list(initial_list)
        while True:
            changed = False
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded, dtype=float)

            for new_column in excluded:
                try:
                    model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
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

            if included:
                try:
                    model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
                    pvalues = model.pvalues.iloc[1:]
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
        """
        import pickle
        models_dir = self.config['paths']['models_dir']
        filepath = models_dir / f'selected_features_{method_name}.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump(features, f)

        logger.info(f"Características {method_name} guardadas en: {filepath}")

    def _save_method_variables(self, variables: List[str], method_name: str) -> None:
        """
        Guarda variables de un método en archivo de texto.
        """
        filename = f"variables_{method_name}.txt"
        with open(filename, "w") as f:
            for var in variables:
                f.write(var + "\n")
        logger.info(f"Variables {method_name} guardadas en: {filename}")

    def _save_shap_importances(self, importances: pd.DataFrame) -> None:
        """
        Guarda importancias SHAP.
        """
        reports_dir = self.config['paths']['reports_dir']
        filepath = reports_dir / 'shap_importances.csv'
        importances.to_csv(filepath, index=False)

        # También guardar top 30 en archivo de texto
        top_30_vars_shap = importances['variable'].head(30).tolist()
        with open("variables_shap.txt", "w") as f:
            for var in top_30_vars_shap:
                f.write(var + "\n")

        logger.info(f"Importancias SHAP guardadas en: {filepath}")

    def _analyze_consensus(self, methods_results: Dict[str, set]) -> None:
        """
        Analiza el consenso entre métodos de selección.
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
