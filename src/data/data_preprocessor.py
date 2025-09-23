# -*- coding: utf-8 -*-
"""
Módulo simplificado para preprocesamiento de datos siguiendo la lógica especificada.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clase para preprocesamiento simplificado de datos."""

    def __init__(self):
        self.config = get_config()
        self.scaler = None

    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline completo de preprocesamiento siguiendo la lógica especificada.

        Args:
            df: DataFrame limpio cargado

        Returns:
            Diccionario con datasets preparados
        """
        logger.info("="*60)
        logger.info("INICIANDO PREPROCESAMIENTO SIMPLIFICADO")
        logger.info("="*60)

        # Asegúrate de tener una copia original si necesitas rescatar algo como Fecha_Reporte
        df_original = df.copy()

        # BLOQUE 2: Definir columnas categóricas preferidas (ELIMINAMOS desc_distrito_local)
        preferidas = [
            col for col in ['desc_barrio_local', 'desc_tipo_acceso_local', 'desc_seccion']
            if col in df.columns
        ]

        logger.info(f"✅ Variables categóricas a procesar (SIN distrito): {preferidas}")

        # BLOQUE 3: Separar variable objetivo
        y = df['actividad']
        logger.info(f"Variable objetivo extraída. Distribución: {y.value_counts().to_dict()}")

        # BLOQUE 4: Filtrar variables categóricas preferidas con baja cardinalidad
        cat_vars = df[preferidas].select_dtypes(include='object').columns
        cardinalidades = df[cat_vars].nunique()
        cat_vars_baja_card = cardinalidades[cardinalidades <= 30].index.tolist()
        logger.info(f"Variables que se transformarán en dummies: {cat_vars_baja_card}")

        # BLOQUE 5: Crear dummies de variables seleccionadas
        df = pd.get_dummies(df, columns=cat_vars_baja_card, drop_first=True)
        logger.info(f"Variables dummy creadas. Nuevas dimensiones: {df.shape}")

        # BLOQUE 6: Eliminar variables no deseadas después de crear dummies
        variables_a_eliminar = [
            'actividad', 'rotulo', 'desc_vial_acceso', 'Fecha_Reporte', 'Mes', 'Año',
            'num_acceso', 'cal_acceso', 'latitud_local', 'longitud_local',
            'id_local', 'id_distrito_local', 'cod_barrio_local',
            'id_tipo_acceso_local', 'id_seccion', 'id_epigrafe', 'desc_epigrafe',
            'id_division', 'desc_division', 'desc_situacion_local',
            'clase_vial_acceso', 'nom_acceso'
        ]

        # IMPORTANTE: También eliminar TODAS las variables de distrito que se hayan creado
        dummies_a_excluir = [col for col in df.columns if col.startswith((
            'id_', 'cod_barrio_local_', 'latitud_local', 'longitud_local', 'desc_epigrafe_',
            'desc_division_', 'id_seccion_', 'desc_distrito_local_'  # ← CLAVE: Eliminar variables distrito
        ))]

        variables_a_eliminar += dummies_a_excluir
        df = df.drop(columns=[col for col in variables_a_eliminar if col in df.columns], errors='ignore')

        # VERIFICACIÓN: Asegurar que no quedan variables de distrito
        distrito_cols_restantes = [col for col in df.columns if 'distrito' in col.lower()]
        if distrito_cols_restantes:
            logger.info(f"⚠️ ELIMINANDO variables de distrito restantes: {distrito_cols_restantes}")
            df = df.drop(columns=distrito_cols_restantes, errors='ignore')

        # BLOQUE 7: Eliminar columnas no numéricas que aún queden (seguridad)
        non_numeric_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if non_numeric_cols:
            logger.info(f"⚠️ Eliminando columnas no numéricas restantes: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)

        # BLOQUE 8: Añadir Fecha_Reporte para particionar
        df['Fecha_Reporte'] = df_original['Fecha_Reporte']

        # BLOQUE 9: Separar train y test
        X_train = df[df['Fecha_Reporte'] < 202401].drop(columns=['Fecha_Reporte'])
        X_test = df[df['Fecha_Reporte'] >= 202401].drop(columns=['Fecha_Reporte'])
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        logger.info(f"División temporal completada:")
        logger.info(f"Train set: {X_train.shape} (antes de {202401})")
        logger.info(f"Test set: {X_test.shape} (desde {202401})")
        logger.info(f"Distribución train - Activos: {y_train.mean():.2%}")
        logger.info(f"Distribución test - Activos: {y_test.mean():.2%}")

        # BLOQUE 10: Alinear columnas entre train y test
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # VERIFICACIÓN FINAL: Confirmar que Renta_Media está presente
        if 'Renta_Media' in X_train.columns:
            logger.info("✅ Renta_Media confirmada en el dataset")
        else:
            logger.warning("⚠️ ADVERTENCIA: Renta_Media no encontrada en el dataset")

        # BLOQUE 11: Estandarización
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        self.scaler = scaler

        # BLOQUE 12: Mostrar variables finales
        logger.info(f"\n📋 Variables finales en X_train_scaled ({len(X_train_scaled.columns)} variables):")
        for i, col in enumerate(X_train_scaled.columns, start=1):
            logger.info(f"{i:3}: {col}")

        # VERIFICACIÓN DE MULTICOLINEALIDAD
        logger.info("\n🔍 Verificando correlaciones con Renta_Media:")
        if 'Renta_Media' in X_train_scaled.columns:
            correlaciones_altas = []
            for col in X_train_scaled.columns:
                if col != 'Renta_Media':
                    corr = abs(X_train_scaled['Renta_Media'].corr(X_train_scaled[col]))
                    if corr > 0.7:  # Correlación alta
                        correlaciones_altas.append((col, corr))

            if correlaciones_altas:
                logger.info("⚠️ Variables con correlación alta con Renta_Media (>0.7):")
                for col, corr in sorted(correlaciones_altas, key=lambda x: x[1], reverse=True):
                    logger.info(f"   {col}: {corr:.4f}")
            else:
                logger.info("✅ No se detectaron correlaciones altas con Renta_Media")
        else:
            logger.info("⚠️ No se pudo verificar correlaciones - Renta_Media no encontrada")

        # Guardar datasets procesados
        self.save_processed_datasets({
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        })

        logger.info("="*60)
        logger.info("PREPROCESAMIENTO COMPLETADO")
        logger.info(f"Variables finales: {X_train_scaled.shape[1]}")
        logger.info(f"Train samples: {len(X_train_scaled)}")
        logger.info(f"Test samples: {len(X_test_scaled)}")
        logger.info("="*60)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X_train_scaled.columns)
        }

    def save_processed_datasets(self, datasets: Dict[str, Any]) -> None:
        """
        Guarda datasets procesados.

        Args:
            datasets: Diccionario con datasets y objetos
        """
        logger.info("Guardando datasets procesados...")

        models_dir = self.config['paths']['models_dir']

        # Guardar cada componente
        for name, data in datasets.items():
            filepath = models_dir / f'{name}.joblib'
            joblib.dump(data, filepath)

        logger.info(f"Datasets guardados en: {models_dir}")

    def load_processed_datasets(self) -> Dict[str, Any]:
        """
        Carga datasets procesados previamente guardados.

        Returns:
            Diccionario con datasets cargados
        """
        try:
            models_dir = self.config['paths']['models_dir']

            datasets = {}
            dataset_files = [
                'X_train.joblib', 'X_test.joblib',
                'X_train_scaled.joblib', 'X_test_scaled.joblib',
                'y_train.joblib', 'y_test.joblib',
                'scaler.joblib'
            ]

            for filename in dataset_files:
                filepath = models_dir / filename
                if filepath.exists():
                    name = filename.replace('.joblib', '')
                    datasets[name] = joblib.load(filepath)

            if len(datasets) == len(dataset_files):
                logger.info("Datasets procesados cargados exitosamente")
                return datasets
            else:
                logger.warning("No se encontraron todos los datasets procesados")
                return None

        except Exception as e:
            logger.error(f"Error cargando datasets procesados: {e}")
            return None

    def validate_processed_data(self, datasets: Dict[str, Any]) -> bool:
        """
        Valida que los datasets procesados sean correctos.

        Args:
            datasets: Diccionario con datasets

        Returns:
            True si la validación es exitosa
        """
        required_keys = ['X_train_scaled', 'X_test_scaled', 'y_train', 'y_test']

        # Verificar que existan las claves requeridas
        if not all(key in datasets for key in required_keys):
            logger.error("Faltan datasets requeridos")
            return False

        X_train = datasets['X_train_scaled']
        X_test = datasets['X_test_scaled']
        y_train = datasets['y_train']
        y_test = datasets['y_test']

        # Verificar shapes
        if len(X_train) != len(y_train):
            logger.error("Shape mismatch entre X_train y y_train")
            return False

        if len(X_test) != len(y_test):
            logger.error("Shape mismatch entre X_test y y_test")
            return False

        if X_train.shape[1] != X_test.shape[1]:
            logger.error("Número de características diferentes entre train y test")
            return False

        # Verificar que no hay NaNs
        if X_train.isna().any().any() or X_test.isna().any().any():
            logger.error("Valores NaN encontrados en datasets")
            return False

        # Verificar que y es binario
        unique_train = set(y_train.unique())
        unique_test = set(y_test.unique())

        if not unique_train.issubset({0, 1}) or not unique_test.issubset({0, 1}):
            logger.error("Variable objetivo no es binaria")
            return False

        logger.info("Validación de datasets exitosa")
        return True
