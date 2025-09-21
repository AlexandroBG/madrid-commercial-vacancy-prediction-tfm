"""
Módulo para preprocesamiento de datos antes del modelado.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from unidecode import unidecode
import joblib
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clase para preprocesamiento completo de datos."""

    def __init__(self):
        self.config = get_config()
        self.categorical_vars = self.config['categorical_variables']
        self.scaler = None

    def merge_with_renta_data(self, df_actividades: pd.DataFrame,
                             df_renta: pd.DataFrame) -> pd.DataFrame:
        """
        Fusiona datos de actividades con datos de renta y población.

        Args:
            df_actividades: DataFrame de actividades económicas
            df_renta: DataFrame con renta y población

        Returns:
            DataFrame fusionado
        """
        logger.info("Fusionando datos de actividades con renta y población...")

        # Normalizar nombres de barrios
        articulos = {'el', 'la', 'los', 'las'}

        def normalizar_avanzado(texto):
            texto = unidecode(str(texto)).lower().strip()
            palabras = texto.split()
            if palabras and palabras[0] in articulos:
                palabras = palabras[1:]
            return ''.join(palabras)

        # Crear mapeo para barrios
        mapa_barrios_df = {
            normalizar_avanzado(barrio): barrio
            for barrio in df_actividades['desc_barrio_local'].unique()
            if pd.notna(barrio)
        }

        df_renta_copy = df_renta.copy()
        df_renta_copy['Barrio_norm'] = df_renta_copy['Barrio'].apply(
            lambda x: mapa_barrios_df.get(normalizar_avanzado(x), x)
        )

        # Normalizar distritos
        df_actividades_copy = df_actividades.copy()
        df_actividades_copy['distrito_norm'] = df_actividades_copy['desc_distrito_local'].str.strip().str.lower()
        df_renta_copy['distrito_norm'] = df_renta_copy['Distrito'].str.strip().str.lower()

        mapa_distritos = df_actividades_copy.drop_duplicates(subset='distrito_norm').set_index('distrito_norm')['desc_distrito_local'].to_dict()
        df_renta_copy['Distrito_norm'] = df_renta_copy['distrito_norm'].map(mapa_distritos)

        # Preparar columnas para merge
        df_renta_copy['Año'] = df_renta_copy['Año_Poblacion'].astype(str)
        df_actividades_copy['Año_str'] = df_actividades_copy['Año'].astype(str)

        # Realizar merge
        df_merged = df_actividades_copy.merge(
            df_renta_copy[['Barrio_norm', 'Distrito_norm', 'Año', 'Total_Poblacion', 'Renta_Media']],
            how='left',
            left_on=['desc_barrio_local', 'desc_distrito_local', 'Año_str'],
            right_on=['Barrio_norm', 'Distrito_norm', 'Año']
        )

        # Limpiar columnas temporales
        columns_to_drop = ['Barrio_norm', 'Distrito_norm', 'Año_str',
                          'distrito_norm', 'Año']
        df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

        # Ajustar Total_Poblacion (convertir de miles a unidades)
        df_merged['Total_Poblacion'] = (df_merged['Total_Poblacion'] * 1000).astype('Int64')

        logger.info(f"Merge completado. Shape: {df_merged.shape}")
        logger.info(f"Valores nulos en Total_Poblacion: {df_merged['Total_Poblacion'].isna().sum()}")
        logger.info(f"Valores nulos en Renta_Media: {df_merged['Renta_Media'].isna().sum()}")

        return df_merged

    def create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variables dummy para variables categóricas seleccionadas.

        Args:
            df: DataFrame original

        Returns:
            DataFrame con variables dummy
        """
        logger.info("Creando variables dummy...")

        df_processed = df.copy()

        # Filtrar variables categóricas que existen en el DataFrame
        existing_cat_vars = [var for var in self.categorical_vars if var in df_processed.columns]

        # Filtrar por cardinalidad (<=30 categorías)
        cat_vars_to_process = []
        for var in existing_cat_vars:
            if df_processed[var].nunique() <= 30:
                cat_vars_to_process.append(var)

        logger.info(f"Variables categóricas a procesar: {cat_vars_to_process}")

        # Crear variables dummy
        if cat_vars_to_process:
            df_processed = pd.get_dummies(
                df_processed,
                columns=cat_vars_to_process,
                drop_first=True,
                prefix_sep='_'
            )

        logger.info(f"Variables dummy creadas. Nuevas dimensiones: {df_processed.shape}")
        return df_processed

    def remove_unnecessary_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina variables no necesarias para el modelado.

        Args:
            df: DataFrame con todas las variables

        Returns:
            DataFrame con variables relevantes para modelado
        """
        logger.info("Eliminando variables innecesarias...")

        # Variables a eliminar
        variables_to_remove = [
            'actividad', 'rotulo', 'desc_vial_acceso', 'Fecha_Reporte',
            'Mes', 'Año', 'num_acceso', 'cal_acceso', 'latitud_local',
            'longitud_local', 'id_local', 'id_distrito_local', 'cod_barrio_local',
            'id_tipo_acceso_local', 'id_seccion', 'id_epigrafe', 'desc_epigrafe',
            'id_division', 'desc_division', 'desc_situacion_local',
            'desc_situacion_recodificada', 'desc_situacion'
        ]

        # Agregar dummies de variables eliminadas
        dummy_prefixes_to_remove = [
            'id_', 'desc_epigrafe_', 'desc_division_', 'id_seccion_',
            'desc_distrito_local_', 'cod_barrio_local_'
        ]

        # Encontrar columnas que empiecen con los prefijos
        cols_to_remove = []
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in dummy_prefixes_to_remove):
                cols_to_remove.append(col)

        variables_to_remove.extend(cols_to_remove)

        # Eliminar variables que existen
        existing_vars_to_remove = [var for var in variables_to_remove if var in df.columns]
        df_clean = df.drop(columns=existing_vars_to_remove, errors='ignore')

        # Eliminar columnas no numéricas restantes
        non_numeric_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
        if non_numeric_cols:
            logger.info(f"Eliminando columnas no numéricas: {non_numeric_cols}")
            df_clean = df_clean.drop(columns=non_numeric_cols)

        logger.info(f"Variables eliminadas: {len(existing_vars_to_remove)}")
        logger.info(f"Dimensiones finales: {df_clean.shape}")

        return df_clean

    def split_train_test_temporal(self, df: pd.DataFrame, y: pd.Series,
                                 split_date: int = 202401) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide datos en train/test usando criterio temporal.

        Args:
            df: DataFrame con variable Fecha_Reporte
            y: Variable objetivo
            split_date: Fecha de corte en formato YYYYMM

        Returns:
            Tupla con (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Dividiendo datos temporalmente. Corte: {split_date}")

        # Asegurar que Fecha_Reporte esté disponible
        if 'Fecha_Reporte' not in df.columns:
            raise ValueError("Columna Fecha_Reporte no encontrada")

        # Crear máscaras temporales
        train_mask = df['Fecha_Reporte'] < split_date
        test_mask = df['Fecha_Reporte'] >= split_date

        # Dividir datasets
        X_train = df[train_mask].drop(columns=['Fecha_Reporte'])
        X_test = df[test_mask].drop(columns=['Fecha_Reporte'])
        y_train = y[train_mask]
        y_test = y[test_mask]

        # Alinear columnas (en caso de diferencias)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Distribución train - Activos: {y_train.mean():.2%}")
        logger.info(f"Distribución test - Activos: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Escala las variables usando StandardScaler.

        Args:
            X_train: Dataset de entrenamiento
            X_test: Dataset de prueba

        Returns:
            Tupla con (X_train_scaled, X_test_scaled, scaler)
        """
        logger.info("Escalando variables...")

        # Crear y ajustar scaler solo en train
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transformar test con el mismo scaler
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        self.scaler = scaler

        logger.info("Escalado completado")
        return X_train_scaled, X_test_scaled, scaler

    def prepare_for_modeling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline completo de preparación para modelado.

        Args:
            df: DataFrame limpio

        Returns:
            Diccionario con datasets preparados y objetos auxiliares
        """
        logger.info("="*60)
        logger.info("INICIANDO PREPARACIÓN PARA MODELADO")
        logger.info("="*60)

        # 1. Extraer variable objetivo
        if 'actividad' not in df.columns:
            raise ValueError("Variable objetivo 'actividad' no encontrada")

        y = df['actividad']
        logger.info(f"Variable objetivo extraída. Distribución: {y.value_counts().to_dict()}")

        # 2. Crear variables dummy
        df_with_dummies = self.create_dummy_variables(df)

        # 3. Conservar Fecha_Reporte para división temporal
        fecha_reporte = df_with_dummies['Fecha_Reporte'].copy()

        # 4. Eliminar variables innecesarias
        df_clean = self.remove_unnecessary_variables(df_with_dummies)

        # 5. Reincorporar Fecha_Reporte
        df_clean['Fecha_Reporte'] = fecha_reporte

        # 6. División temporal train/test
        X_train, X_test, y_train, y_test = self.split_train_test_temporal(df_clean, y)

        # 7. Escalado de variables
        X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)

        # 8. Guardar datasets procesados
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
        logger.info("PREPARACIÓN COMPLETADA")
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
