"""
Módulo para cargar datos de diferentes fuentes.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from src.utils.config import get_config, get_data_file_path

logger = logging.getLogger(__name__)

class DataLoader:
    """Clase para manejar la carga de datos del proyecto."""

    def __init__(self):
        self.config = get_config()

    def load_actividades_economicas(self, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Carga el dataset limpio de actividades económicas de Madrid.

        Args:
            encoding: Codificación del archivo

        Returns:
            DataFrame con los datos de actividades económicas ya limpios
        """
        file_path = get_data_file_path('actividades')

        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            logger.info(f"Dataset limpio de actividades cargado: {df.shape}")
            logger.info(f"Columnas disponibles: {list(df.columns)}")
            logger.info(f"Variable objetivo 'actividad' presente: {'actividad' in df.columns}")
            return df
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar actividades económicas: {e}")
            raise e

    def load_renta_poblacion(self) -> pd.DataFrame:
        """
        Carga el dataset de renta y población.

        Returns:
            DataFrame con datos de renta per cápita y población
        """
        file_path = get_data_file_path('renta_poblacion')

        try:
            df = pd.read_excel(file_path)
            logger.info(f"Dataset de renta/población cargado: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar renta/población: {e}")
            # Intentar con ruta alternativa para Windows
            alt_path = str(file_path).replace("/Users/alexandrobazan", "C:\\Users\\alex_")
            try:
                df = pd.read_excel(alt_path)
                logger.info(f"Dataset cargado desde ruta alternativa: {df.shape}")
                return df
            except:
                raise e

    def load_clean_data(self) -> pd.DataFrame:
        """
        Carga el dataset limpio que ya incluye datos de renta y población.

        Returns:
            DataFrame con datos limpios y completos
        """
        logger.info("Cargando dataset limpio...")

        df_clean = self.load_actividades_economicas()

        # Verificar que tenga las columnas esperadas
        required_cols = ['actividad', 'Renta_Media', 'Total_Poblacion', 'Fecha_Reporte']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]

        if missing_cols:
            logger.error(f"Faltan columnas requeridas: {missing_cols}")
            raise ValueError(f"Dataset incompleto. Faltan: {missing_cols}")

        logger.info("Dataset limpio cargado exitosamente")
        logger.info(f"Distribución de actividad: {df_clean['actividad'].value_counts().to_dict()}")

        return df_clean

    def validate_data_files(self) -> bool:
        """
        Valida que todos los archivos de datos existan.

        Returns:
            True si todos los archivos existen, False en caso contrario
        """
        missing_files = []

        for file_key, file_path in self.config['data_files'].items():
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"Archivos faltantes: {missing_files}")
            return False

        logger.info("Todos los archivos de datos están disponibles")
        return True

    def get_data_info(self, df: pd.DataFrame, name: str) -> dict:
        """
        Obtiene información básica de un DataFrame.

        Args:
            df: DataFrame a analizar
            name: Nombre del dataset

        Returns:
            Diccionario con información del dataset
        """
        info = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicated_rows': df.duplicated().sum()
        }

        logger.info(f"Info de {name}: Shape {info['shape']}, "
                   f"Nulos: {sum(info['null_counts'].values())}, "
                   f"Duplicados: {info['duplicated_rows']}")

        return info

def load_processed_data(file_name: str) -> Optional[pd.DataFrame]:
    """
    Carga datos procesados desde la carpeta processed.

    Args:
        file_name: Nombre del archivo (con extensión)

    Returns:
        DataFrame si el archivo existe, None en caso contrario
    """
    config = get_config()
    file_path = config['paths']['processed_data_dir'] / file_name

    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith('.pkl') or file_name.endswith('.pickle'):
            df = pd.read_pickle(file_path)
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"Formato de archivo no soportado: {file_name}")
            return None

        logger.info(f"Datos procesados cargados: {file_name} - Shape: {df.shape}")
        return df

    except FileNotFoundError:
        logger.warning(f"Archivo procesado no encontrado: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error al cargar datos procesados {file_name}: {e}")
        return None

def save_processed_data(df: pd.DataFrame, file_name: str, format: str = 'pickle') -> bool:
    """
    Guarda datos procesados en la carpeta processed.

    Args:
        df: DataFrame a guardar
        file_name: Nombre del archivo (sin extensión)
        format: Formato de salida ('pickle', 'csv', 'parquet')

    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    config = get_config()
    processed_dir = config['paths']['processed_data_dir']

    try:
        if format == 'pickle':
            file_path = processed_dir / f"{file_name}.pkl"
            df.to_pickle(file_path)
        elif format == 'csv':
            file_path = processed_dir / f"{file_name}.csv"
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            file_path = processed_dir / f"{file_name}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            logger.error(f"Formato no soportado: {format}")
            return False

        logger.info(f"Datos guardados: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error al guardar datos procesados: {e}")
        return False
