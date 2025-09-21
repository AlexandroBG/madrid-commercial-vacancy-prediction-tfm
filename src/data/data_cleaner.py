"""
Módulo para limpieza y normalización de datos.
"""

import pandas as pd
import numpy as np
import unicodedata
import re
import logging
from typing import Dict, List, Optional, Tuple
from unidecode import unidecode
from pyproj import Proj, Transformer
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clase para manejar la limpieza de datos."""

    def __init__(self):
        self.config = get_config()
        self.utm_zone = 30
        self.proj_utm = Proj(proj='utm', zone=self.utm_zone, hemisphere='north', datum='WGS84')
        self.proj_latlon = Proj(proj='latlong', datum='WGS84')
        self.transformer = Transformer.from_proj(self.proj_utm, self.proj_latlon)

        # Diccionario de correcciones comunes
        self.text_corrections = {
            'r0tulo': 'rotulo',
            'ratulo': 'rotulo',
            's/r': 'sin rotulo',
            'sr': 'sin rotulo',
            'sin rotulo': 'sin actividad',
            'hosteleraa': 'hosteleria',
            'educacian': 'educacion',
            'cientaficas': 'cientificas',
            'actividades administrativas y servicios auxliares':
                'actividades administrativas y servicios auxiliares',
            'construccian': 'construccion',
            'caaada': 'cañada',
            'caaaada': 'cañada'
        }

    def clean_text(self, texto: str) -> str:
        """
        Limpia y normaliza texto: minúsculas, sin espacios, sin acentos.

        Args:
            texto: Texto a limpiar

        Returns:
            Texto limpio y normalizado
        """
        if not isinstance(texto, str) or pd.isna(texto):
            return texto

        texto = texto.strip().lower()
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', errors='ignore').decode('utf-8')
        return texto

    def correct_text_errors(self, texto: str) -> str:
        """
        Corrige errores comunes de codificación y ortografía.

        Args:
            texto: Texto a corregir

        Returns:
            Texto corregido
        """
        if pd.isna(texto):
            return texto

        texto = str(texto).lower()

        # Reemplazos de caracteres comunes
        texto = texto.replace('0', 'o')
        texto = re.sub(r'[âãáàä]', 'a', texto)
        texto = re.sub(r'[éèë]', 'e', texto)
        texto = re.sub(r'[íìï]', 'i', texto)
        texto = re.sub(r'[óòö]', 'o', texto)
        texto = re.sub(r'[úùü]', 'u', texto)
        texto = re.sub(r'[ñ]', 'n', texto)

        # Aplicar correcciones específicas
        for error, correccion in self.text_corrections.items():
            texto = texto.replace(error, correccion)

        # Limpiar espacios
        texto = texto.strip()
        texto = re.sub(r'\s+', ' ', texto)

        return texto

    def clean_dataframe_text_columns(self, df: pd.DataFrame,
                                   text_columns: List[str]) -> pd.DataFrame:
        """
        Aplica limpieza de texto a múltiples columnas de un DataFrame.

        Args:
            df: DataFrame a procesar
            text_columns: Lista de nombres de columnas de texto

        Returns:
            DataFrame con columnas de texto limpiadas
        """
        df_clean = df.copy()

        for col in text_columns:
            if col in df_clean.columns:
                logger.info(f"Limpiando columna de texto: {col}")
                df_clean[col] = df_clean[col].apply(self.clean_text)
                df_clean[col] = df_clean[col].apply(self.correct_text_errors)

        return df_clean

    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina columnas innecesarias del DataFrame.

        Args:
            df: DataFrame original

        Returns:
            DataFrame sin columnas innecesarias
        """
        columns_to_drop = self.config['columns_to_drop']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

        if existing_cols_to_drop:
            logger.info(f"Eliminando {len(existing_cols_to_drop)} columnas innecesarias")
            df = df.drop(columns=existing_cols_to_drop)

        return df

    def merge_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fusiona columnas duplicadas (ej: id_local y ?id_local).

        Args:
            df: DataFrame con posibles columnas duplicadas

        Returns:
            DataFrame con columnas fusionadas
        """
        df_clean = df.copy()

        # Fusionar id_local con ?id_local si existe
        if '?id_local' in df_clean.columns:
            df_clean['id_local'] = df_clean['id_local'].combine_first(df_clean['?id_local'])
            df_clean.drop(columns='?id_local', inplace=True)
            logger.info("Columnas id_local fusionadas")

        return df_clean

    def fill_missing_values_by_id(self, df: pd.DataFrame,
                                id_column: str,
                                fill_columns: List[str]) -> pd.DataFrame:
        """
        Rellena valores faltantes usando registros con el mismo ID.

        Args:
            df: DataFrame a procesar
            id_column: Nombre de la columna ID
            fill_columns: Columnas a rellenar

        Returns:
            DataFrame con valores imputados
        """
        df_clean = df.copy()

        # Identificar filas con valores nulos en las columnas especificadas
        df_nulos = df_clean[df_clean[fill_columns].isnull().any(axis=1)]

        for idx, row in df_nulos.iterrows():
            id_value = row[id_column]

            # Buscar registros con el mismo ID que tengan información completa
            posibles = df_clean[
                (df_clean[id_column] == id_value) &
                df_clean[fill_columns].notnull().all(axis=1)
            ]

            if not posibles.empty:
                datos = posibles.iloc[0]
                for col in fill_columns:
                    df_clean.loc[idx, col] = datos[col]

        logger.info(f"Valores imputados por {id_column} en {len(fill_columns)} columnas")
        return df_clean

    def utm_to_latlon(self, row: pd.Series) -> pd.Series:
        """
        Convierte coordenadas UTM a latitud/longitud.

        Args:
            row: Fila con coordenadas UTM

        Returns:
            Serie con latitud y longitud
        """
        x = row['coordenada_x_local']
        y = row['coordenada_y_local']

        if pd.isna(x) or pd.isna(y) or x == 0.0 or y == 0.0:
            return pd.Series([None, None])

        try:
            lon, lat = self.transformer.transform(x, y)
            return pd.Series([lat, lon])
        except Exception:
            return pd.Series([None, None])

    def process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa coordenadas geográficas: convierte UTM a lat/lon.

        Args:
            df: DataFrame con coordenadas UTM

        Returns:
            DataFrame con coordenadas convertidas
        """
        df_clean = df.copy()

        # Convertir a numérico
        df_clean['coordenada_x_local'] = pd.to_numeric(
            df_clean['coordenada_x_local'], errors='coerce'
        )
        df_clean['coordenada_y_local'] = pd.to_numeric(
            df_clean['coordenada_y_local'], errors='coerce'
        )

        # Convertir UTM a latitud/longitud
        logger.info("Convirtiendo coordenadas UTM a latitud/longitud")
        df_clean[['latitud_local', 'longitud_local']] = df_clean.apply(
            self.utm_to_latlon, axis=1
        )

        return df_clean

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea la variable objetivo binaria 'actividad'.

        Args:
            df: DataFrame con variable de situación

        Returns:
            DataFrame con variable objetivo creada
        """
        df_clean = df.copy()

        # Recodificar situaciones específicas
        situacion_mapping = {
            'abierto': 'Abierto',
            'cerrado': 'Inactivo',
            'baja reunificacion': 'Inactivo',
            'baja reunificacia3n': 'baja reunificacion',
            'baja reunificaciaa3n': 'baja reunificacion',
            'baja': 'Inactivo',
            'uso vivienda': 'Inactivo',
            'en obras': 'Inactivo',
            'baja pc asociado': 'Inactivo'
        }

        df_clean['desc_situacion_recodificada'] = df_clean['desc_situacion_local'].replace(
            situacion_mapping
        )

        # Crear variable binaria
        df_clean['actividad'] = df_clean['desc_situacion_recodificada'].apply(
            lambda x: 1 if x == 'Abierto' else 0
        )

        logger.info("Variable objetivo 'actividad' creada")
        logger.info(f"Distribución: {df_clean['actividad'].value_counts().to_dict()}")

        return df_clean

    def process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa fechas y crea variables temporales.

        Args:
            df: DataFrame con fechas

        Returns:
            DataFrame con fechas procesadas
        """
        df_clean = df.copy()

        if 'Fecha_Reporte' in df_clean.columns:
            # Convertir a datetime si es necesario
            if df_clean['Fecha_Reporte'].dtype == 'object':
                df_clean['Fecha_Reporte'] = pd.to_datetime(df_clean['Fecha_Reporte'])

            # Convertir a formato YYYYMM
            df_clean['Fecha_Reporte'] = pd.to_datetime(df_clean['Fecha_Reporte'], errors='coerce').dt.strftime('%Y%m').astype(int)

            # Crear variables de año y mes
            df_clean['Año'] = df_clean['Fecha_Reporte'] // 100
            df_clean['Mes'] = df_clean['Fecha_Reporte'] % 100

            logger.info("Fechas procesadas y variables temporales creadas")

        return df_clean

    def normalize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza variables de identificación.

        Args:
            df: DataFrame con IDs

        Returns:
            DataFrame con IDs normalizados
        """
        df_clean = df.copy()

        id_columns = ['id_seccion', 'id_division', 'id_epigrafe']

        for col in id_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()

        logger.info("IDs normalizados")
        return df_clean

    def clean_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todo el pipeline de limpieza al dataset.

        Args:
            df: DataFrame original

        Returns:
            DataFrame completamente limpio
        """
        logger.info("Iniciando limpieza completa del dataset")

        # 1. Eliminar columnas innecesarias
        df_clean = self.remove_unnecessary_columns(df)

        # 2. Fusionar columnas duplicadas
        df_clean = self.merge_duplicate_columns(df_clean)

        # 3. Limpiar texto en columnas categóricas
        text_columns = [
            'desc_barrio_local', 'desc_distrito_local', 'desc_tipo_acceso_local',
            'desc_situacion_local', 'clase_vial_edificio', 'desc_vial_edificio',
            'clase_vial_acceso', 'desc_vial_acceso', 'rotulo', 'desc_seccion',
            'desc_division', 'desc_epigrafe'
        ]
        existing_text_columns = [col for col in text_columns if col in df_clean.columns]
        df_clean = self.clean_dataframe_text_columns(df_clean, existing_text_columns)

        # 4. Rellenar valores faltantes por ID
        if 'id_local' in df_clean.columns:
            fill_columns = [
                'id_distrito_local', 'desc_distrito_local',
                'id_barrio_local', 'desc_barrio_local'
            ]
            existing_fill_columns = [col for col in fill_columns if col in df_clean.columns]
            df_clean = self.fill_missing_values_by_id(
                df_clean, 'id_local', existing_fill_columns
            )

        # 5. Procesar coordenadas
        if 'coordenada_x_local' in df_clean.columns:
            df_clean = self.process_coordinates(df_clean)

        # 6. Crear variable objetivo
        if 'desc_situacion_local' in df_clean.columns:
            df_clean = self.create_target_variable(df_clean)

        # 7. Procesar fechas
        df_clean = self.process_dates(df_clean)

        # 8. Normalizar IDs
        df_clean = self.normalize_ids(df_clean)

        # 9. Correcciones específicas
        if 'desc_barrio_local' in df_clean.columns:
            df_clean.loc[
                df_clean['id_barrio_local'] == 206, 'desc_barrio_local'
            ] = 'palos de la frontera'

        logger.info(f"Limpieza completa finalizada. Shape final: {df_clean.shape}")
        return df_clean
