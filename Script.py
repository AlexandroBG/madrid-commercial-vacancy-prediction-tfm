#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proyecto: Predicción de Actividad Comercial en Madrid
Autor: Alexandro Bazán
Ruta del proyecto: /Users/alexandrobazan/Desktop/madrid-commercial-prediction/
"""

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import unicodedata
import re
import folium
from folium.plugins import MarkerCluster
from pyproj import Proj, Transformer
from unidecode import unidecode
from IPython.display import display, HTML

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV, SequentialFeatureSelector
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                       roc_auc_score, roc_curve, confusion_matrix, classification_report,
                       ConfusionMatrixDisplay, RocCurveDisplay)

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Feature Selection y Interpretabilidad
from boruta import BorutaPy
import statsmodels.api as sm
import shap
import joblib
import pickle

warnings.filterwarnings('ignore')

# =============================================================================
# 2. CONFIGURACIÓN Y RUTAS
# =============================================================================
# Rutas principales del proyecto
RUTA_PROYECTO = "/Users/alexandrobazan/Desktop/madrid-commercial-prediction/"
RUTA_ACTIVIDADES = RUTA_PROYECTO + "Actividades Economicas de Madrid.csv"
RUTA_RENTA_POB = RUTA_PROYECTO + "RentaPOB.xlsx"
RUTA_GUARDADO = RUTA_PROYECTO + "variables_seleccionadas/"

# Crear directorio para guardar resultados si no existe
import os
os.makedirs(RUTA_GUARDADO, exist_ok=True)

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def limpiar_texto(texto):
"""Limpia y normaliza texto: minúsculas, sin espacios, sin acentos"""
if isinstance(texto, str):
    texto = texto.strip().lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', errors='ignore').decode('utf-8')
    return texto
return texto

def corregir_texto(texto):
"""Corrige errores comunes de codificación y ortografía"""
if pd.isna(texto):
    return texto

texto = texto.lower()

# Reemplazos comunes
texto = texto.replace('0', 'o')
texto = re.sub(r'[âãáàä]', 'a', texto)
texto = re.sub(r'[éèë]', 'e', texto)
texto = re.sub(r'[íìï]', 'i', texto)
texto = re.sub(r'[óòö]', 'o', texto)
texto = re.sub(r'[úùü]', 'u', texto)
texto = re.sub(r'[ñ]', 'n', texto)

# Corregir errores específicos
reemplazos = {
    'r0tulo': 'rotulo',
    'ratulo': 'rotulo',
    's/r': 'sin rotulo',
    'sr': 'sin rotulo',
    'sin rotulo': 'sin actividad',
    'hosteleraa': 'hosteleria',
    'educacian': 'educacion',
    'cientaficas': 'cientificas',
    'actividades administrativas y servicios auxliares': 'actividades administrativas y servicios auxiliares',
    'construccian': 'construccion',
    'caaada': 'cañada',
    'caaaada': 'cañada'
}

for k, v in reemplazos.items():
    texto = texto.replace(k, v)

texto = texto.strip()
texto = re.sub(r'\s+', ' ', texto)

return texto

def stepwise_selection_verbose(X, y, initial_list=[], threshold_in=0.001, threshold_out=0.01, verbose=True):
"""Selección stepwise de variables"""
included = list(initial_list)
while True:
    changed = False
    excluded = list(set(X.columns) - set(included))
    if excluded:
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
                print(f'Add {best_feature} with p-value {best_pval:.6f}')
    if included:
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval:.6f}')
    if not changed:
        break
return included

# =============================================================================
# 4. CARGA DE DATOS
# =============================================================================
print("="*80)
print("CARGA DE DATOS")
print("="*80)

# Cargar dataset de actividades económicas
try:
df = pd.read_csv(RUTA_ACTIVIDADES, encoding='latin1')
print(f"✓ Dataset de actividades cargado: {df.shape}")
except Exception as e:
print(f"Error al cargar actividades: {e}")
df = pd.read_csv(RUTA_ACTIVIDADES.replace("/Users/alexandrobazan", "C:\\Users\\alex_"), encoding='latin1')

# Cargar dataset de renta y población
try:
df_renta = pd.read_excel(RUTA_RENTA_POB)
print(f"✓ Dataset de renta/población cargado: {df_renta.shape}")
except Exception as e:
print(f"Error al cargar renta: {e}")
df_renta = pd.read_excel(RUTA_RENTA_POB.replace("/Users/alexandrobazan", "C:\\Users\\alex_"))

# =============================================================================
# 5. LIMPIEZA Y PREPARACIÓN DE DATOS
# =============================================================================
print("\n" + "="*80)
print("LIMPIEZA Y PREPARACIÓN DE DATOS")
print("="*80)

# 5.1 Información inicial
print("\nInformación del dataset:")
print(df.info())
print("\nPrimeras filas:")
print(df.head())
print("\nValores nulos por columna:")
print(df.isnull().sum().sort_values(ascending=False))

# 5.2 Eliminar columnas innecesarias
columnas_eliminar = ['coordenada_y_agrupacion', 'fx_carga', 'fx_datos_fin',
                 'coordenada_x_agrupacion', 'coordenada_y_agrup', 'id_vial_acceso',
                 'id_clase_ndp_edificio', 'id_clase_ndp_acceso', 'id_local_agrupado',
                 'id_ndp_acceso', 'vial_coinciden', 'duplicado', 'id_planta_agrupado_str']
df = df.drop(columns=[col for col in columnas_eliminar if col in df.columns], errors='ignore')

# 5.3 Combinar columnas duplicadas
df['id_local'] = df['id_local'].combine_first(df.get('?id_local', pd.Series()))
if '?id_local' in df.columns:
df.drop(columns='?id_local', inplace=True)

# 5.4 Rellenar valores faltantes usando id_local
print("\nRellenando valores faltantes por id_local...")

# Rellenar información de distrito y barrio
cols_distrito_barrio = ['id_distrito_local', 'desc_distrito_local', 'id_barrio_local', 'desc_barrio_local']
df_nulos = df[df[cols_distrito_barrio].isnull().any(axis=1)]

for idx, row in df_nulos.iterrows():
id_local = row['id_local']
posibles = df[(df['id_local'] == id_local) &
              df[cols_distrito_barrio].notnull().all(axis=1)]

if not posibles.empty:
    datos = posibles.iloc[0]
    for col in cols_distrito_barrio:
        df.loc[idx, col] = datos[col]

# 5.5 Limpiar y normalizar texto
print("\nLimpiando y normalizando texto...")

columnas_texto = ['desc_barrio_local', 'desc_distrito_local', 'desc_tipo_acceso_local',
              'desc_situacion_local', 'clase_vial_edificio', 'desc_vial_edificio',
              'clase_vial_acceso', 'desc_vial_acceso', 'rotulo', 'desc_seccion',
              'desc_division', 'desc_epigrafe']

for col in columnas_texto:
if col in df.columns:
    df[col] = df[col].apply(limpiar_texto)
    df[col] = df[col].apply(corregir_texto)

# 5.6 Correcciones específicas
df.loc[df['id_barrio_local'] == 206, 'desc_barrio_local'] = 'palos de la frontera'
df['desc_situacion_local'] = df['desc_situacion_local'].replace({
'baja reunificacia3n': 'baja reunificacion',
'baja reunificaciaa3n': 'baja reunificacion'
})

# 5.7 Crear variable objetivo (actividad)
df['desc_situacion_recodificada'] = df['desc_situacion_local'].replace({
'abierto': 'Abierto',
'cerrado': 'Inactivo',
'baja reunificacion': 'Inactivo',
'baja': 'Inactivo',
'uso vivienda': 'Inactivo',
'en obras': 'Inactivo',
'baja pc asociado': 'Inactivo'
})

df['actividad'] = df['desc_situacion_recodificada'].apply(lambda x: 1 if x == 'Abierto' else 0)

# 5.8 Gestión de valores nulos
print("\nEliminando filas con valores nulos en columnas críticas...")
columnas_criticas = ['clase_vial_edificio', 'desc_vial_edificio', 'nom_edificio',
                 'num_edificio', 'cal_edificio', 'clase_vial_acceso',
                 'desc_vial_acceso', 'id_ndp_acceso', 'nom_acceso',
                 'num_acceso', 'cal_acceso']
df = df.dropna(subset=[col for col in columnas_criticas if col in df.columns])

# 5.9 Procesamiento de coordenadas
print("\nProcesando coordenadas geográficas...")
df['coordenada_x_local'] = pd.to_numeric(df['coordenada_x_local'], errors='coerce')
df['coordenada_y_local'] = pd.to_numeric(df['coordenada_y_local'], errors='coerce')

# Convertir UTM a latitud/longitud
utm_zone = 30
proj_utm = Proj(proj='utm', zone=utm_zone, hemisphere='north', datum='WGS84')
proj_latlon = Proj(proj='latlong', datum='WGS84')
transformer = Transformer.from_proj(proj_utm, proj_latlon)

def utm_a_latlon(row):
x = row['coordenada_x_local']
y = row['coordenada_y_local']
if pd.isna(x) or pd.isna(y) or x == 0.0 or y == 0.0:
    return pd.Series([None, None])
lon, lat = transformer.transform(x, y)
return pd.Series([lat, lon])

df[['latitud_local', 'longitud_local']] = df.apply(utm_a_latlon, axis=1)

# 5.10 Procesamiento de fechas
df['Fecha_Reporte'] = pd.to_datetime(df['Fecha_Reporte'])
df['Fecha_Reporte'] = df['Fecha_Reporte'].dt.strftime('%Y%m').astype(int)
df['Año'] = df['Fecha_Reporte'] // 100
df['Mes'] = df['Fecha_Reporte'] % 100

# 5.11 Normalización de IDs
df['id_seccion'] = df['id_seccion'].astype(str).str.strip()
df['id_division'] = df['id_division'].astype(str).str.strip()
df['id_epigrafe'] = df['id_epigrafe'].astype(str).str.strip()

# =============================================================================
# 6. MERGE CON DATOS DE RENTA Y POBLACIÓN
# =============================================================================
print("\n" + "="*80)
print("MERGE CON DATOS DE RENTA Y POBLACIÓN")
print("="*80)

# Normalizar nombres en df_renta
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
for barrio in df['desc_barrio_local'].unique()
}

df_renta['Barrio'] = df_renta['Barrio'].map(lambda x: mapa_barrios_df.get(normalizar_avanzado(x), x))

# Normalizar distritos
df['distrito_norm'] = df['desc_distrito_local'].str.strip().str.lower()
df_renta['distrito_norm'] = df_renta['Distrito'].str.strip().str.lower()

mapa_distritos = df.drop_duplicates(subset='distrito_norm').set_index('distrito_norm')['desc_distrito_local'].to_dict()
df_renta['Distrito'] = df_renta['distrito_norm'].map(mapa_distritos)

df.drop(columns=['distrito_norm'], inplace=True)
df_renta.drop(columns=['distrito_norm'], inplace=True)

# Crear columna Año en df_renta
df_renta['Año'] = df_renta['Año_Poblacion'].astype(str)
df['Año_str'] = df['Año'].astype(str)

# Hacer merge
df = df.merge(
df_renta[['Barrio', 'Distrito', 'Año', 'Total_Poblacion', 'Renta_Media']],
how='left',
left_on=['desc_barrio_local', 'desc_distrito_local', 'Año_str'],
right_on=['Barrio', 'Distrito', 'Año']
)

# Limpiar columnas redundantes
df.drop(columns=['Barrio', 'Distrito', 'Año_str'], inplace=True, errors='ignore')

# Ajustar Total_Poblacion (estaba en miles)
df['Total_Poblacion'] = (df['Total_Poblacion'] * 1000).astype('Int64')

print(f"\n✓ Merge completado. Dataset final: {df.shape}")
print(f"Nulos en Total_Poblacion: {df['Total_Poblacion'].isna().sum()}")
print(f"Nulos en Renta_Media: {df['Renta_Media'].isna().sum()}")

# =============================================================================
# 7. ANÁLISIS EXPLORATORIO DE DATOS
# =============================================================================
print("\n" + "="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*80)

# 7.1 Distribución de la variable objetivo
print("\nDistribución de la variable objetivo (actividad):")
print(df['actividad'].value_counts())
print(df['actividad'].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 6))
sns.countplot(x='actividad', data=df)
plt.title('Distribución de la variable actividad')
plt.xticks([0, 1], ['Inactivo', 'Activo'])
plt.show()

# 7.2 Matriz de correlación
df_num = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = df_num.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Matriz de Correlación entre Variables Numéricas')
plt.show()

# 7.3 Análisis por sección
print("\nTop 10 secciones más frecuentes:")
print(df['desc_seccion'].value_counts().head(10))

# 7.4 Análisis temporal
actividad_temporal = df.groupby(['Año', 'Mes'])['actividad'].agg(['sum', 'count', 'mean'])
actividad_temporal.columns = ['activos', 'total', 'tasa_actividad']

plt.figure(figsize=(12, 6))
actividad_temporal['tasa_actividad'].plot(kind='line', marker='o')
plt.title('Evolución de la tasa de actividad en el tiempo')
plt.xlabel('Periodo')
plt.ylabel('Tasa de actividad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Guardar checkpoint
df.to_pickle(os.path.join(RUTA_GUARDADO, "df_limpio.pkl"))
print(f"\n✓ Dataset limpio guardado en: {RUTA_GUARDADO}df_limpio.pkl")

# =============================================================================
# 8. PREPARACIÓN PARA MODELADO
# =============================================================================
print("\n" + "="*80)
print("PREPARACIÓN PARA MODELADO")
print("="*80)

# Cargar dataset limpio
df = pd.read_pickle(os.path.join(RUTA_GUARDADO, "df_limpio.pkl"))
df_original = df.copy()

# 8.1 Definir columnas categóricas preferidas (sin distrito para evitar multicolinealidad)
preferidas = [col for col in ['desc_barrio_local', 'desc_tipo_acceso_local', 'desc_seccion']
          if col in df.columns]
print("Variables categóricas a procesar:", preferidas)

# 8.2 Separar variable objetivo
y = df['actividad']

# 8.3 Crear variables dummy
cat_vars = df[preferidas].select_dtypes(include='object').columns
cardinalidades = df[cat_vars].nunique()
cat_vars_baja_card = cardinalidades[cardinalidades <= 30].index.tolist()

print(f"\nVariables que se transformarán en dummies: {cat_vars_baja_card}")
df = pd.get_dummies(df, columns=cat_vars_baja_card, drop_first=True)

# 8.4 Eliminar variables no deseadas
variables_a_eliminar = [
'actividad', 'rotulo', 'desc_vial_acceso', 'Fecha_Reporte', 'Mes', 'Año',
'num_acceso', 'cal_acceso', 'latitud_local', 'longitud_local',
'id_local', 'id_distrito_local', 'cod_barrio_local',
'id_tipo_acceso_local', 'id_seccion', 'id_epigrafe', 'desc_epigrafe',
'id_division', 'desc_division', 'desc_situacion_local',
'desc_situacion_recodificada', 'desc_situacion'
]

# Eliminar también variables de distrito
dummies_a_excluir = [col for col in df.columns if col.startswith((
'id_', 'cod_barrio_local_', 'latitud_local', 'longitud_local',
'desc_epigrafe_', 'desc_division_', 'id_seccion_', 'desc_distrito_local_'
))]

variables_a_eliminar += dummies_a_excluir
df = df.drop(columns=[col for col in variables_a_eliminar if col in df.columns], errors='ignore')

# Asegurar que no quedan variables de distrito
distrito_cols_restantes = [col for col in df.columns if 'distrito' in col.lower()]
if distrito_cols_restantes:
df = df.drop(columns=distrito_cols_restantes, errors='ignore')

# 8.5 Eliminar columnas no numéricas restantes
non_numeric_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if non_numeric_cols:
print(f"Eliminando columnas no numéricas restantes: {non_numeric_cols}")
df = df.drop(columns=non_numeric_cols)

# 8.6 Añadir Fecha_Reporte para particionar
df['Fecha_Reporte'] = df_original['Fecha_Reporte']

# 8.7 Separar train y test temporalmente
X_train = df[df['Fecha_Reporte'] < 202401].drop(columns=['Fecha_Reporte'])
X_test = df[df['Fecha_Reporte'] >= 202401].drop(columns=['Fecha_Reporte'])
y_train = y.loc[X_train.index]
y_test = y.loc[X_test.index]

# 8.8 Alinear columnas
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 8.9 Verificar Renta_Media
if 'Renta_Media' in X_train.columns:
print("✓ Renta_Media confirmada en el dataset")
else:
print("⚠ ADVERTENCIA: Renta_Media no encontrada en el dataset")

# 8.10 Estandarización
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                          columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                         columns=X_test.columns, index=X_test.index)

print(f"\nDimensiones finales:")
print(f"X_train: {X_train_scaled.shape}")
print(f"X_test: {X_test_scaled.shape}")
print(f"\nVariables finales ({len(X_train_scaled.columns)} variables):")
for i, col in enumerate(X_train_scaled.columns[:20], start=1):
print(f"{i:3}: {col}")
if len(X_train_scaled.columns) > 20:
print(f"... y {len(X_train_scaled.columns) - 20} variables más")

# Guardar datos preparados
joblib.dump(X_train, os.path.join(RUTA_GUARDADO, 'X_train_raw.joblib'))
joblib.dump(X_test, os.path.join(RUTA_GUARDADO, 'X_test_raw.joblib'))
joblib.dump(y_train, os.path.join(RUTA_GUARDADO, 'y_train.joblib'))
joblib.dump(y_test, os.path.join(RUTA_GUARDADO, 'y_test.joblib'))
joblib.dump(scaler, os.path.join(RUTA_GUARDADO, 'scaler.joblib'))
joblib.dump(X_train_scaled, os.path.join(RUTA_GUARDADO, 'X_train_scaled.joblib'))
joblib.dump(X_test_scaled, os.path.join(RUTA_GUARDADO, 'X_test_scaled.joblib'))

# =============================================================================
# 9. SELECCIÓN DE CARACTERÍSTICAS
# =============================================================================
print("\n" + "="*80)
print("SELECCIÓN DE CARACTERÍSTICAS")
print("="*80)

# 9.1 BORUTA
print("\n--- Método 1: BORUTA ---")
X_boruta_sample = X_train_scaled.sample(n=min(100000, len(X_train_scaled)), random_state=12345)
y_boruta_sample = y_train.loc[X_boruta_sample.index]

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=12345)
boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=1, random_state=12345)
boruta_selector.fit(X_boruta_sample.values, y_boruta_sample.values)

selected_features_boruta = X_train_scaled.columns[boruta_selector.support_].tolist()
print(f"Variables seleccionadas por Boruta: {len(selected_features_boruta)}")

# 9.2 RFECV
print("\n--- Método 2: RFECV ---")
X_sample = X_train_scaled.sample(n=50000, random_state=42)
y_sample = y_train.loc[X_sample.index]

model = RandomForestClassifier(n_estimators=100, random_state=123)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
selector = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy', n_jobs=-1)
selector.fit(X_sample, y_sample)

selected_features_rfecv = X_sample.columns[selector.support_].tolist()
print(f"Variables seleccionadas por RFECV: {len(selected_features_rfecv)}")

# 9.3 Stepwise
print("\n--- Método 3: Stepwise ---")
X_step_sample, _, y_step_sample, _ = train_test_split(
X_train_scaled, y_train, train_size=min(500000, len(X_train_scaled)),
stratify=y_train, random_state=12345
)

selector_var = VarianceThreshold(threshold=1e-5)
X_step_sample_filtered = pd.DataFrame(selector_var.fit_transform(X_step_sample),
                                  columns=X_step_sample.columns[selector_var.get_support()])
X_step_sample_filtered.index = X_step_sample.index

# Eliminar variables altamente correlacionadas
corr_matrix = X_step_sample_filtered.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_step_sample_filtered.drop(columns=to_drop, inplace=True)

vars_stepwise = stepwise_selection_verbose(X_step_sample_filtered, y_step_sample, verbose=False)
print(f"Variables seleccionadas por Stepwise: {len(vars_stepwise)}")

# 9.4 Sequential Backward Selection (SBF)
print("\n--- Método 4: SBF ---")
X_sbf, _, y_sbf, _ = train_test_split(
X_train_scaled, y_train, train_size=50000, stratify=y_train, random_state=12345
)

model_sbf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=12345)
sbf = SequentialFeatureSelector(
estimator=model_sbf, direction='backward', n_features_to_select='auto',
scoring='roc_auc', cv=3, n_jobs=-1
)
sbf.fit(X_sbf, y_sbf)

vars_sbf = X_sbf.columns[sbf.get_support()].tolist()
print(f"Variables seleccionadas por SBF: {len(vars_sbf)}")

# 9.5 SHAP
print("\n--- Método 5: SHAP ---")
model_shap = xgb.XGBClassifier(
n_estimators=100, max_depth=6, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8, random_state=12345, n_jobs=-1
)
model_shap.fit(X_train_scaled, y_train)

explainer = shap.Explainer(model_shap, X_train_scaled)
shap_values = explainer(X_test_scaled)

shap_importances = pd.DataFrame({
'variable': X_test_scaled.columns,
'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

top_30_vars_shap = shap_importances['variable'].head(30).tolist()
print(f"Top 30 variables seleccionadas por SHAP: {len(top_30_vars_shap)}")

# Guardar variables seleccionadas
pickle.dump(selected_features_boruta, open(os.path.join(RUTA_GUARDADO, "selected_features_boruta.pkl"), "wb"))
pickle.dump(selected_features_rfecv, open(os.path.join(RUTA_GUARDADO, "selected_features_rfecv.pkl"), "wb"))
pickle.dump(vars_stepwise, open(os.path.join(RUTA_GUARDADO, "vars_stepwise.pkl"), "wb"))
pickle.dump(vars_sbf, open(os.path.join(RUTA_GUARDADO, "vars_sbf.pkl"), "wb"))
pickle.dump(top_30_vars_shap, open(os.path.join(RUTA_GUARDADO, "top_30_vars_shap.pkl"), "wb"))

# =============================================================================
# 10. ENTRENAMIENTO DE MODELOS
# =============================================================================
print("\n" + "="*80)
print("ENTRENAMIENTO DE MODELOS")
print("="*80)

# Usar variables seleccionadas por Boruta
X_train_boruta = X_train_scaled[selected_features_boruta]
X_test_boruta = X_test_scaled[selected_features_boruta]

resultados_modelos = {}

# 10.1 Regresión Logística
print("\n--- Modelo 1: Regresión Logística ---")
param_dist_lr = {
'C': [0.001, 0.01, 0.1, 1, 10],
'penalty': ['l2'],
'solver': ['liblinear', 'saga']
}

lr_base = LogisticRegression(random_state=12345, max_iter=1000)
lr_search = RandomizedSearchCV(lr_base, param_dist_lr, n_iter=10, cv=3,
                           scoring='f1', random_state=123, n_jobs=-1)
lr_search.fit(X_train_boruta, y_train)

y_pred_lr = lr_search.predict(X_test_boruta)
y_proba_lr = lr_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['Logistic Regression'] = {'pred': y_pred_lr, 'proba': y_proba_lr}

# 10.2 Árbol de Decisión
print("\n--- Modelo 2: Árbol de Decisión ---")
param_grid_dt = {
'max_depth': [3, 5, 10, 20, None],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=12345)
dt_search = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
dt_search.fit(X_train_boruta, y_train)

y_pred_dt = dt_search.predict(X_test_boruta)
y_proba_dt = dt_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['Decision Tree'] = {'pred': y_pred_dt, 'proba': y_proba_dt}

# 10.3 Random Forest
print("\n--- Modelo 3: Random Forest ---")
X_train_sample = X_train_boruta.sample(n=min(100000, len(X_train_boruta)), random_state=12345)
y_train_sample = y_train.loc[X_train_sample.index]

param_grid_rf = {
'n_estimators': [100, 150],
'max_depth': [5, 10],
'min_samples_split': [2],
'min_samples_leaf': [1]
}

rf = RandomForestClassifier(random_state=12345)
rf_search = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1', n_jobs=1, verbose=1)
rf_search.fit(X_train_sample, y_train_sample)

rf_search.best_estimator_.fit(X_train_boruta, y_train)
y_pred_rf = rf_search.predict(X_test_boruta)
y_proba_rf = rf_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['Random Forest'] = {'pred': y_pred_rf, 'proba': y_proba_rf}

# 10.4 XGBoost
print("\n--- Modelo 4: XGBoost ---")
param_grid_xgb = {
'max_depth': [3, 5],
'learning_rate': [0.1],
'n_estimators': [100, 200],
'subsample': [0.8]
}

xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=12345)
xgb_search = GridSearchCV(xgb_base, param_grid_xgb, scoring='roc_auc', cv=3, n_jobs=1, verbose=1)
xgb_search.fit(X_train_boruta, y_train)

y_pred_xgb = xgb_search.predict(X_test_boruta)
y_proba_xgb = xgb_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['XGBoost'] = {'pred': y_pred_xgb, 'proba': y_proba_xgb}

# 10.5 KNN
print("\n--- Modelo 5: KNN ---")
X_sample_knn, _, y_sample_knn, _ = train_test_split(
X_train_boruta, y_train, train_size=min(100000, len(X_train_boruta)),
stratify=y_train, random_state=12345
)

param_grid_knn = {
'n_neighbors': [3, 5],
'weights': ['distance'],
'metric': ['euclidean']
}

from sklearn.pipeline import Pipeline
knn_pipe = Pipeline([
('scaler', StandardScaler()),
('knn', KNeighborsClassifier())
])

knn_search = GridSearchCV(knn_pipe, param_grid_knn, cv=3, scoring='accuracy', n_jobs=2, verbose=1)
knn_search.fit(X_sample_knn, y_sample_knn)

y_pred_knn = knn_search.predict(X_test_boruta)
y_proba_knn = knn_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['KNN'] = {'pred': y_pred_knn, 'proba': y_proba_knn}

# 10.6 SVM
print("\n--- Modelo 6: SVM ---")
X_train_small = X_train_boruta.sample(n=min(100000, len(X_train_boruta)), random_state=12345)
y_train_small = y_train.loc[X_train_small.index]

param_grid_svm = {
'C': [0.1, 1],
'kernel': ['linear']
}

svm_base = SVC(random_state=12345)
svm_search = GridSearchCV(svm_base, param_grid_svm, cv=2, scoring='f1', n_jobs=2, verbose=1)
svm_search.fit(X_train_small, y_train_small)

y_pred_svm = svm_search.predict(X_test_boruta)
scores_svm = svm_search.best_estimator_.decision_function(X_test_boruta)
resultados_modelos['SVM'] = {'pred': y_pred_svm, 'proba': scores_svm}

# 10.7 MLP (Red Neuronal)
print("\n--- Modelo 7: MLP (Red Neuronal) ---")
X_train_sub, _, y_train_sub, _ = train_test_split(
X_train_boruta, y_train, train_size=0.5, random_state=42
)

param_grid_mlp = {
'hidden_layer_sizes': [(50, 100)],
'activation': ['relu'],
'alpha': [1e-4, 1e-3],
'learning_rate': ['adaptive']
}

mlp_base = MLPClassifier(solver='adam', max_iter=300, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=10, random_state=12345)
mlp_search = GridSearchCV(mlp_base, param_grid_mlp, scoring='f1', cv=3, verbose=2, n_jobs=1)
mlp_search.fit(X_train_sub, y_train_sub)

mlp_search.best_estimator_.fit(X_train_boruta, y_train)
y_pred_mlp = mlp_search.predict(X_test_boruta)
y_proba_mlp = mlp_search.predict_proba(X_test_boruta)[:, 1]
resultados_modelos['MLP'] = {'pred': y_pred_mlp, 'proba': y_proba_mlp}

# 10.8 Voting Classifier (Ensamble)
print("\n--- Modelo 8: Voting Classifier ---")
X_train_boruta_np = X_train_boruta.to_numpy(copy=True)
X_test_boruta_np = X_test_boruta.to_numpy(copy=True)
y_train_np = y_train.to_numpy(copy=True)
y_test_np = y_test.to_numpy(copy=True)

voting_clf = VotingClassifier(
estimators=[
    ('rf', rf_search.best_estimator_),
    ('xgb', xgb_search.best_estimator_),
    ('mlp', mlp_search.best_estimator_)
],
voting='soft',
weights=[3, 2, 1],
n_jobs=-1
)
voting_clf.fit(X_train_boruta_np, y_train_np)

y_pred_voting = voting_clf.predict(X_test_boruta_np)
y_proba_voting = voting_clf.predict_proba(X_test_boruta_np)[:, 1]
resultados_modelos['VotingClassifier'] = {'pred': y_pred_voting, 'proba': y_proba_voting}

# 10.9 Stacking Classifier
print("\n--- Modelo 9: Stacking Classifier ---")
stacking_clf = StackingClassifier(
estimators=[
    ('rf', rf_search.best_estimator_),
    ('xgb', xgb_search.best_estimator_),
    ('mlp', mlp_search.best_estimator_)
],
final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=12345),
cv=3,
n_jobs=1,
passthrough=False
)
stacking_clf.fit(X_train_boruta_np, y_train_np)

y_pred_stacking = stacking_clf.predict(X_test_boruta_np)
y_proba_stacking = stacking_clf.predict_proba(X_test_boruta_np)[:, 1]
resultados_modelos['StackingClassifier'] = {'pred': y_pred_stacking, 'proba': y_proba_stacking}

# =============================================================================
# 11. EVALUACIÓN Y COMPARACIÓN DE MODELOS
# =============================================================================
print("\n" + "="*80)
print("EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("="*80)

# Crear DataFrame con métricas
metricas = []

for nombre, datos in resultados_modelos.items():
y_pred = datos['pred']
y_proba = datos['proba']

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

# Calcular especificidad
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

metricas.append({
    "Modelo": nombre,
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "AUC": auc_score,
    "Especificidad": spec
})

df_metricas = pd.DataFrame(metricas).sort_values("AUC", ascending=False)
print("\n=== COMPARACIÓN DE MÉTRICAS ===")
print(df_metricas.round(4))

# Guardar tabla de métricas
df_metricas.to_csv(os.path.join(RUTA_GUARDADO, "comparacion_modelos.csv"), index=False)

# Graficar curvas ROC
plt.figure(figsize=(10, 8))

for nombre, datos in resultados_modelos.items():
fpr, tpr, _ = roc_curve(y_test, datos['proba'])
roc_auc = roc_auc_score(y_test, datos['proba'])
plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Comparación de Curvas ROC')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GUARDADO, "curvas_roc_comparacion.pdf"), dpi=300)
plt.show()

# Matrices de confusión
n_modelos = len(resultados_modelos)
cols = 3
rows = -(-n_modelos // cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

for ax, (nombre, datos) in zip(axes.flatten(), resultados_modelos.items()):
cm = confusion_matrix(y_test, datos['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title(nombre)
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')

for i in range(n_modelos, rows * cols):
fig.delaxes(axes.flatten()[i])

plt.suptitle("Matrices de Confusión por Modelo", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(RUTA_GUARDADO, "matrices_confusion.pdf"), dpi=300)
plt.show()

# =============================================================================
# 12. INTERPRETABILIDAD - ANÁLISIS SHAP
# =============================================================================
print("\n" + "="*80)
print("INTERPRETABILIDAD - ANÁLISIS SHAP")
print("="*80)

# Usar el mejor modelo (según AUC)
mejor_modelo_nombre = df_metricas.iloc[0]['Modelo']
print(f"\nUsando el mejor modelo para análisis SHAP: {mejor_modelo_nombre}")

# Por simplicidad, usaremos el modelo MLP para SHAP
if 'mlp_search' in locals():
best_model = mlp_search.best_estimator_

# Tomar muestra para SHAP
muestra_X_test = X_test_boruta.sample(min(10000, len(X_test_boruta)), random_state=42)

# Función predictora
def predict_proba_fn(X):
    X_df = pd.DataFrame(X, columns=X_train_boruta.columns)
    return best_model.predict_proba(X_df)[:, 1]

# Crear explainer
explainer = shap.KernelExplainer(
    predict_proba_fn,
    shap.sample(X_train_boruta, 100, random_state=42)
)

# Calcular SHAP values en lotes
batch_size = 500
all_shap_values = []

for i in range(0, len(muestra_X_test), batch_size):
    print(f"Procesando filas {i} a {min(i+batch_size, len(muestra_X_test))}...")
    batch = muestra_X_test.iloc[i:i+batch_size]
    shap_vals_batch = explainer.shap_values(batch, nsamples=100)
    all_shap_values.append(shap_vals_batch)

shap_values_calc = np.vstack(all_shap_values)

# Gráfico resumen SHAP
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_calc, muestra_X_test, plot_type="dot",
                  max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_GUARDADO, "shap_summary_plot.pdf"), dpi=300)
plt.show()

# Ranking de importancia
importancia_media = np.abs(shap_values_calc).mean(axis=0)
ranking = pd.DataFrame({
    'Variable': muestra_X_test.columns,
    'Impacto_SHAP_Medio': importancia_media
}).sort_values(by='Impacto_SHAP_Medio', ascending=False)

print("\n📊 Top 15 variables más influyentes según SHAP:")
print(ranking.head(15))
ranking.to_csv(os.path.join(RUTA_GUARDADO, "ranking_variables_shap.csv"), index=False)

# =============================================================================
# 13. GUARDAR MODELOS Y RESULTADOS FINALES
# =============================================================================
print("\n" + "="*80)
print("GUARDANDO MODELOS Y RESULTADOS")
print("="*80)

# Guardar mejores modelos
modelos_a_guardar = {
'logistic_regression': lr_search.best_estimator_ if 'lr_search' in locals() else None,
'decision_tree': dt_search.best_estimator_ if 'dt_search' in locals() else None,
'random_forest': rf_search.best_estimator_ if 'rf_search' in locals() else None,
'xgboost': xgb_search.best_estimator_ if 'xgb_search' in locals() else None,
'knn': knn_search.best_estimator_ if 'knn_search' in locals() else None,
'svm': svm_search.best_estimator_ if 'svm_search' in locals() else None,
'mlp': mlp_search.best_estimator_ if 'mlp_search' in locals() else None,
'voting': voting_clf if 'voting_clf' in locals() else None,
'stacking': stacking_clf if 'stacking_clf' in locals() else None
}

for nombre, modelo in modelos_a_guardar.items():
if modelo is not None:
    joblib.dump(modelo, os.path.join(RUTA_GUARDADO, f"modelo_{nombre}.pkl"))
    print(f"✓ Modelo {nombre} guardado")

# Guardar resultados finales
resultados_finales = {
'metricas': df_metricas,
'variables_boruta': selected_features_boruta,
'mejor_modelo': mejor_modelo_nombre,
'fecha_ejecucion': pd.Timestamp.now()
}

with open(os.path.join(RUTA_GUARDADO, "resultados_finales.pkl"), 'wb') as f:
pickle.dump(resultados_finales, f)

print(f"\n✓ Todos los resultados guardados en: {RUTA_GUARDADO}")

# =============================================================================
# 14. ANÁLISIS DE CASOS ESPECÍFICOS Y VISUALIZACIONES
# =============================================================================
print("\n" + "="*80)
print("ANÁLISIS DE CASOS ESPECÍFICOS")
print("="*80)

# Agregar predicciones al dataset original
df_test_pred = df_original.loc[X_test_boruta.index].copy()
df_test_pred['prediccion_mlp'] = y_pred_mlp
df_test_pred['probabilidad_mlp'] = y_proba_mlp

# Caso 1: Locales activos de comercio en Chamartín
print("\n--- Caso 1: Locales activos de comercio en Chamartín ---")
df_caso1 = df_test_pred[
(df_test_pred['prediccion_mlp'] == 1) &
(df_test_pred['desc_tipo_acceso_local'].str.lower() == 'puerta calle') &
(df_test_pred['desc_seccion'].str.lower().str.contains('comercio')) &
(df_test_pred['desc_distrito_local'].str.strip().str.lower() == 'chamartin') &
(df_test_pred['Fecha_Reporte'] == 202412)
]

print(f"Locales encontrados: {len(df_caso1)}")

# Resumen por barrio
resumen_barrio = df_caso1.groupby(['desc_distrito_local', 'desc_barrio_local']).size().reset_index(name='cantidad')
resumen_barrio = resumen_barrio.sort_values(by='cantidad', ascending=False)
print("\nTop barrios con más actividad comercial:")
print(resumen_barrio.head())

# Crear mapa para visualización (si hay menos de 1000 puntos)
if len(df_caso1) < 1000 and len(df_caso1) > 0:
mapa_caso1 = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles='CartoDB positron')
marker_cluster = MarkerCluster().add_to(mapa_caso1)

for idx, row in df_caso1.iterrows():
    if pd.notnull(row['latitud_local']) and pd.notnull(row['longitud_local']):
        popup_text = f"""
        <b>Barrio:</b> {row['desc_barrio_local']}<br>
        <b>Epígrafe:</b> {row['desc_epigrafe']}<br>
        <b>Rótulo:</b> {row['rotulo']}<br>
        <b>Calle:</b> {row['desc_vial_acceso']} {row.get('num_acceso', '')}
        """
        folium.Marker(
            location=[row['latitud_local'], row['longitud_local']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

mapa_caso1.save(os.path.join(RUTA_GUARDADO, "mapa_comercio_chamartin.html"))
print(f"\n✓ Mapa guardado en: {RUTA_GUARDADO}mapa_comercio_chamartin.html")

# Caso 2: Locales inactivos de industria manufacturera
print("\n--- Caso 2: Locales inactivos de industria manufacturera ---")
df_caso2 = df_test_pred[
(df_test_pred['prediccion_mlp'] == 0) &
(df_test_pred['desc_tipo_acceso_local'].str.lower() == 'interior') &
(df_test_pred['desc_seccion'].str.lower() == 'industria manufacturera') &
(df_test_pred['Fecha_Reporte'] == 202412)
]

print(f"Locales encontrados: {len(df_caso2)}")

# Top epígrafes inactivos
top_epigrafes = df_caso2.groupby(['desc_distrito_local', 'desc_barrio_local', 'desc_epigrafe']).size().reset_index(name='conteo')
top_epigrafes = top_epigrafes.sort_values(by='conteo', ascending=False)
print("\nTop epígrafes de locales inactivos:")
print(top_epigrafes.head(10))

# =============================================================================
# 15. RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "="*80)
print("RESUMEN EJECUTIVO DEL PROYECTO")
print("="*80)

print(f"""
📊 PROYECTO: Predicción de Actividad Comercial en Madrid

📁 DATOS:
- Dataset principal: {df_original.shape[0]:,} registros
- Variables originales: {df_original.shape[1]}
- Variables finales para modelado: {len(X_train_scaled.columns)}
- Periodo temporal: {df_original['Año'].min()} - {df_original['Año'].max()}

🎯 VARIABLE OBJETIVO:
- Actividad (1: Activo, 0: Inactivo)
- Distribución: {(y.value_counts(normalize=True) * 100).round(1).to_dict()}

🔍 SELECCIÓN DE CARACTERÍSTICAS:
- Método utilizado: Boruta
- Variables seleccionadas: {len(selected_features_boruta)}
- Variable clave incluida: {'Renta_Media' if 'Renta_Media' in selected_features_boruta else 'Renta_Media NO incluida'}

🏆 MEJOR MODELO: {mejor_modelo_nombre}
- AUC: {df_metricas.iloc[0]['AUC']:.4f}
- F1 Score: {df_metricas.iloc[0]['F1 Score']:.4f}
- Accuracy: {df_metricas.iloc[0]['Accuracy']:.4f}

📈 TOP 5 VARIABLES MÁS IMPORTANTES:
""")

if 'ranking' in locals():
for i, row in ranking.head(5).iterrows():
    print(f"   {i+1}. {row['Variable']}: {row['Impacto_SHAP_Medio']:.4f}")

print(f"\n📂 Resultados guardados en: {RUTA_GUARDADO}")
print("\n✅ ANÁLISIS COMPLETADO CON ÉXITO")
print("="*80)
