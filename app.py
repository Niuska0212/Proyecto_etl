import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
import base64
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_score, 
                            recall_score, f1_score, roc_auc_score)
import joblib
import json

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema de Análisis de Datos Hotel"

# ==================== FUNCIONES DE PROCESAMIENTO ====================
# (Estas son adaptaciones de las funciones originales para trabajar en Dash)

def load_data(contents, filename):
    """Carga datos desde diferentes formatos"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename:
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    
    return df

def clean_data(df):
    """Limpieza de datos adaptada para Dash"""
    if df is None or df.empty:
        return None
    
    try:
        # Manejo de valores nulos
        if 'children' in df.columns:
            df['children'] = df['children'].fillna(0)
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('Unknown')
        
        # Conversión de fechas
        if 'reservation_status_date' in df.columns:
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
        
        # Eliminación de columnas
        cols_to_drop = ['agent', 'company']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        
        # Rellenar valores numéricos
        numeric_columns = ['adr', 'lead_time', 'stays_in_week_nights', 'stays_in_weekend_nights']
        for col in numeric_columns:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
                
        return df
    except Exception as e:
        print(f"Error en limpieza: {e}")
        return None

def transform_data(df):
    """Transformación de datos adaptada para Dash"""
    if df is None:
        return None
    
    try:
        # One-hot encoding
        if 'hotel' in df.columns:
            df = pd.get_dummies(df, columns=['hotel'], prefix=['hotel'])
        
        # Label encoding
        if 'meal' in df.columns:
            le = LabelEncoder()
            df['meal'] = le.fit_transform(df['meal'])
        
        # Normalización
        numeric_columns = [col for col in ['adr', 'lead_time', 'stays_in_week_nights', 
                                         'stays_in_weekend_nights'] if col in df.columns]
        if numeric_columns:
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Creación de nuevas columnas
        if all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            df['total_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        
        if 'is_canceled' in df.columns:
            df['is_canceled'] = df['is_canceled'].apply(lambda x: 1 if x == 1 else 0)
        
        if 'country' in df.columns:
            df['region'] = df['country'].apply(lambda x: 'Europe' if x in ['PRT', 'ESP', 'FRA'] else 'Other')
            
        return df
    except Exception as e:
        print(f"Error en transformación: {e}")
        return None

def prepare_model_data(df):
    """Prepara los datos para el modelo predictivo"""
    if df is None:
        return None, None, None, None
    
    try:
        # Eliminar columnas problemáticas
        cols_to_drop = ['reservation_status', 'reservation_status_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Convertir booleanos y categóricas
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
            
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Dividir datos
        if 'is_canceled' in df.columns:
            X = df.drop(columns=['is_canceled'])
            y = df['is_canceled']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
        else:
            return None, None, None, None
    except Exception as e:
        print(f"Error en preparación de datos: {e}")
        return None, None, None, None

def train_evaluate_model(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo Random Forest"""
    try:
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        return rf_model, metrics, cm
    except Exception as e:
        print(f"Error en modelo: {e}")
        return None, None, None

# ==================== INTERFAZ DE LA APLICACIÓN ====================
app.layout = dbc.Container([
    html.H1("Sistema de Análisis de Datos Hotel", className="mb-4 text-center"),
    
    dcc.Tabs([
        # Pestaña 1: Carga de datos
        dcc.Tab(label='Carga de Datos', children=[
            html.Div([
                html.H3("Cargar Archivo", className="mt-3"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastra y suelta o ',
                        html.A('Selecciona un Archivo')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    multiple=False
                ),
                
                html.Div(id='output-data-upload'),
                
                html.Div([
                    html.H4("Resumen de Datos", className="mt-4"),
                    html.Div(id='data-summary')
                ]),
                
                html.Div([
                    html.H4("Vista Previa de Datos", className="mt-4"),
                    html.Div(id='data-preview')
                ])
            ])
        ]),
        
        # Pestaña 2: Proceso ETL
        dcc.Tab(label='Proceso ETL', children=[
            html.Div([
                html.H3("Proceso de Extracción, Transformación y Carga", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Datos Originales"),
                        html.Div(id='original-data-stats')
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Datos Transformados"),
                        html.Div(id='transformed-data-stats')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Comparación de Columnas"),
                        html.Div(id='column-comparison')
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Visualización de Transformación"),
                        dcc.Dropdown(
                            id='etl-column-selector',
                            options=[],
                            placeholder="Selecciona una columna para visualizar"
                        ),
                        dcc.Graph(id='etl-comparison-graph')
                    ], width=12)
                ])
            ])
        ]),
        
        # Pestaña 3: Análisis Exploratorio
        dcc.Tab(label='Análisis Exploratorio', children=[
            html.Div([
                html.H3("Análisis Exploratorio de Datos", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Seleccionar Tipo de Gráfico"),
                        dcc.Dropdown(
                            id='eda-plot-type',
                            options=[
                                {'label': 'Distribución de Hoteles', 'value': 'hotel_dist'},
                                {'label': 'Distribución de ADR', 'value': 'adr_dist'},
                                {'label': 'Países de Origen', 'value': 'country_dist'},
                                {'label': 'Cancelaciones vs Lead Time', 'value': 'cancel_lead'},
                                {'label': 'Matriz de Correlación', 'value': 'corr_matrix'}
                            ],
                            value='hotel_dist'
                        )
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='eda-plot')
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Estadísticas Descriptivas"),
                        html.Div(id='descriptive-stats')
                    ], width=12)
                ])
            ])
        ]),
        
        # Pestaña 4: Minería de Datos
        dcc.Tab(label='Minería de Datos', children=[
            html.Div([
                html.H3("Modelo Predictivo de Cancelaciones", className="mt-3"),
                
                dbc.Button("Entrenar Modelo", id='train-model-btn', color="primary", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Métricas del Modelo"),
                        html.Div(id='model-metrics')
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Matriz de Confusión"),
                        dcc.Graph(id='confusion-matrix')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Importancia de Características"),
                        dcc.Graph(id='feature-importance')
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Predicciones de Ejemplo"),
                        html.Div(id='predictions-sample')
                    ], width=12)
                ])
            ])
        ]),
        
        # Pestaña 5: Toma de Decisiones
        dcc.Tab(label='Toma de Decisiones', children=[
            html.Div([
                html.H3("Recomendaciones para la Toma de Decisiones", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Factores Clave en Cancelaciones"),
                        html.Div(id='key-factors')
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Recomendaciones Estratégicas"),
                        html.Div(id='strategic-recommendations')
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Visualización de Impacto"),
                        dcc.Dropdown(
                            id='impact-factor-selector',
                            options=[],
                            placeholder="Selecciona un factor para analizar"
                        ),
                        dcc.Graph(id='impact-analysis-graph')
                    ], width=12)
                ])
            ])
        ])
    ]),
    
    # Almacenamiento de datos entre callbacks
    dcc.Store(id='stored-original-data'),
    dcc.Store(id='stored-cleaned-data'),
    dcc.Store(id='stored-transformed-data'),
    dcc.Store(id='stored-model-data'),
    dcc.Store(id='stored-model')
])

# ==================== CALLBACKS ====================

# Callback para cargar y mostrar datos
@app.callback(
    [Output('stored-original-data', 'data'),
     Output('output-data-upload', 'children'),
     Output('data-summary', 'children'),
     Output('data-preview', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return None, "Suba un archivo para comenzar", "", ""
    
    df = load_data(contents, filename)
    if df is None:
        return None, "Error al cargar el archivo", "", ""
    
    # Guardar datos originales
    original_data = df.to_dict('records')
    
    # Mensaje de éxito
    success_msg = html.Div([
        html.H5(f"Archivo cargado: {filename}"),
        html.P(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
    ])
    
    # Resumen de datos
    summary = html.Div([
        html.H5("Resumen Estadístico"),
        dash_table.DataTable(
            data=df.describe().reset_index().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.describe().reset_index().columns],
            style_table={'overflowX': 'auto'}
        )
    ])
    
    # Vista previa
    preview = html.Div([
        html.H5("Primeras 10 filas"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            page_size=10
        )
    ])
    
    return original_data, success_msg, summary, preview

# Callback para el proceso ETL
@app.callback(
    [Output('stored-cleaned-data', 'data'),
     Output('stored-transformed-data', 'data'),
     Output('original-data-stats', 'children'),
     Output('transformed-data-stats', 'children'),
     Output('column-comparison', 'children'),
     Output('etl-column-selector', 'options')],
    [Input('stored-original-data', 'data')]
)
def process_etl(data):
    if data is None:
        return None, None, "", "", "", []
    
    df = pd.DataFrame.from_dict(data)
    
    # Limpieza de datos
    df_cleaned = clean_data(df)
    if df_cleaned is None:
        return None, None, "", "", "", []
    
    # Transformación de datos
    df_transformed = transform_data(df_cleaned)
    if df_transformed is None:
        return None, None, "", "", "", []
    
    # Estadísticas originales
    original_stats = html.Div([
        html.P(f"Filas: {len(df)}"),
        html.P(f"Columnas: {len(df.columns)}"),
        html.P(f"Valores nulos: {df.isnull().sum().sum()}"),
        html.P(f"Duplicados: {df.duplicated().sum()}")
    ])
    
    # Estadísticas transformadas
    transformed_stats = html.Div([
        html.P(f"Filas: {len(df_transformed)}"),
        html.P(f"Columnas: {len(df_transformed.columns)}"),
        html.P(f"Valores nulos: {df_transformed.isnull().sum().sum()}"),
        html.P(f"Duplicados: {df_transformed.duplicated().sum()}")
    ])
    
    # Comparación de columnas
    original_cols = set(df.columns)
    transformed_cols = set(df_transformed.columns)
    added_cols = transformed_cols - original_cols
    removed_cols = original_cols - transformed_cols
    
    comparison = html.Div([
        html.H5("Cambios en Columnas"),
        html.P(f"Columnas añadidas: {len(added_cols)}"),
        html.P(f"Columnas eliminadas: {len(removed_cols)}"),
        html.P("Nuevas columnas:"),
        html.Ul([html.Li(col) for col in added_cols]),
        html.P("Columnas eliminadas:"),
        html.Ul([html.Li(col) for col in removed_cols])
    ])
    
    # Opciones para el selector de columnas
    common_cols = list(original_cols & transformed_cols)
    options = [{'label': col, 'value': col} for col in common_cols]
    
    return (df_cleaned.to_dict('records'), 
            df_transformed.to_dict('records'), 
            original_stats, 
            transformed_stats, 
            comparison, 
            options)

# Callback para gráfico de comparación ETL
@app.callback(
    Output('etl-comparison-graph', 'figure'),
    [Input('etl-column-selector', 'value'),
     Input('stored-cleaned-data', 'data'),
     Input('stored-transformed-data', 'data')]
)
def update_etl_graph(selected_col, cleaned_data, transformed_data):
    if not selected_col or cleaned_data is None or transformed_data is None:
        return go.Figure()
    
    df_cleaned = pd.DataFrame.from_dict(cleaned_data)
    df_transformed = pd.DataFrame.from_dict(transformed_data)
    
    if selected_col not in df_cleaned.columns or selected_col not in df_transformed.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    # Agregar datos originales
    if pd.api.types.is_numeric_dtype(df_cleaned[selected_col]):
        fig.add_trace(go.Histogram(
            x=df_cleaned[selected_col],
            name='Original',
            opacity=0.75
        ))
    else:
        counts = df_cleaned[selected_col].value_counts().nlargest(10)
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            name='Original'
        ))
    
    # Agregar datos transformados
    if pd.api.types.is_numeric_dtype(df_transformed[selected_col]):
        fig.add_trace(go.Histogram(
            x=df_transformed[selected_col],
            name='Transformado',
            opacity=0.75
        ))
    else:
        counts = df_transformed[selected_col].value_counts().nlargest(10)
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            name='Transformado'
        ))
    
    fig.update_layout(
        title=f'Comparación de {selected_col}',
        barmode='overlay'
    )
    
    return fig

# Callback para análisis exploratorio - VERSIÓN CORREGIDA
@app.callback(
    [Output('eda-plot', 'figure'),
     Output('descriptive-stats', 'children')],
    [Input('eda-plot-type', 'value'),
     Input('stored-transformed-data', 'data')]
)
def update_eda_plot(plot_type, data):
    if data is None:
        return go.Figure(), ""
    
    df = pd.DataFrame.from_dict(data)
    
    if plot_type == 'hotel_dist' and 'hotel' in df.columns:
        fig = px.histogram(df, x='hotel', title='Distribución de Tipos de Hotel')
        stats = df['hotel'].value_counts().to_frame('Conteo').reset_index()
        stats.columns = ['Hotel', 'Conteo']
        
    elif plot_type == 'adr_dist' and 'adr' in df.columns:
        fig = px.histogram(df, x='adr', title='Distribución de ADR (Average Daily Rate)')
        stats = df['adr'].describe().to_frame('ADR').reset_index()
        stats.columns = ['Estadística', 'ADR']
        
    elif plot_type == 'country_dist' and 'country' in df.columns:
        top_countries = df['country'].value_counts().nlargest(15)
        fig = px.bar(top_countries, title='Top 15 Países de Origen')
        stats = top_countries.to_frame('Conteo').reset_index()
        stats.columns = ['País', 'Conteo']
        
    elif plot_type == 'cancel_lead' and all(col in df.columns for col in ['lead_time', 'is_canceled']):
        fig = px.box(df, x='is_canceled', y='lead_time', 
                     title='Lead Time vs Cancelaciones',
                     labels={'is_canceled': 'Cancelada (0: No, 1: Sí)', 'lead_time': 'Lead Time (días)'})
        stats = df.groupby('is_canceled')['lead_time'].describe().reset_index()
        
    elif plot_type == 'corr_matrix':
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(corr, title='Matriz de Correlación')
            stats = corr.stack().reset_index()
            stats.columns = ['Variable 1', 'Variable 2', 'Correlación']
            stats = stats[stats['Variable 1'] != stats['Variable 2']]  # Eliminar diagonal
        else:
            fig = go.Figure()
            stats = pd.DataFrame({'Mensaje': ["No hay suficientes columnas numéricas para matriz de correlación"]})
    else:
        fig = go.Figure()
        stats = pd.DataFrame({'Mensaje': ["Datos no disponibles para este gráfico"]})
    
    # Asegurarnos de que stats siempre sea un DataFrame
    if not isinstance(stats, pd.DataFrame):
        stats = pd.DataFrame(stats)
    
    stats_table = dash_table.DataTable(
        data=stats.to_dict('records'),
        columns=[{'name': str(col), 'id': str(col)} for col in stats.columns],
        style_table={'overflowX': 'auto'}
    )
    
    return fig, stats_table

# Callback para minería de datos
@app.callback(
    [Output('stored-model-data', 'data'),
     Output('stored-model', 'data'),
     Output('model-metrics', 'children'),
     Output('confusion-matrix', 'figure'),
     Output('feature-importance', 'figure'),
     Output('predictions-sample', 'children')],
    [Input('train-model-btn', 'n_clicks'),
     Input('stored-transformed-data', 'data')]
)
def train_and_evaluate_model(n_clicks, data):
    if n_clicks is None or n_clicks == 0 or data is None:
        return None, None, "", go.Figure(), go.Figure(), ""
    
    df = pd.DataFrame.from_dict(data)
    X_train, X_test, y_train, y_test = prepare_model_data(df)
    
    if X_train is None:
        return None, None, "Error al preparar datos para el modelo", go.Figure(), go.Figure(), ""
    
    model, metrics, cm = train_evaluate_model(X_train, X_test, y_train, y_test)
    if model is None:
        return None, None, "Error al entrenar el modelo", go.Figure(), go.Figure(), ""
    
    # Guardar datos del modelo
    model_data = {
        'X_test': X_test.to_dict('records'),
        'y_test': y_test.to_dict(),
        'features': list(X_train.columns)
    }
    
    # Guardar modelo (serializado como string usando pickle)
    model_str = pickle.dumps(model).hex()
    
    # Métricas del modelo
    metrics_display = html.Div([
        html.P(f"Exactitud: {metrics['accuracy']:.2f}"),
        html.P(f"Precisión: {metrics['precision']:.2f}"),
        html.P(f"Sensibilidad: {metrics['recall']:.2f}"),
        html.P(f"F1-Score: {metrics['f1']:.2f}"),
        html.P(f"ROC-AUC: {metrics['roc_auc']:.2f}")
    ])
    
    # Matriz de confusión
    cm_fig = px.imshow(cm, text_auto=True,
                      labels=dict(x="Predicho", y="Real", color="Conteo"),
                      x=['No Cancelado', 'Cancelado'],
                      y=['No Cancelado', 'Cancelado'],
                      title="Matriz de Confusión")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    fi_fig = px.bar(feature_importance, x='importance', y='feature',
                   title='Top 10 Características más Importantes')
    
    # Predicciones de ejemplo
    sample_preds = X_test.copy()
    sample_preds['Actual'] = y_test
    sample_preds['Predicho'] = model.predict(X_test)
    sample_preds['Probabilidad'] = model.predict_proba(X_test)[:, 1]
    
    predictions_sample = html.Div([
        html.H5("Ejemplo de Predicciones (primeras 10)"),
        dash_table.DataTable(
            data=sample_preds.head(10).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in ['Actual', 'Predicho', 'Probabilidad']],
            style_table={'overflowX': 'auto'}
        )
    ])
    
    return model_data, model_str, metrics_display, cm_fig, fi_fig, predictions_sample

# Callback para toma de decisiones - VERSIÓN CORREGIDA
@app.callback(
    [Output('key-factors', 'children'),
     Output('strategic-recommendations', 'children'),
     Output('impact-factor-selector', 'options')],
    [Input('stored-model', 'data'),
     Input('stored-transformed-data', 'data')]
)
def generate_recommendations(model_str, transformed_data):
    if model_str is None or transformed_data is None:
        return "Entrene el modelo primero", "", []
    
    try:
        # Cargar modelo desde string serializado
        model = pickle.loads(bytes.fromhex(model_str))
        df = pd.DataFrame.from_dict(transformed_data)
        
        # Factores clave (basados en importancia de características)
        if hasattr(model, 'feature_importances_'):
            # Obtener nombres de características del modelo
            # (Asumiendo que el modelo fue entrenado con las columnas originales)
            feature_names = df.columns.tolist()
            
            # Verificar que tenemos suficientes importancias
            if len(model.feature_importances_) != len(feature_names):
                return "Las características del modelo no coinciden con los datos", "", []
            
            # Obtener las 5 características más importantes
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            key_factors = html.Ul([html.Li(f"{row['feature']}: {row['importance']:.3f}") 
                                 for _, row in feature_importance.iterrows()])
        else:
            key_factors = "No se pudo determinar la importancia de características"
        
        # Recomendaciones estratégicas
        recommendations = html.Ul([
            html.Li("Enfoque en reducir el lead time para reservas con alta probabilidad de cancelación"),
            html.Li("Ofrecer promociones para estancias de fin de semana (mayor ADR)"),
            html.Li("Monitorear reservas de ciertos países con mayor tasa de cancelación"),
            html.Li("Implementar políticas de depósito para reservas con mucho tiempo de anticipación")
        ])
        
        # Opciones para el selector de factores de impacto
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        options = [{'label': col, 'value': col} for col in numeric_cols]
        
        return key_factors, recommendations, options
    except Exception as e:
        print(f"Error en generación de recomendaciones: {e}")
        return f"Error: {str(e)}", "", []

# Callback para análisis de impacto
@app.callback(
    Output('impact-analysis-graph', 'figure'),
    [Input('impact-factor-selector', 'value'),
     Input('stored-transformed-data', 'data')]
)
def update_impact_analysis(selected_factor, data):
    if not selected_factor or data is None:
        return go.Figure()
    
    df = pd.DataFrame.from_dict(data)
    
    if selected_factor not in df.columns:
        return go.Figure()
    
    if 'is_canceled' in df.columns:
        fig = px.box(df, x='is_canceled', y=selected_factor,
                    title=f'Impacto de {selected_factor} en Cancelaciones',
                    labels={'is_canceled': 'Cancelada (0: No, 1: Sí)'})
    else:
        fig = px.histogram(df, x=selected_factor, 
                          title=f'Distribución de {selected_factor}')
    
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)