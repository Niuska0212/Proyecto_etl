import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_score, 
                            recall_score, f1_score, roc_auc_score)
from sklearn.cluster import KMeans
import json


# Inicializar  Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema ETL para Reservas Hoteleras"

# ============ Funciones de procesamiento ============

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
        
        # Eliminar columnas problemáticas
        cols_to_drop = ['agent', 'company', 'reservation_status_date']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        
        # Rellenar valores numéricos con medianas
        numeric_cols = ['adr', 'lead_time', 'stays_in_week_nights', 'stays_in_weekend_nights']
        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        return df
    except Exception as e:
        print(f"Error en limpieza: {e}")
        return None

def transform_data(df):
    if df is None:
        return None
    try:
        # 1. Creación de total_nights (como ya lo teníamos)
        if all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            if 'total_nights' not in df.columns:
                df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        
        # 2. Creación de arrival_date
        date_cols = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
        if all(col in df.columns for col in date_cols):
            # Convertir mes a numérico si es texto
            if pd.api.types.is_string_dtype(df['arrival_date_month']):
                month_map = {
                    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                }
                df['arrival_date_month'] = df['arrival_date_month'].map(month_map)
                
                # Si algún mes no se mapeó, intentar convertir directamente a número
                df['arrival_date_month'] = pd.to_numeric(df['arrival_date_month'], errors='coerce')
            
            # make sure que todos los componentes son numéricos
            df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce')
            
            # Filtrar filas con valores nulos en las columnas de fecha
            df = df.dropna(subset=date_cols)
            
            # Crear la fecha de llegada con verificación de errores
            try:
                df['arrival_date'] = pd.to_datetime(
                    df['arrival_date_year'].astype(str) + '-' +
                    df['arrival_date_month'].astype(str) + '-' +
                    df['arrival_date_day_of_month'].astype(str),
                    errors='coerce'
                )
                
                # Eliminar filas con fechas inválidas
                df = df[df['arrival_date'].notna()]
            except Exception as e:
                print(f"Error creando arrival_date: {e}")
        else:
            print(f"Faltan columnas de fecha. Se necesitan: {date_cols}")

        # 3. Codificación de variables categóricas
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                          if col != 'country' and df[col].nunique() < 20]
        
        for col in categorical_cols:
            df[col] = df[col].fillna('MISSING')
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            except:
                df[col] = df[col].astype('category').cat.codes

        # 4. Normalización de columnas numéricas con verificación
        numeric_cols = [col for col in ['adr', 'lead_time', 'total_nights'] 
                       if col in df.columns and col != 'country']
        scaler = MinMaxScaler()
        
        for col in numeric_cols:
            if col in df.columns:
                # Verificar si la columna existe y tiene valores válidos
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        continue
                
                # Rellenar nulos con la mediana antes de normalizar
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
                
                # Normalizar solo si hay valores no nulos
                if not df[col].empty:
                    try:
                        df[col] = scaler.fit_transform(df[[col]])
                    except ValueError as e:
                        print(f"Error normalizando {col}: {str(e)}")
                        continue
        required_cols = ['arrival_date', 'deposit_type', 'is_canceled']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Faltan columnas requeridas para el análisis: {missing_cols}")
            return None
        return df

    except Exception as e:
        print(f"Error en transformación: {e}")
        return None

def train_random_forest(df):
    # Entrenamiento de un modelo Random Forest para clasificación
    if df is None or 'is_canceled' not in df.columns:
        return None, None, None
    try:
        X = df.drop(columns=['is_canceled'])
        y = df['is_canceled']
        X = X.select_dtypes(include=['number'])  # Solo columnas numéricas
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=1000,   # Número de árboles
            max_depth=15,        # Profundidad máxima de los árboles
            min_samples_split=150,  # Mínimo de muestras para dividir un nodo
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluación
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        return model, metrics, cm
    except Exception as e:
        print(f"Error en modelado: {e}")
        return None, None, None

def generate_column_stats(df, title):
    # Genera gráficos de distribución para columnas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return html.Div("No hay columnas numéricas para mostrar")
    
    # Mostrar solo las primeras 6 columnas para no saturar
    show_cols = numeric_cols[:6]
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=show_cols)
    
    for i, col in enumerate(show_cols):
        row = (i // 3) + 1
        col_num = (i % 3) + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col),
            row=row, col=col_num
        )
    
    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=False
    )
    
    return dcc.Graph(figure=fig)

def generate_comparison_plots(original_df, transformed_df):
    # Genera gráficos comparativos lado a lado
    common_cols = list(set(original_df.columns) & set(transformed_df.columns))
    plots = []
    
    for col in common_cols[:6]:  # Mostrar máximo 6 columnas
        if original_df[col].dtype.kind in 'iufc' and transformed_df[col].dtype.kind in 'iufc':
            # Para columnas numéricas
            fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Original - {col}", f"Transformado - {col}"])
            
            fig.add_trace(
                go.Histogram(x=original_df[col], name='Original'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=transformed_df[col], name='Transformado'),
                row=1, col=2
            )
            
            fig.update_layout(height=300, showlegend=False)
            plots.append(dcc.Graph(figure=fig))
        
        elif original_df[col].dtype == 'object' and transformed_df[col].dtype == 'object':
            # Para columnas categóricas
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=[f"Original - {col}", f"Transformado - {col}"],
                              specs=[[{'type':'domain'}, {'type':'domain'}]])
            
            orig_counts = original_df[col].value_counts().nlargest(5)
            trans_counts = transformed_df[col].value_counts().nlargest(5)
            
            fig.add_trace(
                go.Pie(labels=orig_counts.index, values=orig_counts.values, name='Original'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(labels=trans_counts.index, values=trans_counts.values, name='Transformado'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            plots.append(dcc.Graph(figure=fig))
    
    return html.Div(plots)



# ============ Diseño de la Interfaz de la aplicación ============
app.layout = dbc.Container([
    html.H1("Sistema ETL para Reservas Hoteleras", className="mb-4 text-center"),
    dbc.Tabs([
        # Pestaña 1: Carga de datos
        dbc.Tab([
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.Span('Arrastra y suelta o '),
                        html.A('Selecciona un archivo')
                    ]),
                    style={
                        'width': '80%',
                        'height': '60px',
                        'display': 'flex',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px auto',
                        'alingnItemnts' : 'center'
                    },
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
                html.Div(id='data-preview-container')
            ], className='mt-4')
        ], label="Carga de Datos"),
        
        # Pestaña 2: ETL 
        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Datos Originales"),
                        html.Div(id='original-data-table'),
                        html.Div(id='column-stats-before-container', className='mt-3')
                    ], width=6),
                    dbc.Col([
                        html.H4("Datos Transformados"),
                        html.Div(id='transformed-data-table'),
                        html.Div(id='column-stats-after-container', className='mt-3')
                    ], width=6)
                ]),
                
                # Sección de controles ETL
                dbc.Card([
                    dbc.CardHeader("Controles de Transformación"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Operaciones de Limpieza", className="mb-3"),
                                dcc.Checklist(
                                    id='clean-options',
                                    options=[
                                        {'label': 'Rellenar valores nulos', 'value': 'fillna'},
                                        {'label': 'Eliminar duplicados', 'value': 'drop_duplicates'},
                                        {'label': 'Eliminar filas con >50% nulos', 'value': 'drop_na_rows'}
                                    ],
                                    value=['fillna', 'drop_duplicates']
                                ),
                                html.H5("Columnas a eliminar", className="mt-3"),
                                dcc.Dropdown(
                                    id='columns-to-drop',
                                    multi=True,
                                    placeholder="Selecciona columnas..."
                                )
                            ], width=4),
                            
                            dbc.Col([
                                html.H5("Transformaciones Numéricas", className="mb-3"),
                                dcc.Dropdown(
                                    id='numeric-columns',
                                    multi=True,
                                    placeholder="Selecciona columnas numéricas..."
                                ),
                                dbc.RadioItems(
                                    id='numeric-transformation',
                                    options=[
                                        {'label': 'Normalizar (MinMax)', 'value': 'minmax'},
                                        {'label': 'Estandarizar (Z-score)', 'value': 'standard'},
                                        {'label': 'Logaritmo', 'value': 'log'}
                                    ],
                                    value='minmax',
                                    className="mt-2"
                                ),
                                dbc.Button("Aplicar a selección", 
                                         id='apply-numeric-transform',
                                         color="primary",
                                         className="mt-2")
                            ], width=4),
                            
                            dbc.Col([
                                html.H5("Transformaciones de Texto", className="mb-3"),
                                dcc.Dropdown(
                                    id='text-columns',
                                    multi=True,
                                    placeholder="Selecciona columnas de texto..."
                                ),
                                dbc.RadioItems(
                                    id='text-transformation',
                                    options=[
                                        {'label': 'Codificar (LabelEncoder)', 'value': 'label_encode'},
                                        {'label': 'One-Hot Encoding', 'value': 'onehot'},
                                        {'label': 'Extraer características', 'value': 'extract'}
                                    ],
                                    value='label_encode',
                                    className="mt-2"
                                ),
                                dbc.Button("Aplicar a selección", 
                                         id='apply-text-transform',
                                         color="primary",
                                         className="mt-2")
                            ], width=4)
                        ]),
                        
                        # Botón para aplicar todas las transformaciones
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Aplicar Todas las Transformaciones", 
                                         id='apply-all-transforms',
                                         color="success",
                                         className="mt-3 w-100"),
                                width=12
                            )
                        ])
                    ])
                ], className='mt-3'),
                
                # Sección de visualización de cambios
                dbc.Card([
                    dbc.CardHeader("Detalle de Cambios"),
                    dbc.CardBody([
                        dcc.Graph(id='changes-heatmap'),
                        html.Div(id='changes-summary-table', className='mt-3')
                    ])
                ], className='mt-3'),
                
                # Comparación detallada
                dbc.Card([
                    dbc.CardHeader("Comparación Detallada Columna por Columna"),
                    dbc.CardBody([
                        html.Div(id='detailed-comparison')
                    ])
                ], className='mt-3')
            ], className='mt-4')
        ], label="ETL"),
        
        # Pestaña 3: Minería de Datos
        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Técnicas de Minería de Datos"),
                        dbc.Tabs([
                            dbc.Tab([
                                dbc.Button("Entrenar Modelo de Clasificación", 
                                        id='train-model-button', 
                                        className="mt-3 mb-4"),
                                html.Div(id='model-metrics'),
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id='confusion-matrix'), width=6),
                                    dbc.Col(dcc.Graph(id='feature-importance'), width=6)
                                ])
                            ], label="Clasificación (Random Forest)"),
                            
                            dbc.Tab([
                                dbc.Button("Aplicar Clustering", 
                                        id='cluster-button', 
                                        className="mt-3 mb-4"),
                                dcc.Dropdown(
                                    id='cluster-features',
                                    options=[
                                        {'label': 'Lead Time vs ADR', 'value': 'lt_adr'},
                                        {'label': 'Noches Totales vs ADR', 'value': 'nights_adr'},
                                        {'label': 'Lead Time vs Noches Totales', 'value': 'lt_nights'}
                                    ],
                                    value='lt_adr'
                                ),
                                dcc.Graph(id='cluster-plot'),
                                html.Div(id='cluster-summary')
                            ], label="Clustering (K-Means)")
                        ])
                    ], width=8),
                    
                    dbc.Col([
                        html.H4("Análisis Exploratorio"),
                        dcc.Dropdown(
                            id='eda-plot-type',
                            options=[
                                {'label': 'Distribución de Hoteles', 'value': 'hotel_dist'},
                                {'label': 'Distribución de ADR', 'value': 'adr_dist'},
                                {'label': 'Países de Origen', 'value': 'country'},
                                {'label': 'Cancelaciones vs Lead Time', 'value': 'cancel_lead'}
                            ],
                            value='hotel_dist'
                        ),
                        dcc.Graph(id='eda-plot'),
                        html.Div(id='eda-stats')
                    ], width=4)
                ])
            ], className='mt-4')
        ], label="Minería de Datos"),
        
        # Pestaña 4: Toma de Decisiones
        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Indicadores Clave"),
                        html.Div(id='kpi-cards')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Análisis para Toma de Decisiones"),
                        dcc.Dropdown(
                            id='decision-plot-type',
                            options=[
                                {'label': 'Cancelaciones por Temporada', 'value': 'season'},
                                {'label': 'Impacto de ADR en Cancelaciones', 'value': 'adr_impact'},
                                {'label': 'Efectividad de Depósitos', 'value': 'deposit'}
                            ],
                            value='season'
                        ),
                        dcc.Graph(id='decision-plot')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Recomendaciones"),
                        html.Div(id='recommendations')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Exportar Reporte", id='export-report', className="mt-3"),
                        width=12
                    )
                ]),
                dcc.Download(id="download-report")
            ], className='mt-4')
        ], label="Toma de Decisión")
    ])
], fluid=True)

# ============ Callbacks ============
# los callbacks son funciones que se ejecutan cuando se producen eventos en la aplicación
#ejemplo: cuando se carga un archivo, se actualiza un gráfico, etc.
#Tambien se pueden usar para actualizar el contenido de la aplicación en función de la interacción del usuario.


# Callback para cargar y previsualizar datos
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('data-preview-container', 'children'),
     Output('original-data-table', 'children'),
     Output('columns-to-drop', 'options'),
     Output('numeric-columns', 'options'),
     Output('text-columns', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_output(contents, filename, last_modified):
    if contents is None:
        return [html.Div("No se ha cargado ningún archivo."), None, None, [], [], []]
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Intentar diferentes codificaciones
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except:
                df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename:
            # Intentar leer como JSON normal
            try:
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
            except ValueError:
                # Intentar leer como JSON por líneas (NDJSON/JSONL)
                try:
                    df = pd.read_json(io.StringIO(decoded.decode('utf-8')), lines=True)
                except Exception:
                    # Último recurso: normalizar
                    df = pd.json_normalize(json.loads(decoded))
        else:
            return [html.Div("Formato de archivo no soportado"), None, None, [], [], []]
        
        # Validar que el dataframe no esté vacío
        if df.empty:
            return [html.Div("El archivo está vacío"), None, None, [], [], []]
            
    except Exception as e:
        return [html.Div([
            html.P("Error al cargar el archivo"),
            html.P(str(e), style={'color': 'red'})
        ]), None, None, [], [], []]
    
    global original_df, decision_df
    original_df = df.copy()

    # Limpiar y transformar solo para toma de decisiones
    df_decision = clean_data(df.copy())
    df_decision = transform_data(df_decision)
    decision_df = df_decision.copy() if df_decision is not None else None

    # Información básica
    file_info = html.Div([
        html.H5(filename),
        html.Hr(),
        html.P(f"Filas: {df.shape[0]}"),
        html.P(f"Columnas: {df.shape[1]}"),
        html.P("Primeras filas:")
    ])
    
    # Tabla de resumen estadístico
    stats_table = dash_table.DataTable(
        data=df.describe(include='all').reset_index().to_dict('records'),
        columns=[{'name': str(i), 'id': str(i)} for i in ['index'] + list(df.describe(include='all').columns)],
        page_size=10,
        style_table={'overflowX': 'auto', 'marginBottom': '20px'}
    )
    
    # Tabla de previsualización (10 filas)
    preview_table = dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10
    )
    
    # Tabla para la pestaña ETL (10 filas)
    original_table = dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
    
    # Mostrar resumen y previsualización juntos
    preview_content = html.Div([
        html.H4("Resumen Estadístico"),
        stats_table,
        html.H4("Vista Previa de Datos"),
        preview_table
    ])
    
    # Definir columnas protegidas
    protected_columns = [
        'lead_time', 
        'arrival_date', 
        'hotel', 
        'adr', 
        'total_nights', 
        'is_canceled',
        'stays_in_week_nights',
        'stays_in_weekend_nights'
    ]
    
    # Filtrar columnas disponibles excluyendo las protegidas
    available_columns = [col for col in df.columns if col not in protected_columns]
    
    # Opciones para los dropdowns de columnas
    columns = [{'label': col, 'value': col} for col in available_columns]
    
    # Columnas numéricas (excluyendo protegidas)
    numeric_cols = [
        {'label': col, 'value': col} 
        for col in df.select_dtypes(include=['number']).columns
        if col not in protected_columns
    ]
    
    # Columnas de texto (excluyendo protegidas)
    text_cols = [
        {'label': col, 'value': col} 
        for col in df.select_dtypes(include=['object']).columns
        if col not in protected_columns
    ]
    
    return [file_info, preview_content, original_table, columns, numeric_cols, text_cols]


# Callback para aplicar transformaciones
@app.callback(
    [Output('transformed-data-table', 'children'),
     Output('changes-heatmap', 'figure'),
     Output('changes-summary-table', 'children'),
     Output('column-stats-before-container', 'children'),
     Output('column-stats-after-container', 'children'),
     Output('detailed-comparison', 'children')],
    [Input('apply-all-transforms', 'n_clicks'),
     Input('apply-numeric-transform', 'n_clicks'),
     Input('apply-text-transform', 'n_clicks')],
    [State('clean-options', 'value'),
     State('columns-to-drop', 'value'),
     State('numeric-columns', 'value'),
     State('numeric-transformation', 'value'),
     State('text-columns', 'value'),
     State('text-transformation', 'value')],
    prevent_initial_call=True
)
def apply_transformations(all_clicks, num_clicks, text_clicks, 
                        clean_opts, drop_cols, num_cols, num_trans, 
                        text_cols, text_trans):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    if 'original_df' not in globals():
        return None, go.Figure(), html.Div(), None, None, None
    
    df = original_df.copy()
    changes_log = []
    
    # Lista de columnas protegidas que no se pueden eliminar
    protected_columns = ['lead_time', 'arrival_date', 'hotel', 'adr', 'total_nights', 'is_canceled']
    
    # Inicializar listas de columnas seguras
    text_cols_list = list(text_cols) if text_cols is not None else []
    num_cols_list = list(num_cols) if num_cols is not None else []
    
    # Aplicar limpieza básica
    if clean_opts and 'fillna' in clean_opts:
        null_counts_before = df.isnull().sum().sum()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col].fillna('Unknown', inplace=True)
        
        null_counts_after = df.isnull().sum().sum()
        if null_counts_before != null_counts_after:
            changes_log.append({
                'Operación': 'Rellenar nulos',
                'Detalle': f'Se rellenaron {null_counts_before - null_counts_after} valores nulos',
                'Columnas': ', '.join(numeric_cols.union(text_cols))
            })
    
    if clean_opts and 'drop_duplicates' in clean_opts:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        if before != after:
            changes_log.append({
                'Operación': 'Eliminar duplicados',
                'Detalle': f'Se eliminaron {before - after} filas duplicadas',
                'Columnas': 'Todas'
            })
    
    if clean_opts and 'drop_na_rows' in clean_opts:
        before = len(df)
        threshold = len(df.columns) // 2
        df.dropna(thresh=threshold, inplace=True)
        after = len(df)
        if before != after:
            changes_log.append({
                'Operación': 'Eliminar filas con nulos',
                'Detalle': f'Se eliminaron {before - after} filas con >50% nulos',
                'Columnas': 'Todas'
            })
    
    # Eliminar columnas seleccionadas (excluyendo las protegidas)
    if drop_cols:
        # Filtrar para excluir columnas protegidas
        drop_cols = [col for col in drop_cols if col not in protected_columns]
        
        before_cols = set(df.columns)
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        after_cols = set(df.columns)
        dropped = before_cols - after_cols
        if dropped:
            changes_log.append({
                'Operación': 'Eliminar columnas',
                'Detalle': f'Se eliminaron {len(dropped)} columnas',
                'Columnas': ', '.join(dropped)
            })
    
    # Aplicar transformaciones numéricas (con protección para columnas críticas)
    if num_cols_list and num_trans and ctx.triggered[0]['prop_id'] in ('apply-numeric-transform.n_clicks', 'apply-all-transforms.n_clicks'):
        for col in num_cols_list:
            if col in df.columns and col not in protected_columns:  # Solo aplicar a columnas no protegidas
                original_values = df[col].copy()
                
                if num_trans == 'minmax':
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    trans_name = 'Normalización MinMax'
                elif num_trans == 'standard':
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    trans_name = 'Estandarización Z-score'
                elif num_trans == 'log':
                    df[col] = np.log1p(df[col])
                    trans_name = 'Transformación Logarítmica'
                
                if not np.allclose(original_values, df[col], equal_nan=True):
                    changes_log.append({
                        'Operación': trans_name,
                        'Detalle': f'Aplicado a {col}',
                        'Columnas': col
                    })
    
    # Aplicar transformaciones de texto (con protección para columnas críticas)
    if text_cols_list and text_trans and ctx.triggered[0]['prop_id'] in ('apply-text-transform.n_clicks', 'apply-all-transforms.n_clicks'):
        for col in text_cols_list:
            if col in df.columns and col not in protected_columns:  # Solo aplicar a columnas no protegidas
                if text_trans == 'label_encode':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    trans_name = 'Codificación Label'
                elif text_trans == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df.drop(columns=[col], inplace=True)
                    df = pd.concat([df, dummies], axis=1)
                    trans_name = 'One-Hot Encoding'
                elif text_trans == 'extract':
                    df[f'{col}_length'] = df[col].str.len()
                    trans_name = 'Extracción de características'
                
                changes_log.append({
                    'Operación': trans_name,
                    'Detalle': f'Aplicado a {col}',
                    'Columnas': col
                })
    
    # Crear columna total_nights si no existe (protegida)
    if 'total_nights' not in df.columns and all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
        df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        changes_log.append({
            'Operación': 'Crear columna',
            'Detalle': 'Se creó total_nights como suma de stays_in_week_nights y stays_in_weekend_nights',
            'Columnas': 'total_nights'
        })
    
    # Actualizar el dataframe transformado global
    global transformed_df
    transformed_df = df.copy()
    
    # Generar tabla de datos transformados
    table = dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

    # Generar heatmap de cambios
    if 'original_df' in globals() and not original_df.empty and not df.empty:
        common_cols = list(set(original_df.columns) & set(df.columns))
        if common_cols:
            change_data = []
            for col in common_cols:
                if original_df[col].dtype.kind in 'iufc' and df[col].dtype.kind in 'iufc':
                    diff = np.mean(np.abs(original_df[col] - df[col])) / (original_df[col].max() - original_df[col].min() + 1e-10)
                    change_data.append(diff)
                else:
                    changed = not original_df[col].equals(df[col])
                    change_data.append(1 if changed else 0)
            
            heatmap_fig = px.imshow(
                [change_data],
                x=common_cols,
                labels=dict(x="Columna", y="", color="Cambio"),
                color_continuous_scale='Viridis',
                aspect="auto"
            )
            heatmap_fig.update_layout(title='Mapa de Calor de Cambios por Columna')
        else:
            heatmap_fig = go.Figure()
            heatmap_fig.update_layout(
                title='No hay columnas comunes para comparar',
                annotations=[dict(text="No se pueden comparar los datos", showarrow=False)]
            )
    else:
        heatmap_fig = go.Figure()
    
    # Tabla resumen de cambios
    if changes_log:
        changes_table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in ['Operación', 'Detalle', 'Columnas']],
            data=changes_log,
            style_table={'overflowX': 'auto'}
        )
    else:
        changes_table = html.Div("No se aplicaron transformaciones", className='text-muted')
    
    # Gráficos de distribución antes/después
    stats_before = generate_column_stats(original_df, "Distribución Original")
    stats_after = generate_column_stats(df, "Distribución Transformada")
    
    # Comparación detallada
    comparison = generate_comparison_plots(original_df, df)
    
    return table, heatmap_fig, changes_table, stats_before, stats_after, comparison

# Callback para exportar reporte
@app.callback(
    Output("download-report", "data"),
    [Input("export-report", "n_clicks")],
    prevent_initial_call=True
)
def export_report(n_clicks):
    if n_clicks is None or 'transformed_df' not in globals():
        return None
    
    df = transformed_df.copy()
    
    # Crear archivo Excel en memoria
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # 1. HOJA: DATOS COMPLETOS
            df.to_excel(writer, sheet_name='Datos Completos', index=False)
            
            # 2. HOJA: RESUMEN EJECUTIVO
            summary_data = {
                'Métrica': ['Total Reservas', 
                           'Tasa Cancelación' if 'is_canceled' in df.columns else None,
                           'ADR Promedio',
                           'Lead Time Promedio',
                           'Estancia Promedio'],
                'Valor': [
                    len(df),
                    df['is_canceled'].mean() if 'is_canceled' in df.columns else None,
                    df['adr'].mean() if 'adr' in df.columns else None,
                    df['lead_time'].mean() if 'lead_time' in df.columns else None,
                    df['total_nights'].mean() if 'total_nights' in df.columns else None
                ]
            }
            # Eliminar métricas no disponibles
            summary_data = {k: [v for v in vals if v is not None] 
                          for k, vals in summary_data.items() if any(vals)}
            
            pd.DataFrame(summary_data).to_excel(
                writer, 
                sheet_name='Resumen Ejecutivo', 
                index=False
            )
            
            # 3. HOJA: ANÁLISIS ESTADÍSTICO
            stats = df.describe(include='all').round(2)
            stats.to_excel(writer, sheet_name='Estadísticas')
            
            # 4. HOJA: CANCELACIONES
            if 'is_canceled' in df.columns:
                # Análisis básico
                cancel_data = df.groupby('is_canceled').size().to_frame('Count')
                
                # Análisis por segmentos
                cancel_by_segment = pd.DataFrame()
                if 'hotel' in df.columns:
                    cancel_by_segment['Por Hotel'] = df.groupby('hotel')['is_canceled'].mean()
                if 'arrival_date' in df.columns:
                    df['month'] = df['arrival_date'].dt.month
                    cancel_by_segment['Por Mes'] = df.groupby('month')['is_canceled'].mean()
                if 'market_segment' in df.columns:
                    cancel_by_segment['Por Segmento'] = df.groupby('market_segment')['is_canceled'].mean()
                
                # Exportar análisis
                cancel_data.to_excel(writer, sheet_name='Cancelaciones')
                if not cancel_by_segment.empty:
                    cancel_by_segment.to_excel(writer, sheet_name='Cancelaciones', startrow=len(cancel_data)+3)
            
            # 5. HOJA: ANÁLISIS ADICIONAL
            if 'adr' in df.columns and 'total_nights' in df.columns:
                revenue_analysis = df.assign(
                    estimated_revenue=lambda x: x['adr'] * x['total_nights']
                ).groupby('hotel').agg({
                    'estimated_revenue': 'sum',
                    'adr': 'mean',
                    'total_nights': 'mean'
                })
                revenue_analysis.to_excel(writer, sheet_name='Ingresos')
            
            # profesional
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            for sheet in writer.sheets:
                worksheet = writer.sheets[sheet]
                worksheet.freeze_panes(1, 0)  # Congelar encabezados
                worksheet.autofilter(0, 0, 0, df.shape[1]-1)  # Filtros
                
                # Formato de columnas según tipo de dato
                if sheet == 'Resumen Ejecutivo':
                    worksheet.set_column('B:B', 15, workbook.add_format({'num_format': '#,##0.00'}))
                
                # Aplicar formato a encabezados
                for col_num, value in enumerate(df.columns.values if sheet == 'Datos Completos' else []):
                    worksheet.write(0, col_num, value, header_format)
    
    except Exception as e:
        print(f"Error generando reporte: {e}")
        return None
    
    output.seek(0)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    return dcc.send_bytes(output.getvalue(), filename=f"Reporte_Hoteles_{timestamp}.xlsx")

# Callback para gráficos EDA
@app.callback(
    Output('eda-plot', 'figure'),
    [Input('eda-plot-type', 'value')]
)
def update_eda_plot(plot_type):
    if 'transformed_df' not in globals():
        return px.scatter(title="Carga y transforma los datos primero")
    
    df = transformed_df
    
    if plot_type == 'hotel_dist':
        fig = px.bar(df['hotel'].value_counts(), 
                    title='Distribución de Tipos de Hotel')
    elif plot_type == 'adr_dist':
        fig = px.histogram(df, x='adr', 
                          title='Distribución de ADR (Tarifa Diaria Promedio)')
    elif plot_type == 'country':
        if 'country' in df.columns:
            top_countries = df['country'].value_counts().head(15)
            fig = px.bar(top_countries, 
                        title='Top 15 Países de Origen de los Clientes',
                        labels={'value': 'Número de Reservas', 'index': 'País'})
        else:
            fig = px.scatter(title="Datos de país no disponibles")
            fig.add_annotation(text="La columna 'country' no está disponible en los datos",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    elif plot_type == 'cancel_lead' and all(col in df.columns for col in ['is_canceled', 'lead_time']):
        fig = px.box(df, x='is_canceled', y='lead_time',
                    title='Relación entre Lead Time y Cancelaciones',
                    labels={'is_canceled': 'Cancelado', 'lead_time': 'Días de Anticipación'})
    
    else:
        fig = px.scatter(title="Datos no disponibles para este gráfico")
    
    return fig

# Callback para el clustering
@app.callback(
    [Output('cluster-plot', 'figure'),
     Output('cluster-summary', 'children')],
    [Input('cluster-button', 'n_clicks'),
     Input('cluster-features', 'value')],
    prevent_initial_call=True
)
def apply_clustering(n_clicks, feature_pair):
    # Verificación inicial
    if n_clicks is None or 'transformed_df' not in globals():
        return go.Figure(), "Presiona el botón para aplicar clustering"
    
    df = transformed_df.copy()
    
    try:
        # 1. Verificación y creación de total_nights
        if 'total_nights' not in df.columns:
            night_cols = ['stays_in_week_nights', 'stays_in_weekend_nights']
            if all(col in df.columns for col in night_cols):
                # Asegurar que sean numéricas y manejar nulos
                df[night_cols] = df[night_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        
        # 2. Selección de características con validación completa
        feature_mapping = {
            'lt_adr': {
                'required': ['lead_time', 'adr'],
                'x': 'lead_time',
                'y': 'adr'
            },
            'nights_adr': {
                'required': ['total_nights', 'adr'],
                'x': 'total_nights',
                'y': 'adr'
            },
            'lt_nights': {
                'required': ['lead_time', 'total_nights'],
                'x': 'lead_time',
                'y': 'total_nights'
            }
        }
        
        if feature_pair not in feature_mapping:
            raise ValueError(f"Opción de clustering no válida: {feature_pair}")
        
        config = feature_mapping[feature_pair]
        missing_cols = [col for col in config['required'] if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {', '.join(missing_cols)}. Columnas disponibles: {', '.join(df.columns)}")
        
        features = df[config['required']].dropna()
        if len(features) == 0:
            raise ValueError("No hay datos válidos para clustering después de eliminar nulos")
        
        x_col, y_col = config['x'], config['y']
        
        # 3. Normalización con verificación
        scaler = MinMaxScaler()
        try:
            features_scaled = scaler.fit_transform(features)
        except ValueError as e:
            raise ValueError(f"Error al normalizar datos: {str(e)}. Valores mínimos/máximos: {features.min()}/{features.max()}")

        # 4. Aplicar K-Means con configuración adaptable
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
        except Exception as e:
            raise ValueError(f"Error en clustering: {str(e)}")

        # 5. Creación del gráfico con más información
        fig = go.Figure()
        colors = ['#636EFA', '#EF553B', '#00CC96'] 
        
        for cluster_id in range(3):
            cluster_data = features[clusters == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=colors[cluster_id]
                ),
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f'Clustering de {x_col.replace("_", " ")} vs {y_col.replace("_", " ")}',
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
            legend_title='Clusters',
            hovermode='closest'
        )
        
        # 6. Resumen estadístico mejorado
        features['cluster'] = clusters
        cluster_stats = features.groupby('cluster').agg({
            x_col: ['mean', 'median', 'std', 'count'],
            y_col: ['mean', 'median', 'std']
        }).reset_index()
        
        # Formatear nombres de columnas
        cluster_stats.columns = ['_'.join(col).strip('_') for col in cluster_stats.columns.values]
        
        # Interpretación automática de clusters
        interpretations = []
        for cluster_id in range(3):
            x_mean = cluster_stats.loc[cluster_id, f'{x_col}_mean']
            y_mean = cluster_stats.loc[cluster_id, f'{y_col}_mean']
            x_std = cluster_stats.loc[cluster_id, f'{x_col}_std']
            y_std = cluster_stats.loc[cluster_id, f'{y_col}_std']
            x_count = cluster_stats.loc[cluster_id, f'{x_col}_count']

            #comparar con la media global
            x_global_mean = features[x_col].mean()
            y_global_mean = features[y_col].mean()

            # Descripción simple para x_desc e y_desc
            x_desc = "mayor que la media" if x_mean > x_global_mean else "menor o igual a la media"
            y_desc = "mayor que la media" if y_mean > y_global_mean else "menor o igual a la media"
            
            #x_level = "bajos" if x_mean < features[x_col].quantile(0.33) else "medios" if x_mean < features[x_col].quantile(0.66) else "altos"
            #y_level = "bajos" if y_mean < features[y_col].quantile(0.33) else "medios" if y_mean < features[y_col].quantile(0.66) else "altos"
            
            interpretations.append(
                f"Cluster {cluster_id}: {x_count} reservas. "
                f"{x_col.replace('_', ' ').title()} promedio {x_desc} ({x_mean:.2f}, global: {x_global_mean:.2f}), "
                f"{y_col.replace('_', ' ').title()} promedio {y_desc} ({y_mean:.2f}, global: {y_global_mean:.2f})." # type: ignore
            )

            #interpretations.append(f"Cluster {cluster_id}: Valores {x_level} en {x_col.replace('_', ' ')} y {y_level} en {y_col.replace('_', ' ')}")
        
        summary = html.Div([
            html.H5("Resumen de Clusters:", className="mb-3"),
            dash_table.DataTable(
                columns=[{"name": col.replace("_", " ").title(), "id": col} 
                         for col in cluster_stats.columns],
                data=cluster_stats.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px'
                },
                page_size=10,
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ),
            html.Hr(),
            html.H5("Interpretación de clusters:", className="mt-3"),
            html.Ul([html.Li(interpretation) for interpretation in interpretations])
        ])
        
        return fig, summary
        
    except Exception as e:
        error_msg = str(e)
        error_fig = go.Figure()
        error_fig.update_layout(
            title="Error en clustering",
            annotations=[dict(
                text=error_msg,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )]
        )
        
        error_div = html.Div([
            html.H5("Error en clustering:", style={'color': 'red'}),
            html.P(error_msg),
            html.P("Columnas disponibles en los datos:"),
            html.Ul([html.Li(col) for col in df.columns.tolist()])
        ])
        
        return error_fig, error_div

# Callback para entrenar modelo y mostrar resultados
@app.callback(
    [Output('model-metrics', 'children'),
     Output('confusion-matrix', 'figure'),
     Output('feature-importance', 'figure')],
    [Input('train-model-button', 'n_clicks')],
    prevent_initial_call=True
)
def train_and_evaluate_model(n_clicks):
    if n_clicks is None or 'transformed_df' not in globals():
        return [html.Div("Presiona el botón para entrenar el modelo."), 
                go.Figure(), 
                go.Figure()]
    
    df = transformed_df.copy()
    
    try:
        model, metrics, cm = train_random_forest(df)
        
        if model is None:
            raise ValueError("No se pudo entrenar el modelo")
        
        # Mostrar métricas en tarjetas
        metrics_html = html.Div([
            html.H5("Métricas del Modelo:", className="mt-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Exactitud"),
                    dbc.CardBody(f"{metrics['accuracy']:.2%}")
                ], color="primary"), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Precisión"),
                    dbc.CardBody(f"{metrics['precision']:.2%}")
                ], color="success"), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Sensibilidad"),
                    dbc.CardBody(f"{metrics['recall']:.2%}")
                ], color="info"), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("F1-Score"),
                    dbc.CardBody(f"{metrics['f1']:.2%}")
                ], color="warning"), width=3)
            ])
        ])
        
        # Matriz de confusión
        cm_fig = px.imshow(cm, text_auto=True,
                          labels=dict(x="Predicho", y="Real"),
                          title="Matriz de Confusión")
        
        # Importancia de características
        X = df.drop(columns=['is_canceled']).select_dtypes(include=['number'])
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fi_fig = px.bar(feature_imp, x='Importance', y='Feature',
                       title='Top 10 Variables más Importantes',
                       orientation='h')
        
        # Explicación automática de hallazgos
        interpretation = html.Div([
            html.H5("Interpretación de Resultados:", className="mt-4"),
            html.Ul([
                html.Li(f"El modelo tiene una exactitud del {metrics['accuracy']:.1%}, lo que indica que predice correctamente la cancelación de reservas en ese porcentaje de los casos."),
                html.Li(f"La variable más importante para la predicción es: {feature_imp.iloc[0]['Feature']}"),
                html.Li("Una alta sensibilidad indica que el modelo identifica bien las cancelaciones reales, mientras que una alta precisión indica pocos falsos positivos."),
                html.Li("Revisa la matriz de confusión para ver los aciertos y errores del modelo.")
            ])
        ])

        return [html.Div([metrics_html, interpretation]), cm_fig, fi_fig]
        
    
    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error: {str(e)}", style={'color': 'red'}),
            html.P("Asegúrate de que los datos estén transformados correctamente.")
        ])
        return [error_msg, go.Figure(), go.Figure()]

# Callback para gráficos de toma de decisiones
@app.callback( 
    [Output('decision-plot', 'figure'),
     Output('kpi-cards', 'children'),
     Output('recommendations', 'children')],
    [Input('decision-plot-type', 'value')]
)
def update_decision_components(plot_type):
    if 'transformed_df' not in globals() or 'original_df' not in globals():
        return [px.scatter(title="Carga y transforma los datos primero"), None, None]
    
    df = decision_df
    df_orig = original_df
    # Verificación adicional de la columna arrival_date
    if 'arrival_date' not in df.columns:
        print("Columnas disponibles en transformed_df:", df.columns.tolist())
        if all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            print("Columnas de fecha presentes pero arrival_date no creada")
        else:
            print("Faltan columnas necesarias para crear arrival_date")
            
    # KPIs (como ya los tenías)
    cancelation_rate = df['is_canceled'].mean() * 100 if 'is_canceled' in df.columns else 0
    avg_adr = df_orig['adr'].mean() if 'adr' in df_orig.columns else 0
    avg_lead_time = df_orig['lead_time'].mean() if 'lead_time' in df_orig.columns else 0

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Tasa de Cancelación"),
            dbc.CardBody(f"{cancelation_rate:.1f}%")
        ], color="danger", inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("ADR Promedio"),
            dbc.CardBody(f"${avg_adr:.2f}")
        ], color="info", inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Lead Time Promedio"),
            dbc.CardBody(f"{avg_lead_time:.1f} días")
        ], color="warning", inverse=True), width=4)
    ], className="mb-4")
    
    # Gráficos con manejo robusto de datos faltantes
    if plot_type == 'season':
        if 'arrival_date' in df.columns:
            # Extraer mes y año de la fecha
            df['arrival_month'] = df['arrival_date'].dt.month
            df['arrival_year'] = df['arrival_date'].dt.year
            
            # Calcular cancelaciones por mes
            monthly_data = df.groupby(['arrival_year', 'arrival_month'])['is_canceled'].mean().reset_index()
            monthly_data['period'] = monthly_data['arrival_year'].astype(str) + '-' + monthly_data['arrival_month'].astype(str).str.zfill(2)
            
            fig = px.line(monthly_data, x='period', y='is_canceled',
                         title="Tasa de Cancelación por Mes",
                         labels={'period': 'Periodo', 'is_canceled': 'Tasa de Cancelación'})
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(xaxis_tickangle=-45)
        else:
            # Mensaje detallado sobre qué columnas faltan
            missing_cols = []
            for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']:
                if col not in df.columns:
                    missing_cols.append(col)
            
            fig = px.scatter(title="Datos de fecha no disponibles")
            error_msg = "No se puede realizar el análisis estacional porque:"
            if missing_cols:
                error_msg += f"\nFaltan columnas: {', '.join(missing_cols)}"
            else:
                error_msg += "\nLa columna arrival_date no se pudo crear (verifica los valores de fecha)"
            fig.add_annotation(text=error_msg,
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)
            
    elif plot_type == 'adr_impact' and 'adr' in df.columns and 'is_canceled' in df.columns:
        fig = px.box(df, x='is_canceled', y='adr',
                    title="Distribución de ADR por Estado de Reserva",
                    labels={'is_canceled': 'Cancelación', 'adr': 'ADR'})
        
    elif plot_type == 'deposit' and 'deposit_type' in df.columns and 'is_canceled' in df.columns:
        deposit_cancel = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
        fig = px.bar(deposit_cancel, x='deposit_type', y='is_canceled',
                    title="Tasa de Cancelación por Tipo de Depósito",
                    labels={'deposit_type': 'Tipo de Depósito', 'is_canceled': 'Tasa de Cancelación'})
        fig.update_yaxes(tickformat=".0%")
    else:
        fig = px.scatter(title="Datos no disponibles para este gráfico")
        fig.add_annotation(text="Las columnas requeridas para este análisis no están disponibles",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Recomendaciones basadas en análisis disponible
    recommendations = []
    
    if 'arrival_date' in df.columns and 'is_canceled' in df.columns:
        # Análisis estacional
        monthly_cancel = df.groupby(df['arrival_date'].dt.month)['is_canceled'].mean()
        worst_month = monthly_cancel.idxmax()
        recommendations.append(f"Atención especial en el mes {worst_month} (tasa de cancelación más alta)")
    
    if 'lead_time' in df.columns and 'is_canceled' in df.columns:
        # Análisis de lead time
        recommendations.append("Ofrecer descuentos para reservas con alto lead time (mayor probabilidad de cancelación)")
    
    if not recommendations:
        recommendations.append("Carga más datos o realiza transformaciones para obtener recomendaciones")
    
    recommendations_html = html.Div([
        html.H5("Recomendaciones:"),
        html.Ul([html.Li(item) for item in recommendations])
    ])
    
    return [fig, kpi_cards, recommendations_html]

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)