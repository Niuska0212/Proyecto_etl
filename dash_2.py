import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_score, 
                            recall_score, f1_score, roc_auc_score)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema ETL para Reservas Hoteleras"

# ============ Funciones de procesamiento mejoradas ============

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
        # Crear columna de fecha combinada para análisis estacional
        if all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            # Convertir nombre de mes a número si es necesario
            if df['arrival_date_month'].dtype == object:
                df['arrival_date_month'] = df['arrival_date_month'].apply(
                    lambda x: pd.to_datetime(x, format='%B').month if not str(x).isdigit() else int(x)
                )
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' +
                df['arrival_date_month'].astype(str) + '-' +
                df['arrival_date_day_of_month'].astype(str),
                errors='coerce'
            )
        
        # Crear nuevas columnas
        if all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        
        # Codificar variables categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Solo codificar si hay pocas categorías
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        # Normalizar columnas numéricas clave
        numeric_cols = ['adr', 'lead_time', 'total_nights']
        scaler = MinMaxScaler()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = scaler.fit_transform(df[[col]])
        
        return df
    except Exception as e:
        print(f"Error en transformación: {e}")
        return None

def train_random_forest(df):
    """Modelo predictivo mejorado"""
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
            n_estimators=25,   # Número de árboles
            max_depth=4,        # Profundidad máxima de los árboles
            min_samples_split=15,  # Mínimo de muestras para dividir un nodo
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

# ============ Diseño de la aplicación simplificado ============
app.layout = dbc.Container([
    html.H1("Sistema ETL para Reservas Hoteleras", className="mb-4 text-center"),
    dbc.Tabs([
        # Pestaña 1: Carga de datos
        dbc.Tab([
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastra y suelta o ',
                        html.A('Selecciona un archivo')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
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
                        html.Div(id='original-data-table')
                    ], width=6),
                    dbc.Col([
                        html.H4("Datos Transformados"),
                        html.Div(id='transformed-data-table'),
                        html.Div(id='etl-summary')  # Resumen arriba
                    ], width=6)
                ]),
                dbc.Button("Aplicar Transformación", id='transform-button', className="mt-3"),
            ], className='mt-4')
        ], label="ETL"),
        
        # Pestaña 3: Minería de Datos
        dbc.Tab([
            html.Div([
                dbc.Button("Entrenar Modelo", id='train-model-button', className="mt-3 mb-4"),  # <-- Mueve aquí el botón
                dbc.Row([
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
                        dcc.Graph(id='eda-plot')
                    ], width=6),
                    dbc.Col([
                        html.H4("Resultados del Modelo"),
                        html.Div(id='model-metrics'),
                        dcc.Graph(id='confusion-matrix'),
                        dcc.Graph(id='feature-importance')
                    ], width=6)
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
                ])
            ], className='mt-4')
        ], label="Toma de Decisión")
    ])
], fluid=True)

# ============ Callbacks ============

# Callback para cargar y previsualizar datos
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('data-preview-container', 'children'),
     Output('original-data-table', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return [html.Div("No se ha cargado ningún archivo."), None, None]
    
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
            return [html.Div("Formato de archivo no soportado"), None, None]
    except Exception as e:
        return [html.Div(f"Error al cargar el archivo: {str(e)}"), None, None]
    
    global original_df
    original_df = df.copy()
    
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
    
    return [file_info, preview_content, original_table]

# Callback unificado para transformación de datos
@app.callback(
    [Output('transformed-data-table', 'children'),
     Output('etl-summary', 'children'),
     Output('eda-plot', 'figure', allow_duplicate=True)],
    [Input('transform-button', 'n_clicks')],
    [State('original-data-table', 'children')],
    prevent_initial_call=True
)
def transform_data_callback(n_clicks, _):
    if n_clicks is None or 'original_df' not in globals():
        return None, "", px.scatter(title="Datos no cargados")
    
    try:
        df = original_df.copy()
        df_cleaned = clean_data(df)
        if df_cleaned is None:
            raise ValueError("Error en la limpieza de datos")
            
        df_transformed = transform_data(df_cleaned)
        if df_transformed is None:
            raise ValueError("Error en la transformación de datos")
        
        global transformed_df
        transformed_df = df_transformed.copy()
        
        # 1. Tabla de datos transformados
        table = dash_table.DataTable(
            data=df_transformed.head(10).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_transformed.columns],
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
        
        # 2. Resumen ETL
        df_orig = original_df
        stats = html.Div([
            html.H5("Resumen de Transformación"),
            html.P(f"Filas originales: {df_orig.shape[0]}, columnas originales: {df_orig.shape[1]}"),
            html.P(f"Filas transformadas: {df_transformed.shape[0]}, columnas transformadas: {df_transformed.shape[1]}")
        ])

        # Comparación de columnas
        orig_cols = set(df_orig.columns)
        trans_cols = set(df_transformed.columns)
        added = trans_cols - orig_cols
        removed = orig_cols - trans_cols

        comparison = html.Div([
            html.P("Nuevas columnas creadas:"),
            html.Ul([html.Li(col) for col in added] if added else [html.Li("Ninguna")]),
            html.P("Columnas eliminadas:"),
            html.Ul([html.Li(col) for col in removed] if removed else [html.Li("Ninguna")])
        ])

        summary = html.Div([stats, comparison])
        
        # 3. Gráfico EDA inicial
        eda_fig = px.histogram(df_transformed, x='hotel', title='Distribución de Hoteles (Post-Transformación)')
        
        return table, summary, eda_fig
        
    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error: {str(e)}", style={'color': 'red'}),
            html.P("Verifica que tus datos tengan las columnas esperadas.")
        ])
        return error_msg, html.Div(), px.scatter(title=f"Error: {str(e)}")

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
        top_countries = df['country'].value_counts().head(15)
        fig = px.bar(top_countries, 
                    title='Top 15 Países de Origen de los Clientes')
    elif plot_type == 'cancel_lead':
        fig = px.box(df, x='is_canceled', y='lead_time',
                    title='Relación entre Lead Time y Cancelaciones')
    
    return fig

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
        
        return [metrics_html, cm_fig, fi_fig]
    
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
    
    df = transformed_df
    df_orig = original_df

    # KPIs: usar valores originales para lead_time y adr
    cancelation_rate = df['is_canceled'].mean() * 100
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
    
    # Gráficos
    if plot_type == 'season':
        # Agrupar por mes para ver estacionalidad
        if 'arrival_date' in df.columns:
            df['arrival_month'] = pd.to_datetime(df['arrival_date']).dt.month
            monthly_cancel = df.groupby('arrival_month')['is_canceled'].mean().reset_index()
            fig = px.line(monthly_cancel, x='arrival_month', y='is_canceled',
                         title="Tasa de Cancelación por Mes")
            fig.update_yaxes(tickformat=".0%")
        else:
            fig = px.scatter(title="No hay datos de fecha para análisis estacional")
            
    elif plot_type == 'adr_impact':
        fig = px.box(df, x='is_canceled', y='adr',
                    title="Distribución de ADR por Estado de Reserva")
        
    elif plot_type == 'deposit' and 'deposit_type' in df.columns:
        deposit_cancel = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
        fig = px.bar(deposit_cancel, x='deposit_type', y='is_canceled',
                    title="Tasa de Cancelación por Tipo de Depósito")
        fig.update_yaxes(tickformat=".0%")
    else:
        fig = px.scatter(title="Datos no disponibles para este gráfico")
    
    # Recomendaciones basadas en análisis
    recommendations = html.Div([
        html.H5("Recomendaciones:"),
        html.Ul([
            html.Li("Ofrecer descuentos o beneficios para reservas con alto lead time para reducir cancelaciones"),
            html.Li("Revisar política de depósitos, ya que ciertos tipos tienen mayor tasa de cancelación"),
            html.Li("Ajustar precios (ADR) en temporadas con alta cancelación")
        ])
    ])
    
    return [fig, kpi_cards, recommendations]

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)