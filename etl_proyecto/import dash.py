import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import base64
import io
import json

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# _________________________________________________________________
# FUNCIONES DE PROCESAMIENTO (ETL)
# _________________________________________________________________

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
        return df
    except Exception as e:
        print(e)
        return None

def clean_data(df):
    """Limpieza de datos"""
    if df is None or df.empty:
        return None
        
    #los valores nulos
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    
    #convierte las fechas
    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
    
    #elimina columnas no necesarias
    cols_to_drop = ['agent', 'company']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    #rellena los valores numéricos faltantes
    numeric_cols = ['adr', 'lead_time', 'stays_in_week_nights', 'stays_in_weekend_nights']
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def transform_data(df):
    """Transformación de datos"""
    if df is None or df.empty:
        return None
    
    #one-hot encoding para hotel
    if 'hotel' in df.columns:
        df = pd.get_dummies(df, columns=['hotel'], prefix=['hotel'])
    
    #label encoding para meal
    if 'meal' in df.columns:
        le = LabelEncoder()
        df['meal'] = le.fit_transform(df['meal'])
    
    #normalización MinMax
    numeric_cols = ['adr', 'lead_time', 'stays_in_week_nights', 'stays_in_weekend_nights']
    scaler = MinMaxScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])
    
    #crea nuevas columnas
    required_cols = ['stays_in_weekend_nights', 'stays_in_week_nights', 'is_canceled', 'country']
    if all(col in df.columns for col in required_cols):
        df['is_weekend'] = (df['stays_in_weekend_nights'] > 0).astype(int)
        df['total_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        df['is_canceled'] = df['is_canceled'].apply(lambda x: 1 if x == 1 else 0)
        df['region'] = df['country'].apply(lambda x: 'Europe' if x in ['PRT', 'ESP', 'FRA'] else 'Other')
    
    return df

# __________________________________________________________________
# INTERFAZ
# __________________________________________________________________

app.layout = html.Div([
    dcc.Store(id='raw-data-store'),
    dcc.Store(id='clean-data-store'),
    dcc.Store(id='transformed-data-store'),
    
    html.H1('Sistema de Almacén de Datos Interactivo', style={'textAlign': 'center'}),
    
    dcc.Tabs(id='main-tabs', value='data-loading', children=[
        dcc.Tab(label='1. Carga de Datos', value='data-loading'),
        dcc.Tab(label='2. Proceso ETL', value='etl-process'),
        dcc.Tab(label='3. Análisis Exploratorio', value='eda-analysis'),
        dcc.Tab(label='4. Minería de Datos', value='data-mining'),
        dcc.Tab(label='5. Toma de Decisiones', value='decision-making'),
    ]),
    
    html.Div(id='tab-content')
])

#la pestaña de carga de datos
data_loading_layout = html.Div([
    html.H2('Carga de Datos'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arrastra y suelta o ',
            html.A('Selecciona un archivo (CSV, Excel, JSON)')
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
    html.Div(id='upload-message'),
    html.H3('Resumen de Datos Cargados'),
    html.Div(id='data-summary'),
    html.H3('Vista Previa de los Datos'),
    html.Div(id='data-preview')
])

#la pestaña ETL
etl_process_layout = html.Div([
    html.H2('Proceso ETL (Extracción, Transformación y Carga)'),
    dcc.Tabs(id='etl-tabs', value='original-data', children=[
        dcc.Tab(label='Datos Originales', value='original-data'),
        dcc.Tab(label='Datos Limpiados', value='cleaned-data'),
        dcc.Tab(label='Datos Transformados', value='transformed-data'),
    ]),
    html.Div(id='etl-tab-content'),
    html.H3('Comparación de Columnas'),
    html.Div(id='column-comparison')
])

#las pestañas no implementadas
not_implemented_layout = html.Div([
    html.H2('Aun noooo'),
    html.P(' :( )'),
])

# ________________________________________________________________
# CALLBACKS
# ________________________________________________________________

# callback para cambiar entre pestañas principales
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == 'data-loading':
        return data_loading_layout
    elif tab == 'etl-process':
        return etl_process_layout
    else:
        # para las otras pestañas
        return not_implemented_layout

# callback para cargar y procesar datos
@app.callback(
    [Output('raw-data-store', 'data'),
     Output('clean-data-store', 'data'),
     Output('transformed-data-store', 'data'),
     Output('upload-message', 'children'),
     Output('data-summary', 'children'),
     Output('data-preview', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_and_process_data(contents, filename):
    if contents is None:
        return [None, None, None, "", html.Div(), html.Div()]
    
    try:
        # cargar
        df_raw = load_data(contents, filename)
        if df_raw is None:
            return [None, None, None, 
                   html.Div('Error al cargar el archivo. Verifica el formato.', style={'color': 'red'}), 
                   html.Div(), html.Div()]
        
        # limpiar
        df_cleaned = clean_data(df_raw.copy())
        if df_cleaned is None:
            return [None, None, None, 
                   html.Div('Error al limpiar los datos.', style={'color': 'red'}), 
                   html.Div(), html.Div()]
        
        # transformar
        df_transformed = transform_data(df_cleaned.copy())
        if df_transformed is None:
            return [None, None, None, 
                   html.Div('Error al transformar los datos.', style={'color': 'red'}), 
                   html.Div(), html.Div()]
        
        # prepara datos para almacenamiento
        raw_data = df_raw.to_dict('records')
        cleaned_data = df_cleaned.to_dict('records')
        transformed_data = df_transformed.to_dict('records')
        
        # resumen
        summary = html.Div([
            html.P(f'Archivo: {filename}'),
            html.P(f'Número de registros: {len(df_raw)}'),
            html.P(f'Número de columnas: {len(df_raw.columns)}'),
            html.P('Columnas: ' + ', '.join(df_raw.columns.tolist()))
        ])
        
        # vista previa
        preview = dash_table.DataTable(
            data=df_raw.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df_raw.columns],
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
        
        message = html.Div(f'Archivo {filename} cargado exitosamente!', 
                         style={'color': 'green', 'fontWeight': 'bold'})
        
        return [raw_data, cleaned_data, transformed_data, message, summary, preview]
    
    except Exception as e:
        print(f"Error en procesamiento: {str(e)}")
        return [None, None, None, 
               html.Div(f'Error procesando el archivo: {str(e)}', style={'color': 'red'}), 
               html.Div(), html.Div()]

# Callback para mostrar datos ETL
@app.callback(
    [Output('etl-tab-content', 'children'),
     Output('column-comparison', 'children')],
    [Input('etl-tabs', 'value'),
     Input('raw-data-store', 'data'),
     Input('clean-data-store', 'data'),
     Input('transformed-data-store', 'data')]
)
def show_etl_data(tab, raw_data, clean_data, transformed_data):
    if raw_data is None:
        return html.Div('Por favor carga datos primero.'), html.Div()
    
    # Convertir datos a DataFrames
    df_raw = pd.DataFrame(raw_data)
    df_clean = pd.DataFrame(clean_data) if clean_data else None
    df_transformed = pd.DataFrame(transformed_data) if transformed_data else None
    
    # muestra datos según la pestaña
    if tab == 'original-data':
        df = df_raw
        title = 'Datos Originales'
    elif tab == 'cleaned-data' and df_clean is not None:
        df = df_clean
        title = 'Datos Limpiados'
    elif tab == 'transformed-data' and df_transformed is not None:
        df = df_transformed
        title = 'Datos Transformados'
    else:
        return html.Div('Datos no disponibles para esta etapa.'), html.Div()
    
    # muestra tabla de datos
    data_table = dash_table.DataTable(
        data=df.head(20).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
    
    # comparacion de columnas
    comparison = html.Div([
        html.H4('Comparación de Columnas'),
        html.P(f'Original: {len(df_raw.columns)} columnas'),
        html.P(f'Limpiado: {len(df_clean.columns) if df_clean is not None else "N/A"} columnas'),
        html.P(f'Transformado: {len(df_transformed.columns) if df_transformed is not None else "N/A"} columnas')
    ])
    
    return html.Div([html.H3(title), data_table]), comparison

if __name__ == '__main__':
    app.run(debug=True)