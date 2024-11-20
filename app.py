import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configurar la página para diseño ancho
st.set_page_config(
    page_title="Dashboard Interactivo: Análisis de Accesos y Velocidad",
    layout="wide",  # Diseño ancho
    initial_sidebar_state="expanded",
)

# CSS personalizado para reducir espacio y ajustar tamaños
st.markdown("""
    <style>
    h1 {
        font-size: 28px;  /* Título principal */
    }
    h2 {
        font-size: 22px;  /* Subtítulos */
    }
    h3 {
        font-size: 18px;  /* Títulos menores */
    }
    .block-container {
        padding-top: 1rem; /* Reducir espacio en blanco superior */
    }
    </style>
""", unsafe_allow_html=True)

# Cargar el dataset
df1 = pd.read_excel("data/Internet.xlsx")

# Preprocesamiento mejorado
# Convertir columnas críticas a numérico (manejo de errores)
for col in ['Velocidad (Mbps)', 'Accesos']:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')  # Convertir valores no numéricos a NaN

# Eliminar filas con valores NaN en las columnas críticas
df1 = df1.dropna(subset=['Velocidad (Mbps)', 'Accesos'])

# Filtrar valores negativos o extremadamente altos
# Velocidad (Mbps): Debe ser razonable (mayor que 0 y menor o igual a 10,000)
df1 = df1[(df1['Velocidad (Mbps)'] > 0) & (df1['Velocidad (Mbps)'] <= 10000)]

# Accesos: No deben ser negativos ni iguales a 0
df1 = df1[df1['Accesos'] > 0]


# Mapear provincias si son numéricas
if pd.api.types.is_numeric_dtype(df1['Provincia']):
    provincia_mapping = {
        0: "BUENOS AIRES", 1: "CABA", 2: "CATAMARCA", 3: "CHACO", 4: "CHUBUT",
        5: "CORDOBA", 6: "CORRIENTES", 7: "ENTRE RIOS", 8: "FORMOSA", 9: "JUJUY",
        10: "LA PAMPA", 11: "LA RIOJA", 12: "MENDOZA", 13: "MISIONES", 14: "NEUQUEN",
        15: "RIO NEGRO", 16: "SALTA", 17: "SAN JUAN", 18: "SAN LUIS", 19: "SANTA CRUZ",
        20: "SANTA FE", 21: "SANTIAGO DEL ESTERO", 22: "TIERRA DEL FUEGO", 23: "TUCUMAN"
    }
    df1['Provincia'] = df1['Provincia'].map(provincia_mapping)

# Transformaciones logarítmicas
df1['Accesos_log'] = df1['Accesos'].apply(lambda x: np.log(x + 1) if not pd.isnull(x) else np.nan)
df1['Velocidad_log'] = df1['Velocidad (Mbps)'].apply(lambda x: np.log(x + 1) if not pd.isnull(x) else np.nan)

# Coordenadas de las provincias
coordenadas_provincias = pd.DataFrame({
    'Provincia': ["BUENOS AIRES", "CABA", "CATAMARCA", "CHACO", "CHUBUT", "CORDOBA",
                  "CORRIENTES", "ENTRE RIOS", "FORMOSA", "JUJUY", "LA PAMPA", "LA RIOJA",
                  "MENDOZA", "MISIONES", "NEUQUEN", "RIO NEGRO", "SALTA", "SAN JUAN",
                  "SAN LUIS", "SANTA CRUZ", "SANTA FE", "SANTIAGO DEL ESTERO",
                  "TIERRA DEL FUEGO", "TUCUMAN"],
    'Latitud': [-34.61315, -34.607568, -28.46958, -27.46056, -43.29833, -31.41667,
                -27.4806, -32.4825, -26.17753, -24.18578, -36.61667, -29.41306,
                -32.89084, -27.36243, -38.95161, -39.03, -24.78212, -31.5375,
                -33.29501, -51.62261, -31.63306, -27.78241, -54.80191, -26.82414],
    'Longitud': [-58.37723, -58.437089, -65.77954, -58.99028, -65.10228, -64.18333,
                 -58.8341, -60.17639, -58.17814, -65.29712, -64.28333, -66.85578,
                 -68.84584, -55.89608, -68.0591, -67.5833, -65.42323, -68.53836,
                 -66.33601, -69.21813, -60.70001, -64.26612, -68.30295, -65.2226]
})

# Agregar coordenadas al DataFrame principal
df1 = pd.merge(df1, coordenadas_provincias, on='Provincia', how='left')

# Encabezado del dashboard
st.title("Dashboard Interactivo: Análisis de Accesos y Velocidad")
st.markdown("Explora la distribución, correlaciones y relaciones entre los datos del análisis de internet.")

# Layout en tres columnas
col1, col2, col3 = st.columns(3)

 # Primera columna: Filtros y distribución de Velocidad
with col1:
    st.subheader("Distribución de Velocidad (Mbps)")

    # Filtro por Provincia
    selected_provincia_vel = st.selectbox(
        "Selecciona una Provincia (Todos para visualizar todas):",
        options=["Todos"] + list(df1['Provincia'].unique()),
        key="provincia_vel"
    )
    
    # Filtro por Rango de Velocidad
    min_vel, max_vel = st.slider(
        "Rango de Velocidad (Mbps):",
        min_value=float(df1['Velocidad (Mbps)'].min()),
        max_value=float(df1['Velocidad (Mbps)'].max()),
        value=(float(df1['Velocidad (Mbps)'].min()), float(df1['Velocidad (Mbps)'].max())),
        step=0.5
    )

    # Filtrar los datos
    df1_vel = df1 if selected_provincia_vel == "Todos" else df1[df1['Provincia'] == selected_provincia_vel]
    df1_vel = df1_vel[(df1_vel['Velocidad (Mbps)'] >= min_vel) & (df1_vel['Velocidad (Mbps)'] <= max_vel)]

    # Graficar
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df1_vel['Velocidad (Mbps)'], kde=True, bins=20, color="skyblue", ax=ax)
    ax.set_title("Distribución de Velocidad")
    st.pyplot(fig)

    st.subheader("Distribución de Accesos (Log)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df1['Accesos_log'], kde=True, bins=20, color="green", ax=ax)
    ax.set_title("Distribución de Accesos (Log)")
    st.pyplot(fig)

# Segunda columna: Mapa interactivo
with col2:
    st.subheader("Mapa de Distribución Geográfica")
    selected_provincia = st.selectbox(
        "Selecciona una Provincia (Todos para visualizar todas):",
        options=["Todos"] + list(df1['Provincia'].unique())
    )
    apply_multiple_filter = st.checkbox("Filtrar por varias Provincias", key="geo")
    if apply_multiple_filter:
        selected_provincias = st.multiselect(
            "Selecciona Provincias:",
            options=df1['Provincia'].unique(),
            default=df1['Provincia'].unique()
        )
        df1_map = df1[df1['Provincia'].isin(selected_provincias)]
    else:
        df1_map = df1 if selected_provincia == "Todos" else df1[df1['Provincia'] == selected_provincia]

    df1_map = df1_map[df1_map['Accesos'] > 0]

    fig = px.scatter_mapbox(
        df1_map,
        lat="Latitud",
        lon="Longitud",
        size="Accesos",
        color="Velocidad (Mbps)",
        hover_name="Provincia",
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=50,
        zoom=3,
        mapbox_style="carto-positron",
        title="Mapa de Accesos"
    )
    st.plotly_chart(fig)

# Tercera columna: Correlación y relación entre variables
with col3:
    st.subheader("Relación Velocidad vs Accesos")
    selected_provincia_rel = st.selectbox(
        "Selecciona una Provincia (Todos para visualizar todas):",
        options=["Todos"] + list(df1['Provincia'].unique()),
        key="provincia_rel"
    )
    variables = {
        "Velocidad (Mbps)": "Velocidad (Mbps)",
        "Velocidad en Escala Logaritmica": "Velocidad_log",
        "Accesos": "Accesos",
        "Accesos en Escala Logaritmica": "Accesos_log"
    }
    selected_x = st.selectbox("Variable eje X:", options=list(variables.keys()), key="x_axis")
    selected_y = st.selectbox("Variable eje Y:", options=list(variables.keys()), key="y_axis")
    df1_rel = df1 if selected_provincia_rel == "Todos" else df1[df1['Provincia'] == selected_provincia_rel]

    fig = px.scatter(
        df1_rel,
        x=variables[selected_x],
        y=variables[selected_y],
        color="Provincia",
        title=f"Relación entre {selected_x} y {selected_y}",
        hover_data=["Localidad", "Partido"]
    )
    st.plotly_chart(fig)

    st.subheader("Matriz de Correlación")
    fig, ax = plt.subplots(figsize=(6, 4))
    correlation_matrix = df1[['Velocidad (Mbps)', 'Accesos']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.write("Máximo valor en Velocidad (Mbps):", df1['Velocidad (Mbps)'].max())
st.write("Mínimo valor en Velocidad (Mbps):", df1['Velocidad (Mbps)'].min())
