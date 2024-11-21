import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# Configurar la página para diseño ancho
st.set_page_config(
    page_title="Dashboard Interactivo: Análisis de Accesos y Velocidad",
    layout="wide",  # Diseño ancho
    initial_sidebar_state="expanded",
)

# CSS personalizado para ajustar el tamaño de los menús desplegables
st.markdown("""
    <style>
    /* Ajustar los selectboxes */
    div[data-baseweb="select"] {
        height: 15px;
        width: 150px; /* Cambia este valor para ajustar el ancho */
        margin: 0 auto;
    }
    div[data-baseweb="select"] > div {
        font-size: 10px; /* Ajustar el tamaño del texto */
    }
    </style>
""", unsafe_allow_html=True)


# CSS personalizado para reducir espacio y ajustar tamaños
st.markdown("""
    <style>
    h1 {
        font-size: 19px;  /* Título principal */
    }
    h2 {
        font-size: 17px;  /* Subtítulos */
    }
    h3 {
        font-size: 15px;  /* Títulos menores */
    }
    .block-container {
        padding-top: 1rem; /* Reducir espacio en blanco superior */
    }
    </style>
""", unsafe_allow_html=True)


# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas para los archivos Excel
EXCEL_PATH = os.path.join(BASE_DIR, "data", "Internet.xlsx")
# Función para cargar hojas específicas de un archivo Excel
def cargar_hoja(ruta, hoja):
    """Carga una hoja específica desde un archivo Excel."""
    try:
        return pd.read_excel(ruta, sheet_name=hoja)
    except Exception as e:
        st.error(f"No se pudo cargar la hoja '{hoja}' del archivo '{ruta}': {e}")
        return None

# Cargar las hojas necesarias
df1 = cargar_hoja(EXCEL_PATH, "Acc_vel_loc_sinrangos")  # Cambia "Hoja_Principal" por el nombre real de la hoja
df_ph = cargar_hoja(EXCEL_PATH, "Penetracion-hogares")
df_pp = cargar_hoja(EXCEL_PATH, "Penetración-poblacion")
df_v = cargar_hoja(EXCEL_PATH, "Velocidad % por prov")

# Renombramos las columnas 
df1.rename(columns={
    'Partido': 'Provincia',
    'Localidad': 'Partido',
    'link Indec': 'Localidad',
    'Velocidad (Mbps)': 'link Indec',
    'Provincia': 'Velocidad (Mbps)',
    'Accesos': 'Accesos'
}, inplace=True)  # inplace=True aplica los cambios directamente al DataFrame

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

# Asegurarse de que no haya valores -inf o NaN en Accesos_log
df1['Accesos_log'] = df1['Accesos_log'].replace([-np.inf, np.inf], np.nan)
df1 = df1.dropna(subset=['Accesos_log'])

tabs = st.tabs(["Accesos_vel_loc_sinrangos",  "KPIs"])

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

with tabs[0]:
    # Layout en dos columnas
    col1, col2 = st.columns(2)

    # Primera columna: Filtros y distribución de Velocidad
    with col1:
                # Dividir la columna en dos subcolumnas: una para los menús y otra para los gráficos
        col1_left, col1_right = st.columns([1, 2])  # La proporción 1:2 asigna más espacio al gráfico

        st.subheader("Relación Velocidad vs Accesos")
            
        with col1_left: 
                selected_provincia_rel = st.selectbox(
                 "Selecciona una Provincia (Todos para visualizar todas):",
                 options=["Todos"] + list(df1['Provincia'].unique()),
                 key="provincia_rel"
             )
                variables = {
                "Velocidad (Mbps)": "Velocidad (Mbps)",
                "VelocidadLOG": "Velocidad_log",
                "Accesos": "Accesos",
                "AccesosLOG": "Accesos_log"
             }
        
                selected_x = st.selectbox("Variable eje X:", options=list(variables.keys()), key="x_axis",index=1 )
                selected_y = st.selectbox("Variable eje Y:", options=list(variables.keys()), key="y_axis",index=3 )
                
        with col1_right:
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

      
    # Segunda columna: Mapa interactivo
    with col2:
        
        col1_left, col1_right = st.columns([1, 2])
        
        with col1_left:
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

        with col1_right:
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

    with col2:
        st.subheader("Distribución de Accesos (Log)")

            # Filtro por Rango de Accesos (Log)
        min_acc_log, max_acc_log = st.slider(
        "Rango de Accesos (Log):",
        min_value=float(df1['Accesos_log'].min()),
        max_value=float(df1['Accesos_log'].max()),
         value=(float(df1['Accesos_log'].min()), float(df1['Accesos_log'].max())),
         step=0.5,
         key="acc_log_slider"
            )
        
        # Calcular valores originales del rango
        min_original = np.exp(min_acc_log) - 1
        max_original = np.exp(max_acc_log) - 1

            # Mostrar rango original
        st.write(f"Rango de Accesos Originales: {min_original:.2f} - {max_original:.2f}")

            # Filtrar los datos
        df1_acc = df1[(df1['Accesos_log'] >= min_acc_log) & (df1['Accesos_log'] <= max_acc_log)]

            # Graficar
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df1_acc['Accesos_log'], kde=True, bins=20, color="green", ax=ax)
        ax.set_title("Distribución de Accesos (Log)")
        st.pyplot(fig)
            



with tabs[1]:
    
    # Layout en tres columnas
    col1, col2, col3 = st.columns(3)
    
    with col1: 
            # Título de la aplicación
            st.title("KPI 1: Aumentar un 2 porciento el acceso al servicio de internet para el próximo trimestre")

            # Cargar el archivo con los datos en `st.session_state`
            if "df1" not in st.session_state:
                st.session_state.df1 = pd.read_excel("data/Internet.xlsx", sheet_name="Penetracion-hogares")
                st.session_state.df1['Nuevo acceso'] = st.session_state.df1['Accesos por cada 100 hogares']  # Inicialmente igual a accesos actuales

            # Obtener el DataFrame de la sesión
            df1 = st.session_state.df1

            # Asegurarse de que las columnas necesarias existan
            if 'Provincia' in df1.columns and 'Accesos por cada 100 hogares' in df1.columns:
                # Menú desplegable para seleccionar una provincia
                st.subheader("Actualizar 'Nuevo acceso' por provincia")
                provincia_seleccionada = st.selectbox(
                    "Selecciona una provincia para actualizar su acceso:",
                    options=df1['Provincia'].unique()
                )

                # Input para ingresar el nuevo acceso de la provincia seleccionada
                nuevo_acceso_provincia = st.number_input(
                    f"Introduce el nuevo acceso para {provincia_seleccionada}:",
                    min_value=0.0,
                    value=float(df1.loc[df1['Provincia'] == provincia_seleccionada, 'Nuevo acceso'].mean()),
                    step=1.0
                )

                # Actualizar el valor en el DataFrame
                df1.loc[df1['Provincia'] == provincia_seleccionada, 'Nuevo acceso'] = nuevo_acceso_provincia
                st.success(f"Nuevo acceso actualizado para {provincia_seleccionada}: {nuevo_acceso_provincia}")

                # Calcular el incremento esperado
                df1['Incremento esperado'] = df1['Accesos por cada 100 hogares'] * 0.02

                # Calcular los accesos esperados
                df1['Accesos esperados'] = df1['Accesos por cada 100 hogares'] + df1['Incremento esperado']

                # Calcular KPI para todas las provincias
                df1['KPI (%)'] = ((df1['Nuevo acceso'] - df1['Accesos por cada 100 hogares']) / df1['Accesos por cada 100 hogares']) * 100

                # Agrupar por provincia
                df_grouped = df1.groupby('Provincia', as_index=False).agg({
                    'Accesos por cada 100 hogares': 'mean',
                    'Incremento esperado': 'mean',
                    'Accesos esperados': 'mean',
                    'Nuevo acceso': 'mean',
                    'KPI (%)': 'mean'
                })

                # Mostrar los datos procesados con las nuevas columnas
                st.subheader("Datos Procesados Agrupados por Provincia")
                st.write(df_grouped)

                # Gráfico interactivo para todas las provincias
                st.subheader("Gráfico de KPI por Provincia")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(df_grouped['Provincia'], df_grouped['KPI (%)'], color='skyblue')
                ax.set_title('KPI: Incremento en el acceso a Internet por cada 100 hogares (%)', fontsize=14)
                ax.set_xlabel('Provincia', fontsize=12)
                ax.set_ylabel('KPI (%)', fontsize=12)
                ax.set_xticklabels(df_grouped['Provincia'], rotation=45, ha='right')
                st.pyplot(fig)

                # Descargar los datos procesados
                st.subheader("Descarga de Datos Procesados Agrupados")
                csv = df_grouped.to_csv(index=False)
                st.download_button(
                    label="Descargar CSV Agrupado",
                    data=csv,
                    file_name='resultados_kpi_agrupados.csv',
                    mime='text/csv',
                )
            else:
                st.error("El archivo no contiene las columnas necesarias: 'Provincia' y 'Accesos por cada 100 hogares'.")

        
with col2: 
            # Título de la aplicación
            st.title("KPI: Incremento del 5% en Accesos por cada 100 Habitantes")

            # Filtrar los datos del año 2024 y trimestre 2
            df_pp = df_pp[(df_pp["Año"] == 2024) & (df_pp["Trimestre"] == 2)]

            # Calcular el incremento esperado (5%)
            df_pp["Incremento esperado"] = df_pp["Accesos por cada 100 hab"] * 0.05
            df_pp["Acceso esperado"] = df_pp["Accesos por cada 100 hab"] + df_pp["Incremento esperado"]

            # Inicializar columna para el incremento real si no está en `st.session_state`
            if "Incremento real" not in st.session_state:
                st.session_state["Incremento real"] = df_pp["Incremento esperado"].copy()

            df_pp["Incremento real"] = st.session_state["Incremento real"]

            # Actualizar manualmente el incremento real para cada provincia
            st.subheader("Actualizar Incremento Real")
            provincia_seleccionada = st.selectbox("Selecciona una provincia:", df_pp["Provincia"])

            nuevo_incremento_real = st.number_input(
                f"Introduce el incremento real para {provincia_seleccionada}:",
                min_value=0.0,
                value=float(df_pp.loc[df_pp["Provincia"] == provincia_seleccionada, "Incremento real"].values[0]),
                step=0.1
            )

            # Actualizar el valor en el DataFrame y en `st.session_state`
            df_pp.loc[df_pp["Provincia"] == provincia_seleccionada, "Incremento real"] = nuevo_incremento_real
            st.session_state["Incremento real"] = df_pp["Incremento real"]

            # Calcular KPI real en base al incremento real ingresado
            df_pp["KPI real (%)"] = (df_pp["Incremento real"] / df_pp["Accesos por cada 100 hab"]) * 100

            # Mostrar los datos procesados
            st.subheader("Datos Procesados")
            st.write(df_pp[["Provincia", "Accesos por cada 100 hab", "Incremento esperado", "Acceso esperado", "Incremento real", "KPI real (%)"]])

            # Gráfico: Incremento esperado vs. real
            st.subheader("Gráfico: Incremento Esperado vs Real")
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.35
            x = range(len(df_pp["Provincia"]))

            ax.bar(x, df_pp["Incremento esperado"], width=bar_width, label="Incremento esperado", color="skyblue")
            ax.bar(
                [p + bar_width for p in x],
                df_pp["Incremento real"],
                width=bar_width,
                label="Incremento real",
                color="orange"
            )

            ax.set_title("Comparación de Incrementos por Provincia (Trimestre 2, 2024)", fontsize=14)
            ax.set_xlabel("Provincias", fontsize=12)
            ax.set_ylabel("Incremento", fontsize=12)
            ax.set_xticks([p + bar_width / 2 for p in x])
            ax.set_xticklabels(df_pp["Provincia"], rotation=45, ha="right")
            ax.legend()
            st.pyplot(fig)

            # Gráfico: KPI real por provincia
            st.subheader("Gráfico: KPI Real (%) por Provincia")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df_pp["Provincia"], df_pp["KPI real (%)"], color="green")
            ax.set_title("KPI Real (%) por Provincia (Trimestre 2, 2024)", fontsize=14)
            ax.set_xlabel("Provincia", fontsize=12)
            ax.set_ylabel("KPI Real (%)", fontsize=12)
            ax.set_xticklabels(df_pp["Provincia"], rotation=45, ha="right")
            st.pyplot(fig)

            # Descargar los datos procesados
            st.subheader("Descarga de Datos Procesados")
            csv = df_pp.to_csv(index=False)
            st.download_button(
                label="Descargar CSV con KPI",
                data=csv,
                file_name="kpi_incremento_accesos.csv",
                mime="text/csv",
            )

with col3: 
            # Título de la aplicación
            st.title("KPI: Incremento del 10% en la Velocidad Promedio de Internet")

            # Filtrar los datos del año 2024 y trimestre 2
            df_v = df_v[(df_v["Año"] == 2024) & (df_v["Trimestre"] == 2)]

            # Calcular el incremento esperado (10%)
            df_v["Incremento esperado"] = df_v["Mbps (Media de bajada)"] * 0.10
            df_v["Velocidad esperada"] = df_v["Mbps (Media de bajada)"] + df_v["Incremento esperado"]

            # Inicializar columna para el incremento real si no está en `st.session_state`
            if "Incremento real velocidad" not in st.session_state:
                st.session_state["Incremento real velocidad"] = df_v["Incremento esperado"].copy()

            df_v["Incremento real"] = st.session_state["Incremento real velocidad"]

            # Actualizar manualmente el incremento real para cada provincia
            st.subheader("Actualizar Incremento Real")
            provincia_seleccionada = st.selectbox("Selecciona una provincia:", df_v["Provincia"], key="incremento_real")

            nuevo_incremento_real = st.number_input(
                f"Introduce el incremento real para {provincia_seleccionada}:",
                min_value=0.0,
                value=float(df_v.loc[df_v["Provincia"] == provincia_seleccionada, "Incremento real"].values[0]),
                step=0.1
            )

            # Actualizar el valor en el DataFrame y en `st.session_state`
            df_v.loc[df_v["Provincia"] == provincia_seleccionada, "Incremento real"] = nuevo_incremento_real
            st.session_state["Incremento real velocidad"] = df_v["Incremento real"]

            # Calcular KPI real en base al incremento real ingresado
            df_v["KPI real (%)"] = (df_v["Incremento real"] / df_v["Mbps (Media de bajada)"]) * 100

            # Mostrar los datos procesados
            st.subheader("Datos Procesados")
            st.write(df_v[["Provincia", "Mbps (Media de bajada)", "Incremento esperado", "Velocidad esperada", "Incremento real", "KPI real (%)"]])

            # Gráfico: Incremento esperado vs. real
            st.subheader("Gráfico: Incremento Esperado vs Real")
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.35
            x = range(len(df_v["Provincia"]))

            ax.bar(x, df_v["Incremento esperado"], width=bar_width, label="Incremento esperado", color="skyblue")
            ax.bar(
                [p + bar_width for p in x],
                df_v["Incremento real"],
                width=bar_width,
                label="Incremento real",
                color="orange"
            )

            ax.set_title("Comparación de Incrementos por Provincia (Velocidad, Trimestre 2, 2024)", fontsize=14)
            ax.set_xlabel("Provincias", fontsize=12)
            ax.set_ylabel("Incremento (Mbps)", fontsize=12)
            ax.set_xticks([p + bar_width / 2 for p in x])
            ax.set_xticklabels(df_v["Provincia"], rotation=45, ha="right")
            ax.legend()
            st.pyplot(fig)

            # Gráfico: KPI real por provincia
            st.subheader("Gráfico: KPI Real (%) por Provincia")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df_v["Provincia"], df_v["KPI real (%)"], color="green")
            ax.set_title("KPI Real (%) por Provincia (Velocidad, Trimestre 2, 2024)", fontsize=14)
            ax.set_xlabel("Provincia", fontsize=12)
            ax.set_ylabel("KPI Real (%)", fontsize=12)
            ax.set_xticklabels(df_v["Provincia"], rotation=45, ha="right")
            st.pyplot(fig)

            # Descargar los datos procesados
            st.subheader("Descarga de Datos Procesados")
            csv = df_v.to_csv(index=False)
            st.download_button(
                label="Descargar CSV con KPI",
                data=csv,
                file_name="kpi_incremento_velocidad.csv",
                mime="text/csv",
            )
