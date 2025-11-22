import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. CARGAR CEREBROS (Rutas Corregidas) ---
# Al poner solo el nombre, Python busca en la misma carpeta donde está el script.
# Esto es mucho más seguro y fácil.
try:
    model = joblib.load('modelo_corazon.pkl')
    scaler = joblib.load('escalador.pkl')
    model_columns = joblib.load('columnas.pkl')
except FileNotFoundError:
    st.error("⚠️ Error: No se encuentran los archivos .pkl. Asegúrate de que 'modelo_corazon.pkl', 'escalador.pkl' y 'columnas.pkl' estén en la misma carpeta que este archivo 'app.py'.")
    st.stop()

# --- 2. INTERFAZ GRÁFICA ---
st.set_page_config(page_title="CardioPredicción AI", page_icon="❤️")

st.title("Detector de Riesgo Cardíaco")
st.markdown("Ingrese los datos clínicos del paciente para obtener una predicción basada en IA.")

# Dividimos la pantalla en 3 columnas para que se vea ordenado
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad", 20, 100, 50)
    sex = st.selectbox("Sexo", ["M", "F"])
    # Limpiamos las opciones para que sea más fácil procesar
    chest_pain = st.selectbox("Tipo de Dolor de Pecho", ["ASY (Asintomático)", "NAP (No Anginoso)", "ATA (Angina Atípica)", "TA (Angina Típica)"])

with col2:
    resting_bp = st.number_input("Presión Arterial (Reposo)", 80, 200, 130)
    cholesterol = st.number_input("Colesterol", 80, 600, 220)
    fasting_bs = st.selectbox("Azúcar en Ayunas > 120 mg/dl?", ["No (0)", "Sí (1)"])

with col3:
    max_hr = st.number_input("Frecuencia Cardíaca Máx", 60, 220, 140)
    exercise_angina = st.selectbox("¿Dolor al Ejercitar?", ["No", "Sí"])
    oldpeak = st.number_input("Oldpeak (Depresión ST)", 0.0, 6.0, 0.0)
    st_slope = st.selectbox("Pendiente ST", ["Up (Subida)", "Flat (Plana)", "Down (Bajada)"])

# --- 3. LÓGICA DE PROCESAMIENTO ---
def procesar_datos():
    # Mapeo inverso de los selects
    sex_val = 'M' if sex == "M" else 'F'
    angina_val = 'Y' if exercise_angina == "Sí" else 'N'
    bs_val = 1 if "Sí" in fasting_bs else 0
    
    # Extraemos las siglas del string (ej: "ASY (Asintomático)" -> "ASY")
    cp_val = chest_pain.split(" ")[0]
    slope_val = st_slope.split(" ")[0]

    # Crear DF base con los nombres exactos que usaste al entrenar
    input_data = pd.DataFrame([{
        'Age': age, 
        'Sex': sex_val, 
        'ChestPainType': cp_val, 
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol, 
        'FastingBS': bs_val, 
        'RestingECG': 'Normal', # Asumido por defecto para simplificar la interfaz
        'MaxHR': max_hr, 
        'ExerciseAngina': angina_val, 
        'Oldpeak': oldpeak, 
        'ST_Slope': slope_val
    }])

    # Codificación (One-Hot)
    df_proc = pd.get_dummies(input_data)
    
    # TRUCO DE INGENIERÍA: Alinear columnas
    # Si al hacer get_dummies faltan columnas (ej. el paciente no tiene dolor TA),
    # reindexamos usando las columnas guardadas del modelo original.
    df_final = df_proc.reindex(columns=model_columns, fill_value=0)
    
    # Escalado (Usando el scaler guardado)
    cols_numeric = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df_final[cols_numeric] = scaler.transform(df_final[cols_numeric])
    
    return df_final

# --- 4. BOTÓN DE PREDICCIÓN ---
if st.button("Analizar Paciente 🔍", use_container_width=True):
    try:
        datos_listos = procesar_datos()
        prediction = model.predict(datos_listos)[0]
        probability = model.predict_proba(datos_listos)[0][1]

        st.divider()
        
        if prediction == 1:
            st.error(f"⚠️ ALERTA: El modelo detecta ENFERMEDAD CARDÍACA.")
            st.metric(label="Probabilidad de Riesgo", value=f"{probability:.2%}")
            st.write("Se recomienda derivar a cardiología inmediatamente.")
        else:
            st.success(f"✅ El modelo predice que el paciente está SANO.")
            st.metric(label="Probabilidad de Riesgo", value=f"{probability:.2%}")
            
    except Exception as e:
        st.error(f"Ocurrió un error en el procesamiento: {e}")