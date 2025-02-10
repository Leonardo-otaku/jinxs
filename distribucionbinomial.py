import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Calculadora Binomial", layout="wide")
st.title("📊 Calculadora de Distribución Binomial")
st.markdown("---")

# Sidebar para entradas
with st.sidebar:
    st.header("⚙️ Parámetros")
    n = st.number_input("Número de ensayos (n)", min_value=1, value=10, step=1)
    p = st.slider("Probabilidad de éxito (p)", 0.0, 1.0, 0.5)
    q = 1 - p
    st.markdown(f"**Probabilidad de fracaso (q):** {q:.2f}")

# Funciones de cálculo
def probabilidad_binomial(n, p, x):
    return math.comb(n, x) * (p**x) * ((1-p)**(n-x))

def calcular_tabla(n, p):
    datos = []
    for x in range(n+1):
        prob = probabilidad_binomial(n, p, x)
        datos.append([x, prob, sum(probabilidad_binomial(n, p, k) for k in range(x+1))])
    return pd.DataFrame(datos, columns=["x", "P(X=x)", "P(X≤x)"])

# Cálculos principales
df = calcular_tabla(n, p)
esperanza = n * p
varianza = n * p * (1 - p)
desviacion = math.sqrt(varianza)

# Mostrar métricas principales
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Media (μ)", f"{esperanza:.2f}")
with col2:
    st.metric("Varianza (σ²)", f"{varianza:.2f}")
with col3:
    st.metric("Desviación estándar (σ)", f"{desviacion:.2f}")

st.markdown("---")

# Sección de gráficos interactivos con Plotly
col_graf1, col_graf2 = st.columns(2)

# Gráfico interactivo de la Función de Densidad de Probabilidad
with col_graf1:
    st.subheader("Función de Densidad de Probabilidad (Interactiva)")
    
    # Crear figura interactiva
    fig_density = go.Figure()
    
    # Barra de la densidad
    fig_density.add_trace(go.Bar(
        x=df["x"],
        y=df["P(X=x)"],
        marker_color="#1f77b4",
        name="P(X=x)",
        hovertemplate="x: %{x}<br>P(X=x): %{y:.4f}<extra></extra>"
    ))
    
    # Calcular valores en y para la media y para media ± σ usando interpolación
    x_vals = df["x"].to_numpy()
    y_vals = df["P(X=x)"].to_numpy()
    y_media = np.interp(esperanza, x_vals, y_vals)
    y_left = np.interp(esperanza - desviacion, x_vals, y_vals)
    y_right = np.interp(esperanza + desviacion, x_vals, y_vals)
    
    # Agregar líneas verticales delgadas y puntos pequeños al final para cada indicador
    # Para la Media:
    fig_density.add_shape(
        type="line",
        x0=esperanza,
        y0=0,
        x1=esperanza,
        y1=y_media,
        line=dict(color="red", width=1),
    )
    fig_density.add_trace(go.Scatter(
        x=[esperanza],
        y=[y_media],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Media",
        hovertemplate="Media: %{x:.2f}<br>P: %{y:.4f}<extra></extra>"
    ))
    
    # Para Media - σ:
    fig_density.add_shape(
        type="line",
        x0=esperanza - desviacion,
        y0=0,
        x1=esperanza - desviacion,
        y1=y_left,
        line=dict(color="green", width=1),
    )
    fig_density.add_trace(go.Scatter(
        x=[esperanza - desviacion],
        y=[y_left],
        mode="markers",
        marker=dict(color="green", size=8),
        name="Media - σ",
        hovertemplate="Media - σ: %{x:.2f}<br>P: %{y:.4f}<extra></extra>"
    ))
    
    # Para Media + σ:
    fig_density.add_shape(
        type="line",
        x0=esperanza + desviacion,
        y0=0,
        x1=esperanza + desviacion,
        y1=y_right,
        line=dict(color="blue", width=1),
    )
    fig_density.add_trace(go.Scatter(
        x=[esperanza + desviacion],
        y=[y_right],
        mode="markers",
        marker=dict(color="blue", size=8),
        name="Media + σ",
        hovertemplate="Media + σ: %{x:.2f}<br>P: %{y:.4f}<extra></extra>"
    ))
    
    # Agregar anotación con la varianza y la desviación
    fig_density.add_annotation(
        x=1,
        y=1,
        xref="paper",
        yref="paper",
        text=f"Varianza = {varianza:.2f}<br>Desviación = {desviacion:.2f}",
        showarrow=False,
        align="right",
        bordercolor="black",
        borderwidth=1,
        bgcolor="rgb(0, 0, 0)",
        opacity=0.8
    )
    
    # Actualizar diseño de la figura
    fig_density.update_layout(
        title="Función de Densidad de Probabilidad",
        xaxis_title="Número de éxitos (x)",
        yaxis_title="Probabilidad",
        xaxis=dict(tickmode="linear", dtick=1),
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_density, use_container_width=True)

# Gráfico interactivo de la Función de Distribución Acumulada
with col_graf2:
    st.subheader("Función de Distribución Acumulada (Interactiva)")
    
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=df["x"],
        y=df["P(X≤x)"],
        mode="lines+markers",
        line_shape="hv",  # crea un gráfico tipo escalón
        marker=dict(color="#ff7f0e", size=8),
        line=dict(color="#ff7f0e", width=2),
        name="P(X≤x)",
        hovertemplate="x: %{x}<br>P(X≤x): %{y:.4f}<extra></extra>"
    ))
    
    fig_cumulative.update_layout(
        title="Función de Distribución Acumulada",
        xaxis_title="Número de éxitos (x)",
        yaxis_title="Probabilidad acumulada",
        xaxis=dict(tickmode="linear", dtick=1),
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)

# Tabla interactiva
st.markdown("---")
st.subheader("Tabla de Probabilidades")
st.dataframe(df.style.format({"P(X=x)": "{:.4f}", "P(X≤x)": "{:.4f}"}), height=400)

# Cálculo de probabilidades personalizadas
st.markdown("---")
st.subheader("🔍 Cálculo de probabilidades específicas")

col4, col5 = st.columns(2)
with col4:
    x_min = st.number_input("Mínimo de éxitos", min_value=0, max_value=n, value=0)
with col5:
    x_max = st.number_input("Máximo de éxitos", min_value=0, max_value=n, value=n)

prob_acumulada = sum(probabilidad_binomial(n, p, x) for x in range(x_min, x_max+1))
st.metric(f"P({x_min} ≤ X ≤ {x_max})", f"{prob_acumulada:.4f}")

st.markdown("---")
st.caption("Creado por Muerto. Usa las flechas ← para ajustar parámetros")
