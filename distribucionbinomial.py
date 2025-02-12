import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Función para formato científico en valores pequeños
def formato_cientifico(valor, umbral=1e-4):
    """Formatea números menores al umbral a notación científica"""
    if valor < umbral and valor != 0:
        return f"{valor:.2e}".replace("e-0", "e-")
    return f"{valor:.4f}"

# Configuración de la página
st.set_page_config(page_title="Calculadora Binomial", layout="wide")
st.title("📊 Calculadora de Distribución Binomial")
st.markdown("---")

# Sidebar para parámetros
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
        acum = sum(probabilidad_binomial(n, p, k) for k in range(x+1))
        datos.append([x, prob, acum])
    return pd.DataFrame(datos, columns=["x", "P(X=x)", "P(X≤x)"])

# Cálculos principales
df = calcular_tabla(n, p)
esperanza = n * p
varianza = n * p * (1 - p)
desviacion = math.sqrt(varianza)

# Métricas principales
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Media (μ)", f"{esperanza:.2f}")
with col2:
    st.metric("Varianza (σ²)", f"{varianza:.2f}")
with col3:
    st.metric("Desviación estándar (σ)", f"{desviacion:.2f}")

st.markdown("---")

# Gráficos interactivos
col_graf1, col_graf2 = st.columns(2)

# Gráfico de densidad de probabilidad
with col_graf1:
    st.subheader("Función de Densidad de Probabilidad")
    
    # Preparar datos formateados
    df["P(X=x)_fmt"] = df["P(X=x)"].apply(formato_cientifico)
    
    fig_densidad = go.Figure()
    fig_densidad.add_trace(go.Bar(
        x=df["x"],
        y=df["P(X=x)"],
        marker_color="#1f77b4",
        name="P(X=x)",
        hovertemplate="<b>x</b>: %{x}<br><b>Probabilidad</b>: %{customdata}<extra></extra>",
        customdata=df["P(X=x)_fmt"]
    ))
    
    # Líneas de referencia
    for valor, color, nombre in [
        (esperanza, "red", "Media"),
        (esperanza - desviacion, "green", "Media - σ"),
        (esperanza + desviacion, "blue", "Media + σ")
    ]:
        y_val = np.interp(valor, df["x"], df["P(X=x)"])
        fig_densidad.add_shape(
            type="line",
            x0=valor, y0=0, x1=valor, y1=y_val,
            line=dict(color=color, width=1, dash="dot")
        )
        fig_densidad.add_trace(go.Scatter(
            x=[valor], y=[y_val],
            mode="markers",
            marker=dict(color=color, size=8),
            name=nombre,
            hovertemplate=f"<b>{nombre}</b>: {valor:.2f}<extra></extra>"
        ))
    
    fig_densidad.update_layout(
        xaxis_title="Número de éxitos (x)",
        yaxis_title="Probabilidad",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False
    )
    st.plotly_chart(fig_densidad, use_container_width=True)

# Gráfico de distribución acumulada
with col_graf2:
    st.subheader("Función de Distribución Acumulada")
    
    # Preparar datos formateados
    df["P(X≤x)_fmt"] = df["P(X≤x)"].apply(formato_cientifico)
    
    fig_acumulada = go.Figure()
    fig_acumulada.add_trace(go.Scatter(
        x=df["x"],
        y=df["P(X≤x)"],
        mode="lines+markers",
        line_shape="hv",
        marker=dict(color="#ff7f0e", size=6),
        line=dict(color="#ff7f0e", width=2),
        name="P(X≤x)",
        hovertemplate="<b>x</b>: %{x}<br><b>Prob. Acumulada</b>: %{customdata}<extra></extra>",
        customdata=df["P(X≤x)_fmt"]
    ))
    
    fig_acumulada.update_layout(
        xaxis_title="Número de éxitos (x)",
        yaxis_title="Probabilidad Acumulada",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_acumulada, use_container_width=True)

# Tabla de probabilidades con formato
st.markdown("---")
st.subheader("📈 Tabla de Probabilidades")

def formatear_fila(valor):
    if isinstance(valor, float):
        return formato_cientifico(valor)
    return valor

st.dataframe(
    df[["x", "P(X=x)", "P(X≤x)"]].style.format({
        "P(X=x)": formatear_fila,
        "P(X≤x)": formatear_fila
    }),
    height=400,
    use_container_width=True
)

# Cálculo de rango personalizado
st.markdown("---")
st.subheader("🔍 Calculadora de Rango")

col4, col5 = st.columns(2)
with col4:
    min_x = st.number_input("Mínimo de éxitos", min_value=0, max_value=n, value=0)
with col5:
    max_x = st.number_input("Máximo de éxitos", min_value=0, max_value=n, value=n)

prob_rango = sum(probabilidad_binomial(n, p, x) for x in range(min_x, max_x+1))
st.metric(
    label=f"P({min_x} ≤ X ≤ {max_x})",
    value=formato_cientifico(prob_rango)
)

st.markdown("---")
st.caption("✨ Creado con Streamlit y Plotly | Notación científica para valores < 0.0001")

# Validación para distribución degenerada
if p in (0.0, 1.0):
    st.warning("⚠️ Distribución degenerada: Todos los resultados son idénticos.")
