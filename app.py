import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency
from pathlib import Path

st.set_page_config(page_title="AED EPH", layout="wide")

# ---------- Configuración ----------
CH08_LABELS = {
    1: "Obra social",
    2: "Prepaga/mutual/emergencia",
    3: "Plan/seguro público",
    4: "Sin cobertura",
    12: "Obra social + prepaga",
    13: "Obra social + público",
    23: "Prepaga + público",
    123: "Obra social + prepaga + público",
}
CATEGORICAS = ["REGION", "IV1", "IV12_3", "II7", "V5_01", "V15",
               "CH04", "CH07", "NIVEL_ED", "ESTADO", "EMPLEO", "PP04A"]
NUMERICAS = ["CH06", "IX_MEN10", "IPCF", "P47T"]

# ---------- Datos ----------

@st.cache_data
def cargar():
    df = pd.read_parquet(Path(__file__).parent / "eph_limpio.parquet")
    df["COBERTURA"] = df["CH08"].map(CH08_LABELS)
    return df

df_full = cargar()

# ---------- Sidebar ----------
st.sidebar.header("Filtros")
regiones = st.sidebar.multiselect(
    "Región", sorted(df_full["REGION"].dropna().unique()),
    default=sorted(df_full["REGION"].dropna().unique()),
)
ponderar = st.sidebar.toggle("Ponderar por PONDERA", value=True)

df = df_full[df_full["REGION"].isin(regiones)].copy()
df["_w"] = df["PONDERA"] if ponderar else 1

st.title("AED - Cobertura de salud (EPH 1T 2026)")
tabs = st.tabs(["Resumen", "Variable respuesta", "Numéricas",
                "Categóricas vs Y", "Asociación", "Región"])

# ---------- 1. Resumen ----------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", df_full.shape[1])
    c3.metric("Población representada", f"{int(df['PONDERA'].sum()):,}")
    st.subheader("% de nulos por variable")
    nulos = (df_full.isna().mean() * 100).round(2).rename("% nulos").sort_values(ascending=False)
    st.dataframe(nulos, use_container_width=True)
    st.download_button("Descargar dataset limpio (CSV)",
                       df_full.to_csv(index=False).encode("utf-8"),
                       "eph_limpio.csv", "text/csv")

# ---------- 2. Variable respuesta ----------
with tabs[1]:
    st.subheader("Distribución de la cobertura de salud (CH08)")
    dist = (df.groupby("COBERTURA")["_w"].sum() / df["_w"].sum() * 100).sort_values()
    fig = px.bar(dist, orientation="h", labels={"value": "%", "COBERTURA": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3. Numéricas ----------
with tabs[2]:
    var = st.selectbox("Variable numérica", NUMERICAS)
    sub = df.dropna(subset=[var])
    es_ingreso = var in ("IPCF", "P47T")
    if es_ingreso:
        sub = sub[sub[var] > 0]
        usar_log = st.checkbox("Escala logarítmica", value=True)
    else:
        usar_log = False
    col1, col2 = st.columns(2)
    fig_h = px.histogram(sub, x=var, y="_w", histfunc="sum", nbins=50,
                         log_x=usar_log, labels={"_w": "Personas"})
    col1.plotly_chart(fig_h, use_container_width=True)
    fig_b = px.box(sub.dropna(subset=["COBERTURA"]), x="COBERTURA", y=var, log_y=usar_log)
    col2.plotly_chart(fig_b, use_container_width=True)
    st.caption("El boxplot no está ponderado. Se excluyen ingresos = 0 para poder usar escala log.")

# ---------- 4. Categóricas vs Y ----------
with tabs[3]:
    var = st.selectbox("Variable categórica", CATEGORICAS)
    tabla = df.dropna(subset=[var, "COBERTURA"]).pivot_table(
        index=var, columns="COBERTURA", values="_w", aggfunc="sum", fill_value=0)
    pct = tabla.div(tabla.sum(axis=1), axis=0) * 100
    largo = pct.reset_index().melt(id_vars=var, var_name="COBERTURA", value_name="%")
    largo[var] = largo[var].astype(str)
    fig = px.bar(largo, x=var, y="%", color="COBERTURA", barmode="stack")
    st.plotly_chart(fig, use_container_width=True, key="cat_vs_y")
    with st.expander("Ver tabla"):
        st.dataframe(pct.round(1))

# ---------- 5. Asociación ----------
def cramers_v(x, y):
    t = pd.crosstab(x, y)
    k = min(t.shape) - 1
    if k <= 0:
        return np.nan
    return np.sqrt(chi2_contingency(t)[0] / (t.values.sum() * k))

with tabs[4]:
    st.subheader("Categóricas vs CH08: V de Cramér")
    res = {v: cramers_v(df[v], df["CH08"]) for v in CATEGORICAS if df[v].nunique() > 1}
    cv = pd.Series(res, name="V de Cramér").sort_values(ascending=False)
    st.plotly_chart(px.bar(cv, orientation="h"), use_container_width=True)
    st.caption("Calculado sobre la muestra (sin ponderar). 0 = sin asociación, 1 = asociación perfecta.")

    st.subheader("Correlación entre numéricas (Spearman)")
    corr = df[NUMERICAS].corr(method="spearman")
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1), use_container_width=True)

# ---------- 6. Región ----------
with tabs[5]:
    st.subheader("Cobertura de salud por región")
    tabla = df.dropna(subset=["REGION", "COBERTURA"]).pivot_table(
        index="REGION", columns="COBERTURA", values="_w", aggfunc="sum", fill_value=0)
    pct = tabla.div(tabla.sum(axis=1), axis=0) * 100
    largo = pct.reset_index().melt(id_vars="REGION", var_name="COBERTURA", value_name="%")
    st.plotly_chart(px.bar(largo, x="REGION", y="%", color="COBERTURA", barmode="stack"),
                    use_container_width=True, key="region")
    st.dataframe(pct.round(1))
