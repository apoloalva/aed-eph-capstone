import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EDA prepagas", layout="wide")
st.title("Análisis exploratorio: planes de prepagas")

RUTAS = [r"C:\Users\Apolo\Downloads\base_limpia.csv", "base_limpia.csv"]


@st.cache_data
def cargar(fuente):
    df = pd.read_csv(fuente)
    df["edad_media"] = (df["rango_etario_desde"] + df["rango_etario_hasta"]) / 2
    return df


# ---------- Carga ----------
archivo = st.sidebar.file_uploader("Subir otro CSV (opcional)", type="csv")
fuente = archivo or next((r for r in RUTAS if os.path.exists(r)), None)
if fuente is None:
    st.error("No encontré base_limpia.csv. Subilo desde la barra lateral.")
    st.stop()
df = cargar(fuente)

# ---------- Filtros ----------
st.sidebar.header("Filtros")
prest = st.sidebar.multiselect(
    "Prestadora", sorted(df["nombre_comercial"].unique()),
    default=sorted(df["nombre_comercial"].unique()),
)
regiones = st.sidebar.multiselect("Región", sorted(df["region"].unique()))
vmin, vmax = int(df["valor_capital"].min()), int(df["valor_capital"].max())
rango_v = st.sidebar.slider("Valor cápita ($)", vmin, vmax, (vmin, vmax))
log_y = st.sidebar.checkbox("Escala log en precios", value=False)

dff = df[df["nombre_comercial"].isin(prest)]
if regiones:
    dff = dff[dff["region"].isin(regiones)]
dff = dff[dff["valor_capital"].between(*rango_v)]

if dff.empty:
    st.warning("Los filtros no dejan ningún dato.")
    st.stop()

# ---------- Tabs ----------
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Resumen", "Precios", "Edad", "Región", "Coberturas", "Datos"]
)

with t1:
    c = st.columns(5)
    c[0].metric("Filas", f"{len(dff):,}")
    c[1].metric("Planes", dff["plan_id"].nunique())
    c[2].metric("Prestadoras", dff["nombre_comercial"].nunique())
    c[3].metric("Regiones", dff["region"].nunique())
    c[4].metric("Mediana valor cápita", f"${dff['valor_capital'].median():,.0f}")

    l, r = st.columns(2)
    filas = dff["nombre_comercial"].value_counts().reset_index()
    filas.columns = ["prestadora", "filas"]
    l.plotly_chart(
        px.bar(filas, x="prestadora", y="filas", title="Filas por prestadora"),
        width="stretch",
    )
    cap = dff.groupby("nombre_comercial")["cantidad_capitas"].sum().reset_index()
    r.plotly_chart(
        px.bar(cap, x="nombre_comercial", y="cantidad_capitas",
               title="Cápitas totales por prestadora"),
        width="stretch",
    )
    st.subheader("Estadísticos descriptivos")
    num = ["valor_capital", "cantidad_capitas", "rango_etario_desde",
           "rango_etario_hasta", "medicamentos_pct"]
    st.dataframe(dff[num].describe().T, width="stretch")

with t2:
    l, r = st.columns(2)
    l.plotly_chart(
        px.histogram(dff, x="valor_capital", nbins=50, marginal="box",
                     log_y=log_y, title="Distribución del valor cápita"),
        width="stretch",
    )
    r.plotly_chart(
        px.box(dff, x="nombre_comercial", y="valor_capital", log_y=log_y,
               title="Valor cápita por prestadora"),
        width="stretch",
    )
    col = st.selectbox(
        "Valor cápita según…",
        ["modalidad_adhesion", "permite_copago", "habitacion_individual",
         "optica_adultos", "medicamentos_pct"],
    )
    st.plotly_chart(
        px.box(dff, x=dff[col].astype(str), y="valor_capital", log_y=log_y,
               labels={"x": col}),
        width="stretch",
    )

with t3:
    st.caption("Edad media = punto medio del rango etario (desde–hasta).")
    color = st.selectbox("Color", [None, "nombre_comercial", "modalidad_adhesion"])
    st.plotly_chart(
        px.scatter(dff, x="edad_media", y="valor_capital", color=color,
                   opacity=0.5, log_y=log_y,
                   title="Valor cápita vs. edad"),
        width="stretch",
    )
    med = (dff.groupby(["edad_media", "nombre_comercial"])["valor_capital"]
              .median().reset_index())
    st.plotly_chart(
        px.line(med, x="edad_media", y="valor_capital", color="nombre_comercial",
                markers=True, log_y=log_y,
                title="Mediana del valor cápita por edad y prestadora"),
        width="stretch",
    )

with t4:
    top = st.slider("Top regiones (por cantidad de filas)", 5, 52, 15)
    reg = (dff.groupby("region")["valor_capital"]
              .agg(mediana="median", filas="size").reset_index()
              .sort_values("filas", ascending=False).head(top)
              .sort_values("mediana"))
    st.plotly_chart(
        px.bar(reg, x="mediana", y="region", orientation="h", hover_data=["filas"],
               title="Mediana del valor cápita por región"),
        width="stretch",
    )

with t5:
    cat = ["ortodoncia", "cirugia_estetica", "cobertura_viaje", "kinesiologia",
           "psicologia", "cobertura_fallecimiento", "protesis_dental",
           "blanqueamiento_dental", "medico_domicilio", "gym", "telemedicina",
           "fonoaudiologia", "practicas_especiales", "cirugia_refractiva"]
    cat = [c for c in cat if c in dff.columns]
    var = st.selectbox("Cobertura", cat)
    l, r = st.columns(2)
    vc = dff[var].astype(str).value_counts().reset_index()
    vc.columns = [var, "filas"]
    l.plotly_chart(px.bar(vc, x=var, y="filas", title=f"Frecuencia de {var}"),
                   width="stretch")
    r.plotly_chart(
        px.box(dff, x=dff[var].astype(str), y="valor_capital", log_y=log_y,
               labels={"x": var}, title=f"Valor cápita según {var}"),
        width="stretch",
    )
    st.subheader("Cobertura por prestadora (% de filas)")
    ct = pd.crosstab(dff["nombre_comercial"], dff[var].astype(str), normalize="index") * 100
    st.plotly_chart(
        px.imshow(ct.round(1), text_auto=True, color_continuous_scale="Blues",
                  aspect="auto"),
        width="stretch",
    )

with t6:
    st.dataframe(dff, width="stretch")
    st.download_button("Descargar filtrado (CSV)",
                       dff.to_csv(index=False).encode("utf-8"),
                       "base_filtrada.csv", "text/csv")
