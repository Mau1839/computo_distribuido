import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Accidentes Viales · Chihuahua",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #0f172a; }
        #MainMenu, footer, header { visibility: hidden; }

        .hero {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            padding: 2rem 2.5rem; border-radius: 18px; margin-bottom: 1.8rem;
        }
        .hero h1 { color: #fff; font-size: 2rem; margin: 0; font-weight: 800; }
        .hero p  { color: #e9d5ff; margin: .35rem 0 0; font-size: .95rem; }

        .kpi {
            background: #1a2847; border: 2px solid #4c1d95;
            border-radius: 14px; padding: 1.2rem 1.4rem; height: 100%;
            box-shadow: 0 2px 8px rgba(124,58,237,0.2);
        }
        .kpi .label {
            color: #a78bfa; font-size: .72rem; letter-spacing: .12em;
            text-transform: uppercase; font-weight: 700;
        }
        .kpi .value {
            color: #e9d5ff; font-size: 2rem; font-weight: 800; margin-top: .2rem;
        }
        .kpi .delta { font-size: .8rem; margin-top: .15rem; }

        .section-title {
            color: #e9d5ff; font-size: 1.05rem; font-weight: 700;
            border-left: 4px solid #a855f7; padding-left: .65rem;
            margin: 1.8rem 0 .4rem;
        }
        .insight {
            background: rgba(124,58,237,.15); border-left: 4px solid #a855f7;
            padding: .7rem 1rem; border-radius: 8px; color: #d1d5db;
            font-size: .86rem; margin-top: .5rem;
        }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paleta corporativa para Plotly - TONOS MORADOS
PALETTE = ["#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd",
           "#a855f7", "#9333ea", "#d946ef"]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(124,58,237,0.05)",
    font=dict(color="#e9d5ff", family="Segoe UI, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#1a2847", font_size=12, bordercolor="#a855f7"),
)

# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando y limpieza datos...")
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "chihuahua_2015_2024.csv")

    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    df = df.rename(columns={"ID_ENTIDAD": "COD_MUNICIPIO",
                            "ID_MUNICIPIO": "COD_ENTIDAD"})

    for c in ["ANIO", "MES", "ID_HORA", "COD_MUNICIPIO"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["ANIO"].between(2015, 2024)].copy()

    muertos = ["CONDMUERTO", "PASAMUERTO", "PEATMUERTO",
               "CICLMUERTO", "OTROMUERTO", "NEMUERTO"]
    heridos = ["CONDHERIDO", "PASAHERIDO", "PEATHERIDO",
               "CICLHERIDO", "OTROHERIDO", "NEHERIDO"]

    def limpiar(cols):
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        return sub.where(sub < 90, 0).fillna(0).sum(axis=1)

    df["MUERTOS"] = limpiar(muertos).astype(int)
    df["HERIDOS"] = limpiar(heridos).astype(int)
    df["VICTIMAS"] = df["MUERTOS"] + df["HERIDOS"]

    try:
        cat = pd.read_csv(os.path.join(base, "catalogos", "tc_municipio.csv"),
                          dtype=str)
        cat = cat[cat["ID_ENTIDAD"] == "08"].copy()
        cat["cod"] = pd.to_numeric(cat["ID_MUNICIPIO"], errors="coerce")
        mapa = dict(zip(cat["cod"], cat["NOM_MUNICIPIO"]))
        df["MUNICIPIO"] = df["COD_MUNICIPIO"].map(mapa).fillna("Sin especificar")
    except Exception:
        df["MUNICIPIO"] = df["COD_MUNICIPIO"].astype("Int64").astype(str)

    for c in ["CAUSAACCI", "TIPACCID", "DIASEMANA"]:
        df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=False)
def load_predicciones():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "predicciones_test.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[(d["MUERTOS_REAL"] < 90) & (d["MUERTOS_PREDICHO"] < 90)]
        return d
    return None


@st.cache_data(show_spinner=False)
def load_predicciones():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "predicciones_test.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[(d["MUERTOS_REAL"] < 90) & (d["MUERTOS_PREDICHO"] < 90)]
        return d
    return None


# --------------------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error("No se encontró 'chihuahua_2015_2024.csv'. "
             "Ejecuta el dashboard desde la carpeta del proyecto.")
    st.stop()

# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros")

    anios = sorted(df["ANIO"].dropna().unique().astype(int))
    rango = st.select_slider(
        "Rango de años",
        options=anios,
        value=(anios[0], anios[-1]),
    )

    municipios = ["Todos"] + sorted(df["MUNICIPIO"].unique().tolist())
    muni_sel = st.selectbox(
        "Municipio",
        options=municipios,
    )

    causas = ["Todas"] + sorted(
        df["CAUSAACCI"].value_counts().head(15).index.tolist()
    )
    causa_sel = st.selectbox("Causa del accidente", causas)

    st.markdown("---")
    st.caption("Fuente: INEGI · ATUS 1997-2024")
    st.caption("Procesamiento: Ray · XGBoost")

# Aplicar filtros
mask = df["ANIO"].between(rango[0], rango[1])
if muni_sel != "Todos":
    mask &= df["MUNICIPIO"] == muni_sel
if causa_sel != "Todas":
    mask &= df["CAUSAACCI"] == causa_sel
dff = df[mask]

# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>Análisis de Accidentes Viales</h1>
        <p>Estado de Chihuahua · {rango[0]}–{rango[1]} ·
        Cómputo paralelo y distribuido con Ray</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
total_acc = len(dff)
total_mue = int(dff["MUERTOS"].sum())
total_her = int(dff["HERIDOS"].sum())
tasa = (total_mue / total_acc * 100) if total_acc else 0

k1, k2, k3, k4 = st.columns(4)
for col, label, value, delta in [
    (k1, "Total accidentes", f"{total_acc:,}", f"{len(municipios)-1} municipios"),
    (k2, "Personas fallecidas", f"{total_mue:,}", "Variable objetivo del modelo"),
    (k3, "Personas heridas", f"{total_her:,}", "Heridos registrados"),
    (k4, "Tasa de mortalidad", f"{tasa:.1f}%", "Fallecidos / accidentes"),
]:
    col.markdown(
        f"""<div class="kpi">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="delta" style="color:#a78bfa">{delta}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Geografía y Tiempo", "Causas y Víctimas", "Modelo y Rendimiento"]
)

# ========================= TAB 1 ==========================================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">¿Qué municipios concentran '
                    'más accidentes?</div>', unsafe_allow_html=True)
        top_muni = (dff["MUNICIPIO"].value_counts()
                    .head(10).sort_values())
        df_muni = pd.DataFrame({
            "Municipio": top_muni.index.tolist(),
            "Accidentes": top_muni.values.tolist(),
        })
        fig = px.bar(
            df_muni, x="Accidentes", y="Municipio", orientation="h",
            color="Accidentes", color_continuous_scale="Purples",
            labels={"Accidentes": "Accidentes", "Municipio": ""},
        )
        fig.update_layout(**PLOT_LAYOUT, height=380, showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
        if len(top_muni):
            lider = top_muni.index[-1]
            st.markdown(f'<div class="insight"><b>Insight:</b> {lider} '
                        f'encabeza la siniestralidad con '
                        f'{top_muni.iloc[-1]:,} accidentes registrados.'
                        f'</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">¿En qué horarios ocurren '
                    'más accidentes?</div>', unsafe_allow_html=True)
        por_hora = dff.groupby("ID_HORA").size().reindex(range(24),
                                                         fill_value=0)
        # FIX: usar DataFrame + nombres de columna; evita pasar Index
        # nombrado ("ID_HORA") directamente, lo que confunde a Plotly Express.
        df_hora = pd.DataFrame({
            "Hora": list(range(24)),
            "Accidentes": por_hora.values.tolist(),
        })
        fig = px.area(
            df_hora, x="Hora", y="Accidentes",
            labels={"Hora": "Hora del día", "Accidentes": "Accidentes"},
        )
        fig.update_traces(line_color="#8b5cf6",
                          fillcolor="rgba(139,92,246,.18)")
        fig.update_layout(**PLOT_LAYOUT, height=380)
        st.plotly_chart(fig, width='stretch')
        if por_hora.sum():
            hpico = int(por_hora.idxmax())
            st.markdown(f'<div class="insight"><b>Pico máximo:</b> las '
                        f'{hpico:02d}:00 h con {por_hora.max():,} accidentes. '
                        f'Coincide con horarios de mayor tránsito.</div>',
                        unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">¿Qué meses presentan mayor '
                    'incidencia?</div>', unsafe_allow_html=True)
        nombres_mes = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                       "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        por_mes = dff.groupby("MES").size().reindex(range(1, 13), fill_value=0)
        # FIX: DataFrame + nombres de columna para consistencia.
        df_mes = pd.DataFrame({
            "Mes": nombres_mes,
            "Accidentes": por_mes.values.tolist(),
        })
        fig = px.bar(
            df_mes, x="Mes", y="Accidentes",
            color="Accidentes", color_continuous_scale="Blues",
            labels={"Mes": "Mes", "Accidentes": "Accidentes"},
        )
        fig.update_layout(**PLOT_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with c4:
        st.markdown('<div class="section-title">Tendencia anual de '
                    'accidentes y fallecidos</div>', unsafe_allow_html=True)
        anual = dff.groupby("ANIO").agg(
            Accidentes=("ANIO", "size"),
            Fallecidos=("MUERTOS", "sum"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=anual["ANIO"], y=anual["Accidentes"],
                             name="Accidentes", marker_color="#7c3aed"))
        fig.add_trace(go.Scatter(x=anual["ANIO"], y=anual["Fallecidos"],
                                 name="Fallecidos", yaxis="y2",
                                 line=dict(color="#a855f7", width=3)))
        fig.update_layout(**PLOT_LAYOUT, height=360,
                          yaxis2=dict(overlaying="y", side="right",
                                      showgrid=False),
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Patrón de riesgo: día de la '
                'semana vs hora</div>', unsafe_allow_html=True)
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves",
                  "Viernes", "Sábado", "Domingo"]
    heat = dff.pivot_table(index="DIASEMANA", columns="ID_HORA",
                           values="ANIO", aggfunc="size", fill_value=0)
    heat = heat.reindex([d for d in orden_dias if d in heat.index])
    if not heat.empty and heat.shape[1] > 0:
        fig = px.imshow(heat, color_continuous_scale="Plasma",
                        labels=dict(x="Hora", y="", color="Accidentes"),
                        aspect="auto")
        fig.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Sin datos suficientes para el mapa de calor con los filtros actuales.")

# ========================= TAB 2 ==========================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">¿Qué causas son más '
                    'frecuentes?</div>', unsafe_allow_html=True)
        top_causa = dff["CAUSAACCI"].value_counts().head(7)
        fig = px.pie(values=top_causa.values, names=top_causa.index,
                     hole=0.5, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(**PLOT_LAYOUT, height=400,
                          legend=dict(orientation="v", y=0.5))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown('<div class="section-title">Tipos de accidente más '
                    'comunes</div>', unsafe_allow_html=True)
        top_tipo = dff["TIPACCID"].value_counts().head(8).sort_values()
        # FIX: mismo patrón DataFrame + columnas para evitar el error de Plotly.
        df_tipo = pd.DataFrame({
            "Tipo": top_tipo.index.tolist(),
            "Accidentes": top_tipo.values.tolist(),
        })
        fig = px.bar(df_tipo, x="Accidentes", y="Tipo", orientation="h",
                     color="Accidentes", color_continuous_scale="Purples",
                     labels={"Accidentes": "Accidentes", "Tipo": ""})
        fig.update_layout(**PLOT_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">¿Dónde hay más víctimas '
                'fallecidas y heridas?</div>', unsafe_allow_html=True)
    vic = (dff.groupby("MUNICIPIO")
           .agg(Fallecidos=("MUERTOS", "sum"),
                Heridos=("HERIDOS", "sum"))
           .sort_values("Fallecidos", ascending=False)
           .head(10).reset_index())
    fig = go.Figure()
    fig.add_trace(go.Bar(y=vic["MUNICIPIO"], x=vic["Fallecidos"],
                         name="Fallecidos", orientation="h",
                         marker_color="#6d28d9"))
    fig.add_trace(go.Bar(y=vic["MUNICIPIO"], x=vic["Heridos"],
                         name="Heridos", orientation="h",
                         marker_color="#c4b5fd"))
    fig.update_layout(**PLOT_LAYOUT, height=420, barmode="stack",
                      legend=dict(orientation="h", y=1.1),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Tabla comparativa por '
                'municipio</div>', unsafe_allow_html=True)
    tabla = (dff.groupby("MUNICIPIO")
             .agg(Accidentes=("ANIO", "size"),
                  Fallecidos=("MUERTOS", "sum"),
                  Heridos=("HERIDOS", "sum"))
             .sort_values("Accidentes", ascending=False)
             .head(15).reset_index())
    tabla["Tasa mortalidad %"] = (
        tabla["Fallecidos"] / tabla["Accidentes"] * 100).round(1)
    st.dataframe(tabla, width='stretch', hide_index=True)

# ========================= TAB 3 ==========================================
with tab3:
    st.markdown('<div class="section-title">¿Qué tan eficiente es Ray '
                'frente a Pandas secuencial?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        # Datos ilustrativos de benchmark (sustituir por mediciones reales)
        bench = pd.DataFrame({
            "Configuración": ["Pandas\nsecuencial", "Ray\n2 workers",
                              "Ray\n4 workers", "Ray\n8 workers"],
            "Tiempo (s)": [42.7, 18.5, 8.3, 6.2],
        })
        bench["Speedup"] = (bench["Tiempo (s)"].iloc[0]
                            / bench["Tiempo (s)"]).round(1)
        fig = px.bar(bench, x="Configuración", y="Tiempo (s)",
                     color="Tiempo (s)", color_continuous_scale="Purples",
                     text="Speedup")
        fig.update_traces(texttemplate="%{text}x", textposition="outside")
        fig.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown('<div class="insight"><b>Speedup máximo: 6.9x</b><br><br>'
                    'Ray distribuye el procesamiento entre múltiples workers, '
                    'reduciendo el tiempo de 42.7 s a 6.2 s. El cómputo '
                    'distribuido escala de forma casi lineal hasta saturar '
                    'los núcleos disponibles.</div>', unsafe_allow_html=True)
        st.metric("Registros procesados", f"{len(df):,}")
        st.metric("Mejor tiempo (Ray)", "6.2 s", "-85.5% vs Pandas")

    st.caption("Los tiempos de benchmark son ilustrativos. "
               "Sustitúyelos por mediciones reales de tu Ray Cluster.")

    st.markdown('<div class="section-title">Modelo XGBoost · Predicción de '
                'fallecidos</div>', unsafe_allow_html=True)
    pred = load_predicciones()
    if pred is not None and len(pred):
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = px.scatter(
                pred, x="MUERTOS_REAL", y="MUERTOS_PREDICHO",
                opacity=0.5, color_discrete_sequence=["#8b5cf6"],
                labels={"MUERTOS_REAL": "Fallecidos reales",
                        "MUERTOS_PREDICHO": "Fallecidos predichos"},
            )
            lim = max(pred["MUERTOS_REAL"].max(),
                      pred["MUERTOS_PREDICHO"].max())
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim],
                                     mode="lines", name="Predicción ideal",
                                     line=dict(color="#d946ef", dash="dash")))
            fig.update_layout(**PLOT_LAYOUT, height=380)
            st.plotly_chart(fig, width='stretch')
        with c2:
            err = pred["MUERTOS_PREDICHO"] - pred["MUERTOS_REAL"]
            mae = err.abs().mean()
            rmse = np.sqrt((err ** 2).mean())
            ss_res = (err ** 2).sum()
            ss_tot = ((pred["MUERTOS_REAL"]
                       - pred["MUERTOS_REAL"].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot else 0
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R²", f"{r2:.3f}")
            st.markdown('<div class="insight">Métricas calculadas sobre el '
                        'conjunto de prueba (predicciones_test.csv). '
                        'XGBoost se entrenó de forma distribuida con Ray.'
                        '</div>', unsafe_allow_html=True)
    else:
        st.info("No se encontró 'predicciones_test.csv'.")

    base = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(base, "importancia_variables.png")
    if os.path.exists(img):
        st.markdown('<div class="section-title">Importancia de variables '
                    '(métrica gain)</div>', unsafe_allow_html=True)
        st.image(img, width='stretch')

# --------------------------------------------------------------------------
st.markdown("---")
st.caption("Dashboard desarrollado con Streamlit + Plotly · "
           "Proyecto Cómputo Distribuido · INEGI ATUS 2015-2024")