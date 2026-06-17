# =====================================================================
# Dashboard interactivo — Análisis multidimensional de brechas
# socioeconómicas y dinámicas (CASEN 2020–2024)
# Autores: Carla Maureira Venegas, Constanza Luarte Salazar
#
# Refactorizado con diseño SaaS:
# - Sidebar izquierdo para controles y mapa.
# - Grilla para todos los gráficos solicitados.
# - Tarjetas blancas planas (Flat Design).
# =====================================================================

import json
import os 

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================================================================
# 1. Configuración de Página y CSS
# =====================================================================
st.set_page_config(
    page_title="Brechas socioeconómicas — CASEN 2024",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar CSS externo
css_path = os.path.join(BASE_DIR, "assets", "style.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass



# Paleta Cromática — Identidad Magma (naranja → morado oscuro)
# Contraste: celeste (hombre) vs carmesí (mujer)
C_HOMBRE = "#4FC3F7"   # Celeste
C_MUJER = "#E8537A"    # Rosa carmesí
C_NACIONAL = "#9E9E9E" # Gris neutro (referencia)
C_ACENTO = "#F97316"   # Naranja Magma (acento principal)
C_MAGMA_DARK = "#2D1B69" # Morado oscuro Magma
C_MAGMA_MID = "#B5367A"  # Magenta medio Magma
GRID = "rgba(226, 232, 240, 0.6)"
MAP_SCALE = "Sunsetdark"  # Naranja → púrpura (identidad Magma)

# Constantes
REGIONES = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "La Araucanía",
    14: "Los Ríos", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}
ORDEN_NS = [15, 1, 2, 3, 4, 5, 13, 6, 7, 16, 8, 9, 14, 10, 11, 12]

EDUC = {
    0: "Sin educación", 1: "Básica Incompleta", 2: "Básica Completa",
    3: "Media Incompleta", 4: "Media Completa",
    5: "Superior Incompleta", 6: "Superior Completa",
}

SECTORES = {
    6: "Construcción", 2: "Minería", 12: "Inmobiliario",
    4: "Electricidad/Agua", 5: "Electricidad/Agua", 3: "Manufactura",
    13: "Profesional/Científico", 16: "Educación", 17: "Salud",
    11: "Finanzas",
}

OFICIOS = {
    1: "Directivos", 2: "Profesionales", 3: "Técnicos",
    4: "Administrativos", 7: "Oficios/Artesanía",
    8: "Operadores de maq.", 9: "Ocup. Elementales",
}

ADYACENCIA = [
    (15, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 13), (5, 6),
    (13, 6), (6, 7), (7, 16), (16, 8), (8, 9), (9, 14), (14, 10),
    (10, 11), (11, 12),
]

GEOJSON_URLS = [
    "https://raw.githubusercontent.com/fcortes/Chile-GeoJSON/master/Regional.geojson",
    "https://raw.githubusercontent.com/caracena/chile-geojson/master/regiones.json",
]


# =====================================================================
# 2. Carga de Datos
# =====================================================================
@st.cache_data(show_spinner="Cargando datos optimizados...")
def load_data():
    data_dir = os.path.join(BASE_DIR, "data")
    d_mapa = pd.read_parquet(os.path.join(data_dir, "casen_mapa.parquet"))
    d_ing = pd.read_parquet(os.path.join(data_dir, "casen_ingreso_edu.parquet"))
    d_emp = pd.read_parquet(os.path.join(data_dir, "casen_empleo_edu.parquet"))
    d_evo24 = pd.read_parquet(os.path.join(data_dir, "casen_evolucion_24.parquet"))
    d_evo20 = pd.read_parquet(os.path.join(data_dir, "casen_evolucion_20.parquet"))
    d_evo22 = pd.read_parquet(os.path.join(data_dir, "casen_evolucion_22.parquet"))
    d_sec = pd.read_parquet(os.path.join(data_dir, "casen_brecha_sector.parquet"))
    d_ofi = pd.read_parquet(os.path.join(data_dir, "casen_brecha_oficio.parquet"))
    return d_mapa, d_ing, d_emp, d_evo24, d_evo20, d_evo22, d_sec, d_ofi


@st.cache_data(show_spinner=False)
def load_geojson():
    def _key(gj):
        props = gj["features"][0]["properties"]
        key = "codregion" if "codregion" in props else list(props)[0]
        return gj, f"properties.{key}"

    local = os.path.join(BASE_DIR, "regiones.json")
    if os.path.exists(local):
        try:
            with open(local, encoding="utf-8") as f:
                return _key(json.load(f))
        except Exception:
            pass
    for url in GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return _key(r.json())
        except Exception:
            continue
    return None, None


def wavg(df, val, w="expr"):
    d = df.dropna(subset=[val, w])
    if d.empty: return np.nan
    return np.average(d[val], weights=d[w])


# =====================================================================
# 3. Funciones Analíticas y Espaciales
# =====================================================================
@st.cache_data(show_spinner=False)
def moran_lisa(vals_dict, n_perm=999, seed=42):
    regs = list(vals_dict.keys())
    y = np.array([vals_dict[r] for r in regs], dtype=float)
    n = len(regs)
    idx = {r: i for i, r in enumerate(regs)}
    W = np.zeros((n, n))
    for a, b in ADYACENCIA:
        if a in idx and b in idx:
            W[idx[a], idx[b]] = W[idx[b], idx[a]] = 1.0
    W = W / W.sum(axis=1, keepdims=True)

    z = y - y.mean()
    s2 = (z ** 2).sum() / n
    I_obs = (z @ W @ z) / (z ** 2).sum() * n / W.sum()

    rng = np.random.default_rng(seed)
    sims = np.empty(n_perm)
    for k in range(n_perm):
        zp = rng.permutation(z)
        sims[k] = (zp @ W @ zp) / (zp ** 2).sum() * n / W.sum()
    p_glob = (np.sum(sims >= I_obs) + 1) / (n_perm + 1)

    lag = W @ z
    Ii = z * lag / s2
    p_loc = np.empty(n)
    for i in range(n):
        others = np.delete(z, i)
        wi = W[i, np.arange(n) != i]
        sims_i = np.empty(n_perm)
        for k in range(n_perm):
            sims_i[k] = z[i] * (wi @ rng.permutation(others)) / s2
        if Ii[i] >= 0:
            p_loc[i] = (np.sum(sims_i >= Ii[i]) + 1) / (n_perm + 1)
        else:
            p_loc[i] = (np.sum(sims_i <= Ii[i]) + 1) / (n_perm + 1)

    quad = []
    for i in range(n):
        if z[i] >= 0 and lag[i] >= 0: quad.append("Alto-Alto")
        elif z[i] < 0 and lag[i] < 0: quad.append("Bajo-Bajo")
        elif z[i] < 0: quad.append("Bajo-Alto")
        else: quad.append("Alto-Bajo")

    lisa = pd.DataFrame({"region": regs, "Ii": Ii, "p": p_loc, "cluster": quad})
    return I_obs, p_glob, lisa

LISA_COLORS = {
    "Alto-Alto": ("#FEE2E2", "#EF4444", "Alto-Alto (Clúster cálido)"),
    "Bajo-Bajo": ("#DBEAFE", "#3B82F6", "Bajo-Bajo (Clúster frío)"),
    "Bajo-Alto": ("#FDF2F8", "#EC4899", "Bajo-Alto (Valor atípico)"),
    "Alto-Bajo": ("#FEF3C7", "#D97706", "Alto-Bajo (Valor atípico)"),
}

def make_lisa_badges(sig_df):
    if sig_df.empty:
        return "<span style='color:#9CA3AF; font-style:italic;'>Sin clústeres locales significativos</span>"
    
    badges = []
    for r in sig_df.itertuples():
        bg, fg, label = LISA_COLORS.get(r.cluster, ("#F3F4F6", "#6B7280", r.cluster))
        badges.append(
            f"<span style='background-color:{bg}; color:{fg}; padding:4px 8px; border-radius:6px; font-weight:600; font-size:11px; margin-right:6px; display:inline-block; margin-bottom:6px;'>"
            f"{r.nombre}: {label}"
            f"</span>"
        )
    return " ".join(badges)


# =====================================================================
# 4. Funciones de UI y Gráficos
# =====================================================================
def base_layout(fig, h=320, l_margin=30):
    fig.update_layout(
        font=dict(family='Inter, sans-serif', size=11, color='#000000'),
        height=h, margin=dict(l=l_margin, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        transition_duration=500,
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="Inter", bordercolor="rgba(255,255,255,0.1)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=11, color="#000000")
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, title_font=dict(size=12, color='#000000', weight='bold'), tickfont=dict(color='#000000'))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, title_font=dict(size=12, color='#000000', weight='bold'), tickfont=dict(color='#000000'))
    fig.update_polars(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor=GRID), angularaxis=dict(gridcolor=GRID, linecolor=GRID))
    fig.update_geos(bgcolor="rgba(0,0,0,0)")
    return fig

def generate_narrative(reg_name, reg_code, d24_base):
    g_reg = d24_base[d24_base["region"] == reg_code]
    gp_reg = g_reg.dropna(subset=["pobreza", "expr"])
    ing_reg = wavg(g_reg, "ypchtotcor") / 1000
    pob_reg = np.average(gp_reg["pobreza"].isin([1, 2]), weights=gp_reg["expr"]) * 100 if not gp_reg.empty else 0
    
    gp_nac = d24_base.dropna(subset=["pobreza", "expr"])
    ing_nac = wavg(d24_base, "ypchtotcor") / 1000
    pob_nac = np.average(gp_nac["pobreza"].isin([1, 2]), weights=gp_nac["expr"]) * 100 if not gp_nac.empty else 0
    
    ing_pct = ((ing_reg - ing_nac) / ing_nac) * 100
    pob_diff = pob_reg - pob_nac
    
    badges_html = ""
    if ing_pct >= 10:
        badges_html += "<span style='background-color:#D1FAE5; color:#065F46; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Ingreso Alto</span>"
    elif ing_pct <= -10:
        badges_html += "<span style='background-color:#FEE2E2; color:#991B1B; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Ingreso Bajo</span>"
    else:
        badges_html += "<span style='background-color:#F3F4F6; color:#374151; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Ingreso Medio</span>"
        
    if pob_diff >= 3:
        badges_html += " <span style='background-color:#FEF3C7; color:#92400E; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Alta Pobreza</span>"
    elif pob_diff <= -3:
        badges_html += " <span style='background-color:#E0F2FE; color:#0369A1; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Baja Pobreza</span>"
    
    ing_text = f"un <b style='color:#111827'>{abs(ing_pct):.1f}% mayor</b>" if ing_pct >= 0 else f"un <b style='color:#111827'>{abs(ing_pct):.1f}% menor</b>"
    pob_text = f"<b style='color:#111827'>{abs(pob_diff):.1f} pp superior</b>" if pob_diff >= 0 else f"<b style='color:#111827'>{abs(pob_diff):.1f} pp inferior</b>"
        
    return f"""
    <div class='narrative-box' style='border-left: 4px solid #F97316;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;'>
            <h4 style='color: #111827; margin: 0; font-size: 16px; font-weight: 700;'>
                Región de {reg_name}
            </h4>
            <div style='display:flex; gap:6px;'>{badges_html}</div>
        </div>
        <p style='color: #4B5563; font-size: 13.5px; line-height: 1.5; margin: 0;'>
            El ingreso per cápita promedio es de <b style='color:#111827'>${ing_reg:,.0f}K</b> ({ing_text} al país). 
            La tasa de pobreza regional es de <b style='color:#111827'>{pob_reg:.1f}%</b> ({pob_text} vs. media).
        </p>
    </div>
    """

def generate_national_narrative(d24_base):
    ing_nac = wavg(d24_base, "ypchtotcor") / 1000
    gp_nac = d24_base.dropna(subset=["pobreza", "expr"])
    pob_nac = np.average(gp_nac["pobreza"].isin([1, 2]), weights=gp_nac["expr"]) * 100 if not gp_nac.empty else 0
    
    return f"""
    <div class='narrative-box' style='border-left: 4px solid #F97316;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
            <h4 style='color: #111827; margin: 0; font-size: 16px; font-weight: 700;'>
                Promedio Nacional
            </h4>
            <span style='background-color:#F3F4F6; color:#374151; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:600;'>Datos País</span>
        </div>
        <p style='color: #4B5563; font-size: 13.5px; line-height: 1.5; margin: 0;'>
            El ingreso promedio nacional es <b style='color:#111827'>${ing_nac:,.0f}K</b> y la pobreza <b style='color:#111827'>{pob_nac:.1f}%</b>. 
            <b style='color:#F97316'>Selecciona una región en el mapa de la izquierda</b> para activar el análisis territorial profundo.
        </p>
    </div>
    """

# =====================================================================
# 5. Layout Principal (SaaS Dashboard)
# =====================================================================
d_mapa, d_ing, d_emp, d_evo24, d_evo20, d_evo22, d_sec, d_ofi = load_data()

# Estado
for key, default in [('selected_region', "Chile (Nacional)"), ('zona_sel', "Todo"), ('map_var', "Ingreso Medio")]:
    if key not in st.session_state: st.session_state[key] = default

sel = st.session_state.selected_region
nom_a_cod = {v: k for k, v in REGIONES.items()}
cod_sel = None if sel == "Chile (Nacional)" else nom_a_cod.get(sel, None)
filtrado = cod_sel is not None
ambito = sel if filtrado else "Chile"

# Filtro de zona
def apply_area_filter(df):
    if "area" in df.columns:
        if st.session_state.zona_sel == "Todo":
            return df
        return df[df["area"] == (1 if st.session_state.zona_sel == "Urbano" else 2)]
    return df

d_mapa_b = apply_area_filter(d_mapa)
d_ing_b = apply_area_filter(d_ing)
d_emp_b = apply_area_filter(d_emp)
d_evo24_b = apply_area_filter(d_evo24)
d_sec_b = apply_area_filter(d_sec)
d_ofi_b = apply_area_filter(d_ofi)

d_mapa_f = d_mapa_b[d_mapa_b["region"] == cod_sel] if filtrado else d_mapa_b
d_ing_f = d_ing_b[d_ing_b["region"] == cod_sel] if filtrado else d_ing_b
d_emp_f = d_emp_b[d_emp_b["region"] == cod_sel] if filtrado else d_emp_b
d_evo24_f = d_evo24_b[d_evo24_b["region"] == cod_sel] if filtrado else d_evo24_b
d_sec_f = d_sec_b[d_sec_b["region"] == cod_sel] if filtrado else d_sec_b
d_ofi_f = d_ofi_b[d_ofi_b["region"] == cod_sel] if filtrado else d_ofi_b

# Agregados Regionales
reg_stats = []
for r in ORDEN_NS:
    g = d_mapa_b[d_mapa_b["region"] == r]
    gp = g.dropna(subset=["pobreza", "expr"])
    ad_r = g[g["edad"] >= 15]
    reg_stats.append({
        "region": r, "nombre": REGIONES[r],
        "ingreso": wavg(g, "ypchtotcor") / 1000,
        "pobreza": np.average(gp["pobreza"].isin([1, 2]), weights=gp["expr"]) * 100 if not gp.empty else 0,
        "empleo": np.average((ad_r["activ"] == 1).fillna(False), weights=ad_r["expr"]) * 100 if not ad_r.empty else 0,
        "poblacion": g["expr"].sum(),
    })
reg_df = pd.DataFrame(reg_stats)

metric_map = {
    "Ingreso Medio": "ingreso",
    "Tasa de Pobreza": "pobreza",
    "Tasa de Empleo": "empleo"
}
var_name = metric_map[st.session_state.map_var]

I_g, p_g, lisa = moran_lisa({int(row.region): float(getattr(row, var_name)) for row in reg_df.itertuples()})
lisa = lisa.merge(reg_df[["region", "nombre"]], on="region")
sig = lisa[lisa["p"] < 0.05]

# Columnas Principales: 1.2 (Sidebar/Mapa) y 3.8 (Main Content)
col_sidebar, col_main = st.columns([1.2, 3.8], gap="large")

# --- COLUMNA IZQUIERDA: SIDEBAR / MAPA ---
with col_sidebar:
    st.markdown("<div class='sidebar-anchor'></div><h3 style='margin-top:0; margin-bottom:4px; font-size:18px; color:#111827;'>Filtros y Mapa</h3>", unsafe_allow_html=True)
    
    # Controles (sólo si area existe)
    if "area" in d_mapa.columns:
        st.markdown("<p style='font-size:11px;color:#6B7280;margin:0 0 0 0;font-weight:600;text-transform:uppercase;'>Zona Territorial</p>", unsafe_allow_html=True)
        zc1, zc2, zc3 = st.columns([1, 1.25, 1])
        if zc1.button("Todo", use_container_width=True, type="primary" if st.session_state.zona_sel=="Todo" else "secondary"): st.session_state.zona_sel = "Todo"; st.rerun()
        if zc2.button("Urbano", use_container_width=True, type="primary" if st.session_state.zona_sel=="Urbano" else "secondary"): st.session_state.zona_sel = "Urbano"; st.rerun()
        if zc3.button("Rural", use_container_width=True, type="primary" if st.session_state.zona_sel=="Rural" else "secondary"): st.session_state.zona_sel = "Rural"; st.rerun()
    
    st.markdown("<p style='font-size:11px;color:#6B7280;margin:-4px 0 0 0;font-weight:600;text-transform:uppercase;'>Métrica a Visualizar</p>", unsafe_allow_html=True)
    st.selectbox("Métrica", ["Ingreso Medio", "Tasa de Pobreza", "Tasa de Empleo"], label_visibility="collapsed", key="map_var")
    
    st.markdown(f"<div style='margin-top:4px; margin-bottom:4px; border-top:1px solid #E5E7EB;'></div>", unsafe_allow_html=True)
    
    # Render Mapa
    gj, featkey = load_geojson()
    lisa_idx = lisa.set_index("region")

    map_cfg = {
        "Ingreso Medio": {"z": reg_df["ingreso"], "scale": MAP_SCALE, "hover": "Ingreso: $%{z:,.0f}K"},
        "Tasa de Pobreza": {"z": reg_df["pobreza"], "scale": "Reds", "hover": "Pobreza: %{z:.1f}%"},
        "Tasa de Empleo": {"z": reg_df["empleo"], "scale": "Tealgrn", "hover": "Empleo: %{z:.1f}%"}
    }[st.session_state.map_var]

    if gj is not None:
        # Preparar label de LISA para el hover del mapa
        lisa_labels = lisa_idx.loc[reg_df["region"]].apply(lambda x: x["cluster"] if x["p"] < 0.05 else "No significativo", axis=1)
        
        fig_map = go.Figure(go.Choropleth(
            geojson=gj, featureidkey=featkey, locations=reg_df["region"], z=map_cfg["z"],
            colorscale=map_cfg["scale"], marker_line_color="#FFFFFF", marker_line_width=1,
            colorbar=dict(title="", orientation="v", thickness=6, len=0.7, x=0.95, y=0.5, tickfont=dict(size=9, color="#6B7280")),
            customdata=np.stack([reg_df["nombre"], lisa_idx.loc[reg_df["region"], "cluster"], lisa_idx.loc[reg_df["region"], "p"], lisa_labels], axis=-1),
            hovertemplate="<b style='font-size:12px'>%{customdata[0]}</b><br><span style='color:#6B7280'>" + map_cfg["hover"] + "</span><br><span style='color:#8B5CF6'>LISA: %{customdata[3]}</span><extra></extra>",
        ))
        
        # Superponer LISA: Delinear regiones significativas con color celeste/cyan grueso (como en la imagen original)
        if not sig.empty:
            fig_map.add_trace(go.Choropleth(
                geojson=gj, featureidkey=featkey, locations=sig["region"], 
                z=[1]*len(sig), # Dummy z
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], 
                marker_line_color=C_ACENTO, # Naranja Magma para resaltar LISA
                marker_line_width=3, 
                showscale=False, 
                hoverinfo="skip"
            ))
            
        if filtrado:
            sel_row = reg_df[reg_df["region"] == cod_sel]
            fig_map.add_trace(go.Choropleth(
                geojson=gj, featureidkey=featkey, locations=sel_row["region"], z=map_cfg["z"][reg_df["region"] == cod_sel],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker_line_color="#0F172A", marker_line_width=3, showscale=False, hoverinfo="skip",
            ))
            
        # Fijar encuadre exacto como en el mapa original
        fig_map.update_geos(visible=False, projection_type="mercator", lonaxis_range=[-77.5, -65.0], lataxis_range=[-56.5, -17.2])
        fig_map.update_layout(title=dict(text="Distribución del ingreso medio regional", font=dict(size=12, color="#000000"), x=0.5, y=0.95))
        fig_map = base_layout(fig_map, h=380) # Altura precisa para encajar estáticamente
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
        
        ev = st.plotly_chart(fig_map, width='stretch', on_select="rerun", selection_mode="points", config={'scrollZoom':False, 'displayModeBar':False})
        if ev and ev.get('selection',{}).get('points'):
            clicked = ev['selection']['points'][0].get('customdata', [None])[0]
            if clicked in nom_a_cod and clicked != st.session_state.selected_region:
                st.session_state.selected_region = clicked; st.rerun()

    if p_g < 0.05:
        moran_text = "→ agrupamiento espacial significativo"
        moran_color = "#4B5563" 
    else:
        moran_text = "→ sin agrupamiento significativo"
        moran_color = "#9CA3AF"

    # Box de Autocorrelación Espacial estilo imagen original
    st.markdown(f"""
    <div style='background-color:#FFFFFF; border: 1px solid #E5E7EB; border-radius:12px; padding:10px; text-align:center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); margin-top:-10px; margin-bottom:12px;'>
        <div style='font-size:12px; font-weight:700; color:#111827; margin-bottom:2px;'>Autocorrelación espacial:</div>
        <div style='font-size:12px; color:#4B5563; font-weight:700;'>I de Moran = {I_g:.3f} · p = {p_g:.3f}</div>
        <div style='font-size:12px; font-weight:700; color:{moran_color};'>{moran_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Box para indicar qué regiones son los clústeres si existen
    if not sig.empty:
        st.markdown(f"<div style='margin-bottom:12px; text-align:center;'>{make_lisa_badges(sig)}</div>", unsafe_allow_html=True)


# --- COLUMNA DERECHA: MAIN CONTENT ---
with col_main:
    # Header Principal
    st.markdown(f"""
    <div style='margin-bottom: 28px;'>
        <div style='display:flex; justify-content:space-between; align-items:flex-end;'>
            <div>
                <h2 class='gradient-title' style='margin-bottom: 4px;'>Análisis multidimensional de brechas socioeconómicas y dinámicas</h2>
                <span style='font-size:13px; font-weight:700; color:#000000; letter-spacing:0.3px;'>CASEN 2020–2024</span>
            </div>
            <span style='color:#000000; font-size:11px; font-weight:700;'>Carla Maureira — Constanza Luarte</span>
        </div>
        <div style='margin-top:14px; height:2px; background: linear-gradient(90deg, #F97316, #B5367A, rgba(181,54,122,0)); border-radius:2px;'></div>
    </div>
    """, unsafe_allow_html=True)

    # Sparklines: mini-barras de 16 regiones para contexto visual en cada KPI
    def make_sparkline_svg(values, color, highlight_idx=None):
        """Genera un SVG inline ultra-compacto con mini-barras."""
        if not values or all(np.isnan(v) for v in values):
            return ""
        vals = [v if not np.isnan(v) else 0 for v in values]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1
        n = len(vals)
        w, h_svg = 120, 24
        bar_w = max(1, (w - (n - 1) * 2) // n)
        bars = []
        for i, v in enumerate(vals):
            bh = max(2, int(((v - vmin) / rng) * (h_svg - 4)))
            x = i * (bar_w + 2)
            opacity = "1"
            bars.append(f"<rect x='{x}' y='{h_svg - bh}' width='{bar_w}' height='{bh}' rx='1' fill='{color}' opacity='{opacity}'/>")
        return f"<svg width='{n * (bar_w + 2)}' height='{h_svg}' style='display:block;'>{''.join(bars)}</svg>"

    # KPIs Premium
    ing_nac = wavg(d_mapa_b, "ypchtotcor") / 1000
    dp_n = d_mapa_b.dropna(subset=["pobreza", "expr"])
    pob_nac = np.average(dp_n["pobreza"].isin([1, 2]), weights=dp_n["expr"]) * 100 if not dp_n.empty else 0
    ad_nac = d_mapa_b[d_mapa_b["edad"] >= 15]
    emp_nac = np.average((ad_nac["activ"] == 1).fillna(False), weights=ad_nac["expr"]) * 100 if not ad_nac.empty else 0

    # Determinar highlight index para sparklines
    hl_idx = ORDEN_NS.index(cod_sel) if filtrado and cod_sel in ORDEN_NS else None

    # Sparklines por métrica
    spark_ing = make_sparkline_svg(reg_df["ingreso"].tolist(), "#4FC3F7", hl_idx)
    spark_pob = make_sparkline_svg(reg_df["pobreza"].tolist(), "#E8537A", hl_idx)
    spark_emp = make_sparkline_svg(reg_df["empleo"].tolist(), "#F97316", hl_idx)

    if filtrado:
        ing_val, dp_f, ad_f = wavg(d_mapa_f, "ypchtotcor") / 1000, d_mapa_f.dropna(subset=["pobreza", "expr"]), d_mapa_f[d_mapa_f["edad"] >= 15]
        pob_val = np.average(dp_f["pobreza"].isin([1, 2]), weights=dp_f["expr"]) * 100 if not dp_f.empty else 0
        emp_val = np.average((ad_f["activ"] == 1).fillna(False), weights=ad_f["expr"]) * 100 if not ad_f.empty else 0
        
        d_ing, d_pob, d_emp = ing_val - ing_nac, pob_val - pob_nac, emp_val - emp_nac
        s_ing = f"<span style='color:{'#10B981' if d_ing>=0 else '#EF4444'}; font-weight:600; background-color:{'#D1FAE5' if d_ing>=0 else '#FEE2E2'}; padding:2px 6px; border-radius:4px;'>{'↑' if d_ing>=0 else '↓'} {abs(d_ing):.0f}K</span> <span style='color:#000000; font-weight:600;'>vs Nac.</span>"
        s_pob = f"<span style='color:{'#EF4444' if d_pob>=0 else '#10B981'}; font-weight:600; background-color:{'#FEE2E2' if d_pob>=0 else '#D1FAE5'}; padding:2px 6px; border-radius:4px;'>{'↑' if d_pob>=0 else '↓'} {abs(d_pob):.1f}pp</span> <span style='color:#000000; font-weight:600;'>vs Nac.</span>"
        s_emp = f"<span style='color:{'#10B981' if d_emp>=0 else '#EF4444'}; font-weight:600; background-color:{'#D1FAE5' if d_emp>=0 else '#FEE2E2'}; padding:2px 6px; border-radius:4px;'>{'↑' if d_emp>=0 else '↓'} {abs(d_emp):.1f}pp</span> <span style='color:#000000; font-weight:600;'>vs Nac.</span>"
    else:
        ing_val, pob_val, emp_val = ing_nac, pob_nac, emp_nac
        s_ing = s_pob = s_emp = "<span style='color:#000000; font-weight:600;'>Referencia Nacional</span>"

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(3, 1fr);gap:24px;margin-bottom:24px;'>
        <div class='kpi-card kpi-ingreso'>
            <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;'>
                <div>
                    <span style='font-size:11px;color:#000000;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;'>Ingreso Per Cápita</span><br>
                    <span style='font-size:30px;font-weight:800;color:#111827;letter-spacing:-0.5px;line-height:1.2;'>${ing_val:,.0f}K</span>
                </div>
                <div style='margin-bottom:6px;'>{spark_ing}</div>
            </div>
            <div style='font-size:11px;'>{s_ing}</div>
        </div>
        <div class='kpi-card kpi-pobreza'>
            <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;'>
                <div>
                    <span style='font-size:11px;color:#000000;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;'>Tasa de Pobreza</span><br>
                    <span style='font-size:30px;font-weight:800;color:#111827;letter-spacing:-0.5px;line-height:1.2;'>{pob_val:.1f}%</span>
                </div>
                <div style='margin-bottom:6px;'>{spark_pob}</div>
            </div>
            <div style='font-size:11px;'>{s_pob}</div>
        </div>
        <div class='kpi-card kpi-empleo'>
            <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;'>
                <div>
                    <span style='font-size:11px;color:#000000;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;'>Tasa de Empleo</span><br>
                    <span style='font-size:30px;font-weight:800;color:#111827;letter-spacing:-0.5px;line-height:1.2;'>{emp_val:.1f}%</span>
                </div>
                <div style='margin-bottom:6px;'>{spark_emp}</div>
            </div>
            <div style='font-size:11px;'>{s_emp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Función para título de gráfico con badge de ámbito
    def chart_header(title):
        badge = f"<span class='ambito-badge'>{ambito}</span>"
        return f"""<p style='font-weight:700; font-size:14px; margin-bottom:8px; color:#000000; text-align:center;'>{title}{badge}</p>"""

    CH_H = 340

    # Funciones auxiliares para gráficos
    def brecha(df, col, mapa):
        if "activ" in df.columns:
            occ = df[(df["activ"]==1) & (df[col].isin(mapa))].copy()
        else:
            occ = df[df[col].isin(mapa)].copy()
        occ["grupo"] = occ[col].map(mapa)
        out = {}
        for s, g in occ.groupby("grupo"):
            h, m = g[g["sexo"]==1]["ypchtotcor"].mean(), g[g["sexo"]==2]["ypchtotcor"].mean()
            if min((g["sexo"]==1).sum(), (g["sexo"]==2).sum()) >= 30: out[s] = (h - m)/1000
        return out
        
    def fig_brecha(vals_reg, vals_nac, xlab, ylab):
        orden = sorted([s for s in vals_nac if not np.isnan(vals_nac[s])],
                       key=lambda s: vals_nac[s])
        vr = [vals_reg.get(s, np.nan) for s in orden]
        vn = [vals_nac[s] for s in orden]
        
        # 1. Ya no destacamos solo una, todas tienen color vibrante
        
        colores = []
        for i, v in enumerate(vr):
            if np.isnan(v):
                colores.append("rgba(0,0,0,0)")
                continue
                
            es_mujer = v < 0
            # Todas con opacidad completa
            colores.append(C_MUJER if es_mujer else C_HOMBRE)

        fig = go.Figure()
        
        # Tooltip Interactivo Estilizado (HTML) sin comparación nacional
        hovertemplate = (
            "<b>%{y}</b><br>"
            "<span style='color:#cbd5e1;'>Brecha: </span>"
            "<b>%{x:,.0f}K</b><br>"
            "<extra></extra>"
        )

        fig.add_trace(go.Bar(
            x=vr, y=orden, orientation="h",
            marker=dict(
                color=colores,
                cornerradius="30%", # Bordes redondeados tipo píldora (Plotly 5.22+)
                line=dict(color="rgba(255,255,255,0.05)", width=1)
            ),
            name=ambito,
            showlegend=False,
            hovertemplate=hovertemplate))
            
        # Trazos invisibles para forzar la leyenda correcta de colores
        fig.add_trace(go.Bar(x=[None], y=[None], name="Hombres ganan más", marker=dict(color=C_HOMBRE)))
        fig.add_trace(go.Bar(x=[None], y=[None], name="Mujeres ganan más", marker=dict(color=C_MUJER)))
                
        fig.add_vline(x=0, line_color="#E5E7EB", line_dash="dot")
        fig.update_xaxes(title=xlab, title_font=dict(size=10, color="#000000"))
        fig.update_yaxes(title=ylab, title_font=dict(size=10, color="#000000"))
        return base_layout(fig, h=CH_H, l_margin=100)

    # ---------------- Fila 1 de Gráficos ----------------
    r1_c1, r1_c2 = st.columns(2, gap="large")
    
    with r1_c1:
        with st.container(border=True):
            st.markdown(chart_header("Ingreso per cápita del hogar por sexo y nivel educativo (Miles $)"), unsafe_allow_html=True)
            ad = d_ing_f
            ad_n = d_ing_b
            cats = [EDUC[e] for e in range(7)]
            
            rh = [ad[(ad["sexo"]==1)&(ad["educc"]==e)]["ypchtotcor"].mean()/1000 for e in range(7)]
            rm = [ad[(ad["sexo"]==2)&(ad["educc"]==e)]["ypchtotcor"].mean()/1000 for e in range(7)]
            rn = [ad_n[ad_n["educc"]==e]["ypchtotcor"].mean()/1000 for e in range(7)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=rh+[rh[0]], theta=cats+[cats[0]], name="Hombre", line=dict(color=C_HOMBRE, width=3), fill="toself", fillcolor="rgba(79,195,247,0.15)", marker=dict(size=5, color=C_HOMBRE)))
            fig.add_trace(go.Scatterpolar(r=rm+[rm[0]], theta=cats+[cats[0]], name="Mujer", line=dict(color=C_MUJER, width=3), fill="toself", fillcolor="rgba(232,83,122,0.15)", marker=dict(size=5, color=C_MUJER)))
            fig.add_trace(go.Scatterpolar(r=rn+[rn[0]], theta=cats+[cats[0]], name="Nac.", line=dict(color=C_NACIONAL, width=1.5, dash="dash")))
            fig.update_polars(radialaxis=dict(ticksuffix="K", tickprefix="$", gridcolor=GRID), angularaxis=dict(tickfont=dict(size=10, color="#000000"), gridcolor=GRID))
            st.plotly_chart(base_layout(fig, h=CH_H, l_margin=30), width='stretch', config={'displayModeBar':False})

    with r1_c2:
        with st.container(border=True):
            st.markdown(chart_header("Tasa de empleo por educación y sexo"), unsafe_allow_html=True)
            def emp_rates(df):
                out = {}
                for e in range(7):
                    g = df[df["educc"] == e]
                    for s in (1, 2):
                        gs = g[g["sexo"] == s]
                        out[(e, s)] = ((gs["activ"] == 1).mean() * 100 if len(gs) else np.nan)
                return out

            er = emp_rates(d_emp_f)
            er_n = emp_rates(d_emp_b)

            fig = go.Figure()
            for e in range(7):
                fig.add_trace(go.Scatter(x=[er[(e, 2)], er[(e, 1)]], y=[EDUC[e], EDUC[e]], mode="lines", line=dict(color="#E5E7EB", width=3), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[er[(e, 2)] for e in range(7)], y=cats, mode="markers", name="Mujer", marker=dict(color=C_MUJER, size=11), hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=[er[(e, 1)] for e in range(7)], y=cats, mode="markers", name="Hombre", marker=dict(color=C_HOMBRE, size=11), hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            if filtrado:
                fig.add_trace(go.Scatter(x=[er_n[(e, 2)] for e in range(7)], y=cats, mode="markers", name="Mujer Nac.", marker=dict(color=C_MUJER, size=7, symbol="diamond-open"), hovertemplate="Nac: %{x:.1f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=[er_n[(e, 1)] for e in range(7)], y=cats, mode="markers", name="Hombre Nac.", marker=dict(color=C_HOMBRE, size=7, symbol="diamond-open"), hovertemplate="Nac: %{x:.1f}%<extra></extra>"))
            fig.update_xaxes(title="Tasa de empleo (%)", range=[0, 100], title_font=dict(size=10, color="#000000"))
            fig.update_yaxes(title="Nivel educativo", title_font=dict(size=10, color="#000000"))
            st.plotly_chart(base_layout(fig, h=CH_H, l_margin=100), width='stretch', config={'displayModeBar':False})

    # ---------------- Fila 2 de Gráficos ----------------
    r2_c1, r2_c2 = st.columns(2, gap="large")
    
    with r2_c1:
        with st.container(border=True):
            st.markdown(chart_header("Pobreza vs. Ingreso por región"), unsafe_allow_html=True)
            x, y = reg_df["pobreza"], reg_df["ingreso"]
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min() - 1, x.max() + 1, 50)

            # Zonas de cuadrante sombreadas
            x_med, y_med = x.median(), y.median()

            colores = [C_ACENTO if (filtrado and r == cod_sel) else "#CBD5E1" for r in reg_df["region"]]
            fig = go.Figure()
            # Cuadrante favorable (bajo pobreza, alto ingreso)
            fig.add_shape(type="rect", x0=x.min()-2, x1=x_med, y0=y_med, y1=y.max()*1.1, fillcolor="rgba(16,185,129,0.04)", line=dict(width=0), layer="below")
            # Cuadrante desfavorable (alto pobreza, bajo ingreso)
            fig.add_shape(type="rect", x0=x_med, x1=x.max()+2, y0=y.min()*0.9, y1=y_med, fillcolor="rgba(239,68,68,0.04)", line=dict(width=0), layer="below")
            fig.add_trace(go.Scatter(x=xs, y=b1 * xs + b0, mode="lines", name="Tendencia", line=dict(color=C_MUJER, dash="dash", width=1.5)))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers+text", text=reg_df["nombre"], textposition="top center", textfont=dict(size=9, color="#6B7280"), name="Regiones",
                marker=dict(size=np.sqrt(reg_df["poblacion"] / reg_df["poblacion"].max()) * 30 + 6, color=colores, opacity=0.85, line=dict(color="white", width=1.5)),
                hovertemplate=("<b>%{text}</b><br>Pobreza: %{x:.1f}%<br>Ingreso: $%{y:,.0f}K<extra></extra>")
            ))
            fig.update_xaxes(title="Tasa de Pobreza (%)", title_font=dict(size=10, color="#000000"))
            fig.update_yaxes(title="Ingreso per cápita (miles $)", title_font=dict(size=10, color="#000000"))
            st.plotly_chart(base_layout(fig, h=CH_H, l_margin=40), width='stretch', config={'displayModeBar':False})

    with r2_c2:
        with st.container(border=True):
            st.markdown(chart_header("Evolución del ingreso medio y tasa de crecimiento"), unsafe_allow_html=True)
            yrs = [2020, 2022, 2024]
            vn = [wavg(d_evo20, "ypchtotcor")/1000, wavg(d_evo22, "ypchtotcor")/1000, wavg(d_evo24_b, "ypchtotcor")/1000]
            
            fig = go.Figure()
            if filtrado:
                v20r, v24r = wavg(d_evo20[d_evo20["region"]==cod_sel], "ypchtotcor")/1000, wavg(d_evo24_f, "ypchtotcor")/1000
                fig.add_trace(go.Scatter(x=yrs, y=vn, mode="lines+markers", name="Nacional", line=dict(color=C_NACIONAL, width=2, dash="dot")))
                fig.add_trace(go.Scatter(x=[2020, 2024], y=[v20r, v24r], mode="lines+markers+text", name=sel, text=[f"${v20r:,.0f}", f"${v24r:,.0f}"], textposition="top center", textfont=dict(size=12, weight="bold"), line=dict(color=C_ACENTO, width=4), marker=dict(size=10)))
                # Anotación de cambio %
                if v20r and not np.isnan(v20r) and v20r != 0:
                    pct_chg = ((v24r - v20r) / v20r) * 100
                    fig.add_annotation(x=2024, y=v24r, text=f"{'↑' if pct_chg >= 0 else '↓'}{abs(pct_chg):.1f}%", showarrow=False, yshift=22, font=dict(size=11, color="#10B981" if pct_chg >= 0 else "#EF4444", weight="bold"))
            else:
                fig.add_trace(go.Scatter(
                    x=yrs, y=vn,
                    mode="lines+markers+text", name="Nacional",
                    text=[f"${v:,.0f}" for v in vn],
                    textposition=["bottom right", "top left", "top right"],
                    textfont=dict(size=12, weight="bold", color="#111827"),
                    line=dict(color=C_ACENTO, width=4),
                    marker=dict(size=10, color=C_ACENTO)
                ))
                # Anotaciones de crecimiento por período (igual que el PDF)
                if vn[0] and vn[1] and vn[0] != 0 and vn[1] != 0:
                    pct_20_22 = ((vn[1] - vn[0]) / vn[0]) * 100
                    pct_22_24 = ((vn[2] - vn[1]) / vn[1]) * 100
                    for x_pos, y_pos, pct in [
                        (2021, (vn[0]+vn[1])/2, pct_20_22),
                        (2023, (vn[1]+vn[2])/2, pct_22_24),
                    ]:
                        fig.add_annotation(
                            x=x_pos, y=y_pos,
                            text=f"+{pct:.1f}%",
                            showarrow=False,
                            font=dict(size=11, color="#10B981", weight="bold"),
                            bgcolor="rgba(209,250,229,0.9)",
                            bordercolor="#10B981",
                            borderwidth=1, borderpad=4
                        )
            fig.update_xaxes(title="Año", tickvals=yrs, title_font=dict(size=10, color="#000000"))
            fig.update_yaxes(title="Ingreso per cápita (miles $)", title_font=dict(size=10, color="#000000"))
            st.plotly_chart(base_layout(fig, h=CH_H, l_margin=40), width='stretch', config={'displayModeBar':False})

    # ---------------- Fila 3 de Gráficos ----------------
    r3_c1, r3_c2 = st.columns(2, gap="large")
    
    with r3_c1:
        with st.container(border=True):
            st.markdown(chart_header("Brecha en ingresos por sector económico"), unsafe_allow_html=True)
            bs = brecha(d_sec_f, "rama1", SECTORES)
            bs_n = brecha(d_sec_b, "rama1", SECTORES)
            st.plotly_chart(fig_brecha(bs, bs_n, "Ingreso per cápita (miles $)", "Sector económico"), width='stretch', config={'displayModeBar':False})

    with r3_c2:
        with st.container(border=True):
            st.markdown(chart_header("Brecha salarial de género por ocupación"), unsafe_allow_html=True)
            bo = brecha(d_ofi_f, "oficio1_08", OFICIOS)
            bo_n = brecha(d_ofi_b, "oficio1_08", OFICIOS)
            st.plotly_chart(fig_brecha(bo, bo_n, "Diferencia salarial (miles $)", "Tipo de ocupación"), width='stretch', config={'displayModeBar':False})

    # ---------------- Fila 4: Narrativa / Resumen ----------------
    st.markdown(generate_narrative(sel, cod_sel, d_mapa_b) if filtrado else generate_national_narrative(d_mapa_b), unsafe_allow_html=True)
