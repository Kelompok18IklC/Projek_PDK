"""
Dashboard Sea Surface Temperature (SST)
Jalankan dengan: streamlit run Revisi.py
"""

import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard SST",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  TEMA LAUT — CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;600;700&display=swap');

/* ── Background utama ── */
.stApp {
    background: linear-gradient(160deg, #020e1f 0%, #041c33 40%, #062a4a 100%);
    font-family: 'Exo 2', sans-serif;
    color: #d0eaf8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031525 0%, #052238 100%);
    border-right: 1px solid #0a4a6e;
}
[data-testid="stSidebar"] * { color: #a8d8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #7ec8e3 !important; font-weight: 600; }

/* ── Header utama ── */
.sst-header {
    background: linear-gradient(90deg, #041c33, #0a4a6e, #0e7490, #0a4a6e, #041c33);
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #1e6fa0;
    box-shadow: 0 0 32px rgba(14,116,144,0.4), inset 0 0 60px rgba(0,0,0,0.3);
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

.sst-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e0f7ff;
    text-shadow: 0 0 20px rgba(56,189,248,0.7);
    margin: 0 0 6px 0;
    letter-spacing: 1px;
}
.sst-header p {
    color: #7ec8e3;
    margin: 0;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* ── Wave divider ── */
.wave-divider {
    text-align: center;
    font-size: 1.5rem;
    letter-spacing: 4px;
    color: #0e7490;
    margin: 4px 0 20px 0;
    animation: wave 2s ease-in-out infinite;
}
@keyframes wave { 0%,100%{opacity:0.4} 50%{opacity:1} }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #031e35, #052d4a);
    border: 1px solid #1565a0;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 15px rgba(0,80,160,0.25);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,150,220,0.35);
}
[data-testid="stMetricLabel"] { color: #7ec8e3 !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.6rem !important; font-weight: 700; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #031525;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #0a4a6e;
}
[data-testid="stTabs"] button {
    color: #7ec8e3 !important;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #0369a1, #0e7490) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(14,116,144,0.5);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #031e35;
    border: 1px solid #1565a0;
    border-radius: 10px;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid #1565a0; border-radius: 10px; }

/* ── Selectbox / Slider / Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #031e35 !important;
    border: 1px solid #1565a0 !important;
    color: #a8d8f0 !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #0369a1, #0e7490);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #0284c7, #0891b2);
    box-shadow: 0 4px 15px rgba(14,116,144,0.5);
    transform: translateY(-2px);
}

/* ── Caption / footer ── */
.footer-text {
    text-align: center;
    color: #4a9ab5;
    font-size: 0.8rem;
    padding: 12px;
    border-top: 1px solid #0a4a6e;
    margin-top: 20px;
}

/* ── Info/error box ── */
[data-testid="stAlert"] {
    background: #031e35 !important;
    border-left: 4px solid #0e7490 !important;
    color: #a8d8f0 !important;
    border-radius: 8px;
}

/* ── Subheader ── */
h2, h3 { color: #38bdf8 !important; }
h4 { color: #7ec8e3 !important; }

/* ── Plot background ── */
.stPlotlyChart, .stpyplot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KONSTANTA — SESUAIKAN DI SINI
# ─────────────────────────────────────────────
DATA_DIR  = Path(".")
NC_FILES  = {
    "2010-2011": DATA_DIR / "2010-2011.nc",
    "2012-2013": DATA_DIR / "2012-2013.nc",
    "2014-2015": DATA_DIR / "2014-2015.nc",
    "2016-2017": DATA_DIR / "2016-2017.nc",
    "2018-2019": DATA_DIR / "2018-2019.nc",
}

VAR_LAT   = "lat"
VAR_LON   = "lon"
VAR_TIME  = "time"
VAR_MAIN  = "sst"           # ganti sesuai nama variabel di file .nc
VAR_LABEL = "SST (°C)"
COLORMAP  = "RdYlBu_r"      # colormap panas-dingin cocok untuk SST

# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="🌊 Memuat data SST…")
def load_dataset(period: str) -> xr.Dataset:
    return xr.open_dataset(NC_FILES[period], decode_times=True)

@st.cache_data(show_spinner=False)
def get_variable_list(period: str) -> list:
    return list(load_dataset(period).data_vars)

def auto_detect_var(ds: xr.Dataset) -> str:
    if VAR_MAIN in ds:
        return VAR_MAIN
    candidates = [v for v in ds.data_vars if ds[v].ndim >= 2]
    return candidates[0] if candidates else list(ds.data_vars)[0]

def ocean_style_fig(facecolor="#020e1f"):
    """Return matplotlib fig/ax dengan background tema laut."""
    fig, ax = plt.subplots(facecolor=facecolor)
    ax.set_facecolor("#031525")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1565a0")
    ax.tick_params(colors="#7ec8e3")
    ax.xaxis.label.set_color("#7ec8e3")
    ax.yaxis.label.set_color("#7ec8e3")
    ax.title.set_color("#38bdf8")
    ax.grid(True, color="#0a4a6e", alpha=0.5, linestyle="--")
    return fig, ax

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="sst-header">
  <h1>🌊 Dashboard Sea Surface Temperature (SST)</h1>
  <p>Visualisasi & Analisis Data Suhu Permukaan Laut · NetCDF · 2010 – 2019</p>
</div>
<div class="wave-divider">〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
      <span style='font-size:3rem;'>🌊</span><br>
      <span style='font-size:1.1rem; font-weight:700; color:#38bdf8; letter-spacing:1px;'>SST Dashboard</span><br>
      <span style='font-size:0.75rem; color:#4a9ab5;'>PROJEK PDK</span>
    </div>
    <hr style='border-color:#0a4a6e; margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    selected_period = st.selectbox("📅 Pilih Periode", list(NC_FILES.keys()))

    try:
        ds = load_dataset(selected_period)
        var_name = auto_detect_var(ds)
        da = ds[var_name]
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan:\n`{NC_FILES[selected_period]}`")
        st.stop()

    available_vars = get_variable_list(selected_period)
    if len(available_vars) > 1:
        var_name = st.selectbox("📊 Pilih Variabel", available_vars,
                                index=available_vars.index(var_name))
        da = ds[var_name]

    has_time = VAR_TIME in da.dims
    time_index = 0
    if has_time:
        n_times = da.sizes[VAR_TIME]
        time_index = st.slider("🕒 Timestep", 0, n_times - 1, 0)
        try:
            time_val = pd.Timestamp(da[VAR_TIME].values[time_index])
            st.markdown(f"<p style='color:#38bdf8; font-weight:600; text-align:center;'>📆 {time_val.strftime('%d %B %Y')}</p>",
                        unsafe_allow_html=True)
        except Exception:
            st.caption(f"Timestep: {time_index}")

    st.markdown("<hr style='border-color:#0a4a6e;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7ec8e3; font-weight:600; font-size:0.9rem;'>⚖️ Perbandingan Periode</p>",
                unsafe_allow_html=True)
    compare_periods = st.multiselect(
        "Pilih periode",
        list(NC_FILES.keys()),
        default=list(NC_FILES.keys())[:2],
    )

    st.markdown("""
    <hr style='border-color:#0a4a6e;'>
    <p style='text-align:center; color:#4a9ab5; font-size:0.75rem;'>
      🌊 Streamlit · xarray · Cartopy<br>Python 3 · PROJEK PDK
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  METRIC ROW
# ─────────────────────────────────────────────
try:
    _arr_all = da.values.flatten()
    _arr_all = _arr_all[~np.isnan(_arr_all)]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ SST Min",    f"{_arr_all.min():.2f} °C")
    c2.metric("🌡️ SST Max",    f"{_arr_all.max():.2f} °C")
    c3.metric("📊 Rata-rata",  f"{_arr_all.mean():.2f} °C")
    c4.metric("📐 Median",     f"{np.median(_arr_all):.2f} °C")
    c5.metric("📉 Std Dev",    f"{_arr_all.std():.2f} °C")
except Exception:
    pass

st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TAB UTAMA
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Peta SST",
    "📈  Time Series",
    "📊  Statistik & Tabel",
    "⚖️  Perbandingan Tahun",
])

# ══════════════════════════════════════════════
#  TAB 1 — PETA
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"🗺️ Peta Suhu Permukaan Laut — {selected_period}")

    if has_time:
        data_2d = da.isel({VAR_TIME: time_index})
    else:
        data_2d = da.isel(0) if da.ndim > 2 else da

    while data_2d.ndim > 2:
        data_2d = data_2d.isel({data_2d.dims[0]: 0})

    lat_dim = VAR_LAT if VAR_LAT in data_2d.dims else data_2d.dims[-2]
    lon_dim = VAR_LON if VAR_LON in data_2d.dims else data_2d.dims[-1]

    col_map, col_info = st.columns([3, 1])
    with col_map:
        try:
            fig, ax = plt.subplots(
                figsize=(11, 6),
                facecolor="#020e1f",
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            ax.set_facecolor("#031525")
            lat_vals = data_2d[lat_dim].values
            lon_vals = data_2d[lon_dim].values
            vals     = data_2d.values

            im = ax.pcolormesh(lon_vals, lat_vals, vals,
                               transform=ccrs.PlateCarree(),
                               cmap=COLORMAP, shading="auto")
            ax.add_feature(cfeature.OCEAN,     facecolor="#041c33")
            ax.add_feature(cfeature.LAND,      facecolor="#1a2a1a", alpha=0.85)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#38bdf8")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle="--", edgecolor="#4a9ab5")
            gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                              color="#0a4a6e", alpha=0.7)
            gl.xlabel_style = {"color": "#7ec8e3", "size": 8}
            gl.ylabel_style = {"color": "#7ec8e3", "size": 8}
            cb = plt.colorbar(im, ax=ax, label=VAR_LABEL, shrink=0.8, pad=0.02)
            cb.ax.yaxis.label.set_color("#7ec8e3")
            cb.ax.tick_params(colors="#7ec8e3")
            cb.outline.set_edgecolor("#0a4a6e")
            ax.set_title(f"Sea Surface Temperature — {selected_period}",
                         fontsize=13, fontweight="bold", color="#38bdf8", pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Cartopy error: {e} — fallback mode")
            fig, ax = plt.subplots(figsize=(11, 5), facecolor="#020e1f")
            ax.set_facecolor("#031525")
            im = ax.imshow(data_2d.values, origin="lower", aspect="auto", cmap=COLORMAP)
            cb = plt.colorbar(im, ax=ax, label=VAR_LABEL)
            cb.ax.yaxis.label.set_color("#7ec8e3")
            cb.ax.tick_params(colors="#7ec8e3")
            ax.set_title(f"SST — {selected_period}", color="#38bdf8", fontweight="bold")
            ax.tick_params(colors="#7ec8e3")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_info:
        flat = data_2d.values.flatten()
        flat = flat[~np.isnan(flat)]
        st.markdown("**📌 Info Slice**")
        st.metric("Min", f"{flat.min():.2f} °C")
        st.metric("Max", f"{flat.max():.2f} °C")
        st.metric("Mean", f"{flat.mean():.2f} °C")
        st.metric("Std", f"{flat.std():.2f} °C")
        st.markdown(f"**Grid:** `{data_2d.sizes[lat_dim]} × {data_2d.sizes[lon_dim]}`")
        st.markdown(f"**Colormap:** `{COLORMAP}`")

# ══════════════════════════════════════════════
#  TAB 2 — TIME SERIES
# ══════════════════════════════════════════════
with tab2:
    st.subheader(f"📈 Time Series SST — {selected_period}")

    if not has_time:
        st.info("Dataset ini tidak memiliki dimensi waktu.")
    else:
        spatial_dims = [d for d in da.dims if d != VAR_TIME]
        ts = da.mean(dim=spatial_dims)

        try:
            time_vals = pd.to_datetime(ts[VAR_TIME].values)
            df_ts = pd.DataFrame({"Tanggal": time_vals, VAR_LABEL: ts.values})
            x_col = "Tanggal"
        except Exception:
            df_ts = pd.DataFrame({"Timestep": range(len(ts)), VAR_LABEL: ts.values})
            x_col = "Timestep"

        fig, ax = plt.subplots(figsize=(13, 4), facecolor="#020e1f")
        ax.set_facecolor("#031525")
        ax.plot(df_ts[x_col], df_ts[VAR_LABEL], color="#38bdf8", linewidth=1.8, zorder=3)
        ax.fill_between(df_ts[x_col], df_ts[VAR_LABEL],
                        alpha=0.2, color="#0e7490")
        ax.set_xlabel(x_col, color="#7ec8e3")
        ax.set_ylabel(VAR_LABEL, color="#7ec8e3")
        ax.set_title(f"Rata-rata Spasial SST — {selected_period}",
                     fontweight="bold", color="#38bdf8")
        ax.tick_params(colors="#7ec8e3")
        ax.grid(True, color="#0a4a6e", alpha=0.5, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1565a0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("📉 Tampilkan Moving Average"):
            window = st.slider("Window (timestep)", 2, min(60, len(df_ts)//2), 12)
            df_ts["Rolling"] = df_ts[VAR_LABEL].rolling(window, center=True).mean()
            fig2, ax2 = plt.subplots(figsize=(13, 4), facecolor="#020e1f")
            ax2.set_facecolor("#031525")
            ax2.plot(df_ts[x_col], df_ts[VAR_LABEL],
                     alpha=0.35, color="#38bdf8", linewidth=1, label="Original")
            ax2.plot(df_ts[x_col], df_ts["Rolling"],
                     linewidth=2.2, color="#f97316", label=f"MA-{window}")
            ax2.legend(facecolor="#031525", edgecolor="#1565a0", labelcolor="#d0eaf8")
            ax2.tick_params(colors="#7ec8e3")
            ax2.grid(True, color="#0a4a6e", alpha=0.5, linestyle="--")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#1565a0")
            ax2.set_title("Moving Average SST", color="#38bdf8", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

# ══════════════════════════════════════════════
#  TAB 3 — STATISTIK & TABEL
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Statistik Deskriptif Semua Periode")

    rows = []
    for period, path in NC_FILES.items():
        try:
            _ds  = load_dataset(period)
            _var = auto_detect_var(_ds)
            _arr = _ds[_var].values.flatten()
            _arr = _arr[~np.isnan(_arr)]
            rows.append({
                "Periode": period,
                "Min (°C)":    round(float(_arr.min()), 4),
                "Max (°C)":    round(float(_arr.max()), 4),
                "Mean (°C)":   round(float(_arr.mean()), 4),
                "Median (°C)": round(float(np.median(_arr)), 4),
                "Std (°C)":    round(float(_arr.std()), 4),
                "N Grid":      int(len(_arr)),
            })
        except Exception:
            rows.append({"Periode": period, "Min (°C)": "N/A", "Max (°C)": "N/A",
                         "Mean (°C)": "N/A", "Median (°C)": "N/A",
                         "Std (°C)": "N/A", "N Grid": 0})

    df_stat = pd.DataFrame(rows).set_index("Periode")
    st.dataframe(
        df_stat.style
               .format("{:.4f}", subset=["Min (°C)","Max (°C)","Mean (°C)","Median (°C)","Std (°C)"])
               .highlight_max(subset=["Mean (°C)"], color="#0a3a5e")
               .highlight_min(subset=["Mean (°C)"], color="#0a2a3e"),
        use_container_width=True,
    )

    st.markdown("#### 🌊 Distribusi Nilai SST")
    arr_curr = da.values.flatten()
    arr_curr = arr_curr[~np.isnan(arr_curr)]
    fig3, ax3 = plt.subplots(figsize=(11, 4), facecolor="#020e1f")
    ax3.set_facecolor("#031525")
    ax3.hist(arr_curr, bins=70, color="#0e7490", alpha=0.85, edgecolor="#020e1f")
    ax3.axvline(arr_curr.mean(),      color="#f97316", linestyle="--", linewidth=1.8,
                label=f"Mean: {arr_curr.mean():.3f} °C")
    ax3.axvline(np.median(arr_curr),  color="#22d3ee", linestyle="--", linewidth=1.8,
                label=f"Median: {np.median(arr_curr):.3f} °C")
    ax3.set_xlabel(VAR_LABEL, color="#7ec8e3")
    ax3.set_ylabel("Frekuensi", color="#7ec8e3")
    ax3.set_title(f"Distribusi SST — {selected_period}", color="#38bdf8", fontweight="bold")
    ax3.tick_params(colors="#7ec8e3")
    ax3.legend(facecolor="#031525", edgecolor="#1565a0", labelcolor="#d0eaf8")
    ax3.grid(True, color="#0a4a6e", alpha=0.5, linestyle="--")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1565a0")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.download_button(
        "⬇️ Download Statistik CSV",
        df_stat.to_csv().encode(),
        file_name=f"statistik_SST_{selected_period}.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════
#  TAB 4 — PERBANDINGAN
# ══════════════════════════════════════════════
with tab4:
    st.subheader("⚖️ Perbandingan SST Antar Periode")

    if len(compare_periods) < 2:
        st.warning("🌊 Pilih minimal 2 periode di sidebar untuk perbandingan.")
    else:
        means, stds, labels = [], [], []
        for p in compare_periods:
            try:
                _ds  = load_dataset(p)
                _var = auto_detect_var(_ds)
                _arr = _ds[_var].values.flatten()
                _arr = _arr[~np.isnan(_arr)]
                means.append(_arr.mean())
                stds.append(_arr.std())
                labels.append(p)
            except Exception:
                pass

        OCEAN_COLORS = ["#0e7490","#0369a1","#22d3ee","#38bdf8","#7dd3fc"]

        col_bar, col_box = st.columns(2)
        with col_bar:
            fig4, ax4 = plt.subplots(figsize=(7, 5), facecolor="#020e1f")
            ax4.set_facecolor("#031525")
            cols = OCEAN_COLORS[:len(labels)]
            bars = ax4.bar(labels, means, yerr=stds, capsize=5,
                           color=cols, alpha=0.9, edgecolor="#020e1f", linewidth=0.8)
            ax4.set_ylabel(VAR_LABEL, color="#7ec8e3")
            ax4.set_title("Rata-rata ± Std SST per Periode",
                          color="#38bdf8", fontweight="bold")
            ax4.tick_params(colors="#7ec8e3")
            ax4.grid(True, axis="y", color="#0a4a6e", alpha=0.5, linestyle="--")
            for spine in ax4.spines.values():
                spine.set_edgecolor("#1565a0")
            plt.xticks(rotation=30)
            for bar, m in zip(bars, means):
                ax4.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(stds)*0.02,
                         f"{m:.2f}°C", ha="center", va="bottom",
                         fontsize=9, color="#e0f7ff", fontweight="600")
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        with col_box:
            fig5, ax5 = plt.subplots(figsize=(7, 5), facecolor="#020e1f")
            ax5.set_facecolor("#031525")
            box_data = []
            for p in compare_periods:
                try:
                    _ds  = load_dataset(p)
                    _var = auto_detect_var(_ds)
                    _arr = _ds[_var].values.flatten()
                    _arr = _arr[~np.isnan(_arr)]
                    if len(_arr) > 50_000:
                        _arr = np.random.choice(_arr, 50_000, replace=False)
                    box_data.append(_arr)
                except Exception:
                    box_data.append(np.array([]))

            bp = ax5.boxplot(box_data, labels=labels, patch_artist=True, notch=True,
                             medianprops={"color":"#f97316","linewidth":2})
            for patch, c in zip(bp["boxes"], OCEAN_COLORS):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            for element in ["whiskers","caps","fliers"]:
                for item in bp[element]:
                    item.set_color("#7ec8e3")
            ax5.set_ylabel(VAR_LABEL, color="#7ec8e3")
            ax5.set_title("Boxplot SST per Periode", color="#38bdf8", fontweight="bold")
            ax5.tick_params(colors="#7ec8e3")
            ax5.grid(True, axis="y", color="#0a4a6e", alpha=0.5, linestyle="--")
            for spine in ax5.spines.values():
                spine.set_edgecolor("#1565a0")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

        if has_time:
            st.markdown("#### 📈 Overlay Time Series SST Semua Periode")
            fig6, ax6 = plt.subplots(figsize=(13, 5), facecolor="#020e1f")
            ax6.set_facecolor("#031525")
            for p, col in zip(compare_periods, OCEAN_COLORS):
                try:
                    _ds   = load_dataset(p)
                    _var  = auto_detect_var(_ds)
                    _da   = _ds[_var]
                    _dims = [d for d in _da.dims if d != VAR_TIME]
                    _ts   = _da.mean(dim=_dims).values
                    ax6.plot(range(len(_ts)), _ts, label=p, color=col, linewidth=1.8)
                except Exception:
                    pass
            ax6.set_xlabel("Timestep", color="#7ec8e3")
            ax6.set_ylabel(VAR_LABEL, color="#7ec8e3")
            ax6.set_title("Overlay Rata-rata Spasial SST per Periode",
                          color="#38bdf8", fontweight="bold")
            ax6.tick_params(colors="#7ec8e3")
            ax6.legend(facecolor="#031525", edgecolor="#1565a0", labelcolor="#d0eaf8")
            ax6.grid(True, color="#0a4a6e", alpha=0.5, linestyle="--")
            for spine in ax6.spines.values():
                spine.set_edgecolor("#1565a0")
            plt.tight_layout()
            st.pyplot(fig6, use_container_width=True)
            plt.close(fig6)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer-text'>
  🌊 Dashboard Sea Surface Temperature (SST) &nbsp;·&nbsp; PROJEK PDK
  &nbsp;·&nbsp; Streamlit + xarray + Cartopy &nbsp;·&nbsp; Python 3
</div>
""", unsafe_allow_html=True)