"""
FFIEC Neural Network Anomaly Detection Dashboard (Track 2)
==========================================================
Layperson-friendly Streamlit dashboard for exploring per-bank anomaly
detection results.

  - Levels model: pure autoencoder reconstruction error (anomaly_score)
  - QoQ model: autoencoder + LOF ensemble (ensemble_score)

Usage:
    streamlit run NeuralDashboard_0.1.py

Author: Wake Forest MSBA Practicum Team 4
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("per_bank_nn")
LEVELS_DIR = Path("per_bank_nn_levels")
QOQ_RAW_DIR = Path("per_bank_qoq")
FEATURES_RAW_DIR = Path("per_bank_features")
ISO_DIR = Path("iso_output")
FORECAST_DIR = Path("forecasts_1")

# Last quarter of actual reported data — everything after this is a projection
FORECAST_CUTOFF = pd.Timestamp("2025-09-30")

# Map dashboard display names → forecast CSV filenames
FORECAST_BANK_FILES = {
    "Bank of America":       "BofaForecasted.csv",
    "Citibank":              "CitiForecasted.csv",
    "JPMorgan Chase Bank":   "JPMorganForecasted.csv",
    "Goldman Sachs Bank USA": "ffiec_goldman_sachs_arima112_forecast.csv",
    "Wells Fargo Bank":      "ffiec_wells_fargo_arima112_forecast.csv",
    "Morgan Stanley Bank":   "morgan_stanley_forecast.csv",
}

BANK_DISPLAY_NAMES = {
    'bank_of_america': 'Bank of America',
    'jpmorgan_chase_bank': 'JPMorgan Chase Bank',
    'citibank': 'Citibank',
    'wells_fargo_bank': 'Wells Fargo Bank',
    'goldman_sachs_bank_usa': 'Goldman Sachs Bank USA',
    'morgan_stanley_bank': 'Morgan Stanley Bank',
}

# Accent color for flagged items — readable on both light and dark backgrounds
ANOMALY_RED = '#E63946'
NORMAL_BLUE = '#457B9D'
NN_GREEN = '#2A9D8F'
LOF_GOLD = '#E9C46A'


def _score_col(df):
    """Return the primary score column name for a given dataframe.
    QoQ model uses 'ensemble_score'; Levels model uses 'anomaly_score'.
    Falls back to 'reconstruction_error' if neither exists."""
    for col in ('ensemble_score', 'anomaly_score', 'reconstruction_error'):
        if col in df.columns:
            return col
    return None


def _ensure_score_cols(df):
    """Guarantee both 'anomaly_score' and 'ensemble_score' exist, by aliasing
    whichever one is present. Modifies df in place and returns it."""
    if 'ensemble_score' in df.columns and 'anomaly_score' not in df.columns:
        df['anomaly_score'] = df['ensemble_score']
    elif 'anomaly_score' in df.columns and 'ensemble_score' not in df.columns:
        df['ensemble_score'] = df['anomaly_score']
    elif 'reconstruction_error' in df.columns and 'anomaly_score' not in df.columns:
        df['anomaly_score'] = df['reconstruction_error']
        df['ensemble_score'] = df['reconstruction_error']
    return df


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_mdrm_lookup():
    """Load the MDRM data dictionary for translating codes to English.
    Returns (name_dict, type_dict) — name_dict maps code→description,
    type_dict maps code→ItemType (F=financial/$thousands, P=percent, R=ratio, S=count, D=date).
    """
    lookup_path = DATA_DIR / "mdrm_lookup.csv"
    if not lookup_path.exists():
        lookup_path = Path("mdrm_lookup.csv")
    if lookup_path.exists():
        df = pd.read_csv(lookup_path, dtype=str)
        name_dict = dict(zip(df['code'], df['description']))
        type_dict = {}
        if 'item_type' in df.columns:
            type_dict = dict(zip(df['code'], df['item_type'].fillna('F')))
        return name_dict, type_dict
    return {}, {}


# Unit labels by ItemType
UNIT_LABELS = {
    'F': '$ thousands',
    'P': '%',
    'R': 'Ratio',
    'S': 'Count',
    'D': 'Date',
}


def get_unit_label(code, type_dict):
    """Get the unit label for a given MDRM code."""
    clean = code.replace('_qoq', '')
    item_type = type_dict.get(clean, 'F')
    return UNIT_LABELS.get(item_type, '')


def _truncate(text, max_len=55):
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1].rstrip() + "…"


def translate_code(code, mdrm_dict, truncate=False):
    """Translate an MDRM code like RCFD0211_qoq to its English name."""
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    if name:
        display = _truncate(name) if truncate else name
        return f"{display} ({clean})"
    return clean


def chart_label(code, mdrm_dict):
    """Short label for chart axes — truncated name + code."""
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    if name:
        return f"{_truncate(name, 45)} ({clean})"
    return clean


def short_name(code, mdrm_dict):
    """Get just the English name, or the raw code if no translation."""
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    return name if name else clean


def _load_bank_csvs(directory, glob_pattern):
    """Generic loader for per-bank anomaly CSVs."""
    banks = {}
    if not Path(directory).exists():
        return banks

    for filepath in sorted(Path(directory).glob(glob_pattern)):
        if 'flagged' in filepath.stem:
            continue

        # Extract bank slug from filename
        slug = filepath.stem
        for prefix in ['ffiec_']:
            slug = slug.replace(prefix, '')
        for suffix in ['_nn_anomalies', '_nn_levels_anomalies']:
            slug = slug.replace(suffix, '')
        display_name = BANK_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception:
            try:
                df = pd.read_excel(filepath, engine='xlrd')
            except Exception:
                continue

        # Handle transposed format
        if 'feature' in df.columns:
            feature_names = df['feature'].tolist()
            df_t = df.drop(columns=['feature']).T
            df_t.columns = feature_names
            df_t.index.name = 'quarter'
            df_t = df_t.reset_index()
            df_t.rename(columns={'index': 'quarter'}, inplace=True)
            df = df_t

        # Force score columns to numeric
        for col in ['nn_score', 'lof_score', 'ensemble_score',
                    'reconstruction_error', 'anomaly_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Force boolean columns
        for col in ['nn_anomaly', 'lof_anomaly', 'is_anomaly']:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Normalize: ensure both score columns exist regardless of model version
        df = _ensure_score_cols(df)

        # Parse quarter into sortable date
        if 'quarter' in df.columns:
            df['quarter_date'] = pd.to_datetime(
                df['quarter'], format='%m/%d/%Y', errors='coerce'
            )
            df = df.sort_values('quarter_date').reset_index(drop=True)

        banks[display_name] = df

    return banks


@st.cache_data
def load_all_banks():
    """Load QoQ-model anomaly CSVs."""
    return _load_bank_csvs(DATA_DIR, "ffiec_*_nn_anomalies.*")


@st.cache_data
def load_levels_banks():
    """Load levels-model anomaly CSVs."""
    return _load_bank_csvs(LEVELS_DIR, "ffiec_*_nn_levels_anomalies.csv")


@st.cache_data
def load_iso_banks():
    """Load Isolation Forest results from ISO_DIR.
    Splits the all_results CSV by bank into per-bank DataFrames.
    Returns dict of {bank_name: DataFrame}.
    """
    all_path = ISO_DIR / "iso_absolute_all_results.csv"
    if not all_path.exists():
        return {}

    try:
        df = pd.read_csv(all_path)
    except Exception:
        return {}

    # Ensure numeric
    for col in ('anom_score', 'systemic_score', 'adjusted_score'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Boolean flag
    if 'is_anomaly' in df.columns:
        df['is_anomaly'] = df['is_anomaly'].astype(bool)

    # Parse quarter date (format: MM/DD/YYYY)
    if 'quarter' in df.columns:
        df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
        df = df.sort_values(['bank_name', 'quarter_date']).reset_index(drop=True)

    # Alias score columns so the rest of the dashboard can treat it uniformly
    # anom_score → ensemble_score, adjusted_score → anomaly_score
    df['ensemble_score'] = df['anom_score']
    df['anomaly_score'] = df['adjusted_score']

    banks = {}
    for bank_name, group in df.groupby('bank_name'):
        banks[bank_name] = group.reset_index(drop=True)

    return banks


@st.cache_data
def load_raw_data(data_dir, suffix):
    """
    Load raw per-bank data (features or QoQ) from transposed CSVs.
    Returns dict of {bank_name: DataFrame} with rows=quarters, cols=features.
    """
    banks = {}
    if not Path(data_dir).exists():
        return banks

    for filepath in sorted(Path(data_dir).glob(f"ffiec_*_{suffix}.csv")):
        slug = filepath.stem.replace('ffiec_', '').replace(f'_{suffix}', '')
        display_name = BANK_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception:
            continue

        # Transpose if needed (features as rows, quarters as columns)
        if 'feature' in df.columns:
            feature_names = df['feature'].tolist()
            df_t = df.drop(columns=['feature']).T
            df_t.columns = feature_names
            df_t.index.name = 'quarter'
            df_t = df_t.reset_index()
            df_t.rename(columns={'index': 'quarter'}, inplace=True)
            df = df_t

        # Parse quarter dates
        if 'quarter' in df.columns:
            df['quarter_date'] = pd.to_datetime(
                df['quarter'], format='%m/%d/%Y', errors='coerce'
            )
            df = df.sort_values('quarter_date').reset_index(drop=True)

        # Force all data columns numeric
        for col in df.columns:
            if col not in ('quarter', 'quarter_date'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        banks[display_name] = df

    return banks


@st.cache_data
def load_forecasts(bank_name):
    """Load projected values for a bank beyond the forecast cutoff.

    Returns a long-format DataFrame with columns:
        feature, quarter, quarter_date, value
    Only includes quarters after FORECAST_CUTOFF (i.e., Q4 2025 onward).
    Returns None if no forecast file exists for this bank.
    """
    fname = FORECAST_BANK_FILES.get(bank_name)
    if not fname:
        return None
    fpath = FORECAST_DIR / fname
    if not fpath.exists():
        return None

    try:
        df = pd.read_csv(fpath, index_col=0)
    except Exception:
        return None

    # Wide → long: rows=features, cols=quarters
    df = df.reset_index().rename(columns={"index": "feature"})
    df_long = df.melt(id_vars="feature", var_name="quarter", value_name="value")
    df_long["quarter_date"] = pd.to_datetime(df_long["quarter"], errors="coerce")
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Keep only the projected portion
    df_long = df_long[df_long["quarter_date"] > FORECAST_CUTOFF].copy()
    df_long = df_long.sort_values(["feature", "quarter_date"]).reset_index(drop=True)
    return df_long


def parse_year(quarter_str):
    """Extract year from quarter string like '03/31/2008'."""
    try:
        return int(str(quarter_str).split('/')[-1])
    except (ValueError, IndexError):
        return None


def plotly_theme_layout():
    """Return a dict of Plotly layout settings that work in both light/dark."""
    return dict(
        template="plotly_dark" if _is_dark_theme() else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(l=20, r=20, t=30, b=40),
    )


def _is_dark_theme():
    """Best-effort check for Streamlit dark theme."""
    try:
        theme = st.get_option("theme.base")
        return theme == "dark"
    except Exception:
        return True  # Default to dark-friendly


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="FFIEC Anomaly Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only what Streamlit can't do natively
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .anomaly-badge {
        display: inline-block;
        background: #E63946;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL COLOR PALETTE
# =============================================================================

MODEL_COLORS = {
    'QoQ NN':           NORMAL_BLUE,
    'Levels NN':        NN_GREEN,
    'Isolation Forest': LOF_GOLD,
}

# =============================================================================
# MAIN APP
# =============================================================================


def main():
    st.title("FFIEC Anomaly Detection Dashboard")
    st.caption("Multi-model anomaly detection across 6 major U.S. banks")

    # =========================================================================
    # LOAD ALL DATA
    # =========================================================================
    qoq_banks    = load_all_banks()
    levels_banks = load_levels_banks()
    iso_banks    = load_iso_banks()
    raw_features = load_raw_data(FEATURES_RAW_DIR, "features")
    raw_qoq      = load_raw_data(QOQ_RAW_DIR, "qoq")
    mdrm, mdrm_types = load_mdrm_lookup()
    # Forecasts loaded per-bank on demand (cached), so we just pre-load for selected bank
    # after sidebar is rendered (selected_bank is known there)

    has_qoq    = len(qoq_banks) > 0
    has_levels = len(levels_banks) > 0
    has_iso    = len(iso_banks) > 0

    if not has_qoq and not has_levels and not has_iso:
        st.error("No model output found. Check that per_bank_nn/, per_bank_nn_levels/, or iso_output/ contain results.")
        return

    available_models = [m for m, flag in [
        ('QoQ NN',           has_qoq),
        ('Levels NN',        has_levels),
        ('Isolation Forest', has_iso),
    ] if flag]

    all_bank_names = set()
    for d in (qoq_banks, levels_banks, iso_banks):
        all_bank_names.update(d.keys())
    bank_names = sorted(all_bank_names)

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("Settings")

        selected_bank = st.selectbox("Select Bank", bank_names, index=0)

        st.divider()

        # Year range — union of all model quarters
        all_years = set()
        for d in (qoq_banks, levels_banks, iso_banks):
            for df_tmp in d.values():
                if 'quarter' in df_tmp.columns:
                    yrs = df_tmp['quarter'].apply(parse_year).dropna().astype(int)
                    all_years.update(yrs.tolist())
        if all_years:
            min_yr, max_yr = int(min(all_years)), int(max(all_years))
            year_range = st.slider(
                "Year Range", min_yr, max_yr, (min_yr, max_yr),
                help="Filters all charts and tables."
            )
        else:
            year_range = None

        st.divider()

        st.subheader("Projections")
        show_forecast = st.checkbox(
            "Show projected values (2026–2027)",
            value=False,
            help=(
                "Overlays statistical projections for future quarters as a dashed line. "
                "Projections are based on each bank's historical trends and are not guaranteed. "
                "Use as context, not as a definitive outlook."
            ),
        )
        if show_forecast:
            st.caption(
                "Projections run through Q3 2027. "
                "**Dashed line** = projected. **Solid line** = reported data."
            )

        st.divider()

        st.subheader("How to read this dashboard")
        st.markdown(
            "Three models scan each bank's financial data for unusual quarters. "
            "**High-confidence flags** — quarters where 2 or 3 models agree — are the most meaningful.\n\n"
            "- **QoQ NN**: Detects unusual quarter-over-quarter *changes*\n"
            "- **Levels NN**: Detects unusual *absolute values*\n"
            "- **Isolation Forest**: Cross-bank model; separates bank-specific from market-wide events\n\n"
            "Drill into any quarter to see which financial line items were most unusual."
        )

        st.divider()
        st.caption(f"Models loaded: **{', '.join(available_models)}**")
        if mdrm:
            st.caption(f"Line item dictionary: **{len(mdrm):,} codes**")

    # Load forecast for the selected bank (cached, fast after first call)
    bank_forecast_df = load_forecasts(selected_bank) if show_forecast else None

    # =========================================================================
    # HELPER: year filter
    # =========================================================================
    def _filter_years(dataframe):
        if year_range is None or 'quarter' not in dataframe.columns:
            return dataframe
        yrs = dataframe['quarter'].apply(parse_year)
        return dataframe[(yrs >= year_range[0]) & (yrs <= year_range[1])].copy()

    # =========================================================================
    # BUILD CONVERGENCE TABLE (bank × quarter × model flags)
    # =========================================================================
    conv_records = []
    for bank in bank_names:
        qoq_flags    = {}
        levels_flags = {}
        iso_flags    = {}

        if bank in qoq_banks:
            for _, r in _filter_years(qoq_banks[bank]).iterrows():
                qoq_flags[r['quarter']] = bool(r.get('is_anomaly', False))
        if bank in levels_banks:
            for _, r in _filter_years(levels_banks[bank]).iterrows():
                levels_flags[r['quarter']] = bool(r.get('is_anomaly', False))
        if bank in iso_banks:
            for _, r in _filter_years(iso_banks[bank]).iterrows():
                iso_flags[r['quarter']] = bool(r.get('is_anomaly', False))

        all_qs = set(qoq_flags) | set(levels_flags) | set(iso_flags)
        for q in all_qs:
            qf = qoq_flags.get(q, False)
            lf = levels_flags.get(q, False)
            isof = iso_flags.get(q, False)
            n = sum([qf, lf, isof])
            conv_records.append({
                'Bank':             bank,
                'Quarter':          q,
                'quarter_date':     pd.to_datetime(q, format='%m/%d/%Y', errors='coerce'),
                'QoQ NN':           qf,
                'Levels NN':        lf,
                'Isolation Forest': isof,
                'Models Flagged':   n,
            })

    conv_df = pd.DataFrame(conv_records)
    if not conv_df.empty:
        conv_df = conv_df.sort_values(['Bank', 'quarter_date']).reset_index(drop=True)

    # =========================================================================
    # TABS
    # =========================================================================
    tab_all, tab_bank, tab_data = st.tabs(["All Banks", "Bank Detail", "Data Viewer"])

    # =========================================================================
    # TAB 1: ALL BANKS — convergence overview
    # =========================================================================
    with tab_all:
        st.subheader("All Banks at a Glance")

        # Summary metrics per bank
        metric_cols = st.columns(len(bank_names))
        for i, bank in enumerate(bank_names):
            bc = conv_df[conv_df['Bank'] == bank] if not conv_df.empty else pd.DataFrame()
            n_any  = int((bc['Models Flagged'] >= 1).sum()) if len(bc) > 0 else 0
            n_high = int((bc['Models Flagged'] >= 2).sum()) if len(bc) > 0 else 0
            short  = bank.split(' ')[0]
            with metric_cols[i]:
                st.metric(short, f"{n_any} flagged", f"{n_high} high-confidence")

        st.divider()

        # Convergence scatter — only flagged quarters, size/color = models flagged
        st.subheader("Flag Convergence Over Time")
        st.caption(
            "Each dot is a quarter flagged by at least one model. "
            "Larger, redder dots = more models agree. Hover for details."
        )
        flagged_conv = conv_df[conv_df['Models Flagged'] > 0].copy() if not conv_df.empty else pd.DataFrame()
        if not flagged_conv.empty:
            flagged_conv['Models Flagged (label)'] = flagged_conv['Models Flagged'].astype(str)
            fig_heat = px.scatter(
                flagged_conv,
                x='quarter_date',
                y='Bank',
                color='Models Flagged',
                color_continuous_scale=[[0, '#E9C46A'], [0.5, '#F4842A'], [1.0, ANOMALY_RED]],
                range_color=[1, max(len(available_models), 2)],
                size='Models Flagged',
                size_max=20,
                hover_data={
                    'Quarter':          True,
                    'QoQ NN':           True,
                    'Levels NN':        True,
                    'Isolation Forest': True,
                    'quarter_date':     False,
                    'Models Flagged':   True,
                },
                height=350,
            )
            fig_heat.update_layout(**plotly_theme_layout())
            fig_heat.update_layout(coloraxis_colorbar=dict(title="# Models", tickvals=[1,2,3]))
            fig_heat.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_heat.update_yaxes(showgrid=False)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No flagged quarters in the selected year range.")

        st.divider()

        # High-confidence flags table
        st.subheader("High-Confidence Flags (2+ Models)")
        st.caption("These quarters were flagged by multiple models — highest priority for review.")
        hc = conv_df[conv_df['Models Flagged'] >= 2].copy() if not conv_df.empty else pd.DataFrame()
        if not hc.empty:
            hc = hc.sort_values(['Models Flagged', 'quarter_date'], ascending=[False, False])
            disp = hc[['Bank', 'Quarter', 'Models Flagged'] + [m for m in available_models if m in hc.columns]].copy()
            for m in available_models:
                if m in disp.columns:
                    disp[m] = disp[m].map({True: '✓', False: ''})
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No quarters flagged by 2+ models in the selected year range.")

        st.divider()

        # Flag count summary
        st.subheader("Flag Summary by Bank")
        summary_rows = []
        for bank in bank_names:
            bc = conv_df[conv_df['Bank'] == bank] if not conv_df.empty else pd.DataFrame()
            row = {'Bank': bank}
            for m in available_models:
                row[m] = int(bc[m].sum()) if (len(bc) > 0 and m in bc.columns) else 0
            row['Any Model'] = int((bc['Models Flagged'] >= 1).sum()) if len(bc) > 0 else 0
            row['2+ Models'] = int((bc['Models Flagged'] >= 2).sum()) if len(bc) > 0 else 0
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: BANK DETAIL
    # =========================================================================
    with tab_bank:
        st.subheader(selected_bank)

        # Per-model data for this bank
        bank_qoq    = _filter_years(_ensure_score_cols(qoq_banks[selected_bank].copy()))    if selected_bank in qoq_banks    else None
        bank_levels = _filter_years(_ensure_score_cols(levels_banks[selected_bank].copy())) if selected_bank in levels_banks else None
        bank_iso    = _filter_years(iso_banks[selected_bank].copy())                         if selected_bank in iso_banks    else None
        bank_raw    = _filter_years(raw_features[selected_bank].copy())                      if selected_bank in raw_features else None

        bank_conv = conv_df[conv_df['Bank'] == selected_bank].copy() if not conv_df.empty else pd.DataFrame()

        n_total = len(bank_conv)
        n_any   = int((bank_conv['Models Flagged'] >= 1).sum()) if len(bank_conv) > 0 else 0
        n_high  = int((bank_conv['Models Flagged'] >= 2).sum()) if len(bank_conv) > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Quarters", n_total)
        c2.metric("Flagged by Any Model", n_any)
        c3.metric("High Confidence (2+ Models)", n_high)

        st.divider()

        # --- Multi-model score timeline (percentile-normalized) ---
        st.subheader("Model Scores Over Time")
        st.caption(
            "Scores are converted to a 0–100 percentile rank within each model so all three "
            "can be compared on the same chart. **100 = most unusual quarter ever seen by that "
            "model for this bank.** Red diamonds = flagged by that model."
        )

        fig_multi = go.Figure()

        def _add_model_trace(df_m, raw_score_col, model_name, color):
            if df_m is None or raw_score_col not in df_m.columns or 'quarter_date' not in df_m.columns:
                return
            df_m = df_m.copy()
            df_m['_pct'] = df_m[raw_score_col].rank(pct=True) * 100
            normal = ~df_m['is_anomaly'].astype(bool)
            anom   =  df_m['is_anomaly'].astype(bool)

            fig_multi.add_trace(go.Scatter(
                x=df_m.loc[normal, 'quarter_date'],
                y=df_m.loc[normal, '_pct'],
                mode='lines+markers',
                name=model_name,
                marker=dict(color=color, size=4),
                line=dict(color=color, width=1.5),
                legendgroup=model_name,
                hovertemplate='%{customdata[0]}<br>' + model_name + ': %{y:.0f}th percentile<extra></extra>',
                customdata=df_m.loc[normal, ['quarter']].values,
            ))
            if anom.any():
                fig_multi.add_trace(go.Scatter(
                    x=df_m.loc[anom, 'quarter_date'],
                    y=df_m.loc[anom, '_pct'],
                    mode='markers',
                    name=f'{model_name} — flagged',
                    marker=dict(color=ANOMALY_RED, size=12, symbol='diamond',
                                line=dict(width=2, color=color)),
                    legendgroup=model_name,
                    hovertemplate='%{customdata[0]}<br>' + model_name + ': %{y:.0f}th pct<extra>FLAGGED</extra>',
                    customdata=df_m.loc[anom, ['quarter']].values,
                ))

        if bank_qoq is not None:
            sc = _score_col(bank_qoq) or 'ensemble_score'
            _add_model_trace(bank_qoq, sc, 'QoQ NN', MODEL_COLORS['QoQ NN'])
        if bank_levels is not None:
            sc = _score_col(bank_levels) or 'anomaly_score'
            _add_model_trace(bank_levels, sc, 'Levels NN', MODEL_COLORS['Levels NN'])
        if bank_iso is not None:
            _add_model_trace(bank_iso, 'anom_score', 'Isolation Forest', MODEL_COLORS['Isolation Forest'])

        fig_multi.update_layout(**plotly_theme_layout())
        fig_multi.update_layout(
            yaxis_title="Percentile rank (100 = most unusual)",
            height=430,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        fig_multi.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_multi.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[0, 105])
        st.plotly_chart(fig_multi, use_container_width=True)

        st.divider()

        # --- Flagged quarters table (all models as columns) ---
        st.subheader("Flagged Quarters")
        st.caption("✓ = flagged by that model. Sorted by number of models agreeing.")

        if len(bank_conv) > 0:
            flagged_bank = bank_conv[bank_conv['Models Flagged'] >= 1].sort_values(
                ['Models Flagged', 'quarter_date'], ascending=[False, False]
            )[['Quarter', 'Models Flagged'] + [m for m in available_models if m in bank_conv.columns]].copy()
            if len(flagged_bank) > 0:
                for m in available_models:
                    if m in flagged_bank.columns:
                        flagged_bank[m] = flagged_bank[m].map({True: '✓', False: '—'})
                st.dataframe(flagged_bank, use_container_width=True, hide_index=True)
            else:
                st.info("No quarters flagged in the selected year range.")
        else:
            st.info("No data for this bank.")

        st.divider()

        # --- Quarter drill-down (model-agnostic, uses raw absolute features) ---
        st.subheader("Quarter Drill-Down: What Drove the Flag?")
        st.caption(
            "Select any quarter to see which financial line items were most unusual, "
            "based on how far each value was from this bank's own historical average."
        )

        # Offer flagged quarters first, then all others
        if len(bank_conv) > 0:
            flagged_qs = bank_conv[bank_conv['Models Flagged'] >= 1].sort_values(
                ['Models Flagged', 'quarter_date'], ascending=[False, False]
            )['Quarter'].tolist()
            all_qs = bank_conv.sort_values('quarter_date', ascending=False)['Quarter'].tolist()
            ordered_qs = flagged_qs + [q for q in all_qs if q not in set(flagged_qs)]
        else:
            ordered_qs = []

        if not ordered_qs:
            st.info("No quarter data available.")
        else:
            selected_quarter = st.selectbox(
                "Select a quarter to inspect",
                ordered_qs, index=0, key="drill_down_quarter"
            )

            if selected_quarter and len(bank_conv) > 0:
                q_rows = bank_conv[bank_conv['Quarter'] == selected_quarter]
                if len(q_rows) > 0:
                    q_row   = q_rows.iloc[0]
                    flags   = [m for m in available_models if q_row.get(m, False)]
                    n_flags = int(q_row['Models Flagged'])

                    if n_flags == 0:
                        st.info(f"**{selected_quarter}** — Not flagged by any model.")
                    elif n_flags == len(available_models):
                        st.error(f"**{selected_quarter}** — Flagged by ALL {n_flags} models: {', '.join(flags)}")
                    else:
                        st.warning(f"**{selected_quarter}** — Flagged by {n_flags} model(s): {', '.join(flags)}")

                    # ISO systemic breakdown
                    if bank_iso is not None and 'quarter' in bank_iso.columns:
                        iso_q = bank_iso[bank_iso['quarter'] == selected_quarter]
                        if len(iso_q) > 0 and 'systemic_score' in iso_q.columns:
                            r = iso_q.iloc[0]
                            st.caption(
                                f"Isolation Forest — Anomaly score: **{r['anom_score']:.4f}** | "
                                f"Systemic (market-wide): **{r['systemic_score']:.4f}** | "
                                f"Bank-specific excess: **{r['adjusted_score']:.4f}**"
                            )

                # Line item analysis from raw absolute features
                if bank_raw is not None and selected_quarter in bank_raw['quarter'].values:
                    feat_skip = {'quarter', 'quarter_date'}
                    feat_cols = [c for c in bank_raw.columns
                                 if c not in feat_skip and not c.startswith('_')]
                    feat_data  = bank_raw[feat_cols].apply(pd.to_numeric, errors='coerce')
                    bank_means = feat_data.mean()
                    bank_stds  = feat_data.std().replace(0, np.nan)
                    row_vals   = feat_data[bank_raw['quarter'] == selected_quarter].iloc[0]
                    z_scores   = ((row_vals - bank_means) / bank_stds).dropna()

                    n_top = st.slider("Number of top line items to show", 5, 30, 15, key="n_top_features")
                    top_features = z_scores.abs().nlargest(n_top)

                    top_df = pd.DataFrame({
                        'raw_code':                    top_features.index,
                        'Line Item':                   [translate_code(f, mdrm) for f in top_features.index],
                        'Chart Label':                 [chart_label(f, mdrm)    for f in top_features.index],
                        'Value ($ thousands)':         [row_vals[f]   for f in top_features.index],
                        'Bank Average ($ thousands)':  [bank_means[f] for f in top_features.index],
                        'How Unusual':                 [z_scores[f]   for f in top_features.index],
                    })

                    colors = [ANOMALY_RED if z > 0 else NORMAL_BLUE for z in top_df['How Unusual']]
                    fig_drivers = go.Figure()
                    fig_drivers.add_trace(go.Bar(
                        x=top_df['How Unusual'],
                        y=top_df['Chart Label'],
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            '%{customdata[2]}<br>'
                            'How unusual: %{x:.1f} std devs<br>'
                            'Value: %{customdata[0]:,.0f} ($ thousands)<br>'
                            'Bank avg: %{customdata[1]:,.0f} ($ thousands)'
                            '<extra></extra>'
                        ),
                        customdata=top_df[['Value ($ thousands)', 'Bank Average ($ thousands)', 'Line Item']].values,
                    ))
                    fig_drivers.update_layout(**plotly_theme_layout())
                    fig_drivers.update_layout(
                        xaxis_title="How unusual (standard deviations from this bank's average)",
                        height=max(350, n_top * 30),
                        yaxis=dict(autorange='reversed'),
                    )
                    fig_drivers.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    fig_drivers.update_yaxes(showgrid=False)
                    st.plotly_chart(fig_drivers, use_container_width=True)
                    st.caption(
                        "**Red** = unusually high this quarter. **Blue** = unusually low. "
                        "Longer bars = more unusual. All financial values in thousands of dollars."
                    )

                    # Copyable table
                    st.markdown("**Top contributing line items** (select text to copy)")
                    st.caption("Financial values in thousands of dollars ($ thousands).")
                    st.dataframe(
                        top_df[['Line Item', 'Value ($ thousands)', 'Bank Average ($ thousands)', 'How Unusual']].style.format({
                            'Value ($ thousands)':        '{:,.0f}',
                            'Bank Average ($ thousands)': '{:,.0f}',
                            'How Unusual':                '{:.2f}',
                        }),
                        use_container_width=True, hide_index=True,
                    )

                    # Line item time series
                    st.divider()
                    st.subheader("Line Item Over Time")

                    li_options    = [chart_label(c, mdrm) for c in top_df['raw_code']]
                    display_to_raw = {chart_label(c, mdrm): c for c in top_df['raw_code']}

                    selected_li = st.selectbox(
                        "Select line item to graph", li_options, index=0, key="li_drilldown"
                    )
                    if selected_li:
                        raw_code = display_to_raw[selected_li]
                        # Build chart df with anomaly flags merged in from best available model
                        chart_df = bank_raw.copy()
                        flag_src = (
                            iso_banks.get(selected_bank) if has_iso else
                            qoq_banks.get(selected_bank) if has_qoq else
                            levels_banks.get(selected_bank)
                        )
                        if flag_src is not None:
                            chart_df = chart_df.merge(
                                flag_src[['quarter', 'is_anomaly']], on='quarter', how='left'
                            )
                        chart_df['is_anomaly'] = chart_df.get('is_anomaly', pd.Series(False, index=chart_df.index)).fillna(False).astype(bool)
                        _render_line_item_chart(
                            chart_df, raw_code, selected_li, mdrm,
                            highlight_quarter=selected_quarter,
                            is_levels=True,
                            mdrm_types=mdrm_types,
                            forecast_df=bank_forecast_df,
                            show_forecast=show_forecast,
                        )

                elif bank_raw is None:
                    st.warning("Raw features data not available for line-item analysis. Check that per_bank_features/ exists.")
                else:
                    st.warning(f"Quarter {selected_quarter} not found in raw features data.")

    # =========================================================================
    # TAB 3: DATA VIEWER (unchanged)
    # =========================================================================
    with tab_data:
        st.subheader("Raw Data Viewer")
        st.caption(
            "Browse the original financial data by bank, quarter, and line item. "
            "Select a line item to see its exact value at each quarter and graph it over time."
        )

        data_source = st.radio(
            "Data type",
            ["Absolute Values (Levels)", "Quarter-over-Quarter Changes (QoQ)"],
            horizontal=True, key="data_source_radio",
        )

        is_qoq    = "QoQ" in data_source
        raw_banks = raw_qoq if is_qoq else raw_features
        value_label = "QoQ Change (%)" if is_qoq else "Value"

        if not raw_banks:
            src_dir = QOQ_RAW_DIR if is_qoq else FEATURES_RAW_DIR
            st.warning(f"No raw data files found in `{src_dir}/`.")
        elif selected_bank not in raw_banks:
            st.warning(f"No raw data for {selected_bank}.")
        else:
            rdf = _filter_years(raw_banks[selected_bank].copy())
            data_cols = [c for c in rdf.columns if c not in ('quarter', 'quarter_date')]

            viewer_options = []
            viewer_raw_map = {}
            for code in sorted(data_cols):
                display = chart_label(code, mdrm)
                viewer_options.append(display)
                viewer_raw_map[display] = code

            selected_viewer_item = st.selectbox(
                "Search for a line item", viewer_options, index=0, key="viewer_line_item"
            )

            if selected_viewer_item:
                raw_code  = viewer_raw_map[selected_viewer_item]
                full_name = translate_code(raw_code, mdrm)
                unit      = get_unit_label(raw_code, mdrm_types)
                unit_suffix = f" ({unit})" if unit else ""

                st.markdown(f"**{full_name}**")
                if unit:
                    st.caption(f"Unit: **{unit}**")

                vals     = pd.to_numeric(rdf[raw_code], errors='coerce')
                col_label = f"{value_label}{unit_suffix}"
                table_df  = pd.DataFrame({'Quarter': rdf['quarter'], col_label: vals})
                table_df['Change from Prior'] = vals.diff()

                st.divider()
                st.subheader(f"{_truncate(short_name(raw_code, mdrm), 60)} Over Time")

                fig_viewer = go.Figure()
                fig_viewer.add_trace(go.Scatter(
                    x=rdf['quarter_date'], y=vals,
                    mode='lines+markers', name=col_label,
                    marker=dict(color=NORMAL_BLUE, size=5),
                    line=dict(color=NORMAL_BLUE, width=2),
                    hovertemplate='%{customdata[0]}<br>' + col_label + ': %{y:,.2f}<extra></extra>',
                    customdata=rdf[['quarter']].values,
                ))

                # Forecast overlay (levels only — forecasts are absolute values)
                if show_forecast and not is_qoq and bank_forecast_df is not None:
                    fc = bank_forecast_df[bank_forecast_df['feature'] == raw_code].copy()
                    if not fc.empty:
                        last_row = rdf.dropna(subset=['quarter_date']).sort_values('quarter_date').iloc[-1]
                        last_val = pd.to_numeric(last_row[raw_code], errors='coerce')
                        connector = pd.DataFrame([{
                            'quarter_date': last_row['quarter_date'],
                            'quarter':      last_row['quarter'],
                            'value':        last_val,
                        }])
                        fc_plot = pd.concat([connector, fc[['quarter_date', 'quarter', 'value']]], ignore_index=True)
                        fig_viewer.add_trace(go.Scatter(
                            x=fc_plot['quarter_date'], y=fc_plot['value'],
                            mode='lines', name='Projected (2026–2027)',
                            line=dict(color=NORMAL_BLUE, width=2, dash='dash'),
                            opacity=0.65,
                            hovertemplate='%{customdata[0]}<br>Projected: %{y:,.2f}<extra>PROJECTION</extra>',
                            customdata=fc_plot[['quarter']].values,
                        ))
                        fig_viewer.add_vline(
                            x=FORECAST_CUTOFF.timestamp() * 1000,
                            line_dash='dot',
                            line_color='rgba(150,150,150,0.6)',
                            annotation_text='Projections →',
                            annotation_position='top right',
                            annotation_font_size=11,
                        )

                fig_viewer.update_layout(**plotly_theme_layout())
                fig_viewer.update_layout(
                    yaxis_title=col_label, height=400,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                )
                fig_viewer.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig_viewer.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                st.plotly_chart(fig_viewer, use_container_width=True)

                st.divider()
                st.subheader("Values by Quarter")
                st.dataframe(
                    table_df.style.format({col_label: '{:,.2f}', 'Change from Prior': '{:,.2f}'}),
                    use_container_width=True, hide_index=True,
                )

                st.divider()
                st.subheader("Compare Across Banks")
                compare_viewer = st.selectbox(
                    "Overlay another bank",
                    ["None"] + [b for b in bank_names if b != selected_bank],
                    index=0, key="viewer_compare_bank",
                )

                if compare_viewer != "None" and compare_viewer in raw_banks:
                    rdf2 = _filter_years(raw_banks[compare_viewer].copy())
                    if raw_code in rdf2.columns:
                        vals2 = pd.to_numeric(rdf2[raw_code], errors='coerce')
                        fig_vc = go.Figure()
                        fig_vc.add_trace(go.Scatter(
                            x=rdf['quarter_date'], y=vals,
                            mode='lines+markers', name=selected_bank,
                            marker=dict(size=5), line=dict(width=2),
                            hovertemplate='%{customdata[0]}<br>' + col_label + ': %{y:,.2f}<extra>' + selected_bank + '</extra>',
                            customdata=rdf[['quarter']].values,
                        ))
                        fig_vc.add_trace(go.Scatter(
                            x=rdf2['quarter_date'], y=vals2,
                            mode='lines+markers', name=compare_viewer,
                            marker=dict(size=5), line=dict(width=2, dash='dash'),
                            hovertemplate='%{customdata[0]}<br>' + col_label + ': %{y:,.2f}<extra>' + compare_viewer + '</extra>',
                            customdata=rdf2[['quarter']].values,
                        ))
                        fig_vc.update_layout(**plotly_theme_layout())
                        fig_vc.update_layout(
                            title=f"{_truncate(short_name(raw_code, mdrm), 50)} — Comparison",
                            yaxis_title=col_label, height=450,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02),
                        )
                        fig_vc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                        fig_vc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                        st.plotly_chart(fig_vc, use_container_width=True)
                    else:
                        st.warning(f"Line item `{raw_code}` not found in {compare_viewer}'s data.")


def _render_line_item_chart(df, raw_code, display_name, mdrm,
                            highlight_quarter=None, is_levels=False,
                            mdrm_types=None, forecast_df=None, show_forecast=False):
    """Render a time series chart for a single line item with anomaly highlighting.

    If show_forecast is True and forecast_df is provided (long-format with columns
    feature/quarter/quarter_date/value), appends a dashed projection trace.
    """
    if raw_code not in df.columns or 'quarter_date' not in df.columns:
        st.warning("Unable to render chart — data not available.")
        return

    values = pd.to_numeric(df[raw_code], errors='coerce')
    normal_mask = ~df['is_anomaly'].astype(bool)
    anom_mask   =  df['is_anomaly'].astype(bool)

    unit = get_unit_label(raw_code, mdrm_types or {})
    y_label = (f"Value ({unit})" if unit else "Value") if is_levels else \
              (f"QoQ Change ({unit})" if unit else "Quarter-over-Quarter Change (%)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.loc[normal_mask, 'quarter_date'],
        y=values[normal_mask],
        mode='lines+markers', name='Normal quarters',
        marker=dict(color=NORMAL_BLUE, size=5),
        line=dict(color=NORMAL_BLUE, width=1.5),
        hovertemplate='%{customdata[0]}<br>' + y_label + ': %{y:,.2f}<extra></extra>',
        customdata=df.loc[normal_mask, ['quarter']].values,
    ))

    if anom_mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[anom_mask, 'quarter_date'],
            y=values[anom_mask],
            mode='markers', name='Flagged quarters',
            marker=dict(color=ANOMALY_RED, size=11, symbol='diamond',
                        line=dict(width=2, color='white')),
            hovertemplate='%{customdata[0]}<br>' + y_label + ': %{y:,.2f}<extra>FLAGGED</extra>',
            customdata=df.loc[anom_mask, ['quarter']].values,
        ))

    if highlight_quarter:
        hq = df[df['quarter'] == highlight_quarter]
        if len(hq) > 0:
            hq_val = pd.to_numeric(hq[raw_code].iloc[0], errors='coerce')
            fig.add_trace(go.Scatter(
                x=hq['quarter_date'], y=[hq_val],
                mode='markers', name=f'Selected ({highlight_quarter})',
                marker=dict(color='#FFD700', size=16, symbol='star',
                            line=dict(width=2, color='black')),
                hovertemplate=f'{highlight_quarter}<br>{y_label}: {hq_val:,.2f}<extra>SELECTED</extra>',
            ))

    # Forecast overlay
    if show_forecast and forecast_df is not None:
        fc = forecast_df[forecast_df['feature'] == raw_code].copy()
        if not fc.empty:
            # Prepend the last historical point so the dashed line connects smoothly
            last_hist = df.dropna(subset=['quarter_date']).sort_values('quarter_date').iloc[-1]
            last_val  = pd.to_numeric(last_hist[raw_code], errors='coerce')
            connector = pd.DataFrame([{
                'quarter_date': last_hist['quarter_date'],
                'quarter':      last_hist['quarter'],
                'value':        last_val,
            }])
            fc_plot = pd.concat([connector, fc[['quarter_date', 'quarter', 'value']]], ignore_index=True)

            fig.add_trace(go.Scatter(
                x=fc_plot['quarter_date'],
                y=fc_plot['value'],
                mode='lines',
                name='Projected (2026–2027)',
                line=dict(color=NORMAL_BLUE, width=1.5, dash='dash'),
                opacity=0.65,
                hovertemplate='%{customdata[0]}<br>Projected ' + y_label + ': %{y:,.2f}<extra>PROJECTION</extra>',
                customdata=fc_plot[['quarter']].values,
            ))

            # Vertical boundary line at forecast start
            fig.add_vline(
                x=FORECAST_CUTOFF.timestamp() * 1000,  # Plotly expects ms epoch for datetime axes
                line_dash='dot',
                line_color='rgba(150,150,150,0.6)',
                annotation_text='Projections →',
                annotation_position='top right',
                annotation_font_size=11,
            )

    fig.update_layout(**plotly_theme_layout())
    fig.update_layout(
        title=short_name(raw_code, mdrm),
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
