"""
FFIEC Neural Network Anomaly Detection Dashboard (Track 2)
==========================================================
Streamlit dashboard for exploring per-bank anomaly detection results.

Usage:
    1. Place all *_nn_anomalies.xls (or .csv) files in a folder called per_bank_nn/
       (or adjust DATA_DIR below)
    2. Run: streamlit run nn_dashboard.py

Author: Wake Forest MSBA Practicum Team 4
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("per_bank_nn")

BANK_DISPLAY_NAMES = {
    'bank_of_america': 'Bank of America',
    'jpmorgan_chase_bank': 'JPMorgan Chase Bank',
    'citibank': 'Citibank',
    'wells_fargo_bank': 'Wells Fargo Bank',
    'goldman_sachs_bank_usa': 'Goldman Sachs Bank USA',
    'morgan_stanley_bank': 'Morgan Stanley Bank',
}

# Color palette
COLORS = {
    'primary': '#1B2A4A',
    'accent': '#E63946',
    'muted': '#A8DADC',
    'bg': '#F1FAEE',
    'text': '#1D3557',
    'normal': '#457B9D',
    'anomaly': '#E63946',
    'nn': '#2A9D8F',
    'lof': '#E9C46A',
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_mdrm_lookup():
    """Load the MDRM data dictionary if available."""
    lookup_path = DATA_DIR / "mdrm_lookup.csv"
    if not lookup_path.exists():
        # Also check current directory
        lookup_path = Path("mdrm_lookup.csv")
    if lookup_path.exists():
        df = pd.read_csv(lookup_path, dtype=str)
        return dict(zip(df['code'], df['description']))
    return {}


def translate_code(code, mdrm_dict):
    """Translate an MDRM code like RCFD0211_qoq to its English name."""
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    if name:
        return f"{clean}: {name}"
    return clean


@st.cache_data
def load_all_banks():
    """Load all per-bank anomaly CSVs from DATA_DIR."""
    banks = {}
    patterns = [
        DATA_DIR.glob("ffiec_*_nn_anomalies.csv"),
        DATA_DIR.glob("ffiec_*_nn_anomalies.xls"),
    ]

    for pattern in patterns:
        for filepath in sorted(pattern):
            # Skip flagged files
            if 'flagged' in filepath.stem:
                continue

            slug = filepath.stem.replace('ffiec_', '').replace('_nn_anomalies', '')
            display_name = BANK_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())

            try:
                df = pd.read_csv(filepath, low_memory=False)
            except Exception:
                try:
                    df = pd.read_excel(filepath, engine='xlrd')
                except Exception:
                    continue

            # Handle transposed format (features as rows, quarters as columns)
            if 'feature' in df.columns:
                has_qoq = df['feature'].astype(str).str.contains('_qoq').any()
                if has_qoq:
                    feature_names = df['feature'].tolist()
                    df_t = df.drop(columns=['feature']).T
                    df_t.columns = feature_names
                    df_t.index.name = 'quarter'
                    df_t = df_t.reset_index()
                    df_t.rename(columns={'index': 'quarter'}, inplace=True)
                    df = df_t

            # Force score columns to numeric
            score_cols = ['nn_score', 'lof_score', 'ensemble_score',
                          'reconstruction_error', 'anomaly_score']
            for col in score_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Force boolean columns
            bool_cols = ['nn_anomaly', 'lof_anomaly', 'is_anomaly']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].astype(bool)

            # Parse quarter into a sortable date
            if 'quarter' in df.columns:
                df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
                df = df.sort_values('quarter_date').reset_index(drop=True)

            banks[display_name] = df

    return banks


def parse_year(quarter_str):
    """Extract year from quarter string like '03/31/2008'."""
    try:
        return int(str(quarter_str).split('/')[-1])
    except (ValueError, IndexError):
        return None


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="FFIEC Anomaly Detection Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1B2A4A;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1rem;
        color: #457B9D;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #F1FAEE 0%, #A8DADC 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #1B2A4A;
        margin-bottom: 0.75rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B2A4A;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #457B9D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .anomaly-badge {
        display: inline-block;
        background: #E63946;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .section-divider {
        border-top: 2px solid #A8DADC;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">FFIEC Neural Network Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track 2: Per-Bank Autoencoder + LOF Ensemble</div>', unsafe_allow_html=True)

    # Load data
    banks = load_all_banks()
    mdrm = load_mdrm_lookup()

    if not banks:
        st.error(
            f"No anomaly files found in `{DATA_DIR}/`. "
            f"Expected files like `ffiec_bank_of_america_nn_anomalies.csv` (or `.xls`)."
        )
        return

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.markdown("### Settings")

        bank_names = list(banks.keys())
        selected_bank = st.selectbox("Select Bank", bank_names, index=0)

        st.markdown("---")

        # Custom threshold slider
        custom_threshold = st.slider(
            "Anomaly Score Threshold",
            min_value=0.0,
            max_value=100.0,
            value=None,
            help="Override the default top-5% threshold. Leave at 0 to use the model's built-in flags."
        )

        st.markdown("---")
        st.markdown("### Score Components")
        st.markdown(
            "**Ensemble** = 60% Neural Network + 40% LOF\n\n"
            "**NN Score**: High = hard to reconstruct (unusual pattern)\n\n"
            "**LOF Score**: High = isolated in feature space (few similar quarters)"
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Each bank has its own model trained on its own history. "
            "Anomalies reflect quarters unusual **for that bank**, not cross-bank."
        )

        if mdrm:
            st.markdown(f"**MDRM Dictionary:** Loaded ({len(mdrm):,} codes)")
        else:
            st.markdown(
                "**MDRM Dictionary:** Not found. Run `download_mdrm.py` to "
                "translate line item codes to English names."
            )

    # =========================================================================
    # OVERVIEW TAB vs BANK DETAIL TAB
    # =========================================================================

    tab_overview, tab_detail, tab_compare = st.tabs(["Cross-Bank Overview", "Bank Detail", "Score Comparison"])

    # =========================================================================
    # TAB 1: CROSS-BANK OVERVIEW
    # =========================================================================
    with tab_overview:
        st.markdown("### All Banks at a Glance")

        # Summary metrics row
        cols = st.columns(len(banks))
        for i, (name, df) in enumerate(banks.items()):
            with cols[i]:
                n_anom = int(df['is_anomaly'].sum())
                max_score = df['ensemble_score'].max()
                short_name = name.split(' ')[0] if len(name) > 15 else name
                st.metric(
                    label=short_name,
                    value=f"{n_anom} flagged",
                    delta=f"max: {max_score:.1f}",
                    delta_color="inverse",
                )

        st.markdown("---")

        # Timeline heatmap: anomaly scores over time for all banks
        st.markdown("### Ensemble Score Timeline (All Banks)")

        heatmap_data = []
        for name, df in banks.items():
            if 'quarter_date' in df.columns:
                for _, row in df.iterrows():
                    heatmap_data.append({
                        'Bank': name,
                        'Quarter': row['quarter_date'],
                        'Ensemble Score': row['ensemble_score'],
                        'Flagged': row['is_anomaly'],
                    })

        if heatmap_data:
            hm_df = pd.DataFrame(heatmap_data)

            fig_hm = px.scatter(
                hm_df,
                x='Quarter',
                y='Bank',
                size='Ensemble Score',
                color='Ensemble Score',
                color_continuous_scale=['#457B9D', '#E9C46A', '#E63946'],
                hover_data=['Flagged'],
                height=350,
            )
            fig_hm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', size=12),
                margin=dict(l=20, r=20, t=20, b=40),
                coloraxis_colorbar=dict(title="Score"),
            )
            fig_hm.update_xaxes(showgrid=True, gridcolor='#eee')
            fig_hm.update_yaxes(showgrid=False)
            st.plotly_chart(fig_hm, use_container_width=True)

        # Stress period table
        st.markdown("### Stress Period Flags by Bank")
        stress_years = [2008, 2009, 2010, 2020]
        stress_data = []
        for name, df in banks.items():
            df['_year'] = df['quarter'].apply(parse_year)
            row_data = {'Bank': name}
            for yr in stress_years:
                yr_data = df[df['_year'] == yr]
                n_flagged = int(yr_data['is_anomaly'].sum()) if len(yr_data) > 0 else 0
                avg_score = yr_data['ensemble_score'].mean() if len(yr_data) > 0 else 0
                row_data[f'{yr} Flags'] = n_flagged
                row_data[f'{yr} Avg Score'] = round(avg_score, 1)
            stress_data.append(row_data)

        stress_df = pd.DataFrame(stress_data)
        st.dataframe(stress_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: BANK DETAIL
    # =========================================================================
    with tab_detail:
        df = banks[selected_bank].copy()

        st.markdown(f"### {selected_bank}")

        # Top metrics
        n_quarters = len(df)
        n_anomalies = int(df['is_anomaly'].sum())
        max_ensemble = df['ensemble_score'].max()
        max_nn = df['nn_score'].max()
        max_lof = df['lof_score'].max()
        qoq_features = len([c for c in df.columns if c.endswith('_qoq')])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Quarters", n_quarters)
        col2.metric("Anomalies Flagged", n_anomalies)
        col3.metric("Max Ensemble", f"{max_ensemble:.1f}")
        col4.metric("Max NN Score", f"{max_nn:.1f}")
        col5.metric("QoQ Features", qoq_features)

        st.markdown("---")

        # Ensemble score timeline
        st.markdown("#### Ensemble Score Over Time")

        if 'quarter_date' in df.columns:
            fig_timeline = go.Figure()

            # Normal points
            normal_mask = ~df['is_anomaly']
            fig_timeline.add_trace(go.Scatter(
                x=df.loc[normal_mask, 'quarter_date'],
                y=df.loc[normal_mask, 'ensemble_score'],
                mode='markers+lines',
                name='Normal',
                marker=dict(color=COLORS['normal'], size=6),
                line=dict(color=COLORS['normal'], width=1),
                hovertemplate='%{customdata[0]}<br>Score: %{y:.1f}<extra></extra>',
                customdata=df.loc[normal_mask, ['quarter']].values,
            ))

            # Anomaly points
            anom_mask = df['is_anomaly']
            if anom_mask.any():
                fig_timeline.add_trace(go.Scatter(
                    x=df.loc[anom_mask, 'quarter_date'],
                    y=df.loc[anom_mask, 'ensemble_score'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color=COLORS['anomaly'], size=12, symbol='diamond',
                                line=dict(width=2, color='white')),
                    hovertemplate='%{customdata[0]}<br>Score: %{y:.1f}<extra>ANOMALY</extra>',
                    customdata=df.loc[anom_mask, ['quarter']].values,
                ))

            # Custom threshold line
            if custom_threshold and custom_threshold > 0:
                fig_timeline.add_hline(
                    y=custom_threshold,
                    line_dash="dash",
                    line_color=COLORS['accent'],
                    annotation_text=f"Custom threshold: {custom_threshold:.0f}",
                )

            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', size=12),
                margin=dict(l=20, r=20, t=20, b=40),
                yaxis_title="Ensemble Score",
                xaxis_title="Quarter",
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            fig_timeline.update_xaxes(showgrid=True, gridcolor='#eee')
            fig_timeline.update_yaxes(showgrid=True, gridcolor='#eee')
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("---")

        # NN vs LOF scatter
        st.markdown("#### NN Score vs LOF Score")

        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=df.loc[normal_mask, 'nn_score'],
            y=df.loc[normal_mask, 'lof_score'],
            mode='markers',
            name='Normal',
            marker=dict(color=COLORS['normal'], size=7, opacity=0.6),
            hovertemplate='%{customdata[0]}<br>NN: %{x:.1f}<br>LOF: %{y:.1f}<extra></extra>',
            customdata=df.loc[normal_mask, ['quarter']].values,
        ))

        if anom_mask.any():
            fig_scatter.add_trace(go.Scatter(
                x=df.loc[anom_mask, 'nn_score'],
                y=df.loc[anom_mask, 'lof_score'],
                mode='markers+text',
                name='Anomaly',
                marker=dict(color=COLORS['anomaly'], size=12, symbol='diamond',
                            line=dict(width=2, color='white')),
                text=df.loc[anom_mask, 'quarter'].values,
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate='%{text}<br>NN: %{x:.1f}<br>LOF: %{y:.1f}<extra>ANOMALY</extra>',
            ))

        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', size=12),
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Neural Network Score",
            yaxis_title="LOF Score",
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor='#eee')
        fig_scatter.update_yaxes(showgrid=True, gridcolor='#eee')
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Flagged anomalies detail table
        st.markdown("#### Flagged Anomalies Detail")

        display_cols = ['quarter', 'ensemble_score', 'nn_score', 'lof_score',
                        'nn_anomaly', 'lof_anomaly', 'reconstruction_error']
        available_cols = [c for c in display_cols if c in df.columns]

        df_flagged = df[df['is_anomaly']].sort_values('ensemble_score', ascending=False)

        if len(df_flagged) > 0:
            st.dataframe(
                df_flagged[available_cols].style.format({
                    'ensemble_score': '{:.1f}',
                    'nn_score': '{:.1f}',
                    'lof_score': '{:.1f}',
                    'reconstruction_error': '{:.0f}',
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No anomalies flagged for this bank at the current threshold.")

        st.markdown("---")

        # =================================================================
        # ANOMALY DRILL-DOWN: What line items drove the anomaly?
        # =================================================================
        st.markdown("#### Anomaly Drill-Down: Top Contributing Line Items")
        st.markdown(
            "Select a quarter to see which QoQ features had the most extreme values. "
            "These are the line items most likely driving the anomaly flag."
        )

        qoq_cols = sorted([c for c in df.columns if c.endswith('_qoq')])

        # Let user pick any quarter, but default to the top anomaly
        quarter_options = df.sort_values('ensemble_score', ascending=False)['quarter'].tolist()
        selected_quarter = st.selectbox(
            "Select Quarter to Inspect",
            quarter_options,
            index=0,
            key="drill_down_quarter",
        )

        if selected_quarter and qoq_cols:
            row = df[df['quarter'] == selected_quarter].iloc[0]
            is_flagged = row['is_anomaly']
            score = row['ensemble_score']

            flag_label = "FLAGGED ANOMALY" if is_flagged else "Not flagged"
            st.markdown(f"**{selected_quarter}** — Ensemble Score: **{score:.1f}** ({flag_label})")

            # Compute z-scores relative to this bank's own history
            qoq_data = df[qoq_cols].apply(pd.to_numeric, errors='coerce')
            bank_means = qoq_data.mean()
            bank_stds = qoq_data.std().replace(0, np.nan)

            row_values = qoq_data.loc[df['quarter'] == selected_quarter].iloc[0]
            z_scores = ((row_values - bank_means) / bank_stds).dropna()

            # Top N most extreme features by absolute z-score
            n_top = st.slider("Number of top features to show", 5, 30, 15, key="n_top_features")

            top_features = z_scores.abs().nlargest(n_top)
            top_df = pd.DataFrame({
                'Line Item': top_features.index,
                'QoQ Value (%)': [row_values[f] for f in top_features.index],
                'Bank Mean (%)': [bank_means[f] for f in top_features.index],
                'Bank Std (%)': [bank_stds[f] for f in top_features.index],
                'Z-Score': [z_scores[f] for f in top_features.index],
                'Abs Z-Score': top_features.values,
            })
            # Translate MDRM codes to English names
            top_df['Line Item'] = top_df['Line Item'].apply(lambda x: translate_code(x, mdrm))

            # Bar chart of z-scores
            fig_drivers = go.Figure()
            colors = [COLORS['anomaly'] if z > 0 else COLORS['normal'] for z in top_df['Z-Score']]

            fig_drivers.add_trace(go.Bar(
                x=top_df['Z-Score'],
                y=top_df['Line Item'],
                orientation='h',
                marker_color=colors,
                hovertemplate=(
                    '%{y}<br>'
                    'Z-Score: %{x:.2f}<br>'
                    'QoQ: %{customdata[0]:.2f}%<br>'
                    'Bank Mean: %{customdata[1]:.2f}%'
                    '<extra></extra>'
                ),
                customdata=top_df[['QoQ Value (%)', 'Bank Mean (%)']].values,
            ))

            fig_drivers.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', size=11),
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Z-Score (vs bank's own history)",
                height=max(300, n_top * 28),
                yaxis=dict(autorange='reversed'),
            )
            fig_drivers.update_xaxes(showgrid=True, gridcolor='#eee')
            fig_drivers.update_yaxes(showgrid=False)
            st.plotly_chart(fig_drivers, use_container_width=True)

            # Detail table
            with st.expander("View full detail table"):
                st.dataframe(
                    top_df.style.format({
                        'QoQ Value (%)': '{:.2f}',
                        'Bank Mean (%)': '{:.2f}',
                        'Bank Std (%)': '{:.2f}',
                        'Z-Score': '{:.2f}',
                        'Abs Z-Score': '{:.2f}',
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                "**Reading this chart:** Red bars (positive z-score) = the line item was unusually *high* "
                "for this quarter relative to the bank's history. Blue bars (negative) = unusually *low*. "
                "Larger absolute z-scores indicate stronger deviation from normal."
            )

        st.markdown("---")

        # Full score table (expandable)
        with st.expander("View All Quarters (Full Score Table)"):
            st.dataframe(
                df[available_cols].style.format({
                    'ensemble_score': '{:.2f}',
                    'nn_score': '{:.2f}',
                    'lof_score': '{:.2f}',
                    'reconstruction_error': '{:.0f}',
                }),
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================================
    # TAB 3: SCORE COMPARISON
    # =========================================================================
    with tab_compare:
        st.markdown("### Score Distributions Across Banks")

        # Box plot of ensemble scores by bank
        compare_data = []
        for name, df in banks.items():
            for _, row in df.iterrows():
                compare_data.append({
                    'Bank': name,
                    'Ensemble Score': row['ensemble_score'],
                    'NN Score': row['nn_score'],
                    'LOF Score': row['lof_score'],
                })

        if compare_data:
            comp_df = pd.DataFrame(compare_data)

            score_type = st.radio(
                "Score Type",
                ['Ensemble Score', 'NN Score', 'LOF Score'],
                horizontal=True
            )

            fig_box = px.box(
                comp_df,
                x='Bank',
                y=score_type,
                color='Bank',
                color_discrete_sequence=px.colors.qualitative.Set2,
                points='all',
                height=450,
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', size=12),
                margin=dict(l=20, r=20, t=20, b=80),
                showlegend=False,
                xaxis_tickangle=-30,
            )
            fig_box.update_xaxes(showgrid=False)
            fig_box.update_yaxes(showgrid=True, gridcolor='#eee')
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        # Multi-bank timeline overlay
        st.markdown("### Ensemble Score Timeline (All Banks Overlaid)")

        fig_multi = go.Figure()
        color_list = px.colors.qualitative.Set2
        for i, (name, df) in enumerate(banks.items()):
            if 'quarter_date' in df.columns:
                fig_multi.add_trace(go.Scatter(
                    x=df['quarter_date'],
                    y=df['ensemble_score'],
                    mode='lines',
                    name=name,
                    line=dict(color=color_list[i % len(color_list)], width=1.5),
                    opacity=0.8,
                ))

        fig_multi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', size=12),
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis_title="Ensemble Score",
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        fig_multi.update_xaxes(showgrid=True, gridcolor='#eee')
        fig_multi.update_yaxes(showgrid=True, gridcolor='#eee')
        st.plotly_chart(fig_multi, use_container_width=True)


if __name__ == "__main__":
    main()