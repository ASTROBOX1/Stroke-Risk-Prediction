"""
===============================================================================
 🧠 Stroke Prediction — Professional Analytics Dashboard
 Company: Healthcare Analytics Division
 Version: 2.0
 Description: Multi-page interactive dashboard with EDA, model performance,
              prediction engine, and automated reporting.
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
import os

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stroke Prediction Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #E2E8F0 !important;
    }
    
    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    .kpi-card-blue {
        background: linear-gradient(135deg, #2E86AB 0%, #1B5E7F 100%);
        box-shadow: 0 10px 30px rgba(46, 134, 171, 0.3);
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #2ECC71 0%, #1A9A54 100%);
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.3);
    }
    .kpi-card-red {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.3);
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #F39C12 0%, #D68910 100%);
        box-shadow: 0 10px 30px rgba(243, 156, 18, 0.3);
    }
    .kpi-card-purple {
        background: linear-gradient(135deg, #9B59B6 0%, #7D3C98 100%);
        box-shadow: 0 10px 30px rgba(155, 89, 182, 0.3);
    }
    .kpi-card-dark {
        background: linear-gradient(135deg, #2C3E50 0%, #1A252F 100%);
        box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
    }
    .kpi-value {
        font-size: 2.2em;
        font-weight: 800;
        margin: 8px 0 4px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .kpi-label {
        font-size: 0.9em;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-icon {
        font-size: 1.8em;
        margin-bottom: 4px;
    }
    
    /* ── Dividers ── */
    .section-divider {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border-radius: 2px;
        margin: 30px 0;
    }
    
    /* ── Report Card ── */
    .report-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .report-card h4 {
        color: #1E293B;
        margin-top: 0;
    }

    /* ── Risk Gauge ── */
    .risk-high {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.4);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #F39C12, #D68910);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(243, 156, 18, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #2ECC71, #1A9A54);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.4);
    }

    /* ── Hide default footer ── */
    footer {visibility: hidden;}
    
    /* ── Main area ── */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    return df

@st.cache_resource
def load_model():
    return joblib.load('models/best_stroke_model.joblib')

@st.cache_data
def load_metrics():
    with open('models/metrics.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    with open('models/feature_importance.json', 'r') as f:
        return json.load(f)

# Load everything
df_raw = load_data()
df = df_raw.copy()
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df = df[df['gender'] != 'Other']
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

try:
    metrics = load_metrics()
    metrics_loaded = True
except:
    metrics_loaded = False

try:
    fi_data = load_feature_importance()
    fi_loaded = True
except:
    fi_loaded = False


# ═══════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════
COLORS = {
    'primary':     '#667EEA',
    'secondary':   '#764BA2',
    'blue':        '#2E86AB',
    'green':       '#2ECC71',
    'red':         '#E74C3C',
    'orange':      '#F39C12',
    'purple':      '#9B59B6',
    'dark':        '#2C3E50',
    'no_stroke':   '#2ECC71',
    'stroke':      '#E74C3C',
}

PLOTLY_TEMPLATE = 'plotly_white'
MODEL_COLORS = ['#2E86AB', '#A23B72', '#F39C12']


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def kpi_card(icon, value, label, css_class=""):
    st.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    # 🧠 Stroke Analytics
    ### Healthcare AI Platform
    ---
    """)
    
    page = st.radio(
        "📑 Navigation",
        [
            "🏠 Executive Overview",
            "🔍 Data Explorer",
            "📊 Exploratory Analysis",
            "🤖 Model Performance",
            "⚕️ Stroke Risk Predictor",
            "📋 Report Summary"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **Dataset Info**
    - 📊 Records: {:,}
    - 📅 Report: March 2026
    - 🏥 Source: Healthcare DB
    """.format(len(df_raw)))
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; opacity:0.6; font-size:0.8em; color: #94A3B8;">
    Healthcare Analytics Division<br>
    Data Science Team © 2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════

if page == "🏠 Executive Overview":
    st.markdown("# 🏠 Executive Overview")
    st.markdown("*Real-time analytics dashboard for stroke risk assessment and patient demographics.*")
    section_divider()
    
    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    stroke_rate = df['stroke'].mean() * 100
    avg_age = df['age'].mean()
    avg_bmi = df['bmi'].mean()
    avg_glucose = df['avg_glucose_level'].mean()
    hyp_rate = df['hypertension'].mean() * 100
    hd_rate = df['heart_disease'].mean() * 100
    
    with c1:
        kpi_card("👥", f"{len(df):,}", "Total Patients", "kpi-card-blue")
    with c2:
        kpi_card("⚠️", f"{stroke_rate:.1f}%", "Stroke Rate", "kpi-card-red")
    with c3:
        kpi_card("🎂", f"{avg_age:.0f}", "Avg Age", "kpi-card-purple")
    with c4:
        kpi_card("⚖️", f"{avg_bmi:.1f}", "Avg BMI", "kpi-card-orange")
    with c5:
        kpi_card("🩸", f"{hyp_rate:.1f}%", "Hypertension", "kpi-card-dark")
    with c6:
        kpi_card("❤️", f"{hd_rate:.1f}%", "Heart Disease", "kpi-card-dark")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['stroke']==0]['age'], name='No Stroke',
            marker_color=COLORS['no_stroke'], opacity=0.7, nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=df[df['stroke']==1]['age'], name='Stroke',
            marker_color=COLORS['stroke'], opacity=0.8, nbinsx=30
        ))
        fig.update_layout(
            title='Age Distribution by Stroke Status',
            xaxis_title='Age', yaxis_title='Count',
            barmode='overlay', template=PLOTLY_TEMPLATE,
            legend=dict(x=0.02, y=0.98),
            margin=dict(t=50, b=40, l=50, r=20),
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stroke by work type
        ct = pd.crosstab(df['work_type'], df['stroke'], normalize='index') * 100
        ct.columns = ['No Stroke %', 'Stroke %']
        ct = ct.sort_values('Stroke %', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ct.index, x=ct['No Stroke %'], name='No Stroke',
            marker_color=COLORS['no_stroke'], orientation='h', opacity=0.85
        ))
        fig.add_trace(go.Bar(
            y=ct.index, x=ct['Stroke %'], name='Stroke',
            marker_color=COLORS['stroke'], orientation='h', opacity=0.85
        ))
        fig.update_layout(
            title='Stroke Rate by Work Type',
            xaxis_title='Percentage (%)', barmode='stack',
            template=PLOTLY_TEMPLATE,
            legend=dict(x=0.7, y=0.05),
            margin=dict(t=50, b=40, l=100, r=20),
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Glucose vs BMI scatter
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x='avg_glucose_level', y='bmi', color='stroke',
            color_discrete_map={0: COLORS['no_stroke'], 1: COLORS['stroke']},
            labels={'avg_glucose_level': 'Avg Glucose Level', 'bmi': 'BMI', 'stroke': 'Stroke'},
            title='Glucose Level vs BMI',
            opacity=0.6,
            category_orders={'stroke': [0, 1]}
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(t=50, b=40, l=50, r=20),
            height=380,
            legend_title_text='Stroke'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender & marriage breakdown
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                            subplot_titles=('Gender Distribution', 'Marital Status'))
        
        gender_counts = df['gender'].value_counts()
        fig.add_trace(go.Pie(
            labels=gender_counts.index, values=gender_counts.values,
            marker=dict(colors=['#667EEA', '#F472B6']),
            hole=0.45, textinfo='label+percent'
        ), row=1, col=1)
        
        married_counts = df['ever_married'].value_counts()
        fig.add_trace(go.Pie(
            labels=married_counts.index, values=married_counts.values,
            marker=dict(colors=['#2ECC71', '#F39C12']),
            hole=0.45, textinfo='label+percent'
        ), row=1, col=2)
        
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(t=50, b=20, l=20, r=20),
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER
# ═══════════════════════════════════════════════════════════

elif page == "🔍 Data Explorer":
    st.markdown("# 🔍 Data Explorer")
    st.markdown("*Filter, explore, and download the dataset interactively.*")
    section_divider()
    
    # Filters
    with st.expander("🎛️ Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        
        with fc1:
            age_range = st.slider("Age Range", 0, 120, (0, 120))
        with fc2:
            gender_filter = st.multiselect("Gender", df['gender'].unique().tolist(), default=df['gender'].unique().tolist())
        with fc3:
            stroke_filter = st.multiselect("Stroke", [0, 1], default=[0, 1])
        with fc4:
            work_filter = st.multiselect("Work Type", df['work_type'].unique().tolist(), default=df['work_type'].unique().tolist())
    
    # Apply filters
    mask = (
        (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
        (df['gender'].isin(gender_filter)) &
        (df['stroke'].isin(stroke_filter)) &
        (df['work_type'].isin(work_filter))
    )
    df_filtered = df[mask]
    
    # Stats
    st.markdown(f"**Showing {len(df_filtered):,} of {len(df):,} records** ({len(df_filtered)/len(df)*100:.1f}%)")
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        kpi_card("📊", f"{len(df_filtered):,}", "Records", "kpi-card-blue")
    with mc2:
        kpi_card("⚠️", f"{df_filtered['stroke'].mean()*100:.1f}%", "Stroke Rate", "kpi-card-red")
    with mc3:
        kpi_card("🎂", f"{df_filtered['age'].mean():.0f}", "Avg Age", "kpi-card-purple")
    with mc4:
        kpi_card("⚖️", f"{df_filtered['bmi'].mean():.1f}", "Avg BMI", "kpi-card-orange")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=400,
        column_config={
            "stroke": st.column_config.NumberColumn("Stroke", help="1 = stroke, 0 = no stroke"),
            "bmi": st.column_config.NumberColumn("BMI", format="%.1f"),
            "avg_glucose_level": st.column_config.NumberColumn("Glucose", format="%.1f"),
        }
    )
    
    # Download
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        csv, "stroke_data_filtered.csv", "text/csv",
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════
# PAGE 3: EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════

elif page == "📊 Exploratory Analysis":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("*Comprehensive visual analysis of patient demographics and stroke risk factors.*")
    section_divider()
    
    # ── Numerical Distributions ──
    st.markdown("### 📈 Numerical Feature Distributions")
    
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    tabs = st.tabs(["Age", "Glucose Level", "BMI"])
    
    for i, (tab, col) in enumerate(zip(tabs, num_cols)):
        with tab:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['stroke']==0][col], name='No Stroke',
                    marker_color=COLORS['no_stroke'], opacity=0.6, nbinsx=40
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['stroke']==1][col], name='Stroke',
                    marker_color=COLORS['stroke'], opacity=0.8, nbinsx=40
                ))
                fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="#E74C3C",
                              annotation_text=f"Mean: {df[col].mean():.1f}")
                fig.add_vline(x=df[col].median(), line_dash="dot", line_color="#2ECC71",
                              annotation_text=f"Median: {df[col].median():.1f}")
                fig.update_layout(
                    title=f'{col} Distribution by Stroke Status',
                    barmode='overlay', template=PLOTLY_TEMPLATE,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[df['stroke']==0][col], name='No Stroke',
                    marker_color=COLORS['no_stroke']
                ))
                fig.add_trace(go.Box(
                    y=df[df['stroke']==1][col], name='Stroke',
                    marker_color=COLORS['stroke']
                ))
                fig.update_layout(
                    title=f'{col} Box Plot',
                    template=PLOTLY_TEMPLATE,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    section_divider()
    
    # ── Categorical Analysis ──
    st.markdown("### 📊 Categorical Features vs Stroke")
    
    cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']
    
    col1, col2 = st.columns(2)
    
    for i, feat in enumerate(cat_features):
        with col1 if i % 2 == 0 else col2:
            ct = pd.crosstab(df[feat], df['stroke'], normalize='index') * 100
            ct.columns = ['No Stroke', 'Stroke']
            ct = ct.sort_values('Stroke', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ct.index.astype(str), y=ct['No Stroke'], name='No Stroke',
                marker_color=COLORS['no_stroke'], opacity=0.85
            ))
            fig.add_trace(go.Bar(
                x=ct.index.astype(str), y=ct['Stroke'], name='Stroke',
                marker_color=COLORS['stroke'], opacity=0.85
            ))
            fig.update_layout(
                title=f'Stroke Rate by {feat}',
                barmode='stack',
                yaxis_title='Percentage (%)',
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(t=50, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    section_divider()
    
    # ── Correlation Heatmap ──
    st.markdown("### 🔗 Correlation Analysis")
    
    df_corr = df.copy()
    df_corr['gender'] = df_corr['gender'].map({'Male': 1, 'Female': 0})
    df_corr['ever_married'] = df_corr['ever_married'].map({'Yes': 1, 'No': 0})
    df_corr['Residence_type'] = df_corr['Residence_type'].map({'Urban': 1, 'Rural': 0})
    df_corr = pd.get_dummies(df_corr, columns=['work_type', 'smoking_status'], drop_first=True)
    
    corr = df_corr.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
    ))
    fig.update_layout(
        title='Feature Correlation Matrix',
        template=PLOTLY_TEMPLATE,
        height=600,
        margin=dict(t=50, b=50, l=100, r=50)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with stroke
    stroke_corr = corr['stroke'].drop('stroke').sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        y=stroke_corr.index,
        x=stroke_corr.values,
        orientation='h',
        marker=dict(
            color=stroke_corr.values,
            colorscale='RdBu_r',
            cmid=0,
            showscale=True,
            colorbar=dict(title='Correlation')
        ),
        text=[f'{v:.3f}' for v in stroke_corr.values],
        textposition='outside'
    ))
    fig.update_layout(
        title='Feature Correlation with Stroke',
        xaxis_title='Pearson Correlation',
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(t=50, l=200)
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance")
    st.markdown("*Detailed evaluation of trained machine learning models.*")
    section_divider()
    
    if not metrics_loaded:
        st.error("⚠️ Metrics not found. Please run `python train_model.py` first.")
        st.stop()
    
    # Best model banner
    best = metrics[ 'best_model']
    best_auc = metrics['models'][best]['auc_roc']
    st.success(f"🏆 **Best Model: {best}** — AUC-ROC: {best_auc:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics comparison table
    model_names = list(metrics['models'].keys())
    
    mc1, mc2, mc3 = st.columns(3)
    for i, (col, name) in enumerate(zip([mc1, mc2, mc3], model_names)):
        m = metrics['models'][name]
        with col:
            css = "kpi-card-green" if name == best else "kpi-card-blue"
            st.markdown(f"""
            <div class="kpi-card {css}" style="text-align:left; padding:20px;">
                <h3 style="color:white; margin:0 0 16px;">{name} {'🏆' if name == best else ''}</h3>
                <table style="width:100%; color:white;">
                    <tr><td>Accuracy</td><td style="text-align:right; font-weight:700;">{m['accuracy']:.4f}</td></tr>
                    <tr><td>Precision</td><td style="text-align:right; font-weight:700;">{m['precision']:.4f}</td></tr>
                    <tr><td>Recall</td><td style="text-align:right; font-weight:700;">{m['recall']:.4f}</td></tr>
                    <tr><td>F1-Score</td><td style="text-align:right; font-weight:700;">{m['f1_score']:.4f}</td></tr>
                    <tr><td>AUC-ROC</td><td style="text-align:right; font-weight:700;">{m['auc_roc']:.4f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ROC Curves
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for i, name in enumerate(model_names):
            m = metrics['models'][name]
            fig.add_trace(go.Scatter(
                x=m['roc_curve']['fpr'], y=m['roc_curve']['tpr'],
                mode='lines', name=f"{name} (AUC={m['auc_roc']:.4f})",
                line=dict(color=MODEL_COLORS[i], width=3)
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash', width=1)
        ))
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template=PLOTLY_TEMPLATE,
            height=450,
            legend=dict(x=0.5, y=0.05)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        for i, name in enumerate(model_names):
            m = metrics['models'][name]
            fig.add_trace(go.Scatter(
                x=m['pr_curve']['recall'], y=m['pr_curve']['precision'],
                mode='lines', name=name,
                line=dict(color=MODEL_COLORS[i], width=3)
            ))
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            template=PLOTLY_TEMPLATE,
            height=450,
            legend=dict(x=0.5, y=0.98)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    st.markdown("### 📊 Confusion Matrices")
    cm_cols = st.columns(3)
    
    for i, (col, name) in enumerate(zip(cm_cols, model_names)):
        with col:
            cm = np.array(metrics['models'][name]['confusion_matrix'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No', 'Predicted Yes'],
                y=['Actual No', 'Actual Yes'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                showscale=False
            ))
            fig.update_layout(
                title=f'{name}',
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    if fi_loaded:
        section_divider()
        st.markdown("### 🎯 Feature Importance")
        
        fi_sorted = dict(sorted(fi_data.items(), key=lambda x: x[1], reverse=False))
        
        fig = go.Figure(go.Bar(
            y=list(fi_sorted.keys()),
            x=list(fi_sorted.values()),
            orientation='h',
            marker=dict(
                color=list(fi_sorted.values()),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Importance')
            ),
            text=[f'{v:.4f}' for v in fi_sorted.values()],
            textposition='outside'
        ))
        fig.update_layout(
            title=f'Feature Importance — {best}',
            xaxis_title='Importance Score',
            template=PLOTLY_TEMPLATE,
            height=max(400, len(fi_sorted) * 28),
            margin=dict(t=50, l=250, r=80)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison bar chart
    section_divider()
    st.markdown("### 📊 Model Comparison")
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    for i, name in enumerate(model_names):
        m = metrics['models'][name]
        values = [m[mn] for mn in metric_names]
        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=values,
            marker_color=MODEL_COLORS[i],
            opacity=0.85,
            text=[f'{v:.3f}' for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis_title='Score',
        barmode='group',
        template=PLOTLY_TEMPLATE,
        height=450,
        yaxis=dict(range=[0, 1.15]),
        legend=dict(x=0.7, y=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 5: STROKE RISK PREDICTOR
# ═══════════════════════════════════════════════════════════

elif page == "⚕️ Stroke Risk Predictor":
    st.markdown("# ⚕️ Stroke Risk Predictor")
    st.markdown("*Enter patient details to assess stroke risk using our trained ML model.*")
    section_divider()
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please run `python train_model.py` first.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Personal Details")
        gender = st.selectbox("Gender", ['Male', 'Female'], key='pred_gender')
        age = st.number_input("Age", min_value=0, max_value=120, value=50, key='pred_age')
        ever_married = st.selectbox("Ever Married", ['Yes', 'No'], key='pred_married')
        work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], key='pred_work')
        residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'], key='pred_residence')
    
    with col2:
        st.markdown("#### 🩺 Medical History")
        hypertension = st.selectbox("Hypertension", ['No', 'Yes'], key='pred_hyp')
        heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'], key='pred_hd')
        avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, key='pred_gluc')
    
    with col3:
        st.markdown("#### 🏃 Lifestyle")
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.0, key='pred_bmi')
        smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'], key='pred_smoke')
    
    section_divider()
    
    predict_col, gauge_col = st.columns([1, 2])
    
    with predict_col:
        predict_button = st.button("🔬 Analyze Stroke Risk", type="primary", use_container_width=True)
    
    if predict_button:
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100
        
        with gauge_col:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Stroke Risk Probability", 'font': {'size': 20}},
                number={'suffix': '%', 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2},
                    'bar': {'color': "#2C3E50"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'steps': [
                        {'range': [0, 15], 'color': '#2ECC71'},
                        {'range': [15, 40], 'color': '#F39C12'},
                        {'range': [40, 100], 'color': '#E74C3C'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': probability
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
        
        # Result banner
        if prediction == 1 or probability > 40:
            css = "risk-high"
            icon = "🚨"
            level = "HIGH RISK"
            msg = "Immediate medical consultation recommended. The model identifies significant stroke risk factors in this patient profile."
        elif probability > 15:
            css = "risk-moderate"
            icon = "⚠️"
            level = "MODERATE RISK"
            msg = "Regular monitoring advised. Some risk factors are present. Consider lifestyle modifications and regular check-ups."
        else:
            css = "risk-low"
            icon = "✅"
            level = "LOW RISK"
            msg = "Current health profile suggests low stroke risk. Continue maintaining a healthy lifestyle."
        
        st.markdown(f"""
        <div class="{css}">
            <h2 style="margin:0; color:white;">{icon} {level}</h2>
            <p style="font-size:1.3em; margin:10px 0; color:white;">{probability:.2f}% probability of stroke</p>
            <p style="opacity:0.9; color:white;">{msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Patient profile summary
        with st.expander("📋 Patient Profile Summary"):
            prof_col1, prof_col2 = st.columns(2)
            with prof_col1:
                st.markdown("**Demographics**")
                st.write(f"- Gender: {gender}")
                st.write(f"- Age: {age}")
                st.write(f"- Married: {ever_married}")
                st.write(f"- Work: {work_type}")
                st.write(f"- Residence: {residence_type}")
            with prof_col2:
                st.markdown("**Health Metrics**")
                st.write(f"- Hypertension: {hypertension}")
                st.write(f"- Heart Disease: {heart_disease}")
                st.write(f"- Glucose: {avg_glucose_level} mg/dL")
                st.write(f"- BMI: {bmi}")
                st.write(f"- Smoking: {smoking_status}")


# ═══════════════════════════════════════════════════════════
# PAGE 6: REPORT SUMMARY
# ═══════════════════════════════════════════════════════════

elif page == "📋 Report Summary":
    st.markdown("# 📋 Executive Report")
    st.markdown("*Automated summary report of the stroke prediction analysis.*")
    section_divider()
    
    # ── Header ──
    st.markdown("""
    <div class="report-card" style="background: linear-gradient(135deg, #0F172A, #1E293B); color:white; border:none;">
        <h2 style="color:white; margin:0;">Stroke Risk Prediction — Analytics Report</h2>
        <p style="opacity:0.8; margin:5px 0 0;">Healthcare Analytics Division • Data Science Team • March 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Dataset Overview ──
    st.markdown("### 1. Dataset Overview")
    
    st.markdown(f"""
    <div class="report-card">
        <table style="width:100%; border-collapse:collapse;">
            <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Total Records</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{len(df):,}</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Features</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df.shape[1]}</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Stroke Cases</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{int(df['stroke'].sum())} ({df['stroke'].mean()*100:.1f}%)</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Average Age</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df['age'].mean():.1f} years</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Average BMI</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df['bmi'].mean():.1f}</td></tr>
            <tr><td style="padding:8px;"><strong>Avg Glucose Level</strong></td><td style="text-align:right; padding:8px;">{df['avg_glucose_level'].mean():.1f} mg/dL</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Key Findings ──
    st.markdown("### 2. Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="report-card">
            <h4>🎯 Risk Factor Analysis</h4>
            <ul>
                <li><strong>Age</strong> is the strongest predictor — risk jumps dramatically after age 60</li>
                <li><strong>Hypertension</strong> patients have ~2x higher stroke rate</li>
                <li><strong>Heart disease</strong> patients have ~3x higher stroke rate</li>
                <li><strong>High glucose</strong> (>125 mg/dL) strongly correlates with stroke</li>
                <li><strong>BMI</strong> has moderate but significant impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="report-card">
            <h4>⚖️ Data Challenges</h4>
            <ul>
                <li><strong>Class imbalance:</strong> Only ~4.9% positive cases</li>
                <li><strong>Missing data:</strong> BMI has missing values (imputed with median)</li>
                <li><strong>Mitigation:</strong> SMOTE oversampling applied to training data</li>
                <li><strong>Evaluation:</strong> AUC-ROC used as primary metric (robust to imbalance)</li>
                <li><strong>Validation:</strong> Stratified train/test split preserves class ratio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Model Results ──
    st.markdown("### 3. Model Results")
    
    if metrics_loaded:
        results_data = []
        for name in metrics['models']:
            m = metrics['models'][name]
            results_data.append({
                'Model': name,
                'Accuracy': f"{m['accuracy']:.4f}",
                'Precision': f"{m['precision']:.4f}",
                'Recall': f"{m['recall']:.4f}",
                'F1-Score': f"{m['f1_score']:.4f}",
                'AUC-ROC': f"{m['auc_roc']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df.set_index('Model'), use_container_width=True)
        
        st.info(f"**🏆 Selected Model: {metrics['best_model']}** — Selected for highest AUC-ROC score, indicating best overall discrimination between stroke and non-stroke patients.")
    
    # ── Recommendations ──
    st.markdown("### 4. Business Recommendations")
    
    recs = [
        ("🎯", "Priority Screening", "Focus on patients aged 60+ with hypertension or heart disease — they represent the highest risk group"),
        ("📊", "Glucose Monitoring", "Implement regular glucose monitoring for patients with pre-diabetic levels (100-125 mg/dL)"),
        ("⚖️", "Model Deployment", "Deploy Logistic Regression model for interpretable, real-time stroke risk scoring in clinical workflows"),
        ("🔄", "Continuous Learning", "Retrain model quarterly with new patient data. Monitor for concept drift and feature distribution changes"),
        ("📋", "Risk Score Integration", "Integrate composite risk score into patient intake forms and electronic health records for rapid pre-screening"),
    ]
    
    for icon, title, desc in recs:
        st.markdown(f"""
        <div class="report-card">
            <h4>{icon} {title}</h4>
            <p style="margin:0; color:#4A5568;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Limitations ──
    st.markdown("### 5. Limitations & Next Steps")
    
    st.markdown("""
    <div class="report-card" style="border-left: 4px solid #F39C12;">
        <h4>⚠️ Current Limitations</h4>
        <ul style="color:#4A5568;">
            <li>Dataset is limited to ~5,000 records — more data would improve generalization</li>
            <li>Missing BMI values were imputed, which may introduce bias</li>
            <li>No geographic, genetic, or medication data available</li>
            <li>Model performance on real-world clinical data may differ from training metrics</li>
        </ul>
        <h4 style="margin-top:16px;">🚀 Recommended Next Steps</h4>
        <ul style="color:#4A5568;">
            <li>Collect additional patient data from partner hospitals</li>
            <li>Integrate lab test results and medication history</li>
            <li>Develop time-series analysis for recurring patient visits</li>
            <li>Build automated alerting system for high-risk patients</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    section_divider()
    
    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:0.85em;">
        Report generated automatically by the Stroke Prediction Analytics Pipeline<br>
        Healthcare Analytics Division • Data Science Team • March 2026
    </div>
    """, unsafe_allow_html=True)
