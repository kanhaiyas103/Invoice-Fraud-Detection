import streamlit as st
import pandas as pd
import numpy as np

# UI styling
st.set_page_config(page_title="Invoice Fraud Detection", layout="wide")
st.markdown("""
    <style>
        .main-title {
            font-size:32px;
            font-weight:bold;
            color:#007acc;
        }
        .subtitle {
            font-size:20px;
            color:#444;
        }
    </style>
    <div class="main-title">📁 Invoice Fraud Detection</div>
    <div class="subtitle">Built with 🧠 Machine Learning and Streamlit</div>
""", unsafe_allow_html=True)

# Load data
uploaded_file = st.sidebar.file_uploader("Upload Invoices CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Clean and preprocess
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['fraud_risk_score'] = df['fraud_risk_score'].fillna(0.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    # Dropdown filters
    vendor_options = df["Vendor Name"].dropna().unique().tolist()
    vendor_filter = st.sidebar.multiselect("Vendor Name", vendor_options, default=vendor_options)

    department_options = df["Department"].dropna().unique().tolist()
    department_filter = st.sidebar.multiselect("Department", department_options, default=department_options)

    # Risk score slider
    min_score = float(df["fraud_risk_score"].min())
    max_score = float(df["fraud_risk_score"].max())
    score_range = st.sidebar.slider("📊 Fraud Risk Score Range", float(min_score), float(max_score), (min_score, max_score))

    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date])

    # Apply filters
    filtered_df = df[
        (df["Vendor Name"].isin(vendor_filter)) &
        (df["Department"].isin(department_filter)) &
        (df["fraud_risk_score"] >= score_range[0]) &
        (df["fraud_risk_score"] <= score_range[1]) &
        (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
    ]

    # Display filtered data
    st.subheader("📄 Filtered Invoice Records")
    st.dataframe(filtered_df, use_container_width=True)

    # Show fraud count
    fraud_count = (filtered_df['fraud_label'] == 1).sum()
    total = len(filtered_df)
    st.success(f"🔎 {fraud_count} fraud invoices detected out of {total} filtered invoices.")

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", csv, "filtered_invoices.csv", "text/csv")

else:
    st.warning("👈 Please upload a CSV file to begin.")