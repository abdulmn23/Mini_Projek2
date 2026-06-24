import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def detect_columns(df):
    """Detect column names regardless of capitalization or slight variations"""
    columns = df.columns.tolist()
    
    # Common variations of price column
    price_variations = ['Price', 'price', 'PRICE', 'Prices', 'prices', 'Harga', 'harga']
    price_col = None
    for col in columns:
        if col in price_variations or col.lower() in [v.lower() for v in price_variations]:
            price_col = col
            break
    
    # Common variations of company column
    company_variations = ['Company', 'company', 'COMPANY', 'Brand', 'brand', 'Merk', 'merk']
    company_col = None
    for col in columns:
        if col in company_variations or col.lower() in [v.lower() for v in company_variations]:
            company_col = col
            break
    
    # Get all numeric columns (except price)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if price_col and price_col in numeric_cols:
        numeric_cols.remove(price_col)
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if company_col and company_col in categorical_cols:
        categorical_cols.remove(company_col)
    
    return price_col, company_col, numeric_cols, categorical_cols

def clean_price_column(df, price_col):
    """Clean and convert price column to numeric"""
    if price_col not in df.columns:
        return df, None
    
    # Remove any non-numeric characters and convert
    df[price_col] = df[price_col].astype(str).str.replace('.', '', regex=False)
    df[price_col] = df[price_col].str.replace(',', '', regex=False)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    return df, price_col

def show_visual(df):
    """
    Display comprehensive visualizations of the laptop dataset
    """
    st.subheader("📊 Visualisasi Dataset")
    
    # Check if DataFrame is empty
    if df.empty:
        st.warning("Dataset kosong! Tidak ada data untuk divisualisasikan.")
        return
    
    # Clean the price column first
    price_col = None
    for col in df.columns:
        if col.lower() == 'price':
            price_col = col
            df, price_col = clean_price_column(df, col)
            break
    
    # Detect columns
    detected_price_col, company_col, numeric_cols, categorical_cols = detect_columns(df)
    
    # Use detected price column if not found earlier
    if price_col is None and detected_price_col is not None:
        price_col = detected_price_col
        df, price_col = clean_price_column(df, price_col)
    
    # Show detected columns for debugging
    with st.expander("🔍 Informasi Dataset"):
        st.write(f"**Total Data:** {len(df)} baris, {len(df.columns)} kolom")
        st.write(f"**Kolom yang tersedia:** {df.columns.tolist()}")
        st.write(f"**Detected Price Column:** {price_col}")
        st.write(f"**Detected Company Column:** {company_col}")
        st.write(f"**Numeric Columns:** {numeric_cols}")
        st.write("**Sample Data:**")
        st.dataframe(df.head())
    
    if price_col is None or df[price_col].isna().all():
        st.error("❌ Tidak ditemukan kolom 'Price' yang valid dalam dataset. Pastikan dataset memiliki kolom harga dengan format angka.")
        return
    
    # Remove rows with null price
    df_clean = df.dropna(subset=[price_col])
    
    if df_clean.empty:
        st.warning("Tidak ada data harga yang valid.")
        return
    
    st.success(f"✅ Ditemukan {len(df_clean)} data dengan harga yang valid")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribusi Harga", 
        "🏢 Harga per Perusahaan", 
        "📊 Korelasi", 
        "🔍 Detail Features"
    ])
    
    with tab1:
        show_price_distribution(df_clean, price_col)
    
    with tab2:
        show_price_by_company(df_clean, price_col, company_col)
    
    with tab3:
        show_correlation_matrix(df_clean, price_col, numeric_cols)
    
    with tab4:
        show_feature_analysis(df_clean, price_col, numeric_cols, categorical_cols)

def show_price_distribution(df, price_col):
    """Display price distribution histogram"""
    st.subheader("Distribusi Harga Laptop")
    
    # Check if price column has valid data
    if df[price_col].isna().all():
        st.warning(f"Kolom '{price_col}' tidak memiliki data yang valid.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(
            data=df,
            x=price_col,
            kde=True,
            bins=30,
            color='skyblue',
            edgecolor='black',
            ax=ax
        )
        ax.set_title(f'Distribusi {price_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(price_col, fontsize=12)
        ax.set_ylabel('Frekuensi', fontsize=12)
        
        # Add statistics
        mean_price = df[price_col].mean()
        median_price = df[price_col].median()
        ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_price:,.0f}')
        ax.axvline(median_price, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_price:,.0f}')
        ax.legend()
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Display statistics
        st.subheader(f"Statistik {price_col}")
        stats = {
            'Mean': f"{df[price_col].mean():,.0f}",
            'Median': f"{df[price_col].median():,.0f}",
            'Min': f"{df[price_col].min():,.0f}",
            'Max': f"{df[price_col].max():,.0f}",
            'Std Dev': f"{df[price_col].std():,.0f}"
        }
        for key, value in stats.items():
            st.metric(key, value)

def show_price_by_company(df, price_col, company_col):
    """Display price distribution by company"""
    st.subheader(f"{price_col} per {company_col if company_col else 'Perusahaan'}")
    
    # Check if Company column exists
    if company_col is None:
        st.warning("Kolom 'Company' tidak ditemukan dalam dataset.")
        return
    
    # Calculate statistics per company
    company_stats = df.groupby(company_col)[price_col].agg(['mean', 'median', 'count', 'std']).reset_index()
    company_stats.columns = [company_col, 'Mean Price', 'Median Price', 'Count', 'Std Dev']
    company_stats = company_stats.sort_values('Mean Price', ascending=False)
    
    # Show top companies
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Limit to top 20 companies if too many
        top_companies = company_stats.head(20)[company_col].tolist()
        plot_df = df[df[company_col].isin(top_companies)]
        
        sns.boxplot(
            data=plot_df,
            x=company_col,
            y=price_col,
            palette="viridis",
            ax=ax
        )
        plt.xticks(rotation=45, ha='right')
        ax.set_title(f'Distribusi {price_col} per {company_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(company_col, fontsize=12)
        ax.set_ylabel(price_col, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Bar chart of average prices
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_stats = company_stats.head(20)
        bars = ax.bar(
            plot_stats[company_col], 
            plot_stats['Mean Price'],
            color=plt.cm.viridis(np.linspace(0, 1, len(plot_stats))),
            edgecolor='black'
        )
        
        ax.set_title(f'Rata-rata {price_col} per {company_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(company_col, fontsize=12)
        ax.set_ylabel(f'Rata-rata {price_col}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, price in zip(bars, plot_stats['Mean Price']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{price:,.0f}',
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def show_correlation_matrix(df, price_col, numeric_cols):
    """Display correlation matrix of numeric features"""
    st.subheader("Matriks Korelasi Fitur Numerik")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        st.warning("Tidak ada cukup kolom numerik untuk analisis korelasi.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Matriks Korelasi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show strongest correlations with Price
    if price_col in corr_matrix.columns:
        st.subheader(f"Korelasi dengan {price_col}")
        price_corr = corr_matrix[price_col].sort_values(ascending=False)
        
        # Create dataframe for display
        corr_df = pd.DataFrame({
            'Fitur': price_corr.index,
            'Korelasi': price_corr.values
        })
        corr_df = corr_df[corr_df['Fitur'] != price_col]
        
        # Color code correlations
        def color_corr(val):
            if val > 0.5:
                return 'background-color: #90EE90'
            elif val < -0.5:
                return 'background-color: #FFB6C1'
            else:
                return ''
        
        st.dataframe(
            corr_df.style.applymap(color_corr, subset=['Korelasi']),
            use_container_width=True
        )

def show_feature_analysis(df, price_col, numeric_cols, categorical_cols):
    """Display additional feature analysis"""
    st.subheader("Analisis Fitur Detail")
    
    # Select feature to analyze
    all_features = numeric_cols + categorical_cols
    
    if not all_features:
        st.warning("Tidak ada fitur untuk dianalisis.")
        return
    
    selected_feature = st.selectbox(
        "Pilih fitur untuk dianalisis:",
        all_features,
        help="Pilih fitur untuk melihat hubungannya dengan harga"
    )
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Check if feature is numeric or categorical
            if selected_feature in numeric_cols:
                # Scatter plot for numeric features
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    df[selected_feature],
                    df[price_col],
                    alpha=0.6,
                    c=df[price_col],
                    cmap='viridis',
                    edgecolors='black',
                    linewidth=0.5
                )
                ax.set_title(f'Hubungan {selected_feature} vs {price_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(selected_feature, fontsize=12)
                ax.set_ylabel(price_col, fontsize=12)
                plt.colorbar(scatter, label=price_col)
                st.pyplot(fig)
                plt.close()
                
                # Show correlation
                corr_value = df[selected_feature].corr(df[price_col])
                st.metric(f"Korelasi {selected_feature} dengan {price_col}", f"{corr_value:.3f}")
            
            else:
                # Bar plot for categorical features
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by categorical feature
                grouped = df.groupby(selected_feature)[price_col].mean().sort_values(ascending=False)
                
                grouped.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Rata-rata {price_col} berdasarkan {selected_feature}', fontsize=14, fontweight='bold')
                ax.set_xlabel(selected_feature, fontsize=12)
                ax.set_ylabel(f'Rata-rata {price_col}', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            # Boxplot for distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if selected_feature in numeric_cols and df[selected_feature].nunique() < 20:
                # Boxplot for numeric with few unique values
                sns.boxplot(
                    data=df,
                    x=selected_feature,
                    y=price_col,
                    palette='Set2',
                    ax=ax
                )
            elif selected_feature in categorical_cols:
                # Boxplot for categorical
                sns.boxplot(
                    data=df,
                    x=selected_feature,
                    y=price_col,
                    palette='Set2',
                    ax=ax
                )
            else:
                # Histogram for numeric with many unique values
                sns.histplot(
                    data=df,
                    x=selected_feature,
                    kde=True,
                    color='lightblue',
                    ax=ax
                )
            
            ax.set_title(f'Distribusi {selected_feature}', fontsize=14, fontweight='bold')
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel('Frekuensi' if selected_feature in numeric_cols and df[selected_feature].nunique() >= 20 else price_col, fontsize=12)
            
            if selected_feature in categorical_cols or (selected_feature in numeric_cols and df[selected_feature].nunique() < 20):
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Show statistics
        with st.expander(f"Statistik {selected_feature}"):
            if selected_feature in numeric_cols:
                stats_df = df[selected_feature].describe()
                st.dataframe(pd.DataFrame(stats_df).T)
            else:
                stats_df = df[selected_feature].value_counts()
                st.dataframe(pd.DataFrame(stats_df).reset_index().rename(
                    columns={'index': selected_feature, selected_feature: 'Count'}
                ))

def show_data_summary(df):
    """Display comprehensive data summary"""
    st.subheader("📋 Ringkasan Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Baris", f"{len(df):,}")
    with col2:
        st.metric("Total Kolom", f"{len(df.columns)}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    # Show data types
    with st.expander("Lihat Tipe Data"):
        st.dataframe(df.dtypes.reset_index().rename(
            columns={'index': 'Kolom', 0: 'Tipe Data'}
        ))
    
    # Show basic statistics
    with st.expander("Lihat Statistik Deskriptif"):
        st.dataframe(df.describe())

# Main function
def show_visual_enhanced(df):
    """Enhanced version of show_visual with more features"""
    show_data_summary(df)
    show_visual(df)
    
    # Add download option
    st.download_button(
        label="📥 Download Data",
        data=df.to_csv(index=False),
        file_name="laptop_data_clean.csv",
        mime="text/csv"
    )