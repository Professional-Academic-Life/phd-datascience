import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# Load data function (you'll need to replace this with your actual data loading)
@st.cache_data
def load_data():
    houses = pd.read_csv("recife.csv", encoding="ISO-8859-1")

    # Adicionando coluna para aluguel e venda
    houses['operation'] = 'sell'
    houses.loc[(houses['price'] > 100) & (houses['price'] < 30000), 'operation'] = 'rent'

    houses['price_m2'] = houses['price'] / houses['area']

    # Clean data to handle potential NaN values
    required_columns = ['price']
    if 'area' in houses.columns:
        required_columns.append('area')

    # Don't drop rows with NaN in optional columns, just handle them in visualizations
    houses = houses.dropna(subset=required_columns)

    return houses


# Load the data
df = load_data()

# Main title
st.markdown('<h1 class="main-header">🏠 Real Estate Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Price range filter
price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max())),
    format="$%d"
)

# Property type filter
property_types = st.sidebar.multiselect(
    "Property Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)

# City filter
cities = st.sidebar.multiselect(
    "City",
    options=df['city'].unique(),
    default=df['city'].unique()
)

# Operation filter
operations = st.sidebar.multiselect(
    "Operation",
    options=df['operation'].unique(),
    default=df['operation'].unique()
)

# Bedroom filter
bedroom_options = sorted([x for x in df['bedrooms'].unique() if pd.notna(x)])
bedrooms = st.sidebar.multiselect(
    "Number of Bedrooms",
    options=sorted(bedroom_options),
    default=sorted(bedroom_options)
)

# Apply filters
filtered_df = df[
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['type'].isin(property_types)) &
    (df['city'].isin(cities)) &
    (df['operation'].isin(operations)) &
    (df['bedrooms'].isin(bedrooms))
    ]

# Key metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="📊 Total Properties",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df):,} from total"
    )

with col2:
    avg_price = filtered_df['price'].mean()
    st.metric(
        label="💰 Average Price",
        value=f"${avg_price:,.0f}",
        delta=f"${avg_price - df['price'].mean():,.0f}"
    )

with col3:
    median_price = filtered_df['price'].median()
    st.metric(
        label="📈 Median Price",
        value=f"${median_price:,.0f}",
        delta=f"${median_price - df['price'].median():,.0f}"
    )

with col4:
    avg_area = filtered_df['area'].mean()
    st.metric(
        label="📐 Average Area",
        value=f"{avg_area:.0f} m²",
        delta=f"{avg_area - df['area'].mean():.0f} m²"
    )

with col5:
    avg_price_m2 = filtered_df['price_m2'].mean()
    st.metric(
        label="💵 Price per m²",
        value=f"${avg_price_m2:,.0f}",
        delta=f"${avg_price_m2 - df['price_m2'].mean():,.0f}"
    )

st.markdown("---")

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Geographic Analysis", "📊 Market Analysis", "🏠 Property Features", "📈 Price Analytics"])

with tab1:
    st.subheader("Geographic Distribution of Properties")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Interactive map
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            # Prepare data for map - handle NaN values
            map_data = filtered_df.dropna(subset=['latitude', 'longitude', 'price']).copy()

            # Handle size parameter - use area if available, otherwise use a constant size
            if 'area' in map_data.columns:
                # Fill NaN values in area with median area for size calculation
                map_data['area_for_size'] = map_data['area'].fillna(map_data['area'].median())
                size_col = 'area_for_size'
            else:
                # Use constant size if area not available
                map_data['constant_size'] = 10
                size_col = 'constant_size'

            # Create hover data list based on available columns
            hover_cols = ['type', 'price']
            if 'bedrooms' in map_data.columns:
                hover_cols.append('bedrooms')
            if 'bathrooms' in map_data.columns:
                hover_cols.append('bathrooms')
            if 'suburb' in map_data.columns:
                hover_cols.append('suburb')

            if len(map_data) > 0:
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="latitude",
                    lon="longitude",
                    color="price",
                    size=size_col,
                    hover_data=hover_cols,
                    color_continuous_scale="Viridis",
                    mapbox_style="open-street-map",
                    zoom=8,
                    height=500,
                    title="Property Locations with Price and Size"
                )
                fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No valid location data available for mapping")
        else:
            st.info("Map requires latitude and longitude columns")

    with col2:
        # City distribution
        city_counts = filtered_df['city'].value_counts()
        fig_city = px.pie(
            values=city_counts.values,
            names=city_counts.index,
            title="Properties by City",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_city, use_container_width=True)

        # District distribution
        district_counts = filtered_df['district'].value_counts()
        fig_district = px.bar(
            x=district_counts.values,
            y=district_counts.index,
            orientation='h',
            title="Properties by District",
            color=district_counts.values,
            color_continuous_scale="Blues"
        )
        fig_district.update_layout(showlegend=False)
        st.plotly_chart(fig_district, use_container_width=True)

with tab2:
    st.subheader("Market Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution by property type
        fig_box = px.box(
            filtered_df,
            x="type",
            y="price",
            color="type",
            title="Price Distribution by Property Type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        # Sale vs Rent analysis
        operation_stats = filtered_df.groupby('operation')['price'].agg(['mean', 'median', 'count']).reset_index()
        fig_operation = px.bar(
            operation_stats,
            x='operation',
            y='mean',
            title="Average Price by Operation Type",
            color='operation',
            text='count'
        )
        fig_operation.update_traces(texttemplate='Count: %{text}', textposition='outside')
        st.plotly_chart(fig_operation, use_container_width=True)

    with col2:
        # Price vs Area scatter
        if 'area' in filtered_df.columns:
            # Clean data for scatter plot
            scatter_data = filtered_df.dropna(subset=['area', 'price']).copy()

            if len(scatter_data) > 0:
                # Prepare hover data
                hover_cols = ['price', 'area']
                if 'suburb' in scatter_data.columns:
                    hover_cols.append('suburb')
                if 'bathrooms' in scatter_data.columns:
                    hover_cols.append('bathrooms')

                # Handle size parameter for scatter plot
                if 'bedrooms' in scatter_data.columns:
                    scatter_data['bedrooms_for_size'] = scatter_data['bedrooms'].fillna(
                        scatter_data['bedrooms'].median())
                    size_col = 'bedrooms_for_size'
                else:
                    scatter_data['constant_size'] = 5
                    size_col = 'constant_size'

                fig_scatter = px.scatter(
                    scatter_data,
                    x="area",
                    y="price",
                    color="type",
                    size=size_col,
                    hover_data=hover_cols,
                    title="Price vs Area Relationship",
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No valid area and price data available for scatter plot")
        else:
            st.info("Scatter plot requires area column")

        # Price per m² by city
        if 'price_m2' in filtered_df.columns:
            price_m2_city = filtered_df.groupby('city')['price_m2'].mean().sort_values(ascending=False)
            fig_price_m2 = px.bar(
                x=price_m2_city.index,
                y=price_m2_city.values,
                title="Average Price per m² by City",
                color=price_m2_city.values,
                color_continuous_scale="Reds"
            )
            fig_price_m2.update_layout(showlegend=False)
            st.plotly_chart(fig_price_m2, use_container_width=True)
        else:
            st.info("Price per m² analysis requires area column in dataset")
with tab3:
    st.subheader("Property Features Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Bedrooms distribution
        if 'bedrooms' in filtered_df.columns:
            bedroom_counts = filtered_df['bedrooms'].value_counts().sort_index()
            fig_bedrooms = px.bar(
                x=bedroom_counts.index,
                y=bedroom_counts.values,
                title="Distribution of Bedrooms",
                color=bedroom_counts.values,
                color_continuous_scale="Blues"
            )
            fig_bedrooms.update_layout(showlegend=False)
            st.plotly_chart(fig_bedrooms, use_container_width=True)
        else:
            st.info("Bedroom analysis requires bedrooms column in dataset")

        # Parking spaces analysis
        if 'pkspaces' in filtered_df.columns:
            parking_price = filtered_df.groupby('pkspaces')['price'].mean()
            fig_parking = px.line(
                x=parking_price.index,
                y=parking_price.values,
                markers=True,
                title="Average Price by Parking Spaces",
                line_shape='linear'
            )
            st.plotly_chart(fig_parking, use_container_width=True)
        else:
            st.info("Parking analysis requires pkspaces column in dataset")

    with col2:
        # Bathrooms vs Price
        fig_bath_price = px.violin(
            filtered_df,
            x="bathrooms",
            y="price",
            title="Price Distribution by Number of Bathrooms",
            color="bathrooms"
        )
        fig_bath_price.update_layout(showlegend=False)
        st.plotly_chart(fig_bath_price, use_container_width=True)

        # Feature correlation heatmap
        numeric_cols = ['price', 'bedrooms', 'area', 'pkspaces', 'bathrooms', 'ensuites', 'price_m2']
        corr_matrix = filtered_df[numeric_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.subheader("Advanced Price Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution histogram
        fig_hist = px.histogram(
            filtered_df,
            x="price",
            nbins=50,
            title="Price Distribution",
            color_discrete_sequence=["skyblue"]
        )
        fig_hist.add_vline(x=filtered_df['price'].mean(), line_dash="dash", line_color="red", annotation_text="Mean")
        fig_hist.add_vline(x=filtered_df['price'].median(), line_dash="dash", line_color="green",
                           annotation_text="Median")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Price trends by suburb (top 10)
        top_suburbs = filtered_df['suburb'].value_counts().head(10).index
        suburb_data = filtered_df[filtered_df['suburb'].isin(top_suburbs)]
        fig_suburb = px.box(
            suburb_data,
            x="suburb",
            y="price",
            title="Price Distribution by Top 10 Suburbs"
        )
        fig_suburb.update_xaxes(tickangle=45)
        st.plotly_chart(fig_suburb, use_container_width=True)

    with col2:
        # Price per m² distribution
        if 'price_m2' in filtered_df.columns:
            fig_price_m2_hist = px.histogram(
                filtered_df,
                x="price_m2",
                nbins=30,
                title="Price per m² Distribution",
                color_discrete_sequence=["lightcoral"]
            )
            st.plotly_chart(fig_price_m2_hist, use_container_width=True)
        else:
            st.info("Price per m² distribution requires area column")

        # Multi-dimensional analysis
        if 'area' in filtered_df.columns and 'bedrooms' in filtered_df.columns and 'bathrooms' in filtered_df.columns:
            # Clean data for 3D plot - remove NaN values
            plot_3d_data = filtered_df.dropna(subset=['area', 'bedrooms', 'price', 'bathrooms']).copy()

            if len(plot_3d_data) > 0:
                sample_data = plot_3d_data.sample(min(500, len(plot_3d_data)))  # Sample for performance

                # Double-check for any remaining NaN values and fill them
                sample_data['bathrooms_clean'] = sample_data['bathrooms'].fillna(sample_data['bathrooms'].median())

                fig_3d = px.scatter_3d(
                    sample_data,
                    x="area",
                    y="bedrooms",
                    z="price",
                    color="type",
                    size="bathrooms_clean",
                    title="3D Analysis: Area vs Bedrooms vs Price"
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("No complete data available for 3D analysis")
        else:
            st.info("3D analysis requires area, bedrooms, and bathrooms columns")

# Additional insights section
st.markdown("---")
st.subheader("📋 Data Summary & Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏆 Top Performing Suburbs")
    top_suburbs_price = filtered_df.groupby('suburb')['price'].mean().sort_values(ascending=False).head(5)
    for suburb, price in top_suburbs_price.items():
        st.write(f"**{suburb}**: ${price:,.0f}")

with col2:
    st.markdown("### 📊 Market Statistics")
    st.write(f"**Price Range**: ${filtered_df['price'].min():,.0f} - ${filtered_df['price'].max():,.0f}")
    st.write(f"**Most Common Type**: {filtered_df['type'].mode().iloc[0]}")
    st.write(f"**Average Area**: {filtered_df['area'].mean():.0f} m²")
    st.write(f"**Most Common Bedrooms**: {filtered_df['bedrooms'].mode().iloc[0]}")

with col3:
    st.markdown("### 🎯 Quick Insights")
    highest_price_m2_city = filtered_df.groupby('city')['price_m2'].mean().idxmax()
    most_expensive_type = filtered_df.groupby('type')['price'].mean().idxmax()
    st.write(f"**Highest $/m² City**: {highest_price_m2_city}")
    st.write(f"**Most Expensive Type**: {most_expensive_type}")
    st.write(
        f"**Sale vs Rent Ratio**: {(filtered_df['operation'] == 'Sale').sum()}:{(filtered_df['operation'] == 'Rent').sum()}")

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit and Plotly. Data reflects filtered selection.*")

# Optional: Raw data viewer
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_real_estate_data.csv",
        mime="text/csv"
    )