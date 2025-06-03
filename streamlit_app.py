import streamlit as st
import geopandas as gpd
import rasterio
import rasterio.mask
import requests
import tempfile
import os
import gzip
import zipfile
import math
import numpy as np
import pandas as pd
from io import BytesIO
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="CHIRPS Rainfall Analysis",
    page_icon="🌧️",
    layout="wide"
)

# Define country codes globally - THIS MUST BE AT TOP LEVEL
COUNTRY_OPTIONS = {
    "Sierra Leone": "SLE",
    "Gambia": "GMB",
    "Guinea": "GIN", 
    "Mali": "MLI",
    "Burkina Faso": "BFA",
    "Niger": "NER",
    "Ghana": "GHA",
    "Ivory Coast": "CIV",
    "Liberia": "LBR",
    "Senegal": "SEN",
    "Guinea-Bissau": "GNB",
    "Mauritania": "MRT",
    "Nigeria": "NGA",
    "Benin": "BEN",
    "Togo": "TGO",
    "Chad": "TCD",
    "Cameroon": "CMR",
    "Central African Republic": "CAF",
    "Gabon": "GAB",
    "Equatorial Guinea": "GNQ",
    "Republic of the Congo": "COG",
    "Democratic Republic of the Congo": "COD",
    "Angola": "AGO",
    "Zambia": "ZMB",
    "Kenya": "KEN",
    "Tanzania": "TZA",
    "Uganda": "UGA",
    "Rwanda": "RWA",
    "Burundi": "BDI",
    "Ethiopia": "ETH",
    "South Sudan": "SSD",
    "Sudan": "SDN",
    "Madagascar": "MDG",
    "Mozambique": "MOZ",
    "Malawi": "MWI",
    "Zimbabwe": "ZWE",
    "Botswana": "BWA",
    "Namibia": "NAM",
    "South Africa": "ZAF"
}

# Initialize session state variables to avoid NameError
if 'data_source' not in st.session_state:
    st.session_state.data_source = "GADM Database"
if 'country' not in st.session_state:
    st.session_state.country = "Sierra Leone"
if 'country_code' not in st.session_state:
    st.session_state.country_code = "SLE"
if 'admin_level' not in st.session_state:
    st.session_state.admin_level = 1
if 'use_custom_shapefile' not in st.session_state:
    st.session_state.use_custom_shapefile = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}

# Add caching for better performance
@st.cache_data
def check_file_exists(url):
    """Check if a file exists on GitHub with timeout"""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@st.cache_data
def download_shapefile_from_gadm(country_code, admin_level):
    """Download and load shapefiles directly from GADM website"""
    gadm_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{country_code}_shp.zip"
    
    try:
        # Download the zip file
        response = requests.get(gadm_url, timeout=120, stream=True)
        response.raise_for_status()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save zip file
            zip_path = os.path.join(tmpdir, f"gadm41_{country_code}.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Find the correct shapefile for the admin level
            shapefile_name = f"gadm41_{country_code}_{admin_level}.shp"
            shapefile_path = os.path.join(tmpdir, shapefile_name)
            
            if not os.path.exists(shapefile_path):
                available_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                available_levels = []
                for file in available_files:
                    if f"gadm41_{country_code}_" in file:
                        level = file.split('_')[-1].replace('.shp', '')
                        if level.isdigit():
                            available_levels.append(level)
                raise FileNotFoundError(f"Admin level {admin_level} not found for {country_code}. Available levels: {sorted(available_levels)}")
            
            # Load the shapefile
            gdf = gpd.read_file(shapefile_path)
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download from GADM: {str(e)}")
    except zipfile.BadZipFile:
        raise ValueError("Downloaded file is not a valid zip file")
    except Exception as e:
        raise ValueError(f"Failed to process shapefile: {str(e)}")
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    return gdf

def load_uploaded_shapefile(shp_file, shx_file, dbf_file, prj_file=None):
    """Load shapefile from uploaded files with optional projection file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        shp_path = os.path.join(tmpdir, "uploaded.shp")
        shx_path = os.path.join(tmpdir, "uploaded.shx") 
        dbf_path = os.path.join(tmpdir, "uploaded.dbf")
        prj_path = os.path.join(tmpdir, "uploaded.prj")
        
        with open(shp_path, "wb") as f:
            f.write(shp_file.getvalue())
        with open(shx_path, "wb") as f:
            f.write(shx_file.getvalue())
        with open(dbf_path, "wb") as f:
            f.write(dbf_file.getvalue())
        
        # Save projection file if provided
        projection_info = None
        if prj_file is not None:
            with open(prj_path, "wb") as f:
                f.write(prj_file.getvalue())
            
            # Read projection info for display
            try:
                with open(prj_path, "r") as f:
                    projection_info = f.read().strip()
            except Exception:
                projection_info = "Could not read projection file"
        
        try:
            # Load the shapefile (GeoPandas will automatically read .prj if it exists)
            gdf = gpd.read_file(shp_path)
        except Exception as e:
            raise ValueError(f"Failed to read uploaded shapefile: {str(e)}")
    
    # Handle coordinate reference system
    crs_source = None
    if gdf.crs is not None:
        crs_source = "from .prj file" if prj_file is not None else "detected automatically"
    else:
        # No CRS detected - assume WGS84
        st.warning("⚠️ No coordinate reference system detected. Assuming WGS84 (EPSG:4326)")
        gdf = gdf.set_crs("EPSG:4326")
        crs_source = "assumed (WGS84)"
    
    return gdf, crs_source, projection_info

def validate_chirps_date(year, month):
    """Validate if CHIRPS data is available for the given date"""
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    if year < 1981:
        return False, "CHIRPS data starts from 1981"
    if year > current_year or (year == current_year and month > current_month - 2):
        return False, "CHIRPS data has ~2 month delay"
    return True, ""

@st.cache_data
def download_chirps_data(year, month):
    """Download CHIRPS data and return the file path"""
    # Validate date first
    is_valid, error_msg = validate_chirps_date(year, month)
    if not is_valid:
        raise ValueError(error_msg)
    
    link = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz"
    
    try:
        response = requests.get(link, timeout=120, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download CHIRPS data for {year}-{month:02d}: {str(e)}")

    # Return the raw data as bytes
    return response.content

def process_chirps_data(_gdf, year, month):
    """Process CHIRPS rainfall data with improved error handling"""
    
    # Create a copy to avoid modifying the original
    gdf = _gdf.copy()
    
    # Download the CHIRPS data (this part is cached)
    chirps_data = download_chirps_data(year, month)

    with tempfile.TemporaryDirectory() as tmpdir:
        zipped_file_path = os.path.join(tmpdir, "chirps.tif.gz")
        
        # Save the downloaded data
        with open(zipped_file_path, "wb") as f:
            f.write(chirps_data)

        unzipped_file_path = os.path.join(tmpdir, "chirps.tif")
        
        try:
            with gzip.open(zipped_file_path, "rb") as gz:
                with open(unzipped_file_path, "wb") as tif:
                    tif.write(gz.read())
        except gzip.BadGzipFile:
            raise ValueError("Downloaded file is not a valid gzip file")

        try:
            with rasterio.open(unzipped_file_path) as src:
                # Reproject geodataframe to match raster CRS
                gdf_reproj = gdf.to_crs(src.crs)
                
                mean_rains = []
                valid_pixels_count = []
                
                for idx, geom in enumerate(gdf_reproj.geometry):
                    try:
                        masked_data, _ = rasterio.mask.mask(src, [geom], crop=True, nodata=src.nodata)
                        masked_data = masked_data.flatten()
                        
                        # Filter out nodata values
                        valid_data = masked_data[masked_data != src.nodata]
                        valid_data = valid_data[~np.isnan(valid_data)]
                        
                        if len(valid_data) > 0:
                            mean_rains.append(np.mean(valid_data))
                            valid_pixels_count.append(len(valid_data))
                        else:
                            mean_rains.append(np.nan)
                            valid_pixels_count.append(0)
                    except Exception as e:
                        st.warning(f"Error processing geometry {idx}: {str(e)}")
                        mean_rains.append(np.nan)
                        valid_pixels_count.append(0)

                gdf["mean_rain"] = mean_rains
                gdf["valid_pixels"] = valid_pixels_count
        except rasterio.errors.RasterioIOError as e:
            raise ValueError(f"Failed to process raster file: {str(e)}")
    
    return gdf

def create_rainfall_map(gdf, title, color_scheme="Blues"):
    """Create an interactive rainfall map using Plotly"""
    
    # Prepare data for plotting
    plot_data = gdf.copy()
    
    # Handle missing data
    if plot_data['mean_rain'].isna().all():
        st.warning(f"No rainfall data available for {title}")
        return None
    
    # Create the map
    fig = px.choropleth_mapbox(
        plot_data,
        geojson=plot_data.geometry.__geo_interface__,
        locations=plot_data.index,
        color='mean_rain',
        color_continuous_scale=color_scheme,
        labels={'mean_rain': 'Rainfall (mm)'},
        title=title,
        mapbox_style="carto-positron",
        center={"lat": plot_data.geometry.centroid.y.mean(), 
                "lon": plot_data.geometry.centroid.x.mean()},
        zoom=5,
        height=600
    )
    
    fig.update_layout(
        title_font_size=16,
        font=dict(size=12),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_time_series_plot(data_dict, region_name, region_index=None):
    """Create time series plot for a specific region"""
    
    if not data_dict:
        st.warning("No data available for time series plot")
        return None
    
    # Prepare data
    plot_data = []
    for key, gdf in data_dict.items():
        year, month = key
        if region_index is not None and region_index < len(gdf):
            rainfall = gdf.iloc[region_index]['mean_rain']
            if not np.isnan(rainfall):
                plot_data.append({
                    'Year': year,
                    'Month': month,
                    'Date': f"{year}-{month:02d}",
                    'Rainfall': rainfall
                })
    
    if not plot_data:
        st.warning(f"No valid data found for {region_name}")
        return None
    
    df = pd.DataFrame(plot_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Rainfall'],
        mode='lines+markers',
        name=f'Rainfall - {region_name}',
        line=dict(width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'Rainfall Time Series - {region_name}',
        xaxis_title='Date',
        yaxis_title='Rainfall (mm)',
        height=400,
        showlegend=True
    )
    
    return fig

def create_monthly_comparison_plot(data_dict, selected_years=None):
    """Create monthly rainfall comparison across years"""
    
    if not data_dict:
        st.warning("No data available for monthly comparison")
        return None
    
    # Prepare data
    monthly_data = {}
    for key, gdf in data_dict.items():
        year, month = key
        if selected_years is None or year in selected_years:
            mean_rainfall = gdf['mean_rain'].mean()
            if not np.isnan(mean_rainfall):
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append({
                    'Year': year,
                    'Rainfall': mean_rainfall
                })
    
    if not monthly_data:
        st.warning("No valid data for monthly comparison")
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Average Rainfall', 'Seasonal Pattern', 
                       'Year-over-Year Comparison', 'Rainfall Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Monthly averages
    months = sorted(monthly_data.keys())
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_means = []
    for month in months:
        rainfalls = [d['Rainfall'] for d in monthly_data[month]]
        monthly_means.append(np.mean(rainfalls))
    
    fig.add_trace(
        go.Bar(x=[month_names[m-1] for m in months], y=monthly_means,
               name='Monthly Average', marker_color='skyblue'),
        row=1, col=1
    )
    
    # Plot 2: Seasonal pattern (line plot)
    fig.add_trace(
        go.Scatter(x=[month_names[m-1] for m in months], y=monthly_means,
                  mode='lines+markers', name='Seasonal Pattern',
                  line=dict(width=3, color='darkblue')),
        row=1, col=2
    )
    
    # Plot 3: Year-over-year comparison for specific months
    peak_months = [6, 7, 8, 9]  # Jun-Sep
    for month in peak_months:
        if month in monthly_data:
            years = [d['Year'] for d in monthly_data[month]]
            rainfalls = [d['Rainfall'] for d in monthly_data[month]]
            fig.add_trace(
                go.Scatter(x=years, y=rainfalls, mode='lines+markers',
                          name=month_names[month-1]),
                row=2, col=1
            )
    
    # Plot 4: Distribution box plot
    all_rainfalls = []
    all_months = []
    for month in months:
        for d in monthly_data[month]:
            all_rainfalls.append(d['Rainfall'])
            all_months.append(month_names[month-1])
    
    fig.add_trace(
        go.Box(x=all_months, y=all_rainfalls, name='Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    
    return fig

# Main app layout
st.title("🌧️ Enhanced CHIRPS Rainfall Data Analysis")
st.markdown("*Advanced rainfall analysis with interactive visualizations and time series analysis*")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📥 Download Data", "🗺️ Map Visualization", "📈 Time Series Analysis"])

with tab1:
    st.header("Download Rainfall Data")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Data source selection
        data_source = st.radio(
            "📂 Select Data Source", 
            ["GADM Database", "Upload Custom Shapefile"],
            help="Choose between official GADM boundaries or upload your own shapefile"
        )
        st.session_state.data_source = data_source
        
        if data_source == "GADM Database":
            # Country and admin level selection
            country = st.selectbox("🌍 Select Country", list(COUNTRY_OPTIONS.keys()), 
                                  help="Select any African country")
            admin_level = st.selectbox("🏛️ Administrative Level", [0, 1, 2, 3, 4], 
                                      help="0=Country, 1=Regions, 2=Districts, 3=Communes, 4=Localities")
            
            # Set variables for GADM mode
            country_code = COUNTRY_OPTIONS[country]
            use_custom_shapefile = False
            
            # Update session state
            st.session_state.country = country
            st.session_state.country_code = country_code
            st.session_state.admin_level = admin_level
            st.session_state.use_custom_shapefile = False
            
        else:  # Upload Custom Shapefile
            st.markdown("**📁 Upload Shapefile Components**")
            
            # File uploaders
            shp_file = st.file_uploader("🗺️ Shapefile (.shp)", type=['shp'])
            shx_file = st.file_uploader("🔍 Shape Index (.shx)", type=['shx'])
            dbf_file = st.file_uploader("📊 Attribute Table (.dbf)", type=['dbf'])
            prj_file = st.file_uploader("🌐 Projection File (.prj)", type=['prj'])
            
            if shp_file and shx_file and dbf_file:
                use_custom_shapefile = True
                st.success("✅ Required files uploaded!")
            else:
                use_custom_shapefile = False
                st.info("📤 Please upload .shp, .shx, and .dbf files")
        
        # Date selection
        st.subheader("📅 Date Selection")
        current_year = datetime.now().year
        
        # Year multiselect
        available_years = list(range(1981, current_year))
        selected_years = st.multiselect(
            "Select Years",
            options=available_years,
            default=[current_year - 3, current_year - 2, current_year - 1],
            help="Select specific years for analysis"
        )
        
        # Month selection
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        
        selected_months = st.multiselect(
            "📊 Select Months", 
            options=list(month_names.keys()),
            format_func=lambda x: month_names[x],
            default=[6, 7, 8, 9],
            help="Select months for analysis"
        )
    
    # Main download section
    st.subheader("📥 Process & Download Rainfall Data")
    
    # Show selection summary
    if selected_years and selected_months:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Area**: {st.session_state.country}")
        with col2:
            st.info(f"**Years**: {len(selected_years)} selected")
        with col3:
            st.info(f"**Months**: {len(selected_months)} selected")
        
        total_datasets = len(selected_years) * len(selected_months)
        st.warning(f"⚠️ Will process {total_datasets} datasets. Large selections may take time.")
    
    # Process and download section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔄 Process Data & Generate Downloads", type="primary", use_container_width=True):
            if not selected_months:
                st.error("Please select at least one month for analysis.")
            elif not selected_years:
                st.error("Please select at least one year for analysis.")
            elif st.session_state.data_source == "Upload Custom Shapefile" and not use_custom_shapefile:
                st.error("Please upload all required shapefile components")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Load shapefile
                    if st.session_state.data_source == "GADM Database":
                        status_text.text("📥 Loading shapefile from GADM...")
                        gdf = download_shapefile_from_gadm(st.session_state.country_code, st.session_state.admin_level)
                    else:
                        status_text.text("📁 Processing uploaded shapefile...")
                        gdf, crs_source, projection_info = load_uploaded_shapefile(shp_file, shx_file, dbf_file, prj_file)
                    
                    st.session_state.gdf = gdf
                    progress_bar.progress(20)
                    
                    # Process CHIRPS data for each year/month combination
                    total_combinations = len(selected_years) * len(selected_months)
                    processed_count = 0
                    
                    st.session_state.processed_data = {}
                    
                    for year in selected_years:
                        for month in selected_months:
                            try:
                                status_text.text(f"🌧️ Processing {month_names[month]} {year}...")
                                processed_gdf = process_chirps_data(gdf, year, month)
                                st.session_state.processed_data[(year, month)] = processed_gdf
                                
                                processed_count += 1
                                progress = 20 + (60 * processed_count / total_combinations)
                                progress_bar.progress(int(progress))
                                
                            except Exception as e:
                                st.warning(f"Failed to process {month_names[month]} {year}: {str(e)}")
                                continue
                    
                    if st.session_state.processed_data:
                        # Generate download files
                        status_text.text("📦 Preparing download files...")
                        progress_bar.progress(90)
                        
                        # Prepare combined dataset for downloads
                        combined_data = []
                        for key, gdf_processed in st.session_state.processed_data.items():
                            year, month = key
                            df = pd.DataFrame(gdf_processed.drop(columns='geometry'))
                            df['year'] = year
                            df['month'] = month
                            df['month_name'] = month_names[month]
                            df['area_name'] = st.session_state.country
                            df['data_source'] = st.session_state.data_source
                            combined_data.append(df)
                        
                        if combined_data:
                            final_df = pd.concat(combined_data, ignore_index=True)
                            
                            # Reorder columns for better readability
                            column_order = ['area_name', 'data_source', 'year', 'month', 'month_name']
                            name_cols = [col for col in final_df.columns if col.startswith('NAME_')]
                            column_order.extend(sorted(name_cols))
                            column_order.extend(['mean_rain', 'valid_pixels'])
                            remaining_cols = [col for col in final_df.columns if col not in column_order]
                            column_order.extend(remaining_cols)
                            
                            available_cols = [col for col in column_order if col in final_df.columns]
                            final_df = final_df[available_cols]
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Data processing complete!")
                            
                            st.success(f"Successfully processed {len(st.session_state.processed_data)} datasets")
                            
                            # Download section
                            st.subheader("📥 Download Options")
                            
                            # Create three columns for different download formats
                            col_csv, col_excel, col_summary = st.columns(3)
                            
                            with col_csv:
                                st.markdown("**📄 CSV Format**")
                                st.caption("Clean tabular data for analysis")
                                
                                csv_data = final_df.to_csv(index=False)
                                filename_base = f"chirps_rainfall_{st.session_state.country_code}_{min(selected_years)}_{max(selected_years)}"
                                
                                st.download_button(
                                    label="📄 Download CSV",
                                    data=csv_data,
                                    file_name=f"{filename_base}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col_excel:
                                st.markdown("**📊 Excel Format**")
                                st.caption("Multi-sheet workbook with metadata")
                                
                                # Create Excel file in memory
                                excel_buffer = BytesIO()
                                
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    # Main data sheet
                                    final_df.to_excel(writer, sheet_name='Rainfall_Data', index=False)
                                    
                                    # Calculate and add summary statistics
                                    monthly_stats = []
                                    for month in selected_months:
                                        month_data = final_df[final_df['month'] == month]['mean_rain'].dropna()
                                        if len(month_data) > 0:
                                            monthly_stats.append({
                                                'Month': month_names[month],
                                                'Mean_Rainfall': month_data.mean(),
                                                'Std_Rainfall': month_data.std(),
                                                'Min_Rainfall': month_data.min(),
                                                'Max_Rainfall': month_data.max(),
                                                'Data_Points': len(month_data)
                                            })
                                    
                                    if monthly_stats:
                                        stats_df = pd.DataFrame(monthly_stats).round(2)
                                        stats_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
                                    
                                    # Metadata sheet
                                    metadata = pd.DataFrame({
                                        'Parameter': [
                                            'Area Name', 'Data Source', 'Country Code', 
                                            'Admin Level', 'Years Analyzed', 'Months Analyzed',
                                            'CHIRPS Version', 'Boundary Source', 'Generated On',
                                            'Total Records', 'Tool Version'
                                        ],
                                        'Value': [
                                            st.session_state.country,
                                            st.session_state.data_source,
                                            st.session_state.country_code,
                                            str(st.session_state.admin_level) if st.session_state.data_source == "GADM Database" else "Custom",
                                            f"{min(selected_years)}-{max(selected_years)}",
                                            ', '.join([month_names[m] for m in selected_months]),
                                            'CHIRPS v2.0',
                                            'GADM v4.1' if st.session_state.data_source == "GADM Database" else 'User Upload',
                                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            len(final_df),
                                            'Enhanced CHIRPS Tool v3.0'
                                        ]
                                    })
                                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                                
                                excel_buffer.seek(0)
                                
                                st.download_button(
                                    label="📊 Download Excel",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"{filename_base}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col_summary:
                                st.markdown("**📈 Summary Only**")
                                st.caption("Statistical summary without raw data")
                                
                                # Create summary-only file
                                summary_buffer = BytesIO()
                                
                                # Calculate comprehensive statistics
                                monthly_summary = []
                                yearly_summary = []
                                
                                # Monthly stats
                                for month in selected_months:
                                    month_data = final_df[final_df['month'] == month]['mean_rain'].dropna()
                                    if len(month_data) > 0:
                                        monthly_summary.append({
                                            'Month': month_names[month],
                                            'Mean': round(month_data.mean(), 2),
                                            'Std': round(month_data.std(), 2),
                                            'Min': round(month_data.min(), 2),
                                            'Max': round(month_data.max(), 2),
                                            'Q25': round(month_data.quantile(0.25), 2),
                                            'Q75': round(month_data.quantile(0.75), 2),
                                            'Count': len(month_data)
                                        })
                                
                                # Yearly stats
                                for year in selected_years:
                                    year_data = final_df[final_df['year'] == year]['mean_rain'].dropna()
                                    if len(year_data) > 0:
                                        yearly_summary.append({
                                            'Year': year,
                                            'Mean': round(year_data.mean(), 2),
                                            'Std': round(year_data.std(), 2),
                                            'Min': round(year_data.min(), 2),
                                            'Max': round(year_data.max(), 2),
                                            'Q25': round(year_data.quantile(0.25), 2),
                                            'Q75': round(year_data.quantile(0.75), 2),
                                            'Count': len(year_data)
                                        })
                                
                                with pd.ExcelWriter(summary_buffer, engine='openpyxl') as writer:
                                    if monthly_summary:
                                        pd.DataFrame(monthly_summary).to_excel(writer, sheet_name='Monthly_Stats', index=False)
                                    if yearly_summary:
                                        pd.DataFrame(yearly_summary).to_excel(writer, sheet_name='Yearly_Stats', index=False)
                                
                                summary_buffer.seek(0)
                                
                                st.download_button(
                                    label="📈 Download Summary",
                                    data=summary_buffer.getvalue(),
                                    file_name=f"summary_{filename_base}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            # Show data preview
                            with st.expander("👀 Preview Downloaded Data"):
                                st.dataframe(final_df.head(15), use_container_width=True)
                                st.caption(f"Showing first 15 rows of {len(final_df)} total records")
                            
                            # Additional download options for individual datasets
                            with st.expander("📋 Individual Dataset Downloads"):
                                st.caption("Download specific year/month combinations")
                                
                                individual_cols = st.columns(4)
                                for i, (key, gdf_processed) in enumerate(st.session_state.processed_data.items()):
                                    year, month = key
                                    
                                    df_individual = pd.DataFrame(gdf_processed.drop(columns='geometry'))
                                    df_individual['year'] = year
                                    df_individual['month'] = month
                                    df_individual['month_name'] = month_names[month]
                                    
                                    csv_individual = df_individual.to_csv(index=False)
                                    filename_individual = f"chirps_{st.session_state.country_code}_{year}_{month:02d}.csv"
                                    
                                    with individual_cols[i % 4]:
                                        st.download_button(
                                            label=f"{month_names[month][:3]} {year}",
                                            data=csv_individual,
                                            file_name=filename_individual,
                                            mime="text/csv",
                                            key=f"individual_{year}_{month}",
                                            use_container_width=True
                                        )
                        
                        else:
                            st.error("No valid data was processed.")
                    
                    else:
                        st.error("❌ No data could be processed for the selected time periods.")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
    
    with col2:
        if selected_years and selected_months:
            st.metric("📊 Total Datasets", len(selected_years) * len(selected_months))
            st.metric("📅 Years", len(selected_years))
            st.metric("🗓️ Months", len(selected_months))
        else:
            st.info("Select years and months to see processing estimates")

with tab2:
    st.header("Interactive Map Visualization")
    
    if not st.session_state.processed_data:
        st.info("🔄 Please process and download data in the 'Download Data' tab first")
    else:
        # Map controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_years = sorted(list(set([k[0] for k in st.session_state.processed_data.keys()])))
            selected_year = st.selectbox("Select Year", available_years)
        
        with col2:
            available_months = sorted(list(set([k[1] for k in st.session_state.processed_data.keys() if k[0] == selected_year])))
            selected_month = st.selectbox("Select Month", available_months, format_func=lambda x: month_names[x])
        
        with col3:
            color_scheme = st.selectbox("Color Scheme", ["Blues", "viridis", "plasma", "YlOrRd", "Reds"])
        
        # Display map
        if (selected_year, selected_month) in st.session_state.processed_data:
            gdf_to_plot = st.session_state.processed_data[(selected_year, selected_month)]
            title = f"{st.session_state.country} - {month_names[selected_month]} {selected_year}"
            
            # Create static matplotlib map
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Handle missing data
            if not gdf_to_plot['mean_rain'].isna().all():
                gdf_to_plot.plot(
                    column="mean_rain",
                    ax=ax,
                    legend=True,
                    cmap=color_scheme,
                    edgecolor="white",
                    linewidth=0.5,
                    legend_kwds={"shrink": 0.8, "label": "Rainfall (mm)"},
                    missing_kwds={"color": "lightgray", "label": "No data"}
                )
            else:
                gdf_to_plot.plot(ax=ax, color="lightgray", edgecolor="white")
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_axis_off()
            
            st.pyplot(fig)
            
            # Show statistics
            valid_data = gdf_to_plot['mean_rain'].dropna()
            if len(valid_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Rainfall", f"{valid_data.mean():.1f} mm")
                with col2:
                    st.metric("Max Rainfall", f"{valid_data.max():.1f} mm")
                with col3:
                    st.metric("Min Rainfall", f"{valid_data.min():.1f} mm")
                with col4:
                    st.metric("Std Deviation", f"{valid_data.std():.1f} mm")

with tab3:
    st.header("Time Series Analysis")
    
    if not st.session_state.processed_data:
        st.info("🔄 Please process and download data in the 'Download Data' tab first")
    else:
        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Regional Time Series", "Monthly Comparison", "Seasonal Analysis"],
            horizontal=True
        )
        
        if analysis_type == "Regional Time Series":
            st.subheader("📈 Regional Rainfall Time Series")
            
            # Region selection
            if st.session_state.gdf is not None:
                # Get region names for selection
                name_columns = [col for col in st.session_state.gdf.columns if col.startswith('NAME_')]
                
                if name_columns:
                    # Use the most detailed name column available
                    name_col = name_columns[-1]  # Usually the most detailed level
                    region_options = st.session_state.gdf[name_col].tolist()
                    selected_region = st.selectbox("Select Region", region_options)
                    region_index = region_options.index(selected_region)
                else:
                    # Fallback to index-based selection
                    region_index = st.selectbox("Select Region (by index)", range(len(st.session_state.gdf)))
                    selected_region = f"Region {region_index}"
                
                # Create time series plot
                fig = create_time_series_plot(st.session_state.processed_data, selected_region, region_index)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table for selected region
                    with st.expander("📋 View Raw Data"):
                        time_series_data = []
                        for key, gdf in st.session_state.processed_data.items():
                            year, month = key
                            if region_index < len(gdf):
                                rainfall = gdf.iloc[region_index]['mean_rain']
                                if not np.isnan(rainfall):
                                    time_series_data.append({
                                        'Year': year,
                                        'Month': month_names[month],
                                        'Rainfall (mm)': round(rainfall, 2)
                                    })
                        
                        if time_series_data:
                            df_display = pd.DataFrame(time_series_data)
                            st.dataframe(df_display, use_container_width=True)
        
        elif analysis_type == "Monthly Comparison":
            st.subheader("📊 Monthly Rainfall Comparison")
            
            # Year selection for comparison
            available_years = sorted(list(set([k[0] for k in st.session_state.processed_data.keys()])))
            selected_years_comparison = st.multiselect(
                "Select Years to Compare",
                available_years,
                default=available_years[-3:] if len(available_years) >= 3 else available_years
            )
            
            if selected_years_comparison:
                fig = create_monthly_comparison_plot(st.session_state.processed_data, selected_years_comparison)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Seasonal Analysis":
            st.subheader("🌦️ Seasonal Pattern Analysis")
            
            # Calculate seasonal patterns
            seasonal_data = {}
            seasons = {
                'Dry Season (Dec-Feb)': [12, 1, 2],
                'Pre-Wet Season (Mar-May)': [3, 4, 5],
                'Wet Season (Jun-Aug)': [6, 7, 8],
                'Post-Wet Season (Sep-Nov)': [9, 10, 11]
            }
            
            for season_name, season_months in seasons.items():
                seasonal_rainfall = []
                for key, gdf in st.session_state.processed_data.items():
                    year, month = key
                    if month in season_months:
                        mean_rain = gdf['mean_rain'].mean()
                        if not np.isnan(mean_rain):
                            seasonal_rainfall.append(mean_rain)
                
                if seasonal_rainfall:
                    seasonal_data[season_name] = {
                        'mean': np.mean(seasonal_rainfall),
                        'std': np.std(seasonal_rainfall),
                        'min': np.min(seasonal_rainfall),
                        'max': np.max(seasonal_rainfall)
                    }
            
            if seasonal_data:
                # Create seasonal comparison chart
                seasons_list = list(seasonal_data.keys())
                means = [seasonal_data[s]['mean'] for s in seasons_list]
                stds = [seasonal_data[s]['std'] for s in seasons_list]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=seasons_list,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    name='Mean Rainfall',
                    marker_color=['lightblue', 'yellow', 'darkblue', 'orange']
                ))
                
                fig.update_layout(
                    title='Seasonal Rainfall Patterns',
                    xaxis_title='Season',
                    yaxis_title='Mean Rainfall (mm)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show seasonal statistics table
                seasonal_df = pd.DataFrame(seasonal_data).T
                seasonal_df = seasonal_df.round(2)
                st.dataframe(seasonal_df, use_container_width=True)
                        file_name=f"{filename_base}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Create Excel file
                    excel_buffer = BytesIO()
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        final_df.to_excel(writer, sheet_name='Time_Series_Data', index=False)
                        
                        # Add summary statistics
                        summary_stats = []
                        for region_idx in range(len(st.session_state.gdf)):
                            region_data = final_df[final_df.index % len(st.session_state.gdf) == region_idx]
                            if not region_data['mean_rain'].isna().all():
                                stats = {
                                    'Region_Index': region_idx,
                                    'Mean_Rainfall': region_data['mean_rain'].mean(),
                                    'Std_Rainfall': region_data['mean_rain'].std(),
                                    'Min_Rainfall': region_data['mean_rain'].min(),
                                    'Max_Rainfall': region_data['mean_rain'].max(),
                                    'Data_Points': region_data['mean_rain'].count()
                                }
                                
                                # Add region names if available
                                name_cols = [col for col in region_data.columns if col.startswith('NAME_')]
                                for name_col in name_cols:
                                    if not region_data[name_col].isna().all():
                                        stats[name_col] = region_data[name_col].iloc[0]
                                
                                summary_stats.append(stats)
                        
                        if summary_stats:
                            summary_df = pd.DataFrame(summary_stats)
                            summary_df.to_excel(writer, sheet_name='Regional_Summary', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"{filename_base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                # Show preview
                with st.expander("👀 Preview Data"):
                    st.dataframe(final_df.head(20), use_container_width=True)
                    st.caption(f"Showing first 20 rows of {len(final_df)} total records")
        
        elif export_format == "Summary Statistics":
            st.subheader("📈 Summary Statistics Export")
            
            # Calculate comprehensive statistics
            stats_data = []
            
            # Monthly statistics
            monthly_stats = {}
            for key, gdf in st.session_state.processed_data.items():
                year, month = key
                valid_data = gdf['mean_rain'].dropna()
                
                if len(valid_data) > 0:
                    month_key = month_names[month]
                    if month_key not in monthly_stats:
                        monthly_stats[month_key] = []
                    monthly_stats[month_key].extend(valid_data.tolist())
            
            # Create monthly summary
            monthly_summary = []
            for month_name, data in monthly_stats.items():
                if data:
                    monthly_summary.append({
                        'Month': month_name,
                        'Mean_Rainfall': np.mean(data),
                        'Std_Rainfall': np.std(data),
                        'Min_Rainfall': np.min(data),
                        'Max_Rainfall': np.max(data),
                        'Q25_Rainfall': np.percentile(data, 25),
                        'Q75_Rainfall': np.percentile(data, 75),
                        'Data_Points': len(data)
                    })
            
            # Yearly statistics
            yearly_stats = {}
            for key, gdf in st.session_state.processed_data.items():
                year, month = key
                valid_data = gdf['mean_rain'].dropna()
                
                if len(valid_data) > 0:
                    if year not in yearly_stats:
                        yearly_stats[year] = []
                    yearly_stats[year].extend(valid_data.tolist())
            
            yearly_summary = []
            for year, data in yearly_stats.items():
                if data:
                    yearly_summary.append({
                        'Year': year,
                        'Mean_Rainfall': np.mean(data),
                        'Std_Rainfall': np.std(data),
                        'Min_Rainfall': np.min(data),
                        'Max_Rainfall': np.max(data),
                        'Q25_Rainfall': np.percentile(data, 25),
                        'Q75_Rainfall': np.percentile(data, 75),
                        'Data_Points': len(data)
                    })
            
            # Display and download
            if monthly_summary and yearly_summary:
                monthly_df = pd.DataFrame(monthly_summary).round(2)
                yearly_df = pd.DataFrame(yearly_summary).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Monthly Statistics")
                    st.dataframe(monthly_df, use_container_width=True)
                
                with col2:
                    st.subheader("Yearly Statistics")
                    st.dataframe(yearly_df, use_container_width=True)
                
                # Download statistics
                excel_buffer = BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    monthly_df.to_excel(writer, sheet_name='Monthly_Statistics', index=False)
                    yearly_df.to_excel(writer, sheet_name='Yearly_Statistics', index=False)
                    
                    # Metadata
                    metadata = pd.DataFrame({
                        'Parameter': ['Country', 'Data Source', 'Analysis Period', 'Generated On', 'Tool Version'],
                        'Value': [
                            st.session_state.country,
                            st.session_state.data_source,
                            f"{min(yearly_stats.keys())}-{max(yearly_stats.keys())}",
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Enhanced CHIRPS Tool v3.0'
                        ]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Download Summary Statistics",
                    data=excel_buffer.getvalue(),
                    file_name=f"chirps_summary_{st.session_state.country_code}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# Sidebar information (for all tabs)
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Tool Information")
    
    # Current analysis status
    st.metric("📦 Datasets Loaded", len(st.session_state.processed_data))
    
    if st.session_state.processed_data:
        years = sorted(list(set([k[0] for k in st.session_state.processed_data.keys()])))
        months = sorted(list(set([k[1] for k in st.session_state.processed_data.keys()])))
        
        st.write(f"**Year Range**: {years[0]}-{years[-1]}")
        st.write(f"**Months**: {', '.join([month_names[m][:3] for m in months])}")
        
        if st.button("🗑️ Clear All Data"):
            st.session_state.processed_data = {}
            st.session_state.gdf = None
            st.rerun()
    
    st.markdown("---")
    
    with st.expander("📋 Features Overview"):
        st.markdown("""
        **🆕 Enhanced Features:**
        - Interactive map visualization
        - Time series analysis by region
        - Monthly and seasonal comparisons
        - Multiple export formats
        - Batch data processing
        - Statistical summaries
        
        **📊 Analysis Types:**
        - Regional rainfall trends
        - Seasonal pattern analysis
        - Year-over-year comparisons
        - Distribution analysis
        
        **📥 Export Options:**
        - Individual datasets (CSV)
        - Combined time series (CSV/Excel)
        - Summary statistics (Excel)
        - Metadata included
        """)
    
    with st.expander("🔧 Performance Tips"):
        st.markdown("""
        - **Large datasets**: Process fewer years/months at once
        - **Memory**: Clear data between analyses
        - **Speed**: Use lower admin levels for faster processing
        - **Accuracy**: Include .prj files with custom shapefiles
        - **Downloads**: Excel files include more metadata
        """)

# Footer
st.markdown("---")
st.markdown("*Enhanced CHIRPS Rainfall Analysis Tool v3.0*")
st.markdown("**Contact**: Mohamed Sillah Kanu | Northwestern University Malaria Modeling Team")
st.markdown("*New features: Interactive visualizations, time series analysis, and enhanced data export options*")
