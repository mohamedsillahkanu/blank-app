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
from datetime import datetime
import time

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

# Main app layout
st.title("🌧️ CHIRPS Rainfall Data Analysis and Map Generation")
st.markdown("*Analyze seasonal rainfall patterns for malaria intervention planning*")

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
        
        # Show admin level description
        level_descriptions = {
            0: "Country boundary",
            1: "First-level admin (regions/states)",
            2: "Second-level admin (districts/provinces)", 
            3: "Third-level admin (communes/counties)",
            4: "Fourth-level admin (localities/wards)"
        }
        st.caption(f"Level {admin_level}: {level_descriptions.get(admin_level, 'Administrative division')}")
        
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
        st.caption("Upload all required files for your shapefile")
        
        # File uploaders - Required files
        shp_file = st.file_uploader("🗺️ Shapefile (.shp)", type=['shp'], help="Main geometry file (required)")
        shx_file = st.file_uploader("🔍 Shape Index (.shx)", type=['shx'], help="Spatial index file (required)")
        dbf_file = st.file_uploader("📊 Attribute Table (.dbf)", type=['dbf'], help="Attribute data file (required)")
        
        # Optional projection file
        prj_file = st.file_uploader("🌐 Projection File (.prj)", type=['prj'], 
                                   help="Coordinate system definition (optional but recommended)")
        
        # Check if required files are uploaded
        if shp_file and shx_file and dbf_file:
            st.success("✅ Required files uploaded successfully!")
            use_custom_shapefile = True
            
            # Show file details
            with st.expander("📋 File Details"):
                st.write(f"**SHP file**: {shp_file.name} ({shp_file.size:,} bytes)")
                st.write(f"**SHX file**: {shx_file.name} ({shx_file.size:,} bytes)")
                st.write(f"**DBF file**: {dbf_file.name} ({dbf_file.size:,} bytes)")
                if prj_file:
                    st.write(f"**PRJ file**: {prj_file.name} ({prj_file.size:,} bytes) ✅")
                    st.success("Projection file included - coordinate system will be automatically detected!")
                else:
                    st.warning("No projection file - will assume WGS84 coordinate system")
        else:
            use_custom_shapefile = False
            st.info("📤 Please upload .shp, .shx, and .dbf files to proceed")
            
            # Show upload requirements
            with st.expander("ℹ️ Shapefile Upload Requirements"):
                st.markdown("""
                **Required Files:**
                - **.shp**: Contains the geometry data
                - **.shx**: Spatial index for quick access
                - **.dbf**: Attribute table with feature data
                
                **Optional Files:**
                - **.prj**: Coordinate system definition (highly recommended)
                
                **Important Notes:**
                - All files must have the same base name
                - Maximum file size: 200MB per file
                - If no .prj file is provided, WGS84 coordinate system will be assumed
                - Including a .prj file prevents coordinate system errors
                - Ensure your shapefile contains administrative boundaries or areas of interest
                
                **Why include a .prj file?**
                - Defines the exact coordinate system of your data
                - Prevents projection errors during analysis
                - Ensures accurate spatial calculations
                - Required for proper alignment with satellite data
                """)
        
        # Set variables for custom mode
        country = "Custom Area"
        country_code = "CUSTOM"
        admin_level = "Custom"
        
        # Update session state
        st.session_state.country = country
        st.session_state.country_code = country_code
        st.session_state.admin_level = admin_level
        st.session_state.use_custom_shapefile = use_custom_shapefile

    # Date selection with validation
    current_year = datetime.now().year
    year = st.selectbox("📅 Select Year", range(1981, current_year + 1), index=current_year-1981-2)

    # Month selection with names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    selected_months = st.multiselect(
        "📊 Select Months", 
        options=list(month_names.keys()),
        format_func=lambda x: month_names[x],
        default=[6, 7, 8, 9],  # Peak malaria season
        help="Select months for analysis (typically Jun-Sep for West Africa)"
    )

    # Analysis options
    st.subheader("Display Options")
    show_statistics = st.checkbox("📈 Show Statistics", value=True)
    color_scheme = st.selectbox("🎨 Color Scheme", ["Blues", "viridis", "plasma", "YlOrRd"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔄 Generate Analysis", type="primary", use_container_width=True):
        if not selected_months:
            st.error("Please select at least one month for analysis.")
        elif st.session_state.data_source == "Upload Custom Shapefile" and not st.session_state.use_custom_shapefile:
            st.error("Please upload all required shapefile components (.shp, .shx, .dbf)")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load shapefile (GADM or uploaded)
                if st.session_state.data_source == "GADM Database":
                    status_text.text("🔍 Checking GADM database...")
                    progress_bar.progress(10)
                    
                    status_text.text(f"📥 Downloading {st.session_state.country} shapefile from GADM...")
                    progress_bar.progress(20)
                    
                    gdf = download_shapefile_from_gadm(st.session_state.country_code, st.session_state.admin_level)
                    st.success(f"✅ {st.session_state.country} Admin Level {st.session_state.admin_level} shapefile loaded ({len(gdf)} features)")
                    
                else:  # Custom shapefile
                    status_text.text("📁 Processing uploaded shapefile...")
                    progress_bar.progress(10)
                    
                    gdf, crs_source, projection_info = load_uploaded_shapefile(shp_file, shx_file, dbf_file, prj_file)
                    progress_bar.progress(20)
                    st.success(f"✅ Custom shapefile loaded ({len(gdf)} features)")
                    
                    # Show coordinate system info
                    if gdf.crs:
                        st.info(f"📍 Coordinate System: {gdf.crs} ({crs_source})")
                        if projection_info and prj_file:
                            with st.expander("🔍 Projection Details"):
                                st.code(projection_info, language="text")
                    
                    # Show attribute columns
                    attribute_cols = [col for col in gdf.columns if col != 'geometry']
                    if attribute_cols:
                        st.info(f"📊 Available attributes: {', '.join(attribute_cols[:5])}{'...' if len(attribute_cols) > 5 else ''}")
                
                # Show some basic info about the shapefile
                if hasattr(gdf, 'NAME_1') and (st.session_state.data_source == "GADM Database" and st.session_state.admin_level >= 1):
                    region_names = gdf['NAME_1'].unique()[:5]
                    st.info(f"📋 Contains regions: {', '.join(region_names)}{'...' if len(gdf['NAME_1'].unique()) > 5 else ''}")
                elif hasattr(gdf, 'NAME_0'):
                    st.info(f"📋 Country: {gdf['NAME_0'].iloc[0]}")
                elif st.session_state.data_source == "Upload Custom Shapefile":
                    # For custom shapefiles, show basic geometry info
                    geom_types = gdf.geometry.type.unique()
                    st.info(f"📋 Geometry types: {', '.join(geom_types)}")
                
                # Test CHIRPS data availability before processing all months (only for first month)
                status_text.text("🔍 Testing CHIRPS data availability...")
                test_month = selected_months[0]
                test_url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/chirps-v2.0.{year}.{test_month:02d}.tif.gz"
                
                try:
                    test_response = requests.head(test_url, timeout=30)
                    if test_response.status_code != 200:
                        st.error(f"❌ CHIRPS data not available for {year}-{test_month:02d}")
                        st.error("Try selecting a different year or check if the data exists for your selected period.")
                        st.stop()
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Cannot access CHIRPS server: {str(e)}")
                    st.stop()
                
                # Step 2: Process CHIRPS data
                status_text.text("🌧️ Processing CHIRPS rainfall data...")
                processed_data = []
                
                for i, month in enumerate(selected_months):
                    progress = 20 + (60 * (i + 1) / len(selected_months))
                    progress_bar.progress(int(progress))
                    status_text.text(f"🌧️ Processing {month_names[month]} {year}...")
                    
                    try:
                        processed_gdf = process_chirps_data(gdf, year, month)
                        processed_data.append((month, processed_gdf))
                    except Exception as e:
                        st.error(f"❌ Error processing {month_names[month]}: {str(e)}")
                        st.write(f"Debug: Error type: {type(e).__name__}")
                        continue

                if not processed_data:
                    st.error("❌ No data could be processed for the selected months.")
                    st.markdown("""
                    **Possible solutions:**
                    - Try a different year (1981-2023 are most reliable)
                    - Check if the selected months have data available
                    - Ensure internet connection is stable
                    - Try fewer months at once
                    - If using custom shapefile, ensure it has proper coordinate system (.prj file)
                    """)
                    st.stop()

                # Step 3: Generate visualizations
                status_text.text("📊 Generating maps...")
                progress_bar.progress(90)
                
                # Create subplots
                num_months = len(processed_data)
                if num_months == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    axes = [ax]
                else:
                    num_cols = min(3, num_months)
                    num_rows = math.ceil(num_months / num_cols)
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
                    if num_months > 1:
                        axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

                # Plot each month
                all_values = []
                for month, processed_gdf in processed_data:
                    valid_values = processed_gdf['mean_rain'].dropna()
                    all_values.extend(valid_values)

                if all_values:
                    vmin, vmax = np.percentile(all_values, [5, 95])  # Use percentiles for better visualization
                else:
                    vmin, vmax = 0, 100

                for i, (month, processed_gdf) in enumerate(processed_data):
                    ax = axes[i] if len(axes) > 1 else axes[0]
                    
                    # Handle missing data
                    plot_data = processed_gdf.copy()
                    if plot_data['mean_rain'].isna().all():
                        ax.text(0.5, 0.5, f'No data available\nfor {month_names[month]} {year}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{st.session_state.country} - {month_names[month]} {year}")
                        ax.set_axis_off()
                        continue
                    
                    # Create the plot
                    plot_data.plot(
                        column="mean_rain",
                        ax=ax,
                        legend=True,
                        cmap=color_scheme,
                        edgecolor="white",
                        linewidth=0.5,
                        legend_kwds={"shrink": 0.6, "label": "Rainfall (mm)"},
                        vmin=vmin,
                        vmax=vmax,
                        missing_kwds={"color": "lightgray", "label": "No data"}
                    )
                    ax.set_title(f"{st.session_state.country} - {month_names[month]} {year}", fontweight='bold')
                    ax.set_axis_off()

                # Remove empty subplots
                if len(processed_data) < len(axes):
                    for j in range(len(processed_data), len(axes)):
                        fig.delaxes(axes[j])

                plt.tight_layout()
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.pyplot(fig)

                # Show statistics if requested
                if show_statistics:
                    st.subheader("📊 Rainfall Statistics")
                    
                    stats_data = []
                    for month, processed_gdf in processed_data:
                        valid_data = processed_gdf['mean_rain'].dropna()
                        if len(valid_data) > 0:
                            stats_data.append({
                                'Month': month_names[month],
                                'Mean (mm)': f"{valid_data.mean():.1f}",
                                'Std (mm)': f"{valid_data.std():.1f}",
                                'Min (mm)': f"{valid_data.min():.1f}",
                                'Max (mm)': f"{valid_data.max():.1f}",
                                'Valid Areas': len(valid_data)
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)

                # Data download section
                st.subheader("📥 Download Data")
                
                # Prepare combined dataset for download
                combined_data = []
                for month, processed_gdf in processed_data:
                    # Convert to regular DataFrame (drop geometry)
                    df = pd.DataFrame(processed_gdf.drop(columns='geometry'))
                    df['month'] = month
                    df['month_name'] = month_names[month]
                    df['year'] = year
                    df['area_name'] = st.session_state.country
                    df['data_source'] = st.session_state.data_source
                    if st.session_state.data_source == "GADM Database":
                        df['country_code'] = st.session_state.country_code
                        df['admin_level'] = st.session_state.admin_level
                    else:
                        df['country_code'] = "CUSTOM"
                        df['admin_level'] = "Custom"
                        # Add projection info for custom shapefiles
                        if prj_file:
                            df['projection_source'] = "PRJ file provided"
                        else:
                            df['projection_source'] = "Assumed WGS84"
                    
                    # Reorder columns for better readability
                    column_order = ['area_name', 'data_source', 'year', 'month', 'month_name']
                    
                    if st.session_state.data_source == "GADM Database":
                        column_order.extend(['country_code', 'admin_level'])
                    else:
                        column_order.extend(['country_code', 'admin_level', 'projection_source'])
                    
                    # Add name columns if they exist (for GADM data)
                    name_cols = [col for col in df.columns if col.startswith('NAME_')]
                    column_order.extend(sorted(name_cols))
                    
                    # Add other standard GADM columns
                    standard_cols = ['GID_0', 'GID_1', 'GID_2', 'GID_3', 'GID_4', 'CC_0', 'CC_1', 'HASC_1', 'HASC_2', 'TYPE_1', 'TYPE_2', 'ENGTYPE_1', 'ENGTYPE_2']
                    for col in standard_cols:
                        if col in df.columns and col not in column_order:
                            column_order.append(col)
                    
                    # Add rainfall data columns
                    column_order.extend(['mean_rain', 'valid_pixels'])
                    
                    # Add any remaining columns (for custom shapefiles)
                    remaining_cols = [col for col in df.columns if col not in column_order]
                    column_order.extend(remaining_cols)
                    
                    # Reorder DataFrame
                    available_cols = [col for col in column_order if col in df.columns]
                    df = df[available_cols]
                    
                    combined_data.append(df)
                
                if combined_data:
                    # Combine all months into one dataset
                    final_df = pd.concat(combined_data, ignore_index=True)
                    
                    # Create download buttons
                    col_csv, col_excel = st.columns(2)
                    
                    with col_csv:
                        csv_data = final_df.to_csv(index=False)
                        filename_base = f"chirps_rainfall_{st.session_state.country_code}_{year}"
                        if st.session_state.data_source == "Upload Custom Shapefile":
                            filename_base = f"chirps_rainfall_custom_{year}"
                        elif st.session_state.data_source == "GADM Database":
                            filename_base = f"chirps_rainfall_{st.session_state.country_code}_{year}_admin{st.session_state.admin_level}"
                        
                        st.download_button(
                            label="📄 Download as CSV",
                            data=csv_data,
                            file_name=f"{filename_base}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_excel:
                        # Create Excel file in memory
                        excel_buffer = BytesIO()
                        
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            # Main data sheet
                            final_df.to_excel(writer, sheet_name='Rainfall_Data', index=False)
                            
                            # Summary statistics sheet
                            if stats_data:
                                stats_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
                            
                            # Metadata sheet
                            metadata_values = [
                                st.session_state.country, st.session_state.data_source, st.session_state.country_code, 
                                str(st.session_state.admin_level) if st.session_state.data_source == "GADM Database" else "Custom",
                                year,
                                ', '.join([month_names[m] for m in selected_months]),
                                'CHIRPS v2.0',
                                'GADM v4.1' if st.session_state.data_source == "GADM Database" else 'User Upload',
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                len(final_df),
                                'CHIRPS Rainfall Analysis Tool v2.1'
                            ]
                            
                            # Add projection info for custom uploads
                            if st.session_state.data_source == "Upload Custom Shapefile":
                                metadata_values.extend([
                                    crs_source if 'crs_source' in locals() else "Unknown",
                                    "Yes" if prj_file else "No"
                                ])
                            
                            metadata_params = [
                                'Area Name', 'Data Source', 'Country Code', 'Admin Level', 'Year', 
                                'Months Analyzed', 'CHIRPS Source', 'Boundary Source', 'Generated On',
                                'Total Records', 'Tool Version'
                            ]
                            
                            if st.session_state.data_source == "Upload Custom Shapefile":
                                metadata_params.extend(['Coordinate System', 'PRJ File Included'])
                            
                            metadata = pd.DataFrame({
                                'Parameter': metadata_params,
                                'Value': metadata_values
                            })
                            metadata.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="📊 Download as Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"{filename_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    # Show data preview
                    with st.expander("👀 Preview Downloaded Data"):
                        st.dataframe(final_df.head(10), use_container_width=True)
                        st.caption(f"Showing first 10 rows of {len(final_df)} total records")
                    
                    # Data description
                    with st.expander("📖 Data Dictionary"):
                        st.markdown("""
                        **Core Columns:**
                        - `area_name`: Name of analysis area
                        - `data_source`: "GADM Database" or "Upload Custom Shapefile"
                        - `year`: Analysis year
                        - `month`: Month number (1-12)
                        - `month_name`: Month name
                        
                        **GADM Database Columns (when applicable):**
                        - `country_code`: ISO country code
                        - `admin_level`: Administrative level (0-4)
                        - `NAME_X`: Administrative unit names at level X
                        - `GID_X`: Global administrative unit IDs
                        - `CC_X`: Country/region codes
                        - `TYPE_X`: Administrative unit types
                        - `HASC_X`: Hierarchical administrative subdivision codes
                        
                        **Custom Shapefile Columns:**
                        - All original attribute columns from uploaded shapefile
                        - `projection_source`: How coordinate system was determined
                        - Variable names depend on the uploaded data
                        
                        **Rainfall Columns:**
                        - `mean_rain`: Average rainfall in mm for the administrative unit
                        - `valid_pixels`: Number of valid satellite pixels used in calculation
                        
                        **Projection Information (Custom Shapefiles):**
                        - Coordinate system detection improved with .prj file support
                        - Better accuracy when .prj file is included
                        """)

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.write(f"Data Source: {st.session_state.data_source}")
                    if st.session_state.data_source == "GADM Database":
                        st.write(f"Country: {st.session_state.country}")
                        st.write(f"Country Code: {st.session_state.country_code}")
                        st.write(f"Admin Level: {st.session_state.admin_level}")
                    else:
                        st.write(f"Custom Area: {st.session_state.country}")
                        st.write(f"Files Uploaded: {st.session_state.use_custom_shapefile}")
                        if 'prj_file' in locals():
                            st.write(f"PRJ File: {'Yes' if prj_file else 'No'}")
                    st.write(f"Year: {year}")
                    st.write(f"Months: {selected_months}")

with col2:
    st.subheader("ℹ️ About This Tool")
    st.markdown("""
    **CHIRPS Rainfall Analysis Tool**
    
    This tool analyzes satellite-derived rainfall data for malaria intervention planning in Africa.
    
    **Data Sources:**
    - **CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data
    - **GADM**: Global Administrative Areas database (official country boundaries)
    
    **Coverage:**
    - All African countries supported
    - Administrative levels 0-4 (where available)
    - Monthly rainfall data from 1981-present
    - Custom shapefile upload capability with .prj support
    
    **Use Cases:**
    - Seasonal Malaria Chemoprevention planning
    - Intervention targeting
    - Climate-informed public health decisions
    
    **Technical Notes:**
    - Data resolution: ~5km
    - Temporal coverage: 1981-present
    - Update frequency: Monthly (~2 month delay)
    - Improved projection handling with .prj file support
    """)
    
    with st.expander("📋 Usage Tips"):
        st.markdown("""
        - **Peak season**: Jun-Sep for West Africa
        - **Admin levels**: Use level 2 for districts
        - **Color schemes**: Blues for rainfall, viridis for diversity
        - **Performance**: Fewer months = faster processing
        - **Large countries**: Use higher admin levels for better detail
        - **Downloads**: CSV for analysis, Excel for reports with metadata
        - **Projection files**: Always include .prj files with custom shapefiles for best results
        """)
    
    with st.expander("🌐 Projection File (.prj) Benefits"):
        st.markdown("""
        **Why upload a .prj file?**
        - **Accuracy**: Ensures correct coordinate system interpretation
        - **No assumptions**: Prevents automatic WGS84 assumption
        - **Error prevention**: Reduces projection-related analysis errors
        - **Professional results**: Maintains spatial data integrity
        
        **What happens without .prj?**
        - Tool assumes WGS84 coordinate system
        - May cause spatial misalignment
        - Potential analysis inaccuracies
        - Warning messages displayed
        
        **Supported projection formats:**
        - Well-Known Text (WKT) format
        - ESRI projection strings
        - Most GIS software exports compatible .prj files
        """)
    
    with st.expander("💾 Download Features"):
        st.markdown("""
        **CSV Download:**
        - Clean tabular data ready for analysis
        - All administrative attributes included
        - Compatible with R, Python, SPSS, etc.
        - Includes projection information for custom uploads
        
        **Excel Download:**
        - Multiple sheets: Data, Statistics, Metadata
        - Professional format for reports
        - Includes analysis parameters and timestamps
        - Enhanced metadata for custom shapefiles
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        - **Slow loading**: Try fewer months or lower admin level
        - **No data**: Check if recent months (data has ~2 month delay)
        - **Error messages**: Often due to network connectivity
        - **Missing admin levels**: Not all countries have all 5 levels
        - **Large downloads**: Be patient with countries like Nigeria, DRC
        - **Download issues**: Try smaller date ranges if files are too large
        - **Projection errors**: Include .prj file with custom shapefiles
        - **Coordinate system warnings**: Upload .prj file to eliminate assumptions
        """)

    # Show current system info - SAFE VERSION
    with st.expander("📊 System Information"):
        st.write(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Only show country count when using GADM mode to avoid NameError
        if st.session_state.data_source == "GADM Database":
            st.write(f"Available countries: {len(COUNTRY_OPTIONS)}")
        else:
            st.write("Mode: Custom Shapefile Upload")
            st.write("Projection support: .prj files supported")
        st.write("Supported formats: CSV, Excel (.xlsx)")
        st.write("Max recommended: 12 months, Admin level ≤3 for large countries")
        st.write("Tool version: v2.1 (with .prj support)")

# Footer
st.markdown("---")
st.markdown("*Built for malaria researchers and public health professionals*")
st.markdown("**Contact**: Mohamed Sillah Kanu | Informarics Consultancy Firm- Sierra Leone (ICF-SL)")
