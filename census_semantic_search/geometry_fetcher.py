import geopandas as gpd
import pandas as pd
import json
from typing import Dict, List
import os
from .bigquery_client import CensusBigQueryClient

# Use BigQuery geo_us_boundaries to return proper geojson geometries for counties and ZCTAs

class GeometryFetcher:
    def __init__(self, cache_dir: str = "geometry_cache", project_id: str = None):
        self.cache_dir = cache_dir
        self.bq_client = CensusBigQueryClient(project_id)
        os.makedirs(cache_dir, exist_ok=True)
        
    async def fetch_geometries(self, geo_info: dict) -> gpd.GeoDataFrame:
        """
        Fetch geometries for different geographic levels from BigQuery geo_us_boundaries
        
        Args:
            geo_level: 'county', 'zcta', or 'tract'
            state_fips: State FIPS code (required for county and tract levels)
            zip_ranges: ZIP code ranges (required for ZCTA level)
        """
        geo_level = geo_info['geo_level']
        if geo_level == 'county':
            return await self._fetch_county_geometries(geo_info['state_fips'])
        elif geo_level == 'zcta':
            return await self._fetch_zcta_geometries(geo_info['zip_ranges'])
        elif geo_level == 'tract':
            return await self._fetch_tract_geometries(geo_info)
        else:
            raise ValueError(f"Unsupported geo_level: {geo_level}")
    
    async def _fetch_county_geometries(self, state_fips: str) -> gpd.GeoDataFrame:
        """
        Fetch county boundaries for a state from BigQuery geo_us_boundaries
        """
        cache_file = os.path.join(self.cache_dir, f"counties_{state_fips}.geojson")
        
        # Check cache first
        if os.path.exists(cache_file):
            print(f"📁 Loading counties from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print(f"📥 Fetching county boundaries from BigQuery geo_us_boundaries...")
        
        try:
            # Query BigQuery for county boundaries
            geo_filter = f"state_fips_code = '{state_fips}'"
            df = self.bq_client.query_geo_boundaries('counties', geo_filter)
            
            if df.empty:
                print(f"⚠️ No counties found for state FIPS: {state_fips}")
                return gpd.GeoDataFrame()
            
            # Convert to GeoDataFrame
            # The 'county_geom' column contains the geometry in WKT format
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['county_geom']))
            
            # Set CRS (BigQuery geometries are in WGS84)
            gdf.crs = 'EPSG:4326'
            
            # Save to cache
            gdf.to_file(cache_file, driver='GeoJSON')
            
            return gdf
            
        except Exception as e:
            print(f"Error fetching county boundaries from BigQuery: {e}")
            return gpd.GeoDataFrame()
    
    async def _fetch_zcta_geometries(self, zip_ranges: List[str]) -> gpd.GeoDataFrame:
        """
        Fetch ZCTA boundaries from BigQuery geo_us_boundaries
        """
        # Create a cache key based on the zip ranges
        cache_key = "_".join(sorted(zip_ranges))[:50]  # Truncate for filename safety
        cache_file = os.path.join(self.cache_dir, f"zcta_{cache_key}.geojson")
        
        # Check cache first
        if os.path.exists(cache_file):
            print(f"📁 Loading ZCTAs from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print(f"📥 Fetching ZCTA boundaries from BigQuery geo_us_boundaries...")
        
        try:
            # Build filter for ZIP code ranges
            zip_conditions = []
            for zip_range in zip_ranges:
                if '-' in zip_range:
                    start, end = zip_range.split('-')
                    start_padded = start.zfill(5)
                    end_padded = end.zfill(5)
                    zip_conditions.append(f"(zip_code >= '{start_padded}' AND zip_code <= '{end_padded}')")
                else:
                    zip_padded = zip_range.zfill(5)
                    zip_conditions.append(f"zip_code = '{zip_padded}'")
            
            geo_filter = " OR ".join(zip_conditions)
            df = self.bq_client.query_geo_boundaries('zip_codes', geo_filter)
            
            if df.empty:
                print(f"⚠️ No ZIP codes found for ranges: {zip_ranges}")
                return gpd.GeoDataFrame()
            
            # Convert to GeoDataFrame
            # The geometry column contains the geometry in WKT format
            geometry_col = 'zip_code_geom' if 'zip_code_geom' in df.columns else 'geometry'
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geometry_col]))
            
            # Create geo_id to match BigQuery ACS format
            gdf['geo_id'] = gdf['zip_code']
            
            # Set CRS (BigQuery geometries are in WGS84)
            gdf.crs = 'EPSG:4326'
            
            # Save to cache
            gdf.to_file(cache_file, driver='GeoJSON')
            
            return gdf
            
        except Exception as e:
            print(f"Error fetching ZCTA boundaries from BigQuery: {e}")
            return gpd.GeoDataFrame()
    
    async def _fetch_tract_geometries(self, geo_info: dict) -> gpd.GeoDataFrame:
        """
        Fetch census tract boundaries for a state from BigQuery geo_census_tracts
        """
        cache_file = os.path.join(self.cache_dir, f"tracts_{geo_info['state_fips']}.geojson")
        
        # Check cache first
        if os.path.exists(cache_file):
            print(f"📁 Loading census tracts from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print(f"📥 Fetching census tract boundaries from BigQuery geo_census_tracts...")
        
        try:
            state_name = geo_info['state_name']
            
            if not state_name:
                print(f"⚠️ Unknown state: {state_name}")
                return gpd.GeoDataFrame()
            
            # Construct the table name for geo_census_tracts
            table_name = f"census_tracts_{state_name}"
            
            # Query BigQuery for census tract boundaries from geo_census_tracts dataset
            query = f"""
            SELECT *
            FROM `bigquery-public-data.geo_census_tracts.{table_name}`
            """
            
            print(f"Executing BigQuery geometry query:\n{query}")
            df = self.bq_client.client.query(query).to_dataframe()
            
            # Debug: Print available columns to understand the data structure
            print(f"📊 Available columns in {table_name}: {list(df.columns)}")
            if not df.empty:
                # Check if we have geometry data and what type
                geom_cols = [col for col in df.columns if 'geom' in col.lower()]
                if geom_cols:
                    for col in geom_cols:
                        sample_geom = df[col].iloc[0] if not df[col].isna().all() else None
                        if sample_geom:
                            geom_type = "POINT" if "POINT" in str(sample_geom)[:50] else "POLYGON" if "POLYGON" in str(sample_geom)[:50] else "UNKNOWN"
                            print(f"   {col}: {geom_type} geometry")
            
            if df.empty:
                print(f"⚠️ No census tracts found for state: {state_name}")
                return gpd.GeoDataFrame()
            
            # Convert to GeoDataFrame
            # Look for polygon geometry columns first, then fallback to point/centroid
            geometry_col = None
            
            # Priority order: polygon boundaries first, then centroids
            polygon_cols = ['tract_geom', 'geometry', 'geom', 'polygon_geom', 'boundary_geom']
            centroid_cols = ['centroid_geom', 'center_geom', 'point_geom']
            
            # Try polygon columns first
            for col in polygon_cols:
                if col in df.columns:
                    geometry_col = col
                    print(f"📍 Using polygon geometry column: {col}")
                    break
            
            # If no polygon found, try centroid columns
            if not geometry_col:
                for col in centroid_cols:
                    if col in df.columns:
                        geometry_col = col
                        print(f"⚠️ Using centroid geometry column: {col} (polygons not available)")
                        break
            
            # Final fallback - use any column with 'geom' in the name
            if not geometry_col:
                geom_cols = [col for col in df.columns if 'geom' in col.lower()]
                if geom_cols:
                    geometry_col = geom_cols[0]
                    print(f"⚠️ Using fallback geometry column: {geometry_col}")
                else:
                    print(f"❌ No geometry column found in columns: {list(df.columns)}")
                    return gpd.GeoDataFrame()
            
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geometry_col]))
            
            # Create geo_id to match BigQuery ACS format
            # Census tract geo_id format: state_fips + county_fips + tract_code
            if 'geo_id' not in gdf.columns:
                if 'tract_fips_code' in gdf.columns:
                    gdf['geo_id'] = gdf['tract_fips_code']
                elif 'tract_ce' in gdf.columns and 'countyfp' in gdf.columns:
                    gdf['geo_id'] = geo_info['state_fips'] + gdf['countyfp'].astype(str) + gdf['tract_ce'].astype(str)
                elif 'tractce' in gdf.columns and 'countyfp' in gdf.columns:
                    gdf['geo_id'] = geo_info['state_fips'] + gdf['countyfp'].astype(str) + gdf['tractce'].astype(str)
            
            # Set CRS (BigQuery geometries are in WGS84)
            gdf.crs = 'EPSG:4326'
            
            # Save to cache
            gdf.to_file(cache_file, driver='GeoJSON')
            
            return gdf
            
        except Exception as e:
            print(f"Error fetching census tract boundaries from BigQuery: {e}")
            return gpd.GeoDataFrame()
    
    def merge_data_with_geometry(self, census_data: pd.DataFrame, 
                                geometries: gpd.GeoDataFrame) -> Dict:
        """
        Merge census data with geometries to create GeoJSON
        """
        # Check if geometries is empty (fetch failed)
        if geometries.empty or 'geo_id' not in geometries.columns:
            print("⚠️ No geometries available, returning data without spatial information")
            # Create a basic structure with just the data
            return {
                'type': 'FeatureCollection',
                'features': [],
                'metadata': {
                    'source': 'US Census Bureau ACS',
                    'geometry_source': 'None - geometry fetch failed',
                    'features_count': 0,
                    'data_rows': len(census_data)
                },
                'data': census_data.to_dict('records')
            }
        
        # Merge on geo_id
        merged = geometries.merge(
            census_data,
            on='geo_id',
            how='inner'
        )
        
        # Convert to GeoJSON
        geojson = json.loads(merged.to_json())
        
        # Add metadata
        geojson['metadata'] = {
            'source': 'US Census Bureau ACS',
            'geometry_source': 'BigQuery geo_us_boundaries',
            'features_count': len(merged)
        }
        
        return geojson