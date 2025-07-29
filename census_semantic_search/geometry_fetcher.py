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
            geo_level: 'county' or 'zcta'
            state_fips: State FIPS code (required for county level)
            zip_ranges: ZIP code ranges (required for ZCTA level)
        """
        geo_level = geo_info['geo_level']
        if geo_level == 'county':
            return await self._fetch_county_geometries(geo_info['state_fips'])
        elif geo_level == 'zcta':
            return await self._fetch_zcta_geometries(geo_info['zip_ranges'])
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