import geopandas as gpd
import pandas as pd
import httpx
import json
from typing import Dict, List
import os

# Use Census Tiger/Line files to return proper geojson geometries for counties
# Again, only works for county level now

class GeometryFetcher:
    def __init__(self, cache_dir: str = "geometry_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    async def fetch_county_geometries(self, state_fips: str) -> gpd.GeoDataFrame:
        """
        Fetch county boundaries for a state from Census TIGER/Line
        """
        cache_file = os.path.join(self.cache_dir, f"counties_{state_fips}.geojson")
        
        # Check cache first
        if os.path.exists(cache_file):
            print(f"📁 Loading counties from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        # Fetch from Census Bureau
        year = "2021"  # Use recent TIGER/Line files
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
        
        print(f"📥 Downloading county boundaries from Census TIGER/Line...")
        
        # Download and filter to specific state
        try:
            gdf = gpd.read_file(url)
            # Filter to specific state
            state_gdf = gdf[gdf['STATEFP'] == state_fips].copy()
            
            # Create geo_id to match BigQuery format
            state_gdf['geo_id'] = state_gdf['GEOID']
            
            # Save to cache
            state_gdf.to_file(cache_file, driver='GeoJSON')
            
            return state_gdf
            
        except Exception as e:
            print(f"Error fetching from Census Bureau: {e}")
            # Fallback to Nominatim or other service
            return await self._fetch_from_nominatim(state_fips)
    
    async def _fetch_from_nominatim(self, state_fips: str) -> gpd.GeoDataFrame:
        """Fallback to fetch from Nominatim"""
        # This is a simplified version - in production you'd want proper error handling
        print("⚠️ Census TIGER/Line failed, implement Nominatim fallback")
        return gpd.GeoDataFrame()
    
    def merge_data_with_geometry(self, census_data: pd.DataFrame, 
                                geometries: gpd.GeoDataFrame) -> Dict:
        """
        Merge census data with geometries to create GeoJSON
        """
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
            'geometry_source': 'TIGER/Line',
            'features_count': len(merged)
        }
        
        return geojson