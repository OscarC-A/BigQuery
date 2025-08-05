# Intialize BigQuery
# get_acs_tables_metadata: gets acs tables metadata (only needed once, then is saved), 
# get_table_columns: gets all columns from selected table
# query_acs_table: executes final SQL query to BigQuery

import os
from google.cloud import bigquery
from typing import List, Dict, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import json

class CensusBigQueryClient:
    def __init__(self, project_id: Optional[str] = None, boundaries_dir: str = "custom_boundaries"):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.client = bigquery.Client(project=self.project_id)

        # Select few ACS tables to choose from (most comprehensive and commonly used)
        self.acs_tables = {
            'county_2020_5yr': {
                'description': 'County-level demographic, economic, and housing data from 2020 ACS 5-year estimates',
                'geo_level': 'county',
                'year': 2020
            },
            'zcta_2020_5yr': {
                'description': 'Zip code tabulation area-level demographic, economic, and housing data from 2020 ACS 5-year estimates', 
                'geo_level': 'zcta',
                'year': 2020
            },
            'state_2021_1yr': {
                'description': 'State-level demographic, economic, and housing data from 2021 ACS 1-year estimates',
                'geo_level': 'state', 
                'year': 2021
            },
            'censustract_2020_5yr': {
                'description': 'Census tract-level demographic, economic, and housing data from 2020 ACS 5-year estimates',
                'geo_level': 'tract',
                'year': 2020
            }
        }
        
    def load_boundary(self, geojson_dir: str) -> Optional[Dict]:
        """Load a custom boundary from GeoJSON file"""
            
        # Try to find the boundary file
        boundary_file = os.path.join("", f"{geojson_dir}")
        if not os.path.exists(boundary_file):
            # Also check in examples folder
            boundary_file = os.path.join("custom_boundaries", f"{geojson_dir.lower()}")
            if not os.path.exists(boundary_file):
                print("could not find file")
                return None
                
        with open(boundary_file, 'r') as f:
            geojson_data = json.load(f)
            
        # Extract the geometry
        if geojson_data['type'] == 'FeatureCollection':
            # Get the first feature's geometry
            geometry = geojson_data['features'][0]['geometry']
        else:
            geometry = geojson_data
            
        boundary_data = {
            'geometry': geometry,
            'geojson': geojson_data
        }
        
        return boundary_data

    def get_acs_tables_metadata(self) -> pd.DataFrame:
        """Get metadata for all ACS tables"""
        query = """
        SELECT 
            table_name,
            REGEXP_EXTRACT(table_name, r'([A-Z][0-9]+[A-Z]?)') as table_code,
            REGEXP_EXTRACT(table_name, r'(state|county|zcta|tract)') as geo_level,
            REGEXP_EXTRACT(table_name, r'([0-9]{4})') as year
        FROM `bigquery-public-data.census_bureau_acs.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """
        return self.client.query(query).to_dataframe()
    
    def get_table_columns(self, table_name: str) -> pd.DataFrame:
        """Get column names for a specific table"""
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM `bigquery-public-data.census_bureau_acs.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        return self.client.query(query).to_dataframe()
    
    def query_acs_with_geometry(self, table_name: str, variables: List[str],
                                  geo_level: str, 
                                geojson_dir: str, state_name: str,) -> gpd.GeoDataFrame:
        """
        Query ACS data joined with geometry in a single query
        
        Args:
            table_name: ACS table name (e.g., 'county_2020_5yr')
            variables: List of ACS variables to select
            geo_filter: SQL filter for geography
            geo_level: Geographic level ('county', 'zcta', 'tract')
            geo_info: Additional geographic information from geo_resolver
            
        Returns:
            GeoDataFrame with both ACS data and geometries
        """
        # Prepare ACS columns
        acs_columns = ', '.join([f'acs.{var}' for var in variables])

        # Prevent geojson being read as having 4 layers
        # geo_columns = ', '.join([f'geo.{col}' for col in ['state_name', 'state_fips_code', 'county_fips_code', 'tract_ce', 'tract_name', 'lsad_name', 'functional_status', 'area_land_meters', 'area_water_meters', 'internal_point_lat', 'internal_point_lon']]) # 'internal_point_geo'

        # Create shapely geometry from the GeoJSON
        boundary_data = self.load_boundary(str(geojson_dir))
        if boundary_data is None:
            raise FileNotFoundError(f"Could not load boundary file: {geojson_dir}")
        wkt = shape(boundary_data['geometry']).wkt

        # Determine the geometry table and join conditions based on geo_level
        if geo_level == 'state':
            geo_table = 'bigquery-public-data.geo_us_boundaries.states'
            geo_id_field = 'geo_id'
            geom_field = 'state_geom'
            join_condition = 'acs.geo_id = geo.geo_id'

        # CAREFUL: BigQuery only has geom as of the 116th Congress
        elif geo_level =='congressional_district':
            geo_table = 'bigquery-public-data.geo_us_boundaries.congress_district_116'
            geo_id_field = 'geo_id'
            geom_field = 'district_geom'
            join_condition = 'acs.geo_id = geo.geo_id'

        elif geo_level == 'county':
            geo_table = 'bigquery-public-data.geo_us_boundaries.counties'
            geo_id_field = 'geo_id'
            geom_field = 'county_geom'
            join_condition = 'acs.geo_id = geo.geo_id'
            
        elif geo_level == 'zcta':
            geo_table = 'bigquery-public-data.geo_us_boundaries.zip_codes'
            geo_id_field = 'zip_code'
            geom_field = 'zip_code_geom'
            join_condition = 'acs.geo_id = geo.zip_code'

        elif geo_level == 'tract':
            # For tracts, we need state-specific tables
            state_name = state_name.replace(' ', '_')
            # state_name = "brute"
            if state_name == "brute":
                geo_table = "bigquery-public-data.geo_census_tracts.us_census_tracts_national"
            else:
                geo_table = f'bigquery-public-data.geo_census_tracts.census_tracts_{state_name}'
            geo_id_field = 'geo_id'
            geom_field = 'tract_geom'
            join_condition = 'acs.geo_id = geo.geo_id'
        else:
            raise ValueError(f"Unsupported geo_level: {geo_level}")
        
        query = f"""
        WITH boundary_filtered_geo AS (
            SELECT *
            FROM `{geo_table}`
            WHERE ST_INTERSECTS({geom_field}, ST_GEOGFROMTEXT('{wkt}'))
        ),
        acs_data AS (
            SELECT geo_id, {', '.join(variables)}
            FROM `bigquery-public-data.census_bureau_acs.{table_name}`
            WHERE geo_id IN (
                SELECT {geo_id_field} 
                FROM boundary_filtered_geo
            )
        )
        SELECT 
            acs.geo_id,
            {acs_columns},
            ST_ASTEXT(geo.{geom_field}) as geometry_wkt,
            geo.*
        FROM acs_data acs
        INNER JOIN boundary_filtered_geo geo
        ON {join_condition}
        """
        
        print(f"Executing combined ACS + Geometry query:\n{query[:500]}...")
        
        # Execute query
        df = self.client.query(query).to_dataframe()
        
        if df.empty:
            print("Warning: Query returned no results")
            return gpd.GeoDataFrame()
        
        # Fix data types for numeric columns

        # df = self._fix_numeric_columns(df, variables)
        
        # Convert to GeoDataFrame
        # Parse WKT geometry
        from shapely import wkt
        df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # Drop the WKT column
        gdf = gdf.drop(columns=['geometry_wkt'])
        
        return gdf
    
    def _fix_numeric_columns(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """
        Fix data types for numeric columns that may have been returned as strings.
        Common numeric column patterns in ACS data that should be integers or floats.
        """
        for col in variables:
            if col in df.columns and col != "geo_id":
                    try:
                        # Convert to numeric, handling sentinel values and errors
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Replace sentinel values with NaN
                        sentinel_values = [-666666666, -999999999, -888888888, -777777777]
                        df[col] = df[col].replace(sentinel_values, pd.NA)
                        
                        # If column name suggests it should be integer, convert to Int64 (nullable integer)
                        if any(pattern in col.lower() for pattern in ['pop', 'units', 'households', 'families']):
                            df[col] = df[col].astype('Int64')
                        else:
                            # Keep as float for income, age, etc.
                            df[col] = df[col].astype('Float64')
                            
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to numeric: {e}")
                        
        return df