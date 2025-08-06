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
    
    def get_available_acs_tables(self, geo_level: str, years: List[int] = None) -> List[Dict]:
        """
        Dynamically query BigQuery for available ACS tables based on geo_level and years
        
        Args:
            geo_level: Geographic level (county, zcta, tract, state)
            years: List of years to filter for (optional)
        
        Returns:
            List of table metadata dictionaries
        """
        year_filter = ""
        if years:
            year_list = ', '.join([f"'{year}'" for year in years])
            year_filter = f"AND REGEXP_EXTRACT(table_name, r'([0-9]{{4}})') IN ({year_list})"
        
        query = f"""
        SELECT 
            table_name,
            REGEXP_EXTRACT(table_name, r'([A-Z][0-9]+[A-Z]?)') as table_code,
            REGEXP_EXTRACT(table_name, r'(state|county|zcta|tract|congressionaldistrict)') as geo_level,
            REGEXP_EXTRACT(table_name, r'([0-9]{{4}})') as year,
            REGEXP_EXTRACT(table_name, r'([0-9]+yr)') as survey_type
        FROM `bigquery-public-data.census_bureau_acs.INFORMATION_SCHEMA.TABLES`
        WHERE REGEXP_CONTAINS(table_name, r'{geo_level}')
        {year_filter}
        ORDER BY year DESC, survey_type DESC
        """
        
        df = self.client.query(query).to_dataframe()
        
        tables = []
        for _, row in df.iterrows():
            tables.append({
                'table_name': row['table_name'],
                'table_code': row['table_code'],
                'geo_level': row['geo_level'],
                'year': int(row['year']) if row['year'] else None,
                'survey_type': row['survey_type']
            })
        
        return tables

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
    
    def _get_essential_geo_columns(self, geo_level: str) -> List[str]:
        """Get only essential geometry columns based on geographic level"""
        
        # Common columns across all geometry tables
        if geo_level == 'state':
            return ['state_name', 'state_fips_code', 'int_point_lat', 'int_point_lon']
            
        elif geo_level == 'congressional_district':
            return ['district_fips_code', 'state_fips_code', 'int_point_lat', 'int_point_lon']
            
        elif geo_level == 'county':
            return ['county_name', 'county_fips_code', 'state_fips_code', 'int_point_lat', 'int_point_lon']
            
        elif geo_level == 'zcta':
            return ['city', 'county', 'state_name', 'state_fips_code', 'internal_point_lat', 'internal_point_lon']
            
        elif geo_level == 'tract':
            return ['tract_name', 'state_fips_code', 'county_fips_code', 'internal_point_lat', 'internal_point_lon']
            
        else:
            # Fallback to minimal columns
            raise(ValueError("Unknown geo level"))

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

    def _geo_table_join(self, geo_level: str, state_name:str):
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
        
        return geo_table, geo_id_field, geom_field, join_condition

    def query_acs_with_geometry(self, table_name: str, variables: List[str],
                                geo_level: str, geojson_dir: str, 
                                state_name: str,) -> gpd.GeoDataFrame:
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
        
        geo_table, geo_id_field, geom_field, join_condition = self._geo_table_join(geo_level, state_name)
        
        # Pull only essential columns from geo table
        essential_geo_cols = self._get_essential_geo_columns(geo_level)

        # Build geo columns string (excluding geometry_wkt which we handle separately)
        geo_columns_list = [f'geo.{col}' for col in essential_geo_cols if col not in ['geo_id', 'geometry_wkt']]
        geo_columns = ', '.join(geo_columns_list) if geo_columns_list else ''
        
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
            {geo_columns}
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
    
    def query_acs_multi_year(self, table_names: List[str], variables: List[str],
                         geo_level: str, geojson_dir: str, state_name: str) -> gpd.GeoDataFrame:
        """
        Query multiple ACS tables (e.g., different years) and combine results with pivoted columns
        
        Args:
            table_names: List of ACS table names to query
            variables: List of variables to select from each table
            geo_level: Geographic level
            geojson_dir: Path to GeoJSON boundary file
            state_name: State name for filtering
        
        Returns:
            Combined GeoDataFrame with year-suffixed columns (e.g., median_income_2010, median_income_2020)
        """
        # Load boundary once
        boundary_data = self.load_boundary(str(geojson_dir))
        if boundary_data is None:
            raise FileNotFoundError(f"Could not load boundary file: {geojson_dir}")
        wkt = shape(boundary_data['geometry']).wkt
        
        # Determine geometry table configuration based on geo_level
        geo_table, geo_id_field, geom_field, join_condition = self._geo_table_join(geo_level, state_name)
        
        # Extract years from table names and sort them
        import re
        table_years = []
        for table_name in table_names:
            year_match = re.search(r'(\d{4})', table_name)
            if year_match:
                table_years.append((table_name, year_match.group(1)))
        
        # Build CTEs for each year's data
        year_cte_definitions = []  # The actual CTE SQL definitions
        year_cte_names = []  # Just the CTE names for referencing
        for table_name, year in table_years:
            # Create aliased columns for this year
            year_columns = ', '.join([f'{var} as {var}_{year}' for var in variables])
            
            cte_name = f"acs_{year}"
            year_cte = f"""
        {cte_name} AS (
            SELECT 
                geo_id,
                {year_columns}
            FROM `bigquery-public-data.census_bureau_acs.{table_name}`
            WHERE geo_id IN (
                SELECT {geo_id_field}
                FROM `{geo_table}`
                WHERE ST_INTERSECTS({geom_field}, ST_GEOGFROMTEXT('{wkt}'))
            )
        )"""
            year_cte_definitions.append(year_cte)
            year_cte_names.append(cte_name)
        
        # Build the JOIN clause
        if len(year_cte_names) > 1:
            # Start with first table
            join_base = year_cte_names[0]
            join_clauses = []
            for i in range(1, len(year_cte_names)):
                join_clauses.append(f"FULL OUTER JOIN {year_cte_names[i]} ON {join_base}.geo_id = {year_cte_names[i]}.geo_id")
            join_sql = f"""
        FROM {join_base}
        {' '.join(join_clauses)}"""
        else:
            join_sql = f"FROM {year_cte_names[0]}"
        
        # Build SELECT clause with all year-suffixed columns
        select_columns = []
        # Use COALESCE to get geo_id from any table that has it
        geo_id_coalesce = ' '.join([f"COALESCE({cte}.geo_id, " for cte in year_cte_names[:-1]]) + f"{year_cte_names[-1]}.geo_id" + ')' * (len(year_cte_names) - 1)
        select_columns.append(f"{geo_id_coalesce} as geo_id")
        
        # Add all year-suffixed variable columns
        for table_name, year in table_years:
            cte_name = f"acs_{year}"
            for var in variables:
                select_columns.append(f"{cte_name}.{var}_{year}")
        
        # Get essential geo columns
        essential_geo_cols = self._get_essential_geo_columns(geo_level)
        geo_columns_list = [f'geo.{col}' for col in essential_geo_cols if col not in ['geo_id', 'geometry_wkt']]
        geo_columns = ', '.join(geo_columns_list) if geo_columns_list else ''
        
        # Build the complete query
        full_query = f"""
        WITH boundary_filtered_geo AS (
            SELECT *
            FROM `{geo_table}`
            WHERE ST_INTERSECTS({geom_field}, ST_GEOGFROMTEXT('{wkt}'))
        ),
        {','.join(year_cte_definitions)},
        pivoted_data AS (
            SELECT 
                {', '.join(select_columns)}
            {join_sql}
        )
        SELECT 
            pivoted.*,
            ST_ASTEXT(geo.{geom_field}) as geometry_wkt{', ' + geo_columns if geo_columns else ''}
        FROM pivoted_data pivoted
        INNER JOIN boundary_filtered_geo geo
        ON pivoted.geo_id = geo.{geo_id_field}
        """
        
        print(f"Executing pivoted multi-year query for tables: {table_names}")
        print(f"Will create columns: {[f'{var}_{year}' for _, year in table_years for var in variables]}")
        print(f"Executing query: {full_query[:500]}")
        # Execute query
        df = self.client.query(full_query).to_dataframe()
        
        if df.empty:
            print("Warning: Multi-year query returned no results")
            return gpd.GeoDataFrame()
        
        # Convert to GeoDataFrame
        from shapely import wkt
        df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        gdf = gdf.drop(columns=['geometry_wkt'])
        
        print(f"✅ Retrieved {len(gdf)} unique geographic features with data from {len(table_years)} years")
        
        return gdf