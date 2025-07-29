# Intialize BigQuery
# get_acs_tables_metadata: gets acs tables metadata (only needed once, then is saved), 
# get_table_columns: gets all columns from selected table
# query_acs_table: executes final SQL query to BigQuery

import os
from google.cloud import bigquery
from typing import List, Dict, Optional
import pandas as pd

class CensusBigQueryClient:
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.client = bigquery.Client(project=self.project_id)
        
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
    
    def query_acs_data(self, table_name: str, variables: List[str], 
                      geo_filter: str) -> pd.DataFrame:
        """
        Query specific ACS data
        
        Args:
            table_name: e.g., 'county_2020_5yr'
            variables: List of actual column names (not variable codes)
            geo_filter: e.g., "geo_id LIKE '13%'" for Georgia counties
        """
        var_list = ['geo_id'] + variables
        columns = ', '.join(var_list)
        
        query = f"""
        SELECT {columns}
        FROM `bigquery-public-data.census_bureau_acs.{table_name}`
        WHERE {geo_filter}
        """
        
        print(f"Executing BigQuery:\n{query}")
        return self.client.query(query).to_dataframe()
    
    def get_geo_boundaries_tables(self) -> pd.DataFrame:
        """Get available geo boundary tables"""
        query = """
        SELECT 
            table_name,
            table_type,
            creation_time
        FROM `bigquery-public-data.geo_us_boundaries.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """
        return self.client.query(query).to_dataframe()
    
    def query_geo_boundaries(self, table_name: str, geo_filter: str) -> pd.DataFrame:
        """
        Query geo boundary data
        
        Args:
            table_name: e.g., 'counties', 'states', 'zip_codes'
            geo_filter: e.g., "state_fips_code = '13'" for Georgia counties
        """
        query = f"""
        SELECT *
        FROM `bigquery-public-data.geo_us_boundaries.{table_name}`
        WHERE {geo_filter}
        """
        
        print(f"Executing BigQuery geometry query:\n{query}")
        return self.client.query(query).to_dataframe()