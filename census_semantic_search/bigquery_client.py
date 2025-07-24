# Note: This is just using well known table codes for simplicity
# Later, it needs to be fed a more comprehensive list of all codes or 
# use an api to be able to fetch any data. These lists are easily
# found online and shouldnt be too difficult to integrate later on

# Other important note: do we need to be using table codes? Can we just search
# by using columns the llm deems relevant?

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
            REGEXP_EXTRACT(table_name, r'(county|state|tract|place)') as geo_level,
            REGEXP_EXTRACT(table_name, r'([0-9]{4})') as year
        FROM `bigquery-public-data.census_bureau_acs.INFORMATION_SCHEMA.TABLES`
        WHERE table_name LIKE '%1yr'
        ORDER BY table_name
        """
        return self.client.query(query).to_dataframe()
    
    def get_table_columns(self, table_name: str) -> pd.DataFrame:
        """Get column metadata for a specific table"""
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
            table_name: e.g., 'county_2021_1yr'
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