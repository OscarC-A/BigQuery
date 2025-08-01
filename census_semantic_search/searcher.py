import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
from typing import List, Dict, Optional, Tuple
import re
import geopandas as gpd

# Main query processor

# Redesigned approach: 
# 1. Choose from a select few ACS tables first (Kinda hard coded rn but is really all we need for acs data)
# 2. Pick the best table based on query
# 3. Get ALL columns from that specific table  
# 4. Feed all columns to LLM to choose the best subset (could be a LOT easier on chat if we didnt feed all column names)

class CensusSemanticSearcher:
    def __init__(self, indexer, geo_resolver, bq_client):
        self.indexer = indexer
        self.geo_resolver = geo_resolver
        self.bq_client = bq_client
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.openai_client = OpenAI()
        
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
        
    async def select_best_table(self, query: str, intent: Dict) -> str:
        # A lot below is commented out, as llm not needed to select best table at this point, we are just
        # finding the best table based on our geo_level for now

#        """Step 1: Select the most relevant ACS table from our predefined list"""
#         table_descriptions = []
#         for table_name, metadata in self.acs_tables.items():
#             table_descriptions.append(
#                 f"- {table_name}: {metadata['description']} (Geographic level: {metadata['geo_level']}, Year: {metadata['year']})"
#             )
        
#         table_list = "\n".join(table_descriptions)
#         prompt = f"""Select the most relevant ACS table for this query from the provided options.

# Query: "{query}"
# Intent: {json.dumps(intent)}

# Available ACS Tables:
# {table_list}

# Return a JSON object:
# {{
#     "selected_table": "county_2020_5yr",
#     "reasoning": "why this table was selected"
# }}

# SELECTION GUIDELINES:
# 1. Match the geographic level from intent: {intent.get('geo_level', 'county')}
# 2. Prefer recent years unless user specifies otherwise
# 3. For state, county, zip, or tract-level queries, choose from state, county, zcta, censustract tables respectively
# 4. Consider the comprehensiveness - all these tables contain demographic, economic, and housing data"""

#         response = self.openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
        
#         content = response.choices[0].message.content
#         json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        # Line below should be 'if json_match' but set to false for now as we can just 
        # match on geo_level alone
        if False:
            result = json.loads(json_match.group(0))
            selected_table = result.get('selected_table', 'county_2020_5yr')
            print(f"Selected table: {selected_table} - {result.get('reasoning', '')}")
            return selected_table
        else:
            # Fallback based on geo level
            geo_level = intent.get('geo_level', 'county')
            if geo_level == 'county':
                return 'county_2020_5yr'
            elif geo_level == 'zcta':
                return 'zcta_2020_5yr'
            elif geo_level == 'tract':
                return 'censustract_2020_5yr'
            else:
                return 'state_2021_1yr'
    
    def get_all_table_columns(self, table_name: str) -> List[Dict]:
        """Step 2: Get ALL columns from the selected table"""
        print(f"Getting all columns for table: {table_name}")
        
        try:
            # Get column names from BigQuery
            columns_df = self.bq_client.get_table_columns(table_name)
            
            # Convert to list of dicts with column info
            columns = []
            for _, row in columns_df.iterrows():
                # Skip geo_id and standard metadata columns
                if row['column_name'] not in ['geo_id']:
                    columns.append({
                        'name': row['column_name'],
                        'data_type': row['data_type'],
                        'description': row['column_name']  # We'll rely on column names for now
                    })
            
            print(f"Found {len(columns)} data columns in {table_name}")
            return columns
            
        except Exception as e:
            print(f"Error getting columns for {table_name}: {e}")
            # Return some common columns as fallback
            return [
                {'name': 'total_pop', 'data_type': 'FLOAT64', 'description': 'total_pop'},
                {'name': 'white_pop', 'data_type': 'FLOAT64', 'description': 'white_pop'},
                {'name': 'black_pop', 'data_type': 'FLOAT64', 'description': 'black_pop'},
                {'name': 'median_income', 'data_type': 'FLOAT64', 'description': 'median_income'}
            ]
    
    async def analyze_query_intent(self, query: str) -> Dict:
        # Use LLM to understand query intent
        # There are many more geo levels to consider such as (place, puma, cbsa, block group etc.)
        # Cbsa or place would be great for more 'vague' queries (metro areas etc.) but don't want to 
        # overcomplicate things for now
        prompt = f"""Analyze this census data query and extract the intent.

Query: "{query}"

Return a JSON object with:
{{
    "geo_level": "state|county|zcta|tract",
    "point_of_interest": "texas, new york city, tompkins county, deleware, etc."
    "topics": ["list of topics like race, income, housing, etc."],
    "specific_variables": ["any specific variables mentioned"],
    "year_preference": "latest|specific year|null",
    "aggregation": "none|sum|average|percentage",
    "state": "[list of state or states encompassing the point of interest]"
}}

Note that "point_of_interest" should just be the name of the location or area that the query is interested in.
This may be an entire state, a specific county, a metro area, or a city, for example.
"state" is the state or states that the point of interest lies within. For example, if the point of
interest is manhattan, the state would be ["new york"].
Focus on what census data the user wants to see."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            # Print for debug
            print(json.loads(json_match.group(0)))
            return json.loads(json_match.group(0))
        else:
            # Fallback
            return {
                "geo_level": "county",
                "topics": ["demographics"],
                "specific_variables": [],
                "year_preference": "latest",
                "aggregation": "none"
            }
    
    async def select_best_columns(self, query: str, intent: Dict, table_name: str, 
                                 all_columns: List[Dict]) -> Dict:
        """Step 3: Use LLM to select best columns from ALL available columns in the table"""
        
        # Create a comprehensive list of all columns for the LLM
        col_list = "\n".join([
            f"- {col['name']} ({col['data_type']})"
            for col in all_columns
        ])
        print(col_list[:5]) # For debugging
        
        print(f"Presenting {len(all_columns)} columns to LLM for selection...")
        
        prompt = f"""Select the most relevant census columns from this table for the query.

Query: "{query}"
Intent: {json.dumps(intent)}
Selected Table: {table_name}

ALL Available Columns from {table_name}:
{col_list}

Return a JSON object:
{{
    "selected_variables": ["total_pop", "white_pop", "median_income"],
    "reasoning": "why these specific columns were selected"
}}

COLUMN SELECTION GUIDELINES:
1. For race/ethnicity queries: select population counts for different racial groups
2. For income queries: select median household income, per capita income, poverty measures
3. For housing queries: select housing units, occupancy, home values, rent costs
4. For commute queries: select transportation methods, travel times, work locations
5. For education queries: select educational attainment levels
6. For age queries: select age group breakdowns and median age
7. Always include total population (total_pop) for context when relevant
8. Be generous with selections - if someone asks for "all commute information", include ALL commute-related columns
9. Only select columns that exist in the provided list
10. Use the exact column names as shown (without data types)

Choose column names that directly answer the user's query."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group(0))
            
            # Validate that selected columns actually exist
            available_column_names = [col['name'] for col in all_columns]
            valid_variables = []
            
            for var in result.get('selected_variables', []):
                clean_var = var.split(':')[0].strip()
                if clean_var in available_column_names:
                    valid_variables.append(clean_var)
                else:
                    print(f"Warning: Column '{clean_var}' not found in table, skipping")
            
            if not valid_variables:
                # Fallback to first few columns if none are valid
                valid_variables = [col['name'] for col in all_columns[:5]]
                
            return {
                "selected_variables": valid_variables,
                "selected_table": table_name,
                "reasoning": result.get('reasoning', 'Selected based on query relevance')
            }
        else:
            # Fallback to first 10 columns
            fallback_vars = [col['name'] for col in all_columns[:10]]
            return {
                "selected_variables": fallback_vars,
                "selected_table": table_name,
                "reasoning": "Fallback selection due to parsing error"
            }
    
    async def process_query(self, query: str) -> Tuple[gpd.GeoDataFrame, dict, dict]:
        """Main pipeline using the new approach: select table first, then all columns, then LLM selection"""
        print(f"\n🔍 Processing query: '{query}'")
        
        # 1. Analyze intent
        print("📊 Analyzing query intent...")
        intent = await self.analyze_query_intent(query)
        print(f"Intent: {intent}")
        
        # 2. Select best table from our predefined list
        print("🎯 Selecting best ACS table...")
        selected_table = await self.select_best_table(query, intent)
        
        # 3. Get ALL columns from the selected table
        print("📋 Getting all columns from selected table...")
        all_columns = self.get_all_table_columns(selected_table)
        
        # 4. Let LLM choose the best subset of columns
        print("🤖 LLM selecting best columns...")
        selection = await self.select_best_columns(query, intent, selected_table, all_columns)
        print(f"Selected: {len(selection['selected_variables'])} variables from {selected_table}")
        
        # 5. Build geographic filter
        print("🗺️ Building geographic filter...")
        geo_filter = self.geo_resolver.build_geo_filter(query, intent['geo_level'], intent['state'])
        #print(f"Filter: {geo_filter['filter_sql']}")
        
        # 6. Query BigQuery
        print("📊 Querying BigQuery...")
        try:
            # Using custom geo bounds
            gdf = self.bq_client.query_acs_with_geometry(
                selection['selected_table'],
                selection['selected_variables'],
                geo_filter['filter_sql'],
                intent['geo_level'],
                geo_filter
            )

            print(f"✅ Retrieved {len(gdf)} features with {len(selection['selected_variables'])} variables and geometries")
            
            # Return GeoDataFrame directly (no need for separate geometry fetch)
            return gdf, geo_filter, selection
        except Exception as e:
            print(f"❌ Error in combined query: {str(e)}")
            print("Falling back to legacy state-only query boundaries...")
            data = self.bq_client.query_acs_data(
                selection['selected_table'],
                selection['selected_variables'],
                geo_filter['filter_sql']
            )
        
            print(f"✅ Retrieved {len(data)} rows with {len(selection['selected_variables'])} variables")
            return data, geo_filter, selection