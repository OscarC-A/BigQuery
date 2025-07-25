import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import json
from typing import List, Dict, Optional
import re

# Main query processor

# Big change that MUST be made at some point: part of the issue with questionable 
# column selection is that we are not properly selecting the table we want first
# and then selecting from available columns in that table. What is happening is 
# the potential relevant columns are selected from a plethora of tables, and then once the llm
# chooses the table, we filter our available choices by that table. Its a lot slower,
# less accurate, and more costly for the openai api.

# In general this whole ish needs to be altered properly and fixed. Another issue
# is the 'table_list' we are giving the llm to choose from is literally empty??

class CensusSemanticSearcher:
    def __init__(self, indexer, geo_resolver, bq_client):
        self.indexer = indexer
        self.geo_resolver = geo_resolver
        self.bq_client = bq_client
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.openai_client = OpenAI()
        
    def search_tables(self, query: str, k: int = 100) -> pd.DataFrame:
        """Search for relevant tables and variables using semantic search"""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.indexer.index.search(query_embedding.astype('float32'), k)
        
        # Get results
        results = self.indexer.metadata_df.iloc[indices[0]].copy()
        results['score'] = scores[0]
        
        return results
    
    async def analyze_query_intent(self, query: str) -> Dict:
        """Use LLM to understand query intent"""
        prompt = f"""Analyze this census data query and extract the intent.

Query: "{query}"

Return a JSON object with:
{{
    "geo_level": "county|state|tract",
    "topics": ["list of topics like race, income, housing"],
    "specific_variables": ["any specific variables mentioned"],
    "year_preference": "latest|specific year|null",
    "aggregation": "none|sum|average|percentage"
}}

Focus on what census data the user wants to see."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
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
    
    async def filter_results_with_llm(self, query: str, candidates: pd.DataFrame, 
                                     intent: Dict) -> Dict:
        """Use LLM to select best table first, then columns from that table"""
        # Step 1: Select the best table
        tables = candidates[candidates['type'] == 'table']
        
        table_list = "\n".join([
            f"- {row['table_name']} (code: {row['table_code']}, year: {row['year']}, geo_level: {row['geo_level']})"
            for _, row in tables.head(20).iterrows()
        ])
        print(len(table_list))
        print("AAAAAAA")
        table_prompt = f"""Select the most relevant census table for this query.

Query: "{query}"
Intent: {json.dumps(intent)}

Available Tables:
{table_list}

Return a JSON object:
{{
    "selected_table": "county_2021_1yr",
    "reasoning": "why this table was selected"
}}

IMPORTANT TABLE SELECTION GUIDELINES:
1. Prefer recent years (2020-2022) unless user specifies otherwise
2. Match the geographic level from intent: {intent.get('geo_level', 'county')}
3. Choose tables that are most likely to contain the requested information
4. For broad queries, prefer comprehensive tables over specialized ones"""

        table_response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": table_prompt}],
            temperature=0
        )
        
        table_content = table_response.choices[0].message.content
        table_json_match = re.search(r'\{.*\}', table_content, re.DOTALL)
        
        if table_json_match:
            table_result = json.loads(table_json_match.group(0))
            selected_table = table_result['selected_table']
        else:
            # Fallback to most recent table with correct geo level
            filtered_tables = tables[tables['geo_level'] == intent.get('geo_level', 'county')]
            if len(filtered_tables) > 0:
                selected_table = filtered_tables.iloc[0]['table_name']
            else:
                selected_table = tables.iloc[0]['table_name'] if len(tables) > 0 else "county_2021_1yr"
        
        # Step 2: Get columns only from the selected table
        table_columns = candidates[
            (candidates['type'] == 'column') & 
            (candidates['table_name'] == selected_table)
        ]
        
        if len(table_columns) == 0:
            # Fallback: if no columns found for selected table, use all columns
            table_columns = candidates[candidates['type'] == 'column']
        
        col_list = "\n".join([
            f"- {row['column_name']}: {row['description']}"
            for _, row in table_columns.iterrows()
        ])
        print(col_list)
        column_prompt = f"""Select the most relevant census columns from this specific table for the query.

Query: "{query}"
Intent: {json.dumps(intent)}
Selected Table: {selected_table}

Available Columns from {selected_table}:
{col_list}

Return a JSON object:
{{
    "selected_variables": ["total_pop", "white_pop"],
    "reasoning": "why these columns were selected"
}}

IMPORTANT COLUMN SELECTION GUIDELINES:
1. For race/ethnicity queries, prioritize general population columns ending with "_pop" 
2. Choose columns that provide the broadest, most useful demographic information first
3. Always include total_pop for context when selecting demographic subgroups
4. For income/economic queries, prefer median over mean, and general measures over specific breakdowns
5. Do not be afraid to choose many columns. For example, if someone wants all commute related information, return ALL commute related columns from this table
6. Only select columns that actually exist in the provided list

Choose column names (not variable codes) that directly answer the query."""

        column_response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": column_prompt}],
            temperature=0
        )
        
        column_content = column_response.choices[0].message.content
        column_json_match = re.search(r'\{.*\}', column_content, re.DOTALL)
        
        if column_json_match:
            column_result = json.loads(column_json_match.group(0))
            
            # Clean the selected variables to ensure they are valid column names
            cleaned_variables = []
            for var in column_result.get('selected_variables', []):
                # Extract just the column name (before any colon or description)
                clean_var = var.split(':')[0].strip()
                cleaned_variables.append(clean_var)
            
            return {
                "selected_variables": cleaned_variables,
                "selected_table": selected_table,
                "reasoning": f"Table: {table_result.get('reasoning', 'Selected based on relevance')}; Columns: {column_result.get('reasoning', 'Selected based on relevance')}"
            }
        else:
            # Fallback to top columns from selected table
            top_columns = table_columns.head(5)['column_name'].tolist()
            if not top_columns:
                top_columns = ["total_pop"]
            
            return {
                "selected_variables": top_columns,
                "selected_table": selected_table,
                "reasoning": "Fallback selection"
            }
    
    async def process_query(self, query: str) -> pd.DataFrame:
        """Main pipeline to process natural language query"""
        print(f"\n🔍 Processing query: '{query}'")
        
        # 1. Analyze intent
        print("📊 Analyzing query intent...")
        intent = await self.analyze_query_intent(query)
        print(f"Intent: {intent}")
        
        # 2. Semantic search
        print("🔎 Searching for relevant tables and variables...")
        candidates = self.search_tables(query, k=200)
        
        # 3. LLM filtering
        print("🤖 Filtering with LLM...")
        selection = await self.filter_results_with_llm(query, candidates, intent)
        print(f"Selected: {selection}")
        
        # 4. Build geographic filter
        print("🗺️ Building geographic filter...")
        geo_filter = self.geo_resolver.build_geo_filter(query, intent['geo_level'])
        print(f"Filter: {geo_filter['filter_sql']}")
        
        # 5. Query BigQuery
        print("📊 Querying BigQuery...")
        data = self.bq_client.query_acs_data(
            selection['selected_table'],
            selection['selected_variables'],
            geo_filter['filter_sql']
        )
        
        print(f"✅ Retrieved {len(data)} rows")
        return data, geo_filter, selection