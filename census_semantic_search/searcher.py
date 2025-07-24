import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import json
from typing import List, Dict, Optional
import re

# Main query processor

class CensusSemanticSearcher:
    def __init__(self, indexer, geo_resolver, bq_client):
        self.indexer = indexer
        self.geo_resolver = geo_resolver
        self.bq_client = bq_client
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.openai_client = OpenAI()
        
    def search_tables(self, query: str, k: int = 20) -> pd.DataFrame:
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
        """Use LLM to select best tables and variables"""
        # Prepare candidate info
        columns = candidates[candidates['type'] == 'column']
        tables = candidates[candidates['type'] == 'table']
        
        col_list = "\n".join([
            f"- {row['column_name']}: {row['description']} (from {row['table_name']})"
            for _, row in columns.head(15).iterrows()
        ])
        
        table_list = "\n".join([
            f"- {row['table_name']} (code: {row['table_code']}, year: {row['year']})"
            for _, row in tables.head(10).iterrows()
        ])
        
        prompt = f"""Select the most relevant census columns for this query.

Query: "{query}"
Intent: {json.dumps(intent)}

Available Columns:
{col_list}

Available Tables:
{table_list}

Return a JSON object:
{{
    "selected_variables": ["total_pop", "white_pop"],
    "selected_table": "county_2021_1yr",
    "reasoning": "why these were selected"
}}

Choose column names (not variable codes) that directly answer the query. For the table, prefer the most recent year and appropriate geographic level."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            # Fallback to top results
            top_col = columns.iloc[0] if len(columns) > 0 else None
            top_table = tables[tables['geo_level'] == intent['geo_level']].iloc[0] if len(tables) > 0 else None
            
            return {
                "selected_variables": [top_col['column_name']] if top_col is not None else ["total_pop"],
                "selected_table": top_table['table_name'] if top_table is not None else "county_2021_1yr",
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
        candidates = self.search_tables(query, k=30)
        
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