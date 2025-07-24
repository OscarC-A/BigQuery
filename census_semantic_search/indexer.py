import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

# Similar to bigquery_client, do we need to be using table and col codes? How 
# much is this actually aiding our search (idk)

# Also, this is coded so that only works for county level data for now

class ACSMetadataIndexer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        # Fix for segmentation fault: disable multiprocessing
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.metadata_df = None
        
    def build_index(self, bq_client):
        """Build semantic search index for ACS tables and variables"""
        print("🔧 Building ACS metadata index...")
        
        # Get table metadata
        tables_df = bq_client.get_acs_tables_metadata()
        
        # Create rich descriptions for semantic search
        descriptions = []
        metadata_records = []
        
        # For each table, get actual column names from BigQuery schema
        for _, table in tables_df.iterrows():
            if table['geo_level'] == 'county':  # Focus on county for now
                table_name = table['table_name']
                print(f"   Getting columns for {table_name}...")
                
                # Get actual columns from BigQuery
                columns_df = bq_client.get_table_columns(table_name)
                
                # Add column-level descriptions using actual BigQuery column names
                for _, col in columns_df.iterrows():
                    col_name = col['column_name']
                    
                    # Skip metadata columns
                    if col_name in ['geo_id', 'state', 'county', 'state_code', 'county_code']:
                        continue
                    
                    # Create semantic description for the column
                    desc = self._create_column_description(col_name, table_name, table.get('table_code', ''))
                    descriptions.append(desc)
                    metadata_records.append({
                        'type': 'column',
                        'column_name': col_name,
                        'table_name': table_name,
                        'table_code': table.get('table_code', ''),
                        'geo_level': table['geo_level'],
                        'year': table['year'],
                        'description': desc,
                        'search_text': desc
                    })
        
        for _, table in tables_df.iterrows():
            if table['geo_level'] == 'county':  # Focus on county for now                
                desc = f"{table['table_name']} {table['table_code']} {table['geo_level']} {table['year']}"
                descriptions.append(desc)
                metadata_records.append({
                    'type': 'table',
                    'table_name': table['table_name'],
                    'table_code': table['table_code'],
                    'geo_level': table['geo_level'],
                    'year': table['year'],
                    'search_text': desc
                })
        
        # Create embeddings
        print(f"📊 Creating embeddings for {len(descriptions)} items...")
        # Fix for segmentation fault: disable multiprocessing and use smaller batches
        embeddings = self.model.encode(descriptions, convert_to_numpy=True, 
                                     show_progress_bar=True, batch_size=8)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(embeddings.astype('float32'))
        
        # Save metadata
        self.metadata_df = pd.DataFrame(metadata_records)
        
        # Save to disk
        os.makedirs(self.data_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.data_dir, "acs_tables.index"))
        self.metadata_df.to_parquet(os.path.join(self.data_dir, "acs_metadata.parquet"))
        
        print(f"✅ Index built with {len(self.metadata_df)} items")
    
    def _create_column_description(self, col_name: str, table_name: str, table_code: str) -> str:
        """Create semantic description for a BigQuery column name"""
        # Convert column name to human readable description
        desc_parts = [col_name.replace('_', ' ')]
        
        # Add table context
        if table_code:
            desc_parts.append(f"from {table_code}")
        desc_parts.append(f"table {table_name}")
        
        # Add topic keywords based on column name patterns
        col_lower = col_name.lower()
        
        if any(word in col_lower for word in ['pop', 'population']):
            desc_parts.append('population demographics')
        if any(word in col_lower for word in ['white', 'black', 'asian', 'race', 'ethnicity']):
            desc_parts.append('race ethnicity demographics')
        if any(word in col_lower for word in ['income', 'earnings', 'median']):
            desc_parts.append('income economics earnings')
        if any(word in col_lower for word in ['housing', 'units', 'occupied', 'owner', 'renter']):
            desc_parts.append('housing units occupancy')
        if any(word in col_lower for word in ['education', 'school', 'degree', 'bachelor', 'master']):
            desc_parts.append('education attainment degree')
        if any(word in col_lower for word in ['commute', 'transport', 'work', 'travel']):
            desc_parts.append('commuting transportation work')
        
        return ' '.join(desc_parts)
        
    def load_index(self):
        """Load pre-built index from disk"""
        index_path = os.path.join(self.data_dir, "acs_tables.index")
        metadata_path = os.path.join(self.data_dir, "acs_metadata.parquet")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            self.metadata_df = pd.read_parquet(metadata_path)
            print(f"✅ Loaded index with {len(self.metadata_df)} items")
            return True
        return False