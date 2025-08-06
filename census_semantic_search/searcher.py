from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
from typing import List, Dict, Optional, Tuple
import re
import geopandas as gpd
import httpx
from urllib.parse import quote

# Main query processor

# Redesigned approach: 
# 1. Choose from a select few ACS tables first (Kinda hard coded rn but is really all we need for acs data)
# 2. Pick the best table based on query
# 3. Get ALL columns from that specific table  
# 4. Feed all columns to LLM to choose the best subset (could be a LOT easier on chat if we didnt feed all column names)

class CensusSemanticSearcher:
    def __init__(self, indexer, state_detect, bq_client):
        # self.indexer = indexer
        self.state_detect = state_detect
        self.bq_client = bq_client
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.openai_client = OpenAI()
        
    def _get_default_table(self, geo_level: str) -> str:
        # Fallback based on geo level
        if geo_level == 'congressional_district':
            return 'congressionaldistrict_2020_5yr'
        elif geo_level == 'county':
            return 'county_2020_5yr'
        elif geo_level == 'zcta':
            return 'zcta_2020_5yr'
        elif geo_level == 'tract':
            return 'censustract_2020_5yr'
        else:
            return 'state_2021_1yr'
        
    async def select_best_tables(self, query: str, intent: Dict) -> List[str]:
        """Select best tables based on intent, including multiple years if needed"""
        geo_level = intent.get('geo_level', 'county')
        years = intent.get('years')
        comparison_type = intent.get('comparison_type', 'none')
        
        # Get available tables dynamically
        available_tables = self.bq_client.get_available_acs_tables(geo_level, years)
        print(f"Available tables {available_tables}")
        if not available_tables:
            # Fallback to default table
            print("Defaulting to default table")
            return [self._get_default_table(geo_level)]
        
        # If comparing years, select tables for each year
        if comparison_type in ['year_over_year', 'between_years'] and years and len(years) > 1:
            selected_tables = []
            for year in years:
                # Find best table for each year (prefer 5yr over 1yr for consistency)
                year_tables = [t for t in available_tables if t['year'] == year]
                if year_tables:
                    # Prefer 5yr surveys for better coverage
                    five_yr = [t for t in year_tables if '5yr' in t.get('survey_type', '')]
                    if five_yr:
                        selected_tables.append(five_yr[0]['table_name'])
                    else:
                        selected_tables.append(year_tables[0]['table_name'])
            return selected_tables
        else:
            # Single year or latest - return most recent table
            if available_tables:
                # Prefer 5yr surveys
                five_yr = [t for t in available_tables if '5yr' in t.get('survey_type', '')]
                if five_yr:
                    return [five_yr[0]['table_name']]
                return [available_tables[0]['table_name']]
        
        return [self._get_default_table(geo_level)]
    
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
    "geo_level": "state|congressional_district|county|zcta|tract",
    "point_of_interest": "texas, new york city, tompkins county, deleware, etc."
    "topics": ["list of topics like race, income, housing, etc."],
    "specific_variables": ["any specific variables mentioned"],
    "years": [2020, 2019],  // Extract specific years mentioned or inferred from query
    "comparison_type": "none|year_over_year|between_years",  // Detect if comparing across years
    "aggregation": "none|sum|average|percentage",
    "state": "[list of state or states encompassing the point of interest]"
}}

1) For "point_of_interest", extract the geographic location name from this query for boundary extraction.
   Return the most specific location mentioned (city, county, state, metro area, etc.). 
   If multiple locations, return the most relevant one. Return only the location name, nothing else.
2) "state" is the state or states that the point of interest lies within. For example, if the point of
   interest is manhattan, the state would be ["new york"]. 
3) It is CRUCIAL that you do not hallucinate or make an unsure guess for what "state", "point_of_interest" or 
   any other field is. If you do not know, or are not absolutely sure, return an empty list.
4) Be forgiving towards potential typos. If someone types ".. in Itaca", it is fair to assume that the point of interest should be "Ithaca"
5) If user mentions "change from X to Y" or "between X and Y", set comparison_type appropriately. If there are multipule years,
   but you are unsure what comparison_type it is, default to between_years.
6) Extract the years mentioned (e.g., "from 2010 to 2020" → years: [2010, 2020]). If it is a year over year query, 
   then include all years between the first and last. If it is between years, when just return the 2 years.
   There is no data past 2021, that is the most recent year you can return.
7) If the query does not contain a specific year(s), use years: [2021, 2020] (most recent available)
8) Focus on what census data the user wants to see."""

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
                "point of interest": "none",
                "topics": ["demographics"],
                "specific_variables": [],
                "years": None,
                "comparison_type": "none",
                "aggregation": "none",
                "state": "unknown"
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
        
    async def extract_boundary_from_query(self, query: str, intent=None) -> Optional[str]:
        """Extract geographic boundary from query using Nominatim, save as GeoJSON file"""
        try:
            # Extract location using existing LLM method
            place_str = intent['point_of_interest']
            if not place_str:
                return None
            
            print(f"🌍 Getting boundary for: '{place_str}'")
            
            # Query Nominatim for boundary
            async with httpx.AsyncClient(headers={"User-Agent": "CensusSemanticSearch/1.0"}) as client:
                url = (
                    "https://nominatim.openstreetmap.org/search"
                    f"?q={quote(place_str)}&format=jsonv2&polygon_geojson=1&limit=1"
                )
                print(f"Querying Nominatim API...")
                r = await client.get(url, timeout=20)
                r.raise_for_status()
                items = r.json()
                
                if not items:
                    print(f"   No boundary found for '{place_str}'")
                    return None
                    
                itm = items[0]
                osm_id   = int(itm["osm_id"])
                osm_type = itm["osm_type"]
                geom     = itm.get("geojson")  # may be None
                # print(geom)
                # poly_str = _poly_from_geojson(geom) if geom else ""
                geo = itm['geojson']
                # print(geo)
                print(f"   Found: {itm['display_name']}")
                print(f"   OSM ID: {osm_id} ({osm_type})")

                # Get coordinates for around queries
                lat = float(itm["lat"])
                lon = float(itm["lon"])

                # Get bounding box for square area
                bound_box = itm["boundingbox"] # Ex: ["40.5503390","40.7394340","-74.0566880","-73.8329450"]

                # Create GeoJSON file
                if geom["type"] == 'Point':
                    # We can use the given bounding box, or use the lat lon coords and create a bound
                    # of some size or radius. Using bounding box for now, radius that user could set would
                    # be cool to implement later
                    
                    min_lat, max_lat, min_lon, max_lon = bound_box
                    min_lat = float(min_lat)
                    max_lat = float(max_lat)
                    min_lon = float(min_lon)
                    max_lon = float(max_lon)

                    # Create rectangle coordinates (counter-clockwise) bottom-left bottom-right top-right top-left
                    coordinates = [[min_lon, min_lat], [max_lon, min_lat], 
                                   [max_lon, max_lat], [min_lon, max_lat],  
                                   [min_lon, min_lat] ]
                    geo = {"type": "Polygon",
                                           "coordinates": [coordinates]}
                    print(geo)

                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": geo,
                        "properties": {
                            "name": place_str,
                            "display_name": items[0]['display_name']
                        }
                    }]
                }
                
                # Save to temporary file
                import tempfile
                import json
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False)
                json.dump(geojson_data, temp_file, indent=2)
                temp_file.close()
                
                print(f"   Boundary saved to: {temp_file.name}")
                return temp_file.name
            
        except Exception as e:
            print(f"   Error extracting boundary: {e}")
            return None
    
    async def process_query(self, query: str, geojson_dir=None) -> Tuple[gpd.GeoDataFrame, dict, dict]:
        """Main pipeline with automatic boundary extraction if no geojson_dir provided"""
        print(f"\n🔍 Processing query: '{query}'")
        
        # 1. Analyze intent
        print("📊 Analyzing query intent...")
        intent = await self.analyze_query_intent(query)
        print(f"Intent: {intent}")
        
        # 2. Extract boundary if not provided
        if not geojson_dir:
            print("🗺️ No boundary provided, attempting to extract from query...")
            geojson_dir = await self.extract_boundary_from_query(query, intent)
            if geojson_dir:
                print(f"✅ Auto-extracted boundary: {geojson_dir}")
            else:
                print("⚠️ Could not extract boundary, using state-based filtering")
        
        # 3. Select best tables from our predefined list
        print("🎯 Selecting best ACS table(s)...")
        selected_tables = await self.select_best_tables(query, intent)
        print(f"Selected tables: {selected_tables}")

        # 4. Get columns from the first table (assuming schema is same across different years)
        print("📋 Getting all columns from selected table...")
        all_columns = self.get_all_table_columns(selected_tables[0])
        
        # 5. Let LLM choose the best subset of columns
        print("🤖 LLM selecting best columns...")
        selection = await self.select_best_columns(query, intent, selected_tables[0], all_columns)
        print(f"Selected: {len(selection['selected_variables'])} variables from {selected_tables[0]}: {selection['selected_variables']}")
        
        # 6. Extract state
        print("🗺️ Extracting state")
        state_name = self.state_detect.extract_state_from_query(query, intent['state'], geojson_dir)
        
        # 7. Query BigQuery
        print("📊 Querying BigQuery...")
        try:
            if len(selected_tables) > 1:
                # Multi year query
                print(f"Executing multi-year query for {len(selected_tables)} tables")
                gdf = self.bq_client.query_acs_multi_year(
                    selected_tables,
                    selection['selected_variables'],
                    intent['geo_level'],
                    geojson_dir,
                    state_name
                )
            else:
                # Single year/table query
                gdf = self.bq_client.query_acs_with_geometry(
                    selection['selected_table'],
                    selection['selected_variables'],
                    intent['geo_level'],
                    geojson_dir,
                    state_name
                )

            print(f"✅ Retrieved {len(gdf)} features with {len(selection['selected_variables'])} variables and geometries")
            
            # Clean up temporary boundary file if we created one
            if geojson_dir and geojson_dir.startswith('/tmp'):
                import os
                try:
                    os.unlink(geojson_dir)
                    print(f"🧹 Cleaned up temporary boundary file")
                except:
                    pass
            
            return gdf, selection
        except Exception as e:
            print(f"❌ Error in query: {str(e)}")
            raise