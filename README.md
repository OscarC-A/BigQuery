# Census Semantic Search with BigQuery

A comprehensive Python library for semantic search and visualization of US Census data using Google BigQuery and natural language queries. This library enables users to query census data using plain English and automatically generates map-ready GeoJSON files with both census data and geographic boundaries.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Geographic Levels](#geographic-levels)
- [Custom Boundary Support](#custom-boundary-support)
- [Natural Language Query Processing](#natural-language-query-processing)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)

## Features

- **🗣️ Natural Language Queries**: Ask questions about census data in plain English
- **🔍 Semantic Search**: Uses sentence transformers and FAISS for intelligent variable matching
- **🗺️ Multi-Level Geographic Support**: County, ZIP Code Tabulation Area (ZCTA), Census Tract, and State level queries
- **🎯 Custom Boundary Support**: Query data within custom geographic boundaries (e.g., Manhattan, neighborhoods)
- **⚡ BigQuery Integration**: Optimized queries to US Census data from Google BigQuery public datasets
- **🚀 Single-Query Performance**: Combines census data with geometries in one optimized BigQuery call
- **📊 GeoJSON Output**: Generates map-ready GeoJSON files with census data and geometries
- **🧠 LLM-Powered Analysis**: Uses OpenAI GPT for intelligent query interpretation and variable selection
- **💾 Intelligent Caching**: Geometry and metadata caching for improved performance
- **🏗️ Scalable Architecture**: Modular design supporting extension to new geographic levels and data sources

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Analysis  │───▶│  Table Selection│
│ "Income in NYC" │    │   (OpenAI LLM)   │    │   (Auto-select) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GeoJSON       │◀───│  BigQuery Exec   │◀───│ Column Selection│
│   Output        │    │  (Data + Geom)   │    │   (LLM-guided)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The system processes natural language queries through multiple stages:
1. **Intent Analysis**: LLM extracts geographic level, topics, and location
2. **Table Selection**: Automatically selects appropriate ACS table
3. **Column Discovery**: Retrieves all available columns from selected table
4. **Variable Selection**: LLM selects relevant variables for the query
5. **Geographic Resolution**: Builds spatial filters (state-based or custom boundaries)
6. **Optimized Execution**: Single BigQuery call combining data and geometry
7. **GeoJSON Generation**: Map-ready output with metadata

## Installation & Setup

### Prerequisites

- Python 3.8+
- Google Cloud Project with BigQuery API enabled
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BigQuery

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```bash
google-cloud-bigquery>=3.13.0    # BigQuery client
sentence-transformers>=2.7.0     # Semantic search
faiss-cpu>=1.8.0                 # Vector similarity search
pandas>=2.1.3                    # Data manipulation
geopandas>=1.0.1                 # Geospatial data handling
shapely>=2.0.2                   # Geometric operations
openai>=1.3.0                    # LLM-powered query analysis
python-dotenv>=1.0.0             # Environment configuration
```

### Authentication Setup

#### 1. Google Cloud Authentication

```bash
# Option A: Application Default Credentials
gcloud auth application-default login

# Option B: Service Account (recommended for production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Option C: Set project ID (optional)
export GCP_PROJECT_ID="your-project-id"
```

#### 2. OpenAI API Configuration

Create a `.env` file in the project root:

```bash
# Required for query analysis and variable selection
OPENAI_API_KEY=your_openai_api_key_here

# Optional: specify Google Cloud project
GCP_PROJECT_ID=your_gcp_project_id
```

### Verify Setup

Test your authentication setup:

```bash
# Test BigQuery connection
python examples/test_auth.py

# Test tract functionality (advanced)
python examples/test_tract.py
```

## Quick Start

### Basic Usage

```python
import asyncio
from census_semantic_search import (
    CensusBigQueryClient,
    ACSMetadataIndexer,
    GeographicResolver,
    CensusSemanticSearcher
)

async def main():
    # Initialize components
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    
    # Create searcher (indexer can be optional for basic usage)
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Process natural language query
    query = "show education information for counties in texas"
    result_gdf, geo_info, selection = await searcher.process_query(query, None)
    
    # Save as GeoJSON
    import json
    geojson_data = json.loads(result_gdf.to_json())
    geojson_data['metadata'] = {
        'source': 'US Census Bureau ACS',
        'features_count': len(result_gdf),
        'selected_variables': selection['selected_variables']
    }
    
    with open('texas_income.geojson', 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"✅ Generated {len(result_gdf)} features with variables: {selection['selected_variables']}")

# Run the example
asyncio.run(main())
```

### Using the Demo Script

The included demo script provides a complete example:

```bash
# With custom boundary (requires GeoJSON file)
python examples/demo.py custom_boundaries/manhattan_geometry_ex.geojson
```

## Core Components

### 1. CensusBigQueryClient (`bigquery_client.py`)

**Purpose**: Interface to Google BigQuery census datasets with optimized query performance.

**Key Features**:
- Direct access to ACS (American Community Survey) tables
- Optimized single-query approach combining census data + geometries
- Support for multiple geographic levels
- Built-in geometry joining and spatial operations
- State-specific tract table handling

**Supported ACS Tables**:
```python
acs_tables = {
    'county_2020_5yr': {
        'description': 'County-level demographic, economic, and housing data from 2020 ACS 5-year estimates',
        'geo_level': 'county',
        'year': 2020
    },
    'zcta_2020_5yr': {
        'description': 'ZIP code tabulation area-level data from 2020 ACS 5-year estimates', 
        'geo_level': 'zcta',
        'year': 2020
    },
    'state_2021_1yr': {
        'description': 'State-level data from 2021 ACS 1-year estimates',
        'geo_level': 'state', 
        'year': 2021
    },
    'censustract_2020_5yr': {
        'description': 'Census tract-level data from 2020 ACS 5-year estimates',
        'geo_level': 'tract',
        'year': 2020
    }
}
```

**Core Methods**:

```python
# Get metadata for all ACS tables
tables_df = client.get_acs_tables_metadata()

# Get column schema for specific table
columns_df = client.get_table_columns('county_2020_5yr')

# Optimized query with geometry (recommended)
gdf = client.query_acs_with_geometry(
    table_name='county_2020_5yr',
    variables=['total_pop', 'median_income'],
    geo_filter="geo_id LIKE '13%'",  # Georgia counties
    geo_level='county',
    state_name='georgia'
)

# Legacy data-only query
df = client.query_acs_data(
    table_name='county_2020_5yr',
    variables=['total_pop', 'median_income'],
    geo_filter="geo_id LIKE '13%'"
)
```

**Special Features**:

- **Brute Force Tract Queries**: When state is unknown, queries all US state tract tables
- **Custom Boundary Integration**: Supports `ST_INTERSECTS` spatial filtering
- **Geometry Source Mapping**: Automatically selects correct BigQuery geometry tables
- **Error Handling**: Graceful fallbacks for missing geometry data

### 2. GeographicResolver (`geo_resolver.py`)

**Purpose**: Resolves geographic references from natural language and builds appropriate spatial filters.

**Core Capabilities**:

1. **State Detection**: Multiple fallback methods for determining state
   - Direct mention in query text
   - GeoJSON file analysis via `StateDetector`
   - LLM interpretation from query intent
   - Brute force all-state search

2. **FIPS Code Management**: Complete state FIPS code mapping
3. **ZIP Code Range Handling**: State-specific ZIP code ranges for ZCTA queries
4. **Custom Boundary Integration**: Seamless integration with custom GeoJSON boundaries

**Geographic Filter Examples**:

```python
geo_resolver = GeographicResolver()

# State-based filtering
filter_info = geo_resolver.build_geo_filter(
    query="show income in texas counties",
    geo_level="county",
    state=["texas"],
    geojson_dir=None
)
# Returns: {'filter_sql': "geo_id LIKE '48%'", 'state_name': 'texas'}

# Custom boundary filtering  
filter_info = geo_resolver.build_geo_filter(
    query="show data in manhattan",
    geo_level="tract", 
    state=[],
    geojson_dir="manhattan_geometry_ex.geojson"
)
# Returns: {'filter_sql': "ST_INTERSECTS(tract_geom, ST_GEOGFROMTEXT('...'))", 'state_name': 'new york'}
```

### 3. CensusSemanticSearcher (`searcher.py`)

**Purpose**: Main orchestration engine that processes natural language queries through the complete pipeline.

**Query Processing Pipeline**:

```python
async def process_query(self, query: str, geojson_dir: str) -> Tuple[gpd.GeoDataFrame, dict, dict]:
    # 1. Analyze query intent using OpenAI LLM
    intent = await self.analyze_query_intent(query)
    
    # 2. Select best ACS table based on geographic level
    selected_table = await self.select_best_table(query, intent)
    
    # 3. Get ALL available columns from selected table
    all_columns = self.get_all_table_columns(selected_table)
    
    # 4. LLM selects most relevant variables
    selection = await self.select_best_columns(query, intent, selected_table, all_columns)
    
    # 5. Build geographic filter
    geo_filter = self.geo_resolver.build_geo_filter(
        query, intent['geo_level'], intent['state'], geojson_dir
    )
    
    # 6. Execute optimized BigQuery with geometry
    gdf = self.bq_client.query_acs_with_geometry(
        selection['selected_table'],
        selection['selected_variables'], 
        geo_filter['filter_sql'],
        intent['geo_level'],
        geo_filter['state_name']
    )
    
    return gdf, geo_filter, selection
```

**Query Intent Analysis**:

Uses OpenAI LLM to extract structured information from natural language:

```python
# Input: "show median household income by census tract in manhattan"
# Output:
{
    "geo_level": "tract",
    "point_of_interest": "manhattan",
    "topics": ["income", "economics"],
    "specific_variables": ["median_income"],
    "year_preference": "latest",
    "aggregation": "none",
    "state": ["new york"]
}
```

**Variable Selection Process**:

1. Retrieves ALL columns from selected ACS table (often 1000+ variables)
2. Presents complete column list to LLM with query context
3. LLM selects most relevant subset based on query intent
4. Validates selected variables exist in table schema
5. Returns final variable list with reasoning

### 4. CustomBoundaryHandler (`custom_boundary_handler.py`)

**Purpose**: Enables queries within user-defined geographic boundaries using GeoJSON files.

**Supported Workflow**:

```python
handler = CustomBoundaryHandler()

# Load custom boundary from GeoJSON
boundary_data = handler.load_boundary("manhattan_geometry_ex.geojson")

# Convert to WKT for BigQuery spatial operations
wkt = handler.geometry_to_wkt(boundary_data['geometry'])

# Build spatial filter for BigQuery
spatial_filter = handler.build_intersect_filter("manhattan_geometry_ex.geojson", "tract")
# Returns: "ST_INTERSECTS(tract_geom, ST_GEOGFROMTEXT('POLYGON((-74.0479 40.6829, ...))'))"
```

**File Location Strategy**:
1. Look in current directory
2. Check `custom_boundaries/` directory
3. Graceful error handling for missing files

### 5. StateDetector (`geojson_state_detect.py`)

**Purpose**: Automatically determines which US state(s) a custom GeoJSON boundary belongs to.

**Detection Methods** (in order of priority):

1. **Property Analysis**: Scans GeoJSON properties for state references
2. **Filename Analysis**: Extracts state from filename patterns
3. **City Reference Mapping**: Maps major cities to states
4. **Coordinate-Based Detection**: Uses centroid coordinates (simplified implementation)

**Comprehensive State and City Mapping**:
- Complete US state names and abbreviations
- 50+ major city-to-state mappings
- Common state name variations and aliases

**Usage Example**:

```python
detector = StateDetector()

# Analyze Manhattan GeoJSON
state = detector.find_state_in_geojson("manhattan_geometry_ex.geojson")
# Returns: "new york"

# The detector checks:
# 1. Properties: {"name": "Manhattan", "borough": "Manhattan"}
# 2. Filename: Contains "manhattan" 
# 3. City mapping: "manhattan" → "new york"
# 4. Coordinates: Centroid within NY bounds
```

### 6. ACSMetadataIndexer (`indexer.py`)

**Purpose**: Builds semantic search index for intelligent census variable discovery (currently optional).

**Key Features**:
- Creates semantic embeddings using SentenceTransformers
- FAISS vector similarity search for variable matching
- Rich variable descriptions with topic keywords
- Caching for performance optimization

**Note**: While implemented, the current pipeline relies more heavily on LLM-based variable selection from complete column lists rather than semantic search. This component provides a foundation for future semantic search enhancements.

## Geographic Levels

The system supports multiple geographic levels of analysis:

### 1. County Level (`county_2020_5yr`)
- **Geographic Coverage**: All US counties (~3,100 entities)
- **Data Granularity**: County-level aggregates
- **Geometry Source**: `bigquery-public-data.geo_us_boundaries.counties`
- **Use Cases**: State-wide analysis, regional comparisons
- **Example**: "show median income by county in california"

### 2. ZIP Code Tabulation Areas - ZCTA (`zcta_2020_5yr`)
- **Geographic Coverage**: ZIP code areas (~33,000 entities)
- **Data Granularity**: ZIP code-level aggregates
- **Geometry Source**: `bigquery-public-data.geo_us_boundaries.zip_codes`
- **Use Cases**: Local market analysis, service area planning
- **Example**: "show demographics by zip code in texas"

### 3. Census Tract (`censustract_2020_5yr`)
- **Geographic Coverage**: Census tracts (~85,000 entities)
- **Data Granularity**: Neighborhood-level (1,200-8,000 people per tract)
- **Geometry Source**: `bigquery-public-data.geo_census_tracts.census_tracts_{state}`
- **Special Handling**: State-specific tables, brute force search when state unknown
- **Use Cases**: Detailed neighborhood analysis, urban planning
- **Example**: "show population by census tract in manhattan"

### 4. State Level (`state_2021_1yr`)
- **Geographic Coverage**: All US states (50 states + DC)
- **Data Granularity**: State-level aggregates
- **Use Cases**: National comparisons, state-level policy analysis
- **Example**: "compare education levels across all states"

## Custom Boundary Support

### Overview

Query census data within any custom geographic boundary by providing GeoJSON files. The system automatically:
1. Detects which state(s) the boundary belongs to
2. Finds all census geographies that intersect with the boundary
3. Returns data only for intersecting areas

### Quick Setup

**Important Note from the Developer** (Everett): The key to custom boundaries working is having a known encompassing state. The system tries multiple methods to determine this:

1. **Direct State Mention**: State name in the query
2. **LLM Analysis**: OpenAI determines the state from query context  
3. **GeoJSON Analysis**: Automatic state detection from the file
4. **Fallback**: Brute force search across all states (performance intensive)

### Adding Custom Boundaries

#### Step 1: Create GeoJSON File

Create a properly formatted GeoJSON file:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-74.0479, 40.6829],
            [-73.9067, 40.6829], 
            [-73.9067, 40.8648],
            [-74.0479, 40.8648],
            [-74.0479, 40.6829]
          ]
        ]
      },
      "properties": {
        "name": "Custom Area",
        "state": "New York"
      }
    }
  ]
}
```

**Requirements**:
- Coordinates in WGS84 format (longitude, latitude)
- Properly closed polygons (first coordinate = last coordinate)
- Valid GeoJSON structure
- Include state information in properties when possible

#### Step 2: Save the File

Save in the `custom_boundaries/` directory:

```
custom_boundaries/
├── manhattan_geometry_ex.geojson
├── brooklyn_boundary.geojson
└── your_custom_area.geojson
```

#### Step 3: Use in Queries

```python
# Query with custom boundary
query = "show median household income by census tract in manhattan"
result_gdf, geo_info, selection = await searcher.process_query(
    query, 
    "custom_boundaries/manhattan_geometry_ex.geojson"
)
```

### How Custom Boundaries Work

1. **Boundary Loading**: GeoJSON loaded and converted to WKT format
2. **State Detection**: Multiple methods determine encompassing state
3. **Spatial Filter**: `ST_INTERSECTS` query built using boundary geometry
4. **Geographic Selection**: Finds all census geographies intersecting the boundary
5. **Data Retrieval**: Queries census data for intersecting geographies only

### Custom Boundary with Different Geographic Levels

```python
# County level: Find counties that intersect with custom boundary
query = "show income by county in seattle metro area"

# ZCTA level: Find ZIP codes that intersect with custom boundary  
query = "show population by zip code in manhattan"

# Tract level: Find census tracts that intersect with custom boundary
query = "show demographics by tract in brooklyn"
```

## Natural Language Query Processing

### Query Intent Analysis

The system uses OpenAI's LLM to analyze natural language queries and extract structured intent:

**Input Query**: `"show median household income by census tract in manhattan"`

**Extracted Intent**:
```json
{
    "geo_level": "tract",
    "point_of_interest": "manhattan", 
    "topics": ["income", "economics"],
    "specific_variables": ["median_income"],
    "year_preference": "latest",
    "aggregation": "none",
    "state": ["new york"]
}
```

### Supported Query Patterns

#### Geographic Level Detection
- **County**: "counties in texas", "county-level data", "by county"
- **ZCTA**: "zip codes", "by zip", "zcta areas"  
- **Tract**: "census tracts", "by tract", "tract-level"
- **State**: "states", "state-level", "across states"

#### Topic Recognition
- **Demographics**: "race", "ethnicity", "population", "age", "gender"
- **Economics**: "income", "poverty", "earnings", "wages", "salary"
- **Housing**: "housing units", "home ownership", "rent", "occupancy"
- **Education**: "education", "degree", "bachelor", "school"
- **Transportation**: "commute", "transportation", "work travel"

#### Variable Selection Intelligence

The LLM receives the complete list of available variables (often 1000+) and selects relevant ones:

**Query**: `"show all commute information for texas counties"`

**LLM Selection Process**:
1. Receives 1000+ column names from ACS table
2. Identifies commute-related variables:
   - `commute_auto_alone`
   - `commute_carpool`  
   - `commute_public_transport`
   - `commute_walk`
   - `commute_bike`
   - `commute_work_from_home`
   - `travel_time_to_work`
3. Returns focused variable set with reasoning

## Usage Examples

### County-Level Analysis

```python
# Basic demographic query
query = "show race demographics for counties in california"
result, geo_info, selection = await searcher.process_query(query, None)

# Economic analysis
query = "median household income by county in florida"
result, geo_info, selection = await searcher.process_query(query, None)

# Comprehensive data request
query = "return all commute related information for new york counties"
result, geo_info, selection = await searcher.process_query(query, None)
```

### ZIP Code (ZCTA) Analysis

```python
# Population demographics
query = "show population demographics for zip codes in delaware"
result, geo_info, selection = await searcher.process_query(query, None)

# Housing analysis
query = "housing units and occupancy rates by zip code in massachusetts"
result, geo_info, selection = await searcher.process_query(query, None)
```

### Census Tract Analysis

```python
# Fine-grained neighborhood analysis
query = "show population by census tracts in rhode island"
result, geo_info, selection = await searcher.process_query(query, None)

# Educational attainment
query = "education levels by tract in colorado"
result, geo_info, selection = await searcher.process_query(query, None)
```

### Custom Boundary Queries

```python
# Manhattan analysis
query = "show median household income by census tract in manhattan"
result, geo_info, selection = await searcher.process_query(
    query, 
    "custom_boundaries/manhattan_geometry_ex.geojson"
)

# Multi-level custom boundary analysis
queries = [
    "show income by county in custom area",      # County intersections
    "show population by zip code in custom area", # ZCTA intersections
    "show demographics by tract in custom area"   # Tract intersections
]
```

### Advanced Query Examples

```python
# Complex demographic analysis
query = "what is the racial breakdown and education attainment for census tracts in brooklyn"

# Multi-topic query
query = "show housing costs, income levels, and commute patterns for zip codes in seattle"

# Comparative analysis
query = "compare median income and poverty rates across counties in texas"

# Age-specific demographics
query = "show age distribution and gender breakdown for census tracts in chicago"
```

## Advanced Features

### Brute Force Tract Querying

When the state is unknown for tract-level queries, the system automatically queries all US state tract tables:

```python
# This query triggers brute force search across all 50 states
query = "show population by census tract in unknown area"

# The system queries all tract tables:
# bigquery-public-data.geo_census_tracts.census_tracts_alabama
# bigquery-public-data.geo_census_tracts.census_tracts_alaska
# ... (all 50 states + DC)
```

**Performance Note**: Brute force queries are slower but ensure complete coverage when state cannot be determined.

### Intelligent Fallback Handling

The system includes multiple fallback mechanisms:

1. **Variable Selection**: Falls back to first 10 columns if LLM selection fails
2. **State Detection**: Multiple methods with final brute force fallback
3. **Table Selection**: Geographic level-based fallback if LLM fails
4. **Boundary Loading**: Multiple file path attempts for custom boundaries

### Metadata Analysis Utility

Use the included utility to explore available census variables:

```bash
python analyze_metadata.py
```

**Output Example**:
```
Found 1247 race-related columns out of 15847 total items

=== GENERAL POPULATION COLUMNS (ending with _pop) ===
- total_pop: Total population
- white_pop: White alone population
- black_pop: Black or African American alone population
- asian_pop: Asian alone population

=== SPECIFIC DEMOGRAPHIC COLUMNS (age/gender breakdowns) ===
- white_male_under_5: White alone male under 5 years
- black_female_65_over: Black alone female 65 years and over
```

## File Structure

```
BigQuery/
├── README.md                           # This comprehensive documentation
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup (minimal)
├── analyze_metadata.py                # Census variable exploration utility
│
├── census_semantic_search/            # Main package
│   ├── __init__.py                   # Package exports and version
│   ├── bigquery_client.py            # BigQuery interface and optimization
│   ├── indexer.py                    # Semantic search indexing (FAISS)
│   ├── geo_resolver.py               # Geographic resolution and filtering
│   ├── searcher.py                   # Main query orchestration engine
│   ├── custom_boundary_handler.py    # Custom GeoJSON boundary support
│   └── geojson_state_detect.py       # Automatic state detection
│
├── custom_boundaries/                 # Custom GeoJSON boundaries
│   ├── README.md                     # Boundary format documentation
│   └── manhattan_geometry_ex.geojson # Example Manhattan boundary
│
├── examples/                         # Usage examples and tests
│   ├── demo.py                       # Complete demonstration script
│   ├── test_auth.py                  # BigQuery authentication test
│   └── test_tract.py                 # Census tract functionality test
│
├── data/                             # Cached metadata and indices
│   ├── acs_metadata.parquet          # ACS variable metadata cache
│   ├── acs_tables.index              # FAISS semantic search index
│   └── fips_codes.json               # State FIPS code mappings
│
├── geometry_cache/                   # Cached geometry files (auto-generated)
│   ├── counties_*.geojson            # State county boundaries
│   ├── zcta_*.geojson               # ZIP code boundaries
│   └── tracts_*.geojson             # Census tract boundaries
│
└── results/                          # Query result outputs
    └── *.geojson                     # Generated GeoJSON files
```

## Output Format

Results are returned as GeoDataFrames and saved as GeoJSON with comprehensive metadata:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[...]]]
      },
      "properties": {
        "geo_id": "36061",
        "total_pop": 1628701,
        "median_income": 64993,
        "white_pop": 1074529,
        "black_pop": 310473,
        "county": "New York County",
        "state": "New York"
      }
    }
  ],
  "metadata": {
    "source": "US Census Bureau ACS",
    "geometry_source": "BigQuery geo_us_boundaries", 
    "features_count": 1,
    "selected_variables": ["total_pop", "median_income", "white_pop", "black_pop"],
    "custom_boundary": "manhattan_geometry_ex.geojson",
    "geo_level": "county",
    "query": "show demographics in manhattan",
    "generated_at": "2024-01-15T10:30:00Z"
  }
}
```

## Performance Optimization

### Single-Query Architecture

The system uses an optimized approach that combines census data retrieval with geometry fetching in a single BigQuery call:

**Traditional Approach** (slower):
```
Query 1: Get census data → DataFrame
Query 2: Get geometries → GeoDataFrame  
Step 3: Join data + geometries → Final GeoDataFrame
```

**Optimized Approach** (faster):
```
Query 1: Combined census data + geometries → GeoDataFrame (ready to use)
```

### Caching Strategy

1. **Metadata Caching**: ACS table and column information cached locally
2. **Geometry Caching**: Boundary data cached to avoid repeated BigQuery calls
3. **Index Caching**: FAISS semantic search index built once and reused

### Query Optimization Techniques

1. **Selective Column Loading**: Only queries necessary census variables
2. **Spatial Indexing**: Uses BigQuery's spatial functions efficiently
3. **State-Specific Optimization**: Queries only relevant geographic tables
4. **Batch Processing**: Combines multiple operations into single queries

### Best Practices

```python
# ✅ Good: Build index once, reuse
indexer = ACSMetadataIndexer()
if not indexer.load_index():
    indexer.build_index(bq_client)  # Only run once

# ✅ Good: Use appropriate geographic level
query = "show income by county in texas"  # County level for state analysis
query = "show income by tract in manhattan"  # Tract level for neighborhood analysis

# ✅ Good: Simplify custom boundaries
# Use simplified GeoJSON geometries for better performance

# ❌ Avoid: Rebuilding index repeatedly
# ❌ Avoid: Using tract level for very large areas
# ❌ Avoid: Complex custom boundaries for large-scale analysis
```

## Troubleshooting

### Authentication Issues

```bash
# Test BigQuery authentication
python examples/test_auth.py

# Expected output:
# ✅ BigQuery authentication successful!
# Found 5 tables
```

**Common Solutions**:
```bash
# Reinstall gcloud and authenticate
gcloud auth application-default login

# Check service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Verify project permissions
gcloud projects list
```

### OpenAI API Issues

```python
# Test OpenAI connection
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "test"}]
)
print("✅ OpenAI API working")
```

### Query Processing Issues

**No Results Returned**:
```python
# Debug geographic filter
geo_filter = geo_resolver.build_geo_filter(query, geo_level, state, geojson_dir)
print(f"Filter SQL: {geo_filter['filter_sql']}")
print(f"State detected: {geo_filter['state_name']}")

# Check if state detection worked
if geo_filter['state_name'] == 'brute':
    print("⚠️ State unknown - using brute force search")
```

**Custom Boundary Issues**:
```python
# Test boundary loading
handler = CustomBoundaryHandler()
boundary = handler.load_boundary("your_boundary.geojson")
if boundary:
    print("✅ Boundary loaded successfully")
    wkt = handler.geometry_to_wkt(boundary['geometry'])
    print(f"WKT length: {len(wkt)} characters")
else:
    print("❌ Failed to load boundary")
    print("Check file path and GeoJSON format")
```

**Variable Selection Issues**:
```python
# Debug column retrieval
columns = searcher.get_all_table_columns("county_2020_5yr") 
print(f"Found {len(columns)} columns")
print("First 5 columns:", [col['name'] for col in columns[:5]])
```

### Performance Issues

**Slow Queries**:
1. Check if using appropriate geographic level (county vs tract)
2. Simplify custom boundary geometries
3. Clear stale caches: `rm -rf geometry_cache/ data/`
4. Use state filters instead of brute force when possible

**Memory Issues**:
1. Reduce batch size in semantic indexing
2. Limit query scope (smaller geographic areas)
3. Use county level instead of tract for large areas

### Data Quality Issues

**Missing Geometry Data**:
- Some census tracts may not have geometry in BigQuery
- ZCTA coverage varies by state and year
- Custom boundaries may not intersect any census geographies

**Unexpected Results**:
- Verify state names are spelled correctly
- Check that custom boundaries use correct coordinate system (WGS84)
- Validate GeoJSON format using tools like [geojson.io](https://geojson.io)

## Contributing

### Adding New Geographic Levels

1. **Update ACS Table Configuration**:
```python
# In bigquery_client.py
acs_tables['blockgroup_2020_5yr'] = {
    'description': 'Block group-level data from 2020 ACS 5-year estimates',
    'geo_level': 'blockgroup',
    'year': 2020
}
```

2. **Add Table Selection Logic**:
```python
# In searcher.py select_best_table()
elif geo_level == 'blockgroup':
    return 'blockgroup_2020_5yr'
```

3. **Implement Geometry Mapping**:
```python
# In bigquery_client.py query_acs_with_geometry()
elif geo_level == 'blockgroup':
    geo_table = 'bigquery-public-data.geo_us_boundaries.block_groups'
    geo_id_field = 'geo_id'
    geom_field = 'blockgroup_geom'
```

### Extending Custom Boundary Support

1. **Add New Boundary Types**:
```python
# In custom_boundary_handler.py
def load_boundary(self, boundary_identifier):
    # Add support for different boundary sources
    # e.g., database lookups, API calls, etc.
```

2. **Enhance State Detection**:
```python
# In geojson_state_detect.py
# Add more city mappings, coordinate bounds, etc.
```

### Adding New Data Sources

The architecture supports extension to other data sources beyond ACS:

```python
# Example: Add NHGIS data support
class NHGISBigQueryClient(CensusBigQueryClient):
    def __init__(self):
        super().__init__()
        self.nhgis_tables = {
            # Define NHGIS table structure
        }
```

## Data Sources & Acknowledgments

### Primary Data Sources

- **Census Data**: US Census Bureau American Community Survey (ACS) 
  - Accessed via Google BigQuery Public Datasets
  - Tables: `bigquery-public-data.census_bureau_acs.*`
  - Coverage: Counties, ZCTAs, Census Tracts, States

- **Geometry Data**: US Geographic Boundaries
  - Counties: `bigquery-public-data.geo_us_boundaries.counties`
  - ZIP Codes: `bigquery-public-data.geo_us_boundaries.zip_codes`  
  - Census Tracts: `bigquery-public-data.geo_census_tracts.census_tracts_{state}`

- **Custom Boundaries**: User-provided GeoJSON files

### Technologies

- **Google BigQuery**: Data warehouse and spatial query engine
- **OpenAI GPT**: Natural language understanding and variable selection
- **Sentence Transformers**: Semantic embeddings for variable matching
- **FAISS**: Vector similarity search
- **GeoPandas**: Geospatial data manipulation
- **Shapely**: Geometric operations

### Acknowledgments

- US Census Bureau for comprehensive demographic data collection
- Google Cloud for BigQuery public datasets and spatial functions
- OpenAI for advanced language model capabilities
- Open source geospatial community for tools and libraries

## License

This project is available under the MIT License. See LICENSE file for details.

---

## Developer Notes

**From Everett**: This system emphasizes practical functionality over theoretical completeness. The key insight is that having a known encompassing state is crucial for performance - the system tries multiple methods to determine this automatically, but users can help by being specific about locations in their queries.

The LLM-based approach for variable selection works well because census tables often have 1000+ variables, and semantic matching alone isn't sufficient to understand user intent. The combination of comprehensive column presentation + LLM reasoning provides more accurate results than pure semantic search.

For maximum performance, stick to county-level analysis for large geographic areas and use tract-level only when you need neighborhood-level detail.