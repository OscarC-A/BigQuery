# Census Semantic Search

A comprehensive Python library for semantic search and visualization of US Census data using Google BigQuery and natural language queries. This library enables users to query census data using plain English and automatically generates map-ready GeoJSON files with both census data and geographic boundaries.

## Custom Boundary Support

### Overview

The system supports custom geographic boundaries through GeoJSON files, enabling queries within user-defined areas like boroughs, neighborhoods, or custom regions.

### Adding Custom Boundaries

If your name is Everett just read this, all the 'steps' below is claude bs. All you need to know is that having a known larger encompassing state is integral to any of this working in any scenario. This all happens in geo_resolver.py in the build_geo_filter function, so look if you want to understand whats going on more closely. When using a custom boundary, it first looks to see if there was a name of a state in the natural language query, if there was, boom done. If not, it falls back to to the chat api response in searcher.py, analyze_query_intent function. There it returns what it believes to be the encompassing state or states in a list. Right now we only support one state (so the first state returned in the list) but can be expanded later and shouldnt be too much work to do so. Finally if all that fails, then it falls back to hard coded solutions, which is mainly what is outlined below. Info on the rest of the project is found below all the custom boundary stuff.

#### Step 1: Prepare Your GeoJSON File

Create a properly formatted GeoJSON file with your custom boundary:

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

**Important**: 
- Use `"type": "Polygon"` for simple boundaries
- Use `"type": "MultiPolygon"` for complex boundaries with multiple parts
- Ensure coordinates are in WGS84 (longitude, latitude) format
- Close polygons by repeating the first coordinate as the last coordinate

#### Step 2: Save the GeoJSON File

Save your GeoJSON file in the `custom_boundaries/` directory using the naming convention:

```
custom_boundaries/{boundary_name}.geojson
```

Examples:
- `custom_boundaries/manhattan.geojson`
- `custom_boundaries/lower_manhattan.geojson`
- `custom_boundaries/downtown_seattle.geojson`

#### Step 3: Update CustomBoundaryHandler (This step shouldn't be necessary but revisit if encountering any issues)

Edit `census_semantic_search/custom_boundary_handler.py`:

1. **Add to Known Boundaries List**:
```python
def extract_boundary_from_query(self, query: str) -> Optional[str]:
    """Extract custom boundary names from query"""
    query_lower = query.lower()
    
    # Add your custom boundary to this list
    known_boundaries = [
        'manhattan', 'brooklyn', 'queens', 'bronx', 'staten island',
        'lower_manhattan', 'downtown_seattle'  # Add your boundaries here
    ]
    
    for boundary in known_boundaries:
        if boundary in query_lower:
            return boundary
    
    return None
```

2. **Add State Mapping**:
```python
def get_state_for_boundary(self, boundary_name: str) -> Optional[str]:
    """Get the state name for a custom boundary"""
    boundary_to_state = {
        'manhattan': 'new york',
        'brooklyn': 'new york',
        'queens': 'new york',
        'bronx': 'new york',
        'staten island': 'new york',
        'lower_manhattan': 'new york',  # Add your boundary-to-state mapping
        'downtown_seattle': 'washington'
    }
    return boundary_to_state.get(boundary_name.lower())
```

#### Step 4: Test Your Custom Boundary

```python
# Test your custom boundary
query = "show median household income in lower manhattan"
result, geo_info, selection = await searcher.process_query(query)

# Verify the boundary was recognized
print(f"Custom boundary used: {geo_info.get('custom_boundary')}")
print(f"Features found: {len(result)}")
```

### Advanced Custom Boundary Configuration

#### Supporting Different Geographic Levels

Custom boundaries work with all geographic levels (county, ZCTA, tract). The system automatically:

1. **For County Level**: Finds counties that intersect with your custom boundary
2. **For ZCTA Level**: Finds ZIP codes that intersect with your custom boundary  
3. **For Tract Level**: Finds census tracts that intersect with your custom boundary

#### Example: Adding a Metropolitan Area

```python
# 1. Save as custom_boundaries/seattle_metro.geojson
# 2. Update CustomBoundaryHandler:

known_boundaries = [
    'manhattan', 'brooklyn', 'queens', 'bronx', 'staten island',
    'seattle_metro'  # Add here
]

boundary_to_state = {
    'manhattan': 'new york',
    'brooklyn': 'new york', 
    'queens': 'new york',
    'bronx': 'new york',
    'staten island': 'new york',
    'seattle_metro': 'washington'  # Add mapping
}

# 3. Test with different geographic levels:
queries = [
    "show income by county in seattle metro",      # County level
    "show population by zip code in seattle metro", # ZCTA level
    "show demographics by tract in seattle metro"   # Tract level
]
```

### How Custom Boundaries Work Internally

1. **Query Analysis**: The system detects custom boundary names in natural language queries
2. **Boundary Loading**: GeoJSON files are loaded and converted to WKT format
3. **Spatial Filter**: `ST_INTERSECTS` queries are built using the custom boundary geometry
4. **BigQuery Execution**: The spatial filter is applied to find census geographies that intersect with the custom boundary
5. **Data Merging**: Census data is joined with the intersecting geographies

### Troubleshooting Custom Boundaries

#### Common Issues and Solutions

1. **Boundary Not Recognized**:
   - Ensure the boundary name is added to `known_boundaries` list
   - Check that the query contains the exact boundary name
   - Verify file naming: use lowercase with underscores

2. **No Results Returned**:
   - Verify the GeoJSON coordinates are in the correct format (longitude, latitude)
   - Check that the boundary overlaps with the target geographic level
   - Ensure polygons are properly closed

3. **Invalid Geometry Error**:
   - Validate your GeoJSON using tools like [geojson.io](https://geojson.io)
   - Ensure polygons don't self-intersect
   - Check coordinate precision (too many decimal places can cause issues)

4. **Performance Issues**:
   - Simplify complex geometries for better performance
   - Consider using bounding boxes for very large areas
   - Cache frequently used boundaries

#### Example Debugging Session

```python
# Test boundary loading
from census_semantic_search import CustomBoundaryHandler

handler = CustomBoundaryHandler()

# Test if boundary can be loaded
boundary_data = handler.load_boundary('your_boundary_name')
if boundary_data:
    print("✅ Boundary loaded successfully")
    print(f"Geometry type: {boundary_data['geometry']['type']}")
else:
    print("❌ Failed to load boundary - check file path and format")

# Test WKT conversion
if boundary_data:
    wkt = handler.geometry_to_wkt(boundary_data['geometry'])
    print(f"WKT preview: {wkt[:100]}...")

# Test spatial filter generation
spatial_filter = handler.build_intersect_filter('your_boundary_name', 'county')
print(f"Spatial filter: {spatial_filter}")
```

## Output Format

Results are returned as GeoDataFrames and can be saved as GeoJSON files with the following structure:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      },
      "properties": {
        "geo_id": "48001",
        "total_pop": 4652980,
        "median_income": 64034,
        "white_pop": 2480394,
        "black_pop": 754990
      }
    }
  ],
  "metadata": {
    "source": "US Census Bureau ACS",
    "geometry_source": "BigQuery geo_us_boundaries",
    "features_count": 254,
    "custom_boundary": "manhattan",
    "selected_variables": ["total_pop", "median_income", "white_pop", "black_pop"]
  }
}
```

## Features

- **Natural Language Queries**: Ask questions about census data in plain English
- **Semantic Search**: Uses sentence transformers and FAISS for intelligent variable matching
- **Multi-Level Geographic Support**: County, ZIP Code Tabulation Area (ZCTA), and Census Tract level queries
- **Custom Boundary Support**: Query data within custom geographic boundaries (e.g., Manhattan, Brooklyn)
- **BigQuery Integration**: Queries US Census data from Google BigQuery public datasets
- **Optimized Performance**: Single-query approach combining census data with geometries
- **GeoJSON Output**: Generates map-ready GeoJSON files with census data and geometries
- **Caching**: Intelligent geometry caching for improved performance

## Supported Geographic Levels

- **County**: County-level demographic, economic, and housing data
- **ZCTA (ZIP Code Tabulation Areas)**: ZIP code-level census data
- **Census Tract**: Fine-grained neighborhood-level data
- **Custom Boundaries**: User-defined geographic areas (Manhattan, Brooklyn, etc.)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `google-cloud-bigquery>=3.13.0` - BigQuery client
- `sentence-transformers>=2.7.0` - Semantic search
- `faiss-cpu>=1.8.0` - Vector similarity search
- `pandas>=2.1.3` - Data manipulation
- `geopandas>=1.0.1` - Geospatial data handling
- `shapely>=2.0.2` - Geometric operations
- `openai>=1.3.0` - LLM-powered query analysis
- `python-dotenv>=1.0.0` - Environment configuration

## Setup

### 1. Google Cloud Authentication

Set up Google Cloud authentication for BigQuery access:

```bash
# Install Google Cloud SDK and authenticate
gcloud auth application-default login

# Or set up service account
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

### 2. OpenAI API Key

Configure OpenAI API for query analysis:

```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 3. Optional: Google Cloud Project ID

```bash
# Add to .env file if using a specific project
echo "GCP_PROJECT_ID=your_project_id" >> .env
```

## Quick Start

```python
import asyncio
from census_semantic_search import (
    CensusBigQueryClient,
    ACSMetadataIndexer,
    GeographicResolver,
    CensusSemanticSearcher,
    GeometryFetcher
)

async def main():
    # Initialize components
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    
    # Build or load semantic search index
    if not indexer.load_index():
        indexer.build_index(bq_client)
    
    # Create searcher
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Process natural language query
    query = "show median household income for counties in texas"
    result_gdf, geo_info, selection = await searcher.process_query(query)
    
    # Save as GeoJSON
    import json
    with open('texas_income.geojson', 'w') as f:
        json.dump(json.loads(result_gdf.to_json()), f, indent=2)

asyncio.run(main())
```

## Architecture Overview

### Core Components

#### 1. CensusBigQueryClient (`bigquery_client.py`)
- **Purpose**: Interface to Google BigQuery census datasets
- **Key Methods**:
  - `get_acs_tables_metadata()`: Retrieves metadata for all ACS tables
  - `get_table_columns()`: Gets column information for specific tables
  - `query_acs_with_geometry()`: Optimized single-query approach combining census data with geometries
  - `query_acs_data()`: Legacy method for census data only
- **Supported Tables**: `county_2020_5yr`, `zcta_2020_5yr`, `censustract_2020_5yr`, `state_2021_1yr`

#### 2. ACSMetadataIndexer (`indexer.py`)
- **Purpose**: Builds semantic search index for census variables using FAISS
- **Features**:
  - Creates semantic embeddings for census variables
  - Enables intelligent variable matching from natural language queries
  - Caches index for improved performance
- **Note**: Currently focused on county-level data, extensible to other levels

#### 3. GeographicResolver (`geo_resolver.py`)
- **Purpose**: Resolves geographic references and builds spatial filters
- **Capabilities**:
  - State name to FIPS code mapping
  - ZIP code range management for ZCTA queries
  - Custom boundary integration via `CustomBoundaryHandler`
  - Builds appropriate SQL filters for different geographic levels

#### 4. CensusSemanticSearcher (`searcher.py`)
- **Purpose**: Main search engine orchestrating all components
- **Pipeline**:
  1. Query intent analysis using OpenAI LLM
  2. Automatic ACS table selection based on geographic level
  3. Column extraction and LLM-based variable selection
  4. Geographic filter construction
  5. Optimized BigQuery execution with geometry joining

#### 5. GeometryFetcher (`geometry_fetcher.py`)
- **Purpose**: Retrieves and caches geographic boundaries
- **Data Sources**:
  - Counties: `bigquery-public-data.geo_us_boundaries.counties`
  - ZCTA: `bigquery-public-data.geo_us_boundaries.zip_codes`
  - Census Tracts: `bigquery-public-data.geo_census_tracts.census_tracts_{state}`
- **Features**: Intelligent geometry caching, fallback handling

#### 6. CustomBoundaryHandler (`custom_boundary_handler.py`)
- **Purpose**: Handles custom geographic boundaries from GeoJSON files
- **Features**:
  - Loads custom boundaries from `custom_boundaries/` directory
  - Converts GeoJSON to WKT for BigQuery spatial operations
  - Builds `ST_INTERSECTS` filters for spatial queries

## Usage Examples

### County-Level Queries

```python
# Basic county query
query = "show median household income for counties in california"
result, geo_info, selection = await searcher.process_query(query)

# Race demographics
query = "return race demographic information for new york counties"
result, geo_info, selection = await searcher.process_query(query)

# Education levels
query = "what is education attainment in florida counties"
result, geo_info, selection = await searcher.process_query(query)
```

### ZCTA (ZIP Code) Queries

```python
# ZIP code level analysis
query = "show population demographics for zip codes in delaware"
result, geo_info, selection = await searcher.process_query(query)

# Income by ZIP
query = "median household income by zip code in rhode island"
result, geo_info, selection = await searcher.process_query(query)
```

### Census Tract Queries

```python
# Fine-grained neighborhood analysis
query = "show population by census tracts in rhode island"
result, geo_info, selection = await searcher.process_query(query)

# Tract-level demographics
query = "race demographics for census tracts in connecticut"
result, geo_info, selection = await searcher.process_query(query)
```

### Custom Boundary Queries

```python
# Manhattan-specific analysis
query = "show median household income by county in manhattan"
result, geo_info, selection = await searcher.process_query(query)

# Brooklyn demographics
query = "return race demographics for brooklyn"
result, geo_info, selection = await searcher.process_query(query)
```

## File Structure

```
census-semantic-search/
├── README.md
├── requirements.txt
├── setup.py
├── analyze_metadata.py                 # Utility for analyzing ACS metadata
├── census_semantic_search/            # Main package
│   ├── __init__.py                   # Package exports
│   ├── bigquery_client.py            # BigQuery interface
│   ├── indexer.py                    # Semantic search indexing
│   ├── geo_resolver.py               # Geographic resolution
│   ├── searcher.py                   # Main search orchestration
│   ├── geometry_fetcher.py           # Geometry retrieval and caching
│   └── custom_boundary_handler.py    # Custom boundary support
├── custom_boundaries/                 # Custom GeoJSON boundaries
│   ├── README.md                     # Boundary documentation
│   └── manhattan_geometry_ex.geojson # Example Manhattan boundary
├── examples/                         # Usage examples
│   ├── demo.py                       # Main demonstration script
│   ├── test_auth.py                  # BigQuery authentication test
│   └── test_tract.py                 # Census tract functionality test
├── data/                             # Cached metadata and indices
│   ├── acs_metadata.parquet          # ACS variable metadata
│   ├── acs_tables.index              # FAISS search index
│   └── fips_codes.json               # State FIPS code mappings
├── geometry_cache/                   # Cached geometry files
│   ├── counties_*.geojson            # State county boundaries
│   ├── zcta_*.geojson               # ZIP code boundaries
│   └── tracts_*.geojson             # Census tract boundaries
└── results/                          # Query result outputs
    └── *.geojson                     # Generated GeoJSON files
```

## Query Types and Examples

### Demographic Queries
- `"show race demographics for counties in california"`
- `"return population by ethnicity for zip codes in texas"`
- `"what is the age distribution in new york census tracts"`

### Economic Queries
- `"median household income by county in florida"`
- `"show poverty rates for census tracts in chicago"`
- `"income distribution by zip code in washington state"`

### Housing Queries
- `"housing units and occupancy rates in georgia counties"`
- `"home ownership vs rental rates by tract in massachusetts"`
- `"housing costs by zip code in california"`

### Transportation/Commute Queries
- `"commute patterns for workers in texas counties"`
- `"transportation methods by census tract in new york"`
- `"show all commute-related data for florida zip codes"`

### Education Queries
- `"education attainment levels by county in oregon"`
- `"college graduation rates by tract in north carolina"`
- `"educational demographics for zip codes in colorado"`

### Custom Boundary Queries
- `"show demographics for manhattan"`
- `"income levels in brooklyn by census tract"`
- `"housing data for queens zip codes"`

## Performance Optimization

### Caching Strategy
- **Geometry Caching**: Boundaries are cached locally to avoid repeated BigQuery calls
- **Metadata Caching**: ACS table and column metadata is cached for faster query processing
- **Index Caching**: FAISS semantic search index is built once and cached

### Query Optimization
- **Single Query Approach**: Combines census data and geometry retrieval in one BigQuery call
- **Selective Column Loading**: Only queries necessary census variables
- **Spatial Indexing**: Uses BigQuery's spatial functions for efficient geographic filtering

### Best Practices
1. **Build Index Once**: Run `indexer.build_index()` once and reuse the cached index
2. **Use Appropriate Geographic Levels**: Choose the right level (county/ZCTA/tract) for your analysis
3. **Simplify Custom Boundaries**: Use simplified geometries for better performance
4. **Batch Similar Queries**: Process multiple similar queries together when possible

## Troubleshooting

### Authentication Issues
- Verify Google Cloud SDK installation and authentication
- Check `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Ensure BigQuery API is enabled in your Google Cloud project

### Query Failures
- Verify OpenAI API key is set correctly
- Check that state names are spelled correctly in queries
- Ensure custom boundary files are properly formatted GeoJSON

### Performance Issues
- Clear geometry cache if experiencing stale data: `rm -rf geometry_cache/`
- Rebuild semantic index if getting poor variable matches: delete `data/` folder
- Simplify complex custom boundaries

### Data Issues
- Some census tracts may not have geometry data available
- ZCTA coverage varies by state and year
- Custom boundaries may not intersect with any census geographies

## Contributing

### Adding New Geographic Levels
1. Update `acs_tables` dictionary in `CensusSemanticSearcher`
2. Add table mapping in `query_acs_with_geometry()` method
3. Implement geometry fetching logic in `GeometryFetcher`
4. Update geographic level detection in query analysis

### Extending Custom Boundary Support
1. Add new boundary detection logic in `CustomBoundaryHandler`
2. Update state mapping for new regions
3. Add corresponding GeoJSON files to `custom_boundaries/`
4. Update documentation with new examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Data Sources

- **Census Data**: US Census Bureau American Community Survey (ACS) via Google BigQuery Public Datasets
- **Geometry Data**: US Geographic Boundaries from Google BigQuery Public Datasets (`geo_us_boundaries`, `geo_census_tracts`)
- **Custom Boundaries**: User-provided GeoJSON files

## Acknowledgments

- Google BigQuery for providing public census datasets
- US Census Bureau for the American Community Survey data
- OpenAI for LLM-powered query analysis
- Sentence Transformers for semantic search capabilities