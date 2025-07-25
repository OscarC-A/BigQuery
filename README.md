# Census Semantic Search

A Python library for semantic search and visualization of US Census data using Google BigQuery and natural language queries.

## Note

Currently only functional for county level data retrivial from the ACS.

## Features

- **Natural Language Queries**: Ask questions about census data in plain English
- **Semantic Search**: Uses sentence transformers and FAISS for intelligent variable matching
- **Geographic Resolution**: Automatically resolves geographic references (states, counties)
- **BigQuery Integration**: Queries US Census data from Google BigQuery public datasets
- **GeoJSON Output**: Generates map-ready GeoJSON files with census data and geometries

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Set up Google Cloud authentication for BigQuery access
2. Create a `.env` file with your configuration:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

## Usage

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
    geometry_fetcher = GeometryFetcher()
    
    # Build or load index
    if not indexer.load_index():
        indexer.build_index(bq_client)
    
    # Create searcher
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Process query
    query = "show median household income for counties in texas"
    census_data, geo_info, selection = await searcher.process_query(query)
    
    # Fetch geometries and create GeoJSON
    geometries = await geometry_fetcher.fetch_county_geometries(geo_info['state_fips'])
    geojson = geometry_fetcher.merge_data_with_geometry(census_data, geometries)

asyncio.run(main())
```

## Example Queries

- "show median household income for counties in texas"
- "return race demographic information for new york counties"
- "what is the education level in california counties"
- "find commute-related data for florida counties"

## Components

- **CensusBigQueryClient**: Interface to Google BigQuery census datasets
- **ACSMetadataIndexer**: Builds semantic search index for census variables
- **GeographicResolver**: Resolves location names to FIPS codes
- **CensusSemanticSearcher**: Main search engine combining all components
- **GeometryFetcher**: Retrieves and caches county boundary geometries

## Output

Results are saved as GeoJSON files in the `results/` directory, ready for visualization in mapping applications.