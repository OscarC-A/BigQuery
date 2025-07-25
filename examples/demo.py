import asyncio
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from census_semantic_search import (
    CensusBigQueryClient,
    ACSMetadataIndexer,
    GeographicResolver,
    CensusSemanticSearcher,
    GeometryFetcher
)

from dotenv import load_dotenv

load_dotenv()

async def main():
    # Initialize components
    print("🚀 Initializing Census Semantic Search...")
    
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    geometry_fetcher = GeometryFetcher()
    
    # Build or load index
    if not indexer.load_index():
        print("Building index from scratch...")
        indexer.build_index(bq_client)
    
    # Create searcher
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Example queries
    queries = [
        "return all commute related information for new york counties"
        #"show median household income for counties in texas"
        #"what is the education level in california counties"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            # Process query
            census_data, geo_info, selection = await searcher.process_query(query)
            
            # Fetch geometries
            print("\n📍 Fetching geometries...")
            geometries = await geometry_fetcher.fetch_county_geometries(
                geo_info['state_fips']
            )
            
            # Merge into GeoJSON
            print("🗺️ Creating GeoJSON...")
            geojson = geometry_fetcher.merge_data_with_geometry(
                census_data, 
                geometries
            )
            
            # Save result
            output_file = f"results/{query[:25].replace(' ', '_')}.geojson"
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            print(f"\n✅ Success! GeoJSON saved to: {output_file}")
            print(f"   Features: {geojson['metadata']['features_count']}")
            print(f"   Variables: {selection['selected_variables']}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())