import asyncio
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from census_semantic_search import (
    CensusBigQueryClient,
    ACSMetadataIndexer,
    GeographicResolver,
    CensusSemanticSearcher
)

from dotenv import load_dotenv

load_dotenv()

async def main(geojson_file=None):
    # Check if file exists (if provided)
    if geojson_file and not os.path.exists(geojson_file):
        print(f"❌ Error: File not found: {geojson_file}")
        return

    # Initialize components
    print("🚀 Initializing Census Semantic Search...")
    
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    
    # Create searcher
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Example queries
    queries = [
        "Get me median income by census tract in Manhattan",
        #"Show median income by county in california", 
        #"Show education levels in harris county texas"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            # Process query - now auto-extracts boundaries if no file provided
            result, selection = await searcher.process_query(query, geojson_file)
            
            # Check if we got a GeoDataFrame
            if hasattr(result, 'geometry'):
                print("✅ Retrieved data with geometries!")
                
                # Save as GeoJSON
                output_file = f"results/{query[:25].replace(' ', '_')}.geojson"
                
                # Convert to GeoJSON format
                geojson = json.loads(result.to_json())
                geojson['metadata'] = {
                    'source': 'US Census Bureau ACS',
                    'geometry_source': 'BigQuery geo_us_boundaries + Auto-extracted boundary',
                    'features_count': len(result),
                    'selected_variables': selection['selected_variables']
                }
                
                with open(output_file, 'w') as f:
                    json.dump(geojson, f, indent=2)
                
                print(f"\n✅ Success! GeoJSON saved to: {output_file}")
                print(f"   Features: {len(result)}")
                print(f"   Variables: {selection['selected_variables']}")
            else:
                print("⚠️ Query failed")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    geojson_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if geojson_file:
        print(f"Using provided boundary file: {geojson_file}")
    else:
        print("No boundary file provided - will auto-extract from queries")
    
    asyncio.run(main(geojson_file))