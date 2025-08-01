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

async def main(geojson_file):
    # Check if file exists
    if not os.path.exists(geojson_file):
        print(f"❌ Error: File not found: {geojson_file}")
        return

    # Initialize components
    print("🚀 Initializing Census Semantic Search...")
    
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    
    # Build or load index
    # if not indexer.load_index():
    #     print("Building index from scratch...")
    #     indexer.build_index(bq_client)
    
    # Create searcher
    searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
    
    # Example queries
    queries = [
        #"return all commute related information for new york counties"
        "Show black population by census tract in manhattan"
        #"what is the education level in california counties"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            # Process query - now returns GeoDataFrame or regular DataFrame
            result, geo_info, selection = await searcher.process_query(query, geojson_file)
            
            # Check if we got a GeoDataFrame (combined query succeeded)
            if hasattr(result, 'geometry'):
                print("✅ Retrieved data with geometries in single query!")
                
                # Save as GeoJSON directly
                output_file = f"results/{query[:25].replace(' ', '_')}.geojson"
                
                # Convert to GeoJSON format
                geojson = json.loads(result.to_json())
                geojson['metadata'] = {
                    'source': 'US Census Bureau ACS',
                    'geometry_source': 'BigQuery geo_us_boundaries',
                    'features_count': len(result),
                    'selected_variables': selection['selected_variables']
                }
                
                with open(output_file, 'w') as f:
                    json.dump(geojson, f, indent=2)
                
                print(f"\n✅ Success! GeoJSON saved to: {output_file}")
                print(f"   Features: {len(result)}")
                print(f"   Variables: {selection['selected_variables']}")
            else:
                # Fallback: separate geometry (old approach)
                print("⚠️ Combined query failed")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo.py <geojson_file>")
        print("Example: python demo.py manhattan.geojson")
        sys.exit(1)
    
    asyncio.run(main(sys.argv[1]))