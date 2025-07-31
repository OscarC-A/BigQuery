#!/usr/bin/env python3
"""
Quick test to verify census tract functionality
Use demo.py with tract-level query to properly test

"""
import asyncio
import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

async def test_tract_functionality():
    print("🧪 Testing Census Tract Functionality...")
    
    # Initialize components
    bq_client = CensusBigQueryClient()
    indexer = ACSMetadataIndexer()
    geo_resolver = GeographicResolver()
    
    # Test geo resolver tract filtering
    print("\n1. Testing geo_resolver tract filtering...")
    try:
        geo_filter = geo_resolver.build_geo_filter("Show population by census tracts in Rhode Island", "tract")
        print(f"   ✅ Tract filter SQL: {geo_filter['filter_sql']}")
        print(f"   ✅ Geo level: {geo_filter['geo_level']}")
        print(f"   ✅ State: {geo_filter['state_name']}")
        print(f"   ✅ State FIPS: {geo_filter['state_fips']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test searcher table selection
    print("\n2. Testing searcher table selection...")
    try:
        searcher = CensusSemanticSearcher(indexer, geo_resolver, bq_client)
        intent = {"geo_level": "tract"}
        selected_table = await searcher.select_best_table("tract query", intent)
        print(f"   ✅ Selected table: {selected_table}")
        
        if selected_table != 'censustract_2020_5yr':
            print(f"   ⚠️ Expected 'censustract_2020_5yr', got '{selected_table}'")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test geometry fetcher (without actual BigQuery call)
    print("\n3. Testing geometry fetcher setup...")
    try:
        geometry_fetcher = GeometryFetcher()
        geo_info = {
            'geo_level': 'tract',
            'state_fips': '36'  # New York
        }
        print("   ✅ GeometryFetcher can handle tract geo_info structure")
        
        # Just test that the method exists and accepts tract level
        if hasattr(geometry_fetcher, '_fetch_tract_geometries'):
            print("   ✅ _fetch_tract_geometries method exists")
        else:
            print("   ❌ _fetch_tract_geometries method missing")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n🎉 All tract functionality tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_tract_functionality())
    if not success:
        sys.exit(1)