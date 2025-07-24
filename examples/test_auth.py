# Basic test to see if we are connecting to BigQuery properly

from google.cloud import bigquery

def test_bigquery_auth():
    try:
        client = bigquery.Client()
        
        # Test query on public dataset
        query = """
        SELECT table_name 
        FROM `bigquery-public-data.census_bureau_acs.INFORMATION_SCHEMA.TABLES` 
        LIMIT 5
        """
        
        results = client.query(query).to_dataframe()
        print("✅ BigQuery authentication successful!")
        print(f"Found {len(results)} tables")
        print(results)
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

if __name__ == "__main__":
    test_bigquery_auth()