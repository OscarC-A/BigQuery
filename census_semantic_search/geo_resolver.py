import json
import os
from typing import Dict, List, Optional

# Using dictionary to find state FIPS codes, we can do this for all formats we
# want, but might be a good idea to alter this one way or another down the road
# for considering stuff like census tracts, of which there will be considerbly
# more data to store/access 

class GeographicResolver:
    def __init__(self):
        self.state_fips = {
            'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05',
            'california': '06', 'colorado': '08', 'connecticut': '09',
            'delaware': '10', 'florida': '12', 'georgia': '13', 'hawaii': '15',
            'idaho': '16', 'illinois': '17', 'indiana': '18', 'iowa': '19',
            'kansas': '20', 'kentucky': '21', 'louisiana': '22', 'maine': '23',
            'maryland': '24', 'massachusetts': '25', 'michigan': '26',
            'minnesota': '27', 'mississippi': '28', 'missouri': '29',
            'montana': '30', 'nebraska': '31', 'nevada': '32',
            'new hampshire': '33', 'new jersey': '34', 'new mexico': '35',
            'new york': '36', 'north carolina': '37', 'north dakota': '38',
            'ohio': '39', 'oklahoma': '40', 'oregon': '41', 'pennsylvania': '42',
            'rhode island': '44', 'south carolina': '45', 'south dakota': '46',
            'tennessee': '47', 'texas': '48', 'utah': '49', 'vermont': '50',
            'virginia': '51', 'washington': '53', 'west virginia': '54',
            'wisconsin': '55', 'wyoming': '56'
        }
        
    def extract_state_from_query(self, query: str) -> Optional[str]:
        """Extract state name from query"""
        query_lower = query.lower()
        for state_name, fips in self.state_fips.items():
            if state_name in query_lower:
                return state_name
        return None
    
    def get_state_fips(self, state_name: str) -> Optional[str]:
        """Get FIPS code for state"""
        return self.state_fips.get(state_name.lower())
    
    def build_geo_filter(self, query: str, geo_level: str) -> Dict:
        """
        Build geographic filter for BigQuery
        
        Returns:
            {
                'filter_sql': "geo_id LIKE '13%'",
                'state_fips': '13',
                'state_name': 'georgia',
                'geo_level': 'county'
            }
        """
        state_name = self.extract_state_from_query(query)
        if not state_name:
            raise ValueError(f"Could not extract state from query: {query}")
        
        state_fips = self.get_state_fips(state_name)
        if not state_fips:
            raise ValueError(f"Unknown state: {state_name}")
        
        # Build filter based on geo level
        if geo_level == 'county':
            filter_sql = f"geo_id LIKE '{state_fips}%'"
        elif geo_level == 'state':
            filter_sql = f"geo_id = '{state_fips}'"
        else:
            # Placeholder for other levels
            filter_sql = f"geo_id LIKE '{state_fips}%'"
        
        return {
            'filter_sql': filter_sql,
            'state_fips': state_fips,
            'state_name': state_name,
            'geo_level': geo_level
        }