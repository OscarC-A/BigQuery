import json
import os
from typing import Dict, List, Optional
from .geojson_state_detect import StateDetector

# Using dictionary to find state FIPS codes, we can do this for all formats we
# want, but might be a good idea to alter this one way or another down the road
# for considering stuff like census tracts, of which there will be considerbly
# more data to store/access 

# Main issue here is can only really work with states (find .. for counties/zcta in New York)
# Needs to work within more complex bounds besides just state

class GeographicResolver:
    def __init__(self):
        # State FIPS codes
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

        self.geojson_state_detector = StateDetector()
        
    def extract_state_from_query(self, query: str, intent_state: list, geojson_dir: str) -> Optional[str]:
        """Extract state name from query"""
        query_lower = query.lower()
        
        # Check for DC aliases first
        if any(dc_term in query_lower for dc_term in ['washington dc', 'washington d.c.', 'dc', 'district of columbia', 'd.c.']):
            return 'district of columbia'
        
        state = ""
        
        # First check if name of state is in query. If not, examine for state is geojson file. Then, fall back to 
        # llm interpretation in analyzequeryintent. Finally, fall back to querying every table for geom.
        for state_name, fips in self.state_fips.items():
            if state_name in query_lower:
                return state_name
            elif state == "":
                state = self.geojson_state_detector.find_state_in_geojson(geojson_dir)
                if state != "":
                    return state
            elif state == "" and intent_state != []:
                if intent_state[0] in self.state_fips:
                    state = intent_state[0]
                    return state
            else:
                # No state found, use brute force method
                state = "brute"
                return state
    
        return state