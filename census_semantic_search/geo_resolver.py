import json
import os
from typing import Dict, List, Optional
from .custom_boundary_handler import CustomBoundaryHandler
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
        
        # ZIP code ranges by state, from: https://www.structnet.com/instructions/zip_min_max_by_state.html
        self.state_zip_ranges = {
            'alabama': ['35004-36925'],
            'alaska': ['99501-99950'],
            'arizona': ['85001-86556'],
            'arkansas': ['71601-72959', '75502'],
            'california': ['90001-96162'],
            'colorado': ['80001-81658'],
            'connecticut': ['06001-06389', '06401-06928'],
            'delaware': ['19701-19980'],
            'district of columbia': ['20001-20039', '20042-20599', '20799'],
            'florida': ['32004-34997'],
            'georgia': ['30001-31999', '39901'],
            'hawaii': ['96701-96898'],
            'idaho': ['83201-83876'],
            'illinois': ['60001-62999'],
            'indiana': ['46001-47997'],
            'iowa': ['50001-52809', '68119-68120'],
            'kansas': ['66002-67954'],
            'kentucky': ['40003-42788'],
            'louisiana': ['70001-71232', '71234-71497'],
            'maine': ['03901-04992'],
            'maryland': ['20331', '20335-20797', '20812-21930'],
            'massachusetts': ['01001-02791', '05501-05544'],
            'michigan': ['48001-49971'],
            'minnesota': ['55001-56763'],
            'mississippi': ['38601-39776', '71233'],
            'missouri': ['63001-65899'],
            'montana': ['59001-59937'],
            'north carolina': ['27006-28909'],
            'north dakota': ['58001-58856'],
            'nebraska': ['68001-68118', '68122-69367'],
            'nevada': ['88901-89883'],
            'new hampshire': ['03031-03897'],
            'new jersey': ['07001-08989'],
            'new mexico': ['87001-88441'],
            'new york': ['10001-14975', '06390'],
            'ohio': ['43001-45999'],
            'oklahoma': ['73001-73199', '73401-74966'],
            'oregon': ['97001-97920'],
            'pennsylvania': ['15001-19640'],
            'rhode island': ['02801-02940'],
            'south carolina': ['29001-29948'],
            'south dakota': ['57001-57799'],
            'tennessee': ['37010-38589'],
            'texas': ['73301', '75001', '75503-79999', '88510-88589'],
            'utah': ['84001-84784'],
            'vermont': ['05001-05495', '05601-05907'],
            'virginia': ['20040-20167', '22001-24658'],
            'washington': ['98001-99403'],
            'west virginia': ['24701-26886'],
            'wisconsin': ['53001-54990'],
            'wyoming': ['82001-83128']
        }

        # Initialize custom boundary handler and GeoJson detector
        self.boundary_handler = CustomBoundaryHandler()
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

    
    def get_state_fips(self, state_name: str) -> Optional[str]:
        """Get FIPS code for state"""
        return self.state_fips.get(state_name.lower())
    
    def get_state_zip_ranges(self, state_name: str) -> Optional[List[str]]:
        """Get ZIP code ranges for state"""
        return self.state_zip_ranges.get(state_name.lower())
    
    def build_zcta_filter_sql(self, zip_ranges: List[str]) -> str:
        """
        Build SQL filter for ZCTA based on ZIP code ranges
        
        Args:
            zip_ranges: List of ZIP ranges like ['30001-31999', '39901-39901']
            
        Returns:
            SQL filter string like "(geo_id BETWEEN '30001' AND '31999' OR geo_id = '39901')"
        """
        conditions = []
        for zip_range in zip_ranges:
            if '-' in zip_range:
                start, end = zip_range.split('-')
                start_padded = start.zfill(5)
                end_padded = end.zfill(5)
                
                # Range of ZIP codes
                conditions.append(f"geo_id BETWEEN '{start_padded}' AND '{end_padded}'")
            else:
                # Single ZIP code without range
                zip_padded = zip_range.zfill(5)
                conditions.append(f"geo_id = '{zip_padded}'")
        
        return f"({' OR '.join(conditions)})"
    
    def build_geo_filter(self, query: str, geo_level: str, state: list, geojson_dir: str) -> Dict:
        """
        Build geographic filter for BigQuery with custom bound support
        
        Returns:
            For county levels:
            {
                'filter_sql': "geo_id LIKE '13%'",
                'state_name': 'georgia'
            }
            
            For ZCTA level:
            {
                'filter_sql': "(geo_id BETWEEN '30001' AND '31999' OR geo_id = '39901')",
                'state_name': 'georgia'
            }
            
            For tract level:
            {
                'filter_sql': "geo_id LIKE '13%'",
                'state_name': 'georgia'
            }
        """

        state_name = self.extract_state_from_query(query, state, geojson_dir)
        print("found state name:", state_name)
        result = {
                    "state_name": state_name,
                    "filter_sql": ""
                  }

        # Build filter based on geo level
        if geo_level == 'zcta':
            intersect_filter = self.boundary_handler.build_intersect_filter(geojson_dir, geo_level)
            result['filter_sql'] = intersect_filter
            print(result)
            return result
        else:
            # Handle county and tract levels
            # Use ST_INTERSECTS for custom boundaries
            intersect_filter = self.boundary_handler.build_intersect_filter(geojson_dir, geo_level)
            result['filter_sql'] = intersect_filter
            print(result)
            return result