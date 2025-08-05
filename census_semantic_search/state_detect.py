import json
import os
from typing import Optional, List, Dict, Union
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import geopandas as gpd
from collections import Counter

class StateDetector:
    """Detects which state(s) a custom GeoJSON boundary belongs to"""
    
    def __init__(self):
        # State names and common variations
        self.state_names = {
            'alabama': ['alabama', 'al'],
            'alaska': ['alaska', 'ak'],
            'arizona': ['arizona', 'az'],
            'arkansas': ['arkansas', 'ar'],
            'california': ['california', 'ca', 'calif'],
            'colorado': ['colorado', 'co'],
            'connecticut': ['connecticut', 'ct', 'conn'],
            'delaware': ['delaware', 'de'],
            'florida': ['florida', 'fl', 'fla'],
            'georgia': ['georgia', 'ga'],
            'hawaii': ['hawaii', 'hi'],
            'idaho': ['idaho', 'id'],
            'illinois': ['illinois', 'il'],
            'indiana': ['indiana', 'in'],
            'iowa': ['iowa', 'ia'],
            'kansas': ['kansas', 'ks'],
            'kentucky': ['kentucky', 'ky'],
            'louisiana': ['louisiana', 'la'],
            'maine': ['maine', 'me'],
            'maryland': ['maryland', 'md'],
            'massachusetts': ['massachusetts', 'ma', 'mass'],
            'michigan': ['michigan', 'mi', 'mich'],
            'minnesota': ['minnesota', 'mn', 'minn'],
            'mississippi': ['mississippi', 'ms', 'miss'],
            'missouri': ['missouri', 'mo'],
            'montana': ['montana', 'mt'],
            'nebraska': ['nebraska', 'ne', 'neb'],
            'nevada': ['nevada', 'nv'],
            'new hampshire': ['new hampshire', 'nh'],
            'new jersey': ['new jersey', 'nj'],
            'new mexico': ['new mexico', 'nm'],
            'new york': ['new york', 'ny'],
            'north carolina': ['north carolina', 'nc'],
            'north dakota': ['north dakota', 'nd'],
            'ohio': ['ohio', 'oh'],
            'oklahoma': ['oklahoma', 'ok', 'okla'],
            'oregon': ['oregon', 'or', 'ore'],
            'pennsylvania': ['pennsylvania', 'pa', 'penn'],
            'rhode island': ['rhode island', 'ri'],
            'south carolina': ['south carolina', 'sc'],
            'south dakota': ['south dakota', 'sd'],
            'tennessee': ['tennessee', 'tn', 'tenn'],
            'texas': ['texas', 'tx', 'tex'],
            'utah': ['utah', 'ut'],
            'vermont': ['vermont', 'vt'],
            'virginia': ['virginia', 'va'],
            'washington': ['washington', 'wa', 'wash'],
            'west virginia': ['west virginia', 'wv'],
            'wisconsin': ['wisconsin', 'wi', 'wis'],
            'wyoming': ['wyoming', 'wy']
        }
        
        # Major cities to state mapping for additional context
        self.city_to_state = {
            'manhattan': 'new york',
            'brooklyn': 'new york',
            'queens': 'new york',
            'bronx': 'new york',
            'staten island': 'new york',
            'los angeles': 'california',
            'san francisco': 'california',
            'chicago': 'illinois',
            'houston': 'texas',
            'philadelphia': 'pennsylvania',
            'phoenix': 'arizona',
            'san antonio': 'texas',
            'san diego': 'california',
            'dallas': 'texas',
            'austin': 'texas',
            'boston': 'massachusetts',
            'seattle': 'washington',
            'denver': 'colorado',
            'miami': 'florida',
            'atlanta': 'georgia',
            'detroit': 'michigan',
            'portland': 'oregon',
            'las vegas': 'nevada',
            'baltimore': 'maryland',
            'milwaukee': 'wisconsin',
            'albuquerque': 'new mexico',
            'tucson': 'arizona',
            'nashville': 'tennessee',
            'memphis': 'tennessee',
            'louisville': 'kentucky',
            'new orleans': 'louisiana',
            'cleveland': 'ohio',
            'cincinnati': 'ohio',
            'pittsburgh': 'pennsylvania',
            'st louis': 'missouri',
            'saint louis': 'missouri',
            'kansas city': 'missouri',
            'charlotte': 'north carolina',
            'raleigh': 'north carolina',
            'indianapolis': 'indiana',
            'columbus': 'ohio',
            'jacksonville': 'florida',
            'tampa': 'florida',
            'orlando': 'florida'
        }

    def extract_state_from_query(self, query: str, intent_state: list, geojson_dir: str) -> Optional[str]:
        """Extract state name from query"""
        query_lower = query.lower()
        
        # Check for DC aliases first
        if any(dc_term in query_lower for dc_term in ['washington dc', 'washington d.c.', 'dc', 'district of columbia', 'd.c.']):
            return 'district of columbia'
        
        state = ""
        
        # First check if name of state is in query. If not, examine for state is geojson file. Then, fall back to 
        # llm interpretation in analyzequeryintent. Finally, fall back to querying every table for geom.
        for state_name, fips in self.state_names():
            if state_name in query_lower:
                return state_name
            elif state == "":
                state = self.find_state_in_geojson(geojson_dir)
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
    
    def find_state_in_geojson(self, geojson_path: str) -> Optional[str]:
        """
        Find the state name in a GeoJSON file using multiple methods
        
        Args:
            geojson_path: Path to the GeoJSON file
            
        Returns:
            State name if found, None otherwise
        """
        if not os.path.exists(geojson_path):
            print(f"File not found: {geojson_path}")
            return None
            
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            return None
        
        # Method 1: Check properties
        state = self._check_properties(geojson_data)
        if state:
            print(f"Found state in properties: {state}")
            return state
        
        # Method 2: Check filename
        state = self._check_filename(geojson_path)
        if state:
            print(f"Found state in filename: {state}")
            return state
        
        # Method 3: Check for city names in properties
        state = self._check_city_references(geojson_data)
        if state:
            print(f"Found state via city reference: {state}")
            return state
        
        # Method 4: Use centroid coordinates (requires additional data)
        # state = self._check_by_coordinates(geojson_data)
        # if state:
        #     print(f"Found state via coordinates: {state}")
        #     return state
        
        return ""
    
    def _check_properties(self, geojson_data: Dict) -> Optional[str]:
        """Check GeoJSON properties for state references"""
        # Get all text from properties
        property_text = []
        
        if geojson_data.get('type') == 'FeatureCollection':
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                property_text.extend(str(v).lower() for v in props.values() if v)
        elif 'properties' in geojson_data:
            props = geojson_data.get('properties', {})
            property_text.extend(str(v).lower() for v in props.values() if v)
        
        # Search for state names in properties
        all_text = ' '.join(property_text)
        
        for state_name, variations in self.state_names.items():
            for variation in variations:
                # Look for exact matches or variations with word boundaries
                if f' {variation} ' in f' {all_text} ' or f'{variation},' in all_text:
                    return state_name
        
        return None
    
    def _check_filename(self, filepath: str) -> Optional[str]:
        """Check filename for state references"""
        filename = os.path.basename(filepath).lower()
        filename = filename.replace('.geojson', '').replace('_', ' ').replace('-', ' ')
        
        # Check for state names in filename
        for state_name, variations in self.state_names.items():
            for variation in variations:
                if variation in filename:
                    return state_name
        
        return None
    
    def _check_city_references(self, geojson_data: Dict) -> Optional[str]:
        """Check for city names that can indicate the state"""
        # Get all text from properties and features
        text_content = []
        
        if geojson_data.get('type') == 'FeatureCollection':
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                text_content.extend(str(v).lower() for v in props.values() if v)
        elif 'properties' in geojson_data:
            props = geojson_data.get('properties', {})
            text_content.extend(str(v).lower() for v in props.values() if v)
        
        all_text = ' '.join(text_content)
        
        # Check for known cities
        for city, state in self.city_to_state.items():
            if city in all_text:
                return state
        
        return None
    
    def _check_by_coordinates(self, geojson_data: Dict) -> Optional[str]:
        """
        Use centroid coordinates to determine state
        This is a simplified version - in production, you'd want to use
        actual state boundary data
        """
        try:
            # Get the geometry
            if geojson_data.get('type') == 'FeatureCollection':
                if not geojson_data.get('features'):
                    return None
                geometry = geojson_data['features'][0]['geometry']
            else:
                geometry = geojson_data.get('geometry', geojson_data)
            
            # Create shapely geometry and get centroid
            geom = shape(geometry)
            centroid = geom.centroid
            lon, lat = centroid.x, centroid.y
            
            # Rough state boundaries (simplified for demonstration)
            # In production, use actual state boundary data
            state_bounds = {
                'new york': {'min_lon': -80.0, 'max_lon': -71.0, 'min_lat': 40.0, 'max_lat': 45.5},
                'california': {'min_lon': -125.0, 'max_lon': -114.0, 'min_lat': 32.0, 'max_lat': 42.0},
                'texas': {'min_lon': -107.0, 'max_lon': -93.0, 'min_lat': 25.5, 'max_lat': 36.5},
                'florida': {'min_lon': -88.0, 'max_lon': -79.5, 'min_lat': 24.0, 'max_lat': 31.0},
                'illinois': {'min_lon': -91.5, 'max_lon': -87.0, 'min_lat': 36.5, 'max_lat': 42.5},
                # Add more states as needed
            }
            
            # Check which state bounds contain the centroid
            for state, bounds in state_bounds.items():
                if (bounds['min_lon'] <= lon <= bounds['max_lon'] and 
                    bounds['min_lat'] <= lat <= bounds['max_lat']):
                    return state
            
        except Exception as e:
            print(f"Error checking coordinates: {e}")
        
        return None
    
    def find_states_containing_geometry(self, geojson_path: str, 
                                      state_boundaries_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Find which states contain or intersect with the custom geometry
        
        Args:
            geojson_path: Path to the custom boundary GeoJSON
            state_boundaries_gdf: GeoDataFrame with state boundaries
            
        Returns:
            List of state names that contain or intersect the geometry
        """
        try:
            # Load the custom boundary
            custom_gdf = gpd.read_file(geojson_path)
            
            # Ensure same CRS
            if custom_gdf.crs != state_boundaries_gdf.crs:
                custom_gdf = custom_gdf.to_crs(state_boundaries_gdf.crs)
            
            # Find intersecting states
            intersecting_states = []
            
            for idx, state in state_boundaries_gdf.iterrows():
                if custom_gdf.geometry.iloc[0].intersects(state.geometry):
                    state_name = state.get('name', state.get('state_name', ''))
                    if state_name:
                        intersecting_states.append(state_name.lower())
            
            return intersecting_states
            
        except Exception as e:
            print(f"Error finding intersecting states: {e}")
            return []


# Standalone function for simple use cases
def find_state_in_geojson(geojson_path: str) -> Optional[str]:
    """
    Simple function to find state name in a GeoJSON file
    
    Args:
        geojson_path: Path to the GeoJSON file
        
    Returns:
        State name if found, None otherwise
    """
    detector = StateDetector()
    return detector.find_state_in_geojson(geojson_path)


# Example usage
if __name__ == "__main__":
    # Test with Manhattan example
    manhattan_path = "examples/manhattan_geometry_ex.geojson"
    if os.path.exists(manhattan_path):
        state = find_state_in_geojson(manhattan_path)
        print(f"\nState detected for Manhattan: {state}")
    
    # Test with a custom path
    import sys
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        state = find_state_in_geojson(custom_path)
        print(f"\nState detected for {custom_path}: {state}")