import json
import os
from typing import Dict, Optional, List
from shapely.geometry import shape, Polygon, MultiPolygon
import geopandas as gpd

class CustomBoundaryHandler:
    """Handles custom geographic boundaries like Manhattan, specific neighborhoods, etc."""
    
    def __init__(self, boundaries_dir: str = "custom_boundaries"):
        self.boundaries_dir = boundaries_dir
        self.loaded_boundaries = {}
        
    def load_boundary(self, boundary_name: str) -> Optional[Dict]:
        """Load a custom boundary from GeoJSON file"""
        if boundary_name in self.loaded_boundaries:
            return self.loaded_boundaries[boundary_name]
            
        # Try to find the boundary file
        boundary_file = os.path.join(self.boundaries_dir, f"{boundary_name.lower()}.geojson")
        if not os.path.exists(boundary_file):
            # Also check in examples folder
            boundary_file = os.path.join("custom_boundaries", f"{boundary_name.lower()}_geometry_ex.geojson")
            if not os.path.exists(boundary_file):
                return None
                
        with open(boundary_file, 'r') as f:
            geojson_data = json.load(f)
            
        # Extract the geometry
        if geojson_data['type'] == 'FeatureCollection':
            # Get the first feature's geometry
            geometry = geojson_data['features'][0]['geometry']
        else:
            geometry = geojson_data
            
        self.loaded_boundaries[boundary_name] = {
            'geometry': geometry,
            'geojson': geojson_data
        }
        
        return self.loaded_boundaries[boundary_name]
    
    def extract_boundary_from_query(self, query: str) -> Optional[str]:
        """Extract custom boundary names from query"""
        query_lower = query.lower()
        
        # List of known custom boundaries
        known_boundaries = ['manhattan', 'brooklyn', 'queens', 'bronx', 'staten island']
        
        for boundary in known_boundaries:
            if boundary in query_lower:
                return boundary
                
        return None
    
    def get_state_for_boundary(self, boundary_name: str) -> Optional[str]:
        """Get the state name for a custom boundary"""
        boundary_to_state = {
            'manhattan': 'new york',
            'brooklyn': 'new york', 
            'queens': 'new york',
            'bronx': 'new york',
            'staten island': 'new york'
        }
        return boundary_to_state.get(boundary_name.lower())
    
    def geometry_to_wkt(self, geometry: Dict) -> str:
        """Convert GeoJSON geometry to WKT format for BigQuery"""
        # Create a shapely geometry from the GeoJSON
        geom = shape(geometry)
        
        # Return WKT representation
        return geom.wkt
    
    def build_intersect_filter(self, boundary_name: str, geo_level: str) -> Optional[str]:
        """Build ST_INTERSECTS filter for BigQuery"""
        boundary_data = self.load_boundary(boundary_name)
        if not boundary_data:
            return None
            
        wkt = self.geometry_to_wkt(boundary_data['geometry'])
        
        # Determine the geometry column name based on geo_level
        # Note: These column names may need to be verified against actual BigQuery schema
        geom_column_map = {
            'county': 'county_geom',  # might be 'geometry' or other name
            'zcta': 'zip_code_geom',  # might be 'geometry' or other name
            'tract': 'tract_geom'     # might be 'geometry' or other name
        }
        
        geom_column = geom_column_map.get(geo_level, 'geometry')
        
        # Build the ST_INTERSECTS filter
        # We'll use ST_GEOGFROMTEXT to convert WKT to geography
        return f"ST_INTERSECTS({geom_column}, ST_GEOGFROMTEXT('{wkt}'))"