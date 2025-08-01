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
        
    def load_boundary(self, geojson_dir: str) -> Optional[Dict]:
        """Load a custom boundary from GeoJSON file"""
        if geojson_dir in self.loaded_boundaries:
            return self.loaded_boundaries[geojson_dir]
            
        # Try to find the boundary file
        boundary_file = os.path.join(self.boundaries_dir, f"{geojson_dir.lower()}")
        boundary_file = os.path.join("", f"{geojson_dir.lower()}")
        if not os.path.exists(boundary_file):
            # Also check in examples folder
            boundary_file = os.path.join("custom_boundaries", f"{geojson_dir.lower()}")
            if not os.path.exists(boundary_file):
                print("could not find file")
                return None
                
        with open(boundary_file, 'r') as f:
            geojson_data = json.load(f)
            
        # Extract the geometry
        if geojson_data['type'] == 'FeatureCollection':
            # Get the first feature's geometry
            geometry = geojson_data['features'][0]['geometry']
        else:
            geometry = geojson_data
            
        self.loaded_boundaries[geojson_dir] = {
            'geometry': geometry,
            'geojson': geojson_data
        }
        
        return self.loaded_boundaries[geojson_dir]
    
    def geometry_to_wkt(self, geometry: Dict) -> str:
        """Convert GeoJSON geometry to WKT format for BigQuery"""
        # Create a shapely geometry from the GeoJSON
        geom = shape(geometry)
        
        # Return WKT representation
        return geom.wkt
    
    def build_intersect_filter(self, geojson_dir: str, geo_level: str) -> Optional[str]:
        """Build ST_INTERSECTS filter for BigQuery"""
        boundary_data = self.load_boundary(geojson_dir)
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