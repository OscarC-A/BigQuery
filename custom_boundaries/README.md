# Custom Boundaries

This directory contains custom geographic boundaries for census data queries.

## File Format

Boundaries should be stored as GeoJSON files with the following structure:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon" or "MultiPolygon",
        "coordinates": [...]
      },
      "properties": {}
    }
  ]
}
```

## Naming Convention

- Files should be named `{boundary_name}.geojson`
- Use lowercase names with underscores for spaces
- Examples: `manhattan.geojson`, `brooklyn.geojson`, `lower_manhattan.geojson`

## Supported Boundaries

Currently supported boundaries:
- manhattan
- brooklyn
- queens
- bronx
- staten_island

Add new boundaries by placing properly formatted GeoJSON files in this directory.