"""
add_osm.py - Tool for adding OpenStreetMap data to the map

This module provides a tool to take a natural language query,
use advanced semantic search to find relevant OSM tags,
generate an Overpass query, fetch data from OpenStreetMap,
convert it to GeoJSON, and send it to the map.
"""

import pandas as pd
import numpy as np
import time
import os
import re
import json
import ast
import httpx
import asyncio
import textwrap
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from urllib.parse import quote
from typing import Dict, Optional, Union, List, Any
from pydantic import BaseModel, Field
from .base import Tool
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables.")

# Add parent directory to path for redis_client import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redis_client import get_session_key

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
WEBSOCKET_SERVICE_URL = os.getenv("WEBSOCKET_SERVICE_URL", "http://host.docker.internal:8001")

_required = [DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]
if not all(_required):
    missing = [n for n, v in zip(["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"], _required) if not v]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Global variables for model and data
model = None
index = None
df = None
client = None

def _get_connection():
    """Get database connection."""
    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
    )


def initialize_models():
    """Initialize the embedding model, FAISS index, and OpenAI client."""
    global model, index, df, client
    
    print("🔧 Initializing models...")
    
    # Get the data directory path relative to this file
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    
    # Load pre-trained model and index
    print("📥 Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("📥 Loading FAISS index...")
    index = faiss.read_index(os.path.join(data_dir, "osm_tags.index"))
    
    print("📥 Loading OSM tags metadata...")
    df = pd.read_parquet(os.path.join(data_dir, "osm_tags_meta.parquet"))
    print(f"   Loaded {len(df)} OSM tags")
    
    # Initialize OpenAI client
    print("🔑 Initializing OpenAI client...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)
    print("✅ All models initialized successfully!")

def search_tags(query: str, k: int = 50, post_k: int = 500, alpha: float = 0.75):
    """
    Return k tags ranked by alpha*cos_sim + (1-alpha)*pop_norm
    
    Args:
        query: Natural language query
        k: Number of top results to return
        post_k: Number of candidates to retrieve before reranking
        alpha: Weight for semantic similarity vs popularity (0-1)
    """
    if model is None or index is None or df is None:
        raise ValueError("Models not initialized. Call initialize_models() first.")
    
    print(f"🔍 Searching for tags similar to: '{query}'")
    print(f"   Using alpha={alpha} (similarity vs popularity weight)")
        
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)

    sims, idxs = index.search(q_vec.astype("float32"), post_k)
    sims = sims[0]; idxs = idxs[0]

    candidates = df.iloc[idxs].copy()
    candidates["sim"] = sims
    candidates["score"] = alpha * candidates["sim"] + (1 - alpha) * candidates["pop_norm"]

    result = candidates.sort_values("score", ascending=False).head(k)[["tag", "score", "sim", "pop_norm"]]
    print(f"📊 Found {len(result)} candidate tags")
    return result


def _extract_json_list(txt: str) -> list[str]:
    """
    Find the first [...] block in `txt`, parse it, and return the list.
    Raises ValueError if nothing parseable is found.
    """
    m = re.search(r"\[[^\]]*\]", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON list found in LLM output")

    snippet = m.group(0)
    try:                     # strict JSON first
        return json.loads(snippet)
    except json.JSONDecodeError:
        return ast.literal_eval(snippet)   # fall back to Python-style list


def _extract_json_object(txt: str) -> dict:
    """
    Find the first {...} block in `txt`, parse it, and return the dict.
    Raises ValueError if nothing parseable is found.
    """
    m = re.search(r"\{[^}]*\}", txt, flags=re.S | re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output")

    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return ast.literal_eval(snippet)


def analyze_query_intent(query: str, candidate_tags: list) -> dict:
    """
    Use LLM to analyze query intent and determine if it requires broad key matching vs specific tag matching.
    
    Args:
        query: Natural language query
        candidate_tags: List of candidate tags from semantic search
        
    Returns:
        dict with analysis results including broad_intent flag and reasoning
    """
    if client is None:
        raise ValueError("OpenAI client not initialized. Call initialize_models() first.")
    
    print(f"🧠 Analyzing query intent with LLM: '{query}'")
    
    # Extract key analysis for context
    key_counts = {}
    for tag in candidate_tags:
        if '=' in tag:
            key = tag.split('=')[0]
            key_counts[key] = key_counts.get(key, 0) + 1
    
    # Group tags by key for better analysis
    key_groups = {}
    for tag in candidate_tags:
        if '=' in tag:
            key, value = tag.split('=', 1)
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(value)
    
    # Format key analysis for LLM
    key_analysis = ""
    for key, values in key_groups.items():
        key_analysis += f"- {key}: {', '.join(values[:5])}{'...' if len(values) > 5 else ''} ({len(values)} total)\n"
    
    prompt = f"""You are analyzing a user's OpenStreetMap query to determine the best matching strategy.

User Query: "{query}"

Available OSM tags grouped by key:
{key_analysis}

IMPORTANT RESTRICTION: Only use BROAD matching if the user explicitly asks for "all buildings" or "all roads" (not specific types). 
For all other queries, including specific building types (residential, schools, hospitals, etc.) or road types (highways, freeways, etc.), use SPECIFIC matching.

Your task is to determine if the user wants:
1. BROAD matching (ONLY for explicit "all buildings" or "all roads" queries - use entire key)
2. SPECIFIC matching (for everything else - use exact key=value pairs)

Strict Guidelines:
- BROAD mode ONLY for: "all buildings" or "all roads" (exact phrases or very similar)
- SPECIFIC mode for: "schools", "restaurants", "hospitals", "residential buildings", "commercial buildings", "highways", "freeways", etc.
- SPECIFIC mode for: power infrastructure, amenities, and any other data types

Examples:
- "all buildings" → BROAD (use key "building")
- "all roads" → BROAD (use key "highway")
- "all hospitals" → SPECIFIC (use "amenity=hospital" AND AVOID building=hospital UNLESS you are specifically asking for hospital buildings)
- "residential buildings" → SPECIFIC (use "building=residential") 
- "schools" → SPECIFIC (use "amenity=school", "building=school")
- "highways" → SPECIFIC (use "highway=trunk", "highway=motorway", etc.)

Return a JSON object:
{{
  "broad_intent": true/false,
  "reasoning": "explain your decision",
  "recommended_approach": "broad_key|specific_values", 
  "relevant_keys": ["building"] or ["highway"] // ONLY if query is explicitly "all buildings" or "all roads"
}}

Remember: Be very restrictive with broad_intent. Only return true for explicit "all buildings" or "all roads" queries."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        raw = resp.choices[0].message.content.strip()
        result = _extract_json_object(raw)
        
        # Ensure required fields exist
        if 'broad_intent' not in result:
            result['broad_intent'] = False
        if 'reasoning' not in result:
            result['reasoning'] = "LLM analysis failed, defaulting to specific"
        if 'recommended_approach' not in result:
            result['recommended_approach'] = "specific_values"
        if 'relevant_keys' not in result:
            result['relevant_keys'] = []
        
        # Add metadata
        result['candidate_keys'] = list(key_counts.keys())
        result['key_counts'] = key_counts
        result['key_groups'] = key_groups
        
        print(f"   LLM Analysis: {result['broad_intent']} - {result['reasoning']}")
        print(f"   Approach: {result['recommended_approach']}")
        
        return result
        
    except Exception as e:
        print(f"   LLM analysis failed: {e}")
        # Fallback to simple analysis
        return {
            'broad_intent': False,
            'reasoning': f"LLM analysis failed: {e}",
            'recommended_approach': "specific_values",
            'relevant_keys': [],
            'candidate_keys': list(key_counts.keys()),
            'key_counts': key_counts,
            'key_groups': key_groups
        }


def extract_broad_keys_from_tags(tags: list, intent_analysis: dict) -> list:
    """
    Extract the most relevant broad keys from a list of specific tags.
    
    Args:
        tags: List of specific tags like ['building=yes', 'building=residential']
        intent_analysis: Result from analyze_query_intent()
        
    Returns:
        List of broad keys like ['building']
    """
    if not intent_analysis.get('broad_intent', False):
        return []
    
    print(f"🔑 Extracting broad keys from tags: {tags}")
    
    # Group tags by key and find most frequent/relevant keys
    key_groups = {}
    for tag in tags:
        if '=' in tag:
            key, value = tag.split('=', 1)
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(value)
    
    # Prioritize keys with multiple values (indicates diversity)
    broad_keys = []
    for key, values in key_groups.items():
        if len(values) > 1 or intent_analysis.get('broad_keywords_found', False):
            broad_keys.append(key)
    
    # If no multi-value keys found but broad intent detected, use the most common key
    if not broad_keys and intent_analysis.get('broad_intent', False):
        if intent_analysis.get('key_counts'):
            most_common_key = max(intent_analysis['key_counts'], key=intent_analysis['key_counts'].get)
            broad_keys.append(most_common_key)
    
    print(f"   Extracted broad keys: {broad_keys}")
    return broad_keys
    

def llm_filter(query: str, slice_, intent_analysis: dict, model_name: str = "gpt-4o-mini") -> dict:
    """
    Given a NL query and a DataFrame slice with 'tag' (+optional columns),
    return the tags and/or broad keys the LLM judges relevant.
    
    Returns:
        dict with 'mode' ('specific'|'broad'|'hybrid'), 'tags' (list), 'keys' (list)
    """
    if client is None:
        raise ValueError("OpenAI client not initialized. Call initialize_models() first.")
    
    print(f"🤖 Filtering tags with LLM (model: {model_name})")
    print(f"   Candidate tags: {slice_['tag'].tolist()[:5]}{'...' if len(slice_) > 5 else ''}")
    print(f"   Broad intent detected: {intent_analysis.get('broad_intent', False)}")
    
    # Build enhanced prompt based on intent analysis - heavily favor specific mode
    context_info = ""
    if intent_analysis.get('broad_intent', False):
        # Only allow broad mode for explicit "all buildings" or "all roads" queries
        context_info = (
            "\n\nIMPORTANT: The user query appears to ask for 'all buildings' or 'all roads' specifically.\n"
            "These are the ONLY cases where broad KEY matching should be used.\n"
            "Use key 'building' for all buildings or key 'highway' for all roads.\n"
            "For any other query, use SPECIFIC key=value pairs.\n"
        )
    else:
        context_info = (
            "\n\nIMPORTANT: Use SPECIFIC key=value pairs that directly match what the user wants.\n"
            "Select only the 1-2 MOST RELEVANT tags from the candidate list.\n"
            "Avoid broad keys that would return too many unrelated results.\n"
            "Focus on tags that directly represent the requested entities.\n"
            "For example: 'schools' → use 'building=school' AND/OR 'amenity=school'\n"
            "For example: 'restaurants' → use 'amenity=restaurant'\n"
            "For example: 'hospitals' → use 'amenity=hospital' (AND DO NOT USE building=hospital unless specifically asking for hospital buildings')\n"
        )
    
    prompt = (
        "You are helping choose OpenStreetMap tags to answer the user query below.\n"
        "IMPORTANT: Default to SPECIFIC mode and select only 1-2 most relevant tags.\n"
        "Only use BROAD mode if the query is explicitly asking for 'all buildings' or 'all roads'.\n"
        "Only keep tags that map *DIRECTLY* to the thing the user asked for.\n"
        "Discard unrelated tags (addresses, sources, paths, unrelated POIs).\n"
        f"{context_info}\n"
        "Return a JSON object with this structure:\n"
        "{\n"
        "  \"mode\": \"specific\" | \"broad\",\n"
        "  \"tags\": [\"key=value\", \"key2=value2\"],\n"
        "  \"keys\": [\"key\"]\n"
        "}\n\n"
        "Mode explanations:\n"
        "- 'specific': Use 1-2 exact key=value pairs that best match the query\n"
        "- 'broad': ONLY for 'all buildings' or 'all roads' queries - use key 'building' or 'highway'\n\n"
        "Examples:\n"
        "- Query 'schools' → mode: 'specific', tags: ['amenity=school'] (pick the most relevant)\n"
        "- Query 'residential buildings' → mode: 'specific', tags: ['building=residential']\n"
        "- Query 'all buildings' → mode: 'broad', keys: ['building']\n"
        "- Query 'all roads' → mode: 'broad', keys: ['highway']\n"
        "- Query 'highways' → mode: 'specific', tags: ['highway=trunk'] (pick the most relevant)\n"
        "- Query 'restaurants' → mode: 'specific', tags: ['amenity=restaurant']\n"
        "- Query 'hospitals' → mode: 'specific', tags: ['amenity=hospital'] (unless asking for hospital buildings)\n\n"
        f"User query:\n{query}\n\n"
        "Candidate tags (select only 1-2 most relevant):\n" + "\n".join(slice_["tag"].tolist())
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
    
    try:
        result = _extract_json_object(raw)
        
        # Ensure all required fields exist - let LLM analysis decide
        if 'mode' not in result:
            result['mode'] = 'broad' if intent_analysis.get('broad_intent', False) else 'specific'
        if 'tags' not in result:
            result['tags'] = []
        if 'keys' not in result:
            result['keys'] = []
            
        # Fallback to legacy behavior if structured response failed - limit to 2 tags
        if not result['tags'] and not result['keys']:
            legacy_tags = _extract_json_list(raw)
            result['tags'] = legacy_tags[:2]  # Limit to top 2 tags
            result['mode'] = 'specific'
            
    except (ValueError, json.JSONDecodeError):
        # Fallback to legacy parsing - limit to 2 tags and default to specific
        print("   Falling back to legacy tag parsing")
        try:
            legacy_tags = _extract_json_list(raw)
            result = {
                'mode': 'broad' if intent_analysis.get('broad_intent', False) else 'specific',
                'tags': legacy_tags[:2],  # Limit to top 2 tags
                'keys': []
            }
        except:
            # Ultimate fallback - use top 2 candidates in specific mode
            result = {
                'mode': 'specific',
                'tags': slice_['tag'].tolist()[:2],  # Take top 2 candidates
                'keys': []
            }
    
    print(f"✅ LLM selected mode: {result['mode']}")
    print(f"   Tags: {result['tags']}")
    print(f"   Keys: {result['keys']}")
    return result



# ----------------- helpers -------------------------------------------------

def _area_id(osm_type: str, osm_id: int) -> Optional[int]:
    if   osm_type == "relation": return 3600000000 + osm_id
    elif osm_type == "way":      return 2400000000 + osm_id
    # nodes rarely have proper polygons - let caller use bbox/poly fallback
    return None


def _poly_from_geojson(geom: Dict) -> str:
    """
    Convert a GeoJSON Polygon / MultiPolygon to an Overpass
    (poly:"lat lon lat lon ...") string.

    - Uses only the exterior ring (index 0) of each polygon.
    - Returns "" if geometry is not polygonal.
    - Overpass expects LAT LON pairs (not lon lat).
    """
    if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
        return ""

    # Normalise to a list[ring] where each ring is list[(lon, lat)]
    polys = (
        [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    )

    rings = []
    for poly in polys:
        ring = poly[0]                                     # exterior ring
        # swap to lat lon for each vertex
        rings.append(" ".join(f"{lat} {lon}" for lon, lat in ring))

    return " ".join(rings)



async def get_boundary(
    place_name: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    user_agent: str = "MonarchaGIS/1.0"
) -> Dict[str, Union[str, int, Dict]]:
    """
    Returns dict with:
      - display_name
      - osm_type / osm_id
      - area_id      (None if node only)
      - geojson      (polygon or None)
      - poly_string  (for Overpass (poly:"...") filter; empty if none)
    Raises ValueError if nothing found.
    """
    print(f"🌍 Getting boundary for: '{place_name}'")
    
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(headers={"User-Agent": user_agent})

    try:
        url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={quote(place_name)}&format=jsonv2&polygon_geojson=1&limit=1"
        )
        print(f"   Querying Nominatim API...")
        r = await client.get(url, timeout=20)
        r.raise_for_status()
        items = r.json()
        if not items:
            raise ValueError(f"No match for '{place_name}'")

        itm = items[0]
        osm_id   = int(itm["osm_id"])
        osm_type = itm["osm_type"]
        area_id  = _area_id(osm_type, osm_id)
        geom     = itm.get("geojson")  # may be None
        poly_str = _poly_from_geojson(geom) if geom else ""

        print(f"   Found: {itm['display_name']}")
        print(f"   OSM ID: {osm_id} ({osm_type})")
        print(f"   Area ID: {area_id}")

        # If Nominatim had no polygon but we *do* have a relation - ask Overpass
        if not poly_str and osm_type == "relation":
            print("   Getting polygon from Overpass...")
            q = f"[out:json][timeout:20];rel({osm_id});out geom;"
            o = await client.get(
                "https://overpass-api.de/api/interpreter", data={"data": q}
            )
            if o.status_code == 200:
                ogj = o.json()
                if ogj.get("elements"):
                    geom = ogj["elements"][0].get("geometry")
                    if geom:
                        # Overpass geometry list - linearring lon/lat list
                        poly_str = " ".join(f"{p['lat']} {p['lon']}" for p in geom)

        # Get coordinates for around queries
        lat = float(itm["lat"])
        lon = float(itm["lon"])
        
        return {
            "display_name": itm["display_name"],
            "osm_type":     osm_type,
            "osm_id":       osm_id,
            "area_id":      area_id,
            "geojson":      geom,          # keep raw GeoJSON; handy for map preview
            "poly_string":  poly_str,      # "" if we still couldn't get one
            "lat":          lat,           # for around queries
            "lon":          lon            # for around queries
        }
    finally:
        if own_client:
            await client.aclose()


def _build_specific_query(geo_info: dict, tags: list) -> str:
    """Build Overpass query using specific key=value tags."""
    if geo_info["area_id"]:
        def kv_line(kv: str) -> str:
            key, val = kv.split("=", 1)
            return f'        nwr["{key}"="{val}"](area.a);'
        
        tag_block = "\n".join(kv_line(kv) for kv in tags)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        area({geo_info['area_id']})->.a;
        (
{tag_block}
        );
        out geom;
        """
    elif geo_info["poly_string"]:
        def kv_line(kv: str) -> str:
            key, val = kv.split("=", 1)
            return f'        nwr["{key}"="{val}"](poly:"{geo_info["poly_string"]}");'
        
        tag_block = "\n".join(kv_line(kv) for kv in tags)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """
    else:
        # Around query fallback
        radius = 1000
        lat = geo_info['lat']
        lon = geo_info['lon']
        
        def kv_line(kv: str) -> str:
            key, val = kv.split("=", 1)
            return f'        nwr["{key}"="{val}"](around:{radius},{lat},{lon});'
        
        tag_block = "\n".join(kv_line(kv) for kv in tags)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """


def _build_broad_query(geo_info: dict, keys: list) -> str:
    """Build Overpass query using broad key-only matching."""
    if geo_info["area_id"]:
        def key_line(key: str) -> str:
            return f'        nwr["{key}"](area.a);'
        
        tag_block = "\n".join(key_line(key) for key in keys)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        area({geo_info['area_id']})->.a;
        (
{tag_block}
        );
        out geom;
        """
    elif geo_info["poly_string"]:
        def key_line(key: str) -> str:
            return f'        nwr["{key}"](poly:"{geo_info["poly_string"]}");'
        
        tag_block = "\n".join(key_line(key) for key in keys)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """
    else:
        # Around query fallback
        radius = 1000
        lat = geo_info['lat']
        lon = geo_info['lon']
        
        def key_line(key: str) -> str:
            return f'        nwr["{key}"](around:{radius},{lat},{lon});'
        
        tag_block = "\n".join(key_line(key) for key in keys)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """


def _build_hybrid_query(geo_info: dict, tags: list, keys: list) -> str:
    """Build Overpass query combining both specific tags and broad keys."""
    if geo_info["area_id"]:
        # Specific tags
        specific_lines = []
        for kv in tags:
            key, val = kv.split("=", 1)
            specific_lines.append(f'        nwr["{key}"="{val}"](area.a);')
        
        # Broad keys  
        broad_lines = []
        for key in keys:
            broad_lines.append(f'        nwr["{key}"](area.a);')
        
        all_lines = specific_lines + broad_lines
        tag_block = "\n".join(all_lines)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        area({geo_info['area_id']})->.a;
        (
{tag_block}
        );
        out geom;
        """
    elif geo_info["poly_string"]:
        # Specific tags
        specific_lines = []
        for kv in tags:
            key, val = kv.split("=", 1)
            specific_lines.append(f'        nwr["{key}"="{val}"](poly:"{geo_info["poly_string"]}");')
        
        # Broad keys
        broad_lines = []
        for key in keys:
            broad_lines.append(f'        nwr["{key}"](poly:"{geo_info["poly_string"]}");')
        
        all_lines = specific_lines + broad_lines
        tag_block = "\n".join(all_lines)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """
    else:
        # Around query fallback
        radius = 1000
        lat = geo_info['lat']
        lon = geo_info['lon']
        
        # Specific tags
        specific_lines = []
        for kv in tags:
            key, val = kv.split("=", 1)
            specific_lines.append(f'        nwr["{key}"="{val}"](around:{radius},{lat},{lon});')
        
        # Broad keys
        broad_lines = []
        for key in keys:
            broad_lines.append(f'        nwr["{key}"](around:{radius},{lat},{lon});')
        
        all_lines = specific_lines + broad_lines
        tag_block = "\n".join(all_lines)
        return f"""
        [out:json][timeout:360][maxsize:4294967296];
        (
{tag_block}
        );
        out geom;
        """


def validate_query_scope(query: str, mode: str, tags: list, keys: list) -> dict:
    """
    Estimate query scope and suggest optimizations.
    
    Returns:
        dict with validation results and warnings
    """
    warnings = []
    estimated_size = "unknown"
    
    # Broad queries can be very large
    if mode == "broad" and keys:
        broad_keys_concern = ["building", "highway", "landuse", "natural", "power"]
        if any(key in broad_keys_concern for key in keys):
            warnings.append(f"Broad query for {keys} may return very large results")
            estimated_size = "large"
    
    # Multiple broad keys
    if mode in ["broad", "hybrid"] and len(keys) > 2:
        warnings.append(f"Query uses {len(keys)} broad keys - consider narrowing scope")
        estimated_size = "very_large"
    
    # Geographic scope warnings based on query text
    if any(term in query.lower() for term in ["country", "state", "province", "continent"]):
        warnings.append("Large geographic area detected - query may timeout")
    
    return {
        "warnings": warnings,
        "estimated_size": estimated_size,
        "safe_to_proceed": len(warnings) <= 2,
        "mode": mode,
        "tag_count": len(tags),
        "key_count": len(keys)
    }


async def nl_to_overpass(question: str) -> str:
    """
    Enhanced OSM query generation with broad/specific intent analysis.
    
    1.  search_tags - DataFrame[tag, score]
    2.  analyze_query_intent - detect broad vs specific intent
    3.  llm_filter - structured response with mode/tags/keys
    4.  extract_location - free-text place - get_boundary()
    5.  validate_query_scope - check for potential issues
    6.  build appropriate query (specific/broad/hybrid)
    """
    print(f"\n🚀 Converting NL query to Overpass-QL: '{question}'")
    
    # Step 1
    print("\n📋 Step 1: Finding candidate OSM tags")
    cand_df: pd.DataFrame = search_tags(question, k=50)        # columns: tag, score
    if cand_df.empty:
        raise ValueError("search_tags produced an empty DataFrame.")

    # Step 2
    print("\n📋 Step 2: Analyzing query intent")
    intent_analysis = analyze_query_intent(question, cand_df['tag'].tolist())

    # Step 3
    print("\n📋 Step 3: Filtering tags with enhanced LLM")
    llm_result: dict = llm_filter(question, cand_df, intent_analysis)
    if not llm_result.get('tags') and not llm_result.get('keys'):
        raise ValueError("llm_filter rejected all tag candidates.")

    # Extract results from LLM
    mode = llm_result.get('mode', 'specific')
    selected_tags = llm_result.get('tags', [])
    selected_keys = llm_result.get('keys', [])
    
    # Extract broad keys if LLM detected broad intent
    if intent_analysis.get('broad_intent', False) and not selected_keys:
        selected_keys = extract_broad_keys_from_tags(selected_tags, intent_analysis)
        if selected_keys:
            mode = 'broad' if not selected_tags else 'hybrid'

    print(f"   Final selection - Mode: {mode}, Tags: {selected_tags}, Keys: {selected_keys}")

    # Step 4
    print("\n📋 Step 4: Extracting location and getting boundary")
    place_str  = extract_location(question)                     # e.g. "Berlin, Germany"
    geo_info   = await get_boundary(place_str)                  # area_id / poly_string

    # Step 5
    print("\n📋 Step 5: Validating query scope")
    validation = validate_query_scope(question, mode, selected_tags, selected_keys)
    
    if validation.get('warnings'):
        for warning in validation['warnings']:
            print(f"   ⚠️  Warning: {warning}")
    
    if not validation.get('safe_to_proceed', True):
        print(f"   🚨 Query may be too broad - proceeding with caution")

    # Step 6
    print("\n📋 Step 6: Building Overpass query")
    print(f"   Query mode: {mode}")
    
    # Build query based on mode - heavily favor specific mode
    if mode == 'specific' and selected_tags:
        print("   Using specific key=value matching")
        ql = _build_specific_query(geo_info, selected_tags)
    elif mode == 'broad' and selected_keys:
        print("   Using broad key-only matching")
        ql = _build_broad_query(geo_info, selected_keys)
    elif mode == 'hybrid' and (selected_tags or selected_keys):
        print("   Using hybrid specific+broad matching")
        ql = _build_hybrid_query(geo_info, selected_tags, selected_keys)
    else:
        # Fallback to specific mode with top 2 available tags
        print("   Falling back to specific mode with top 2 tags")
        fallback_tags = selected_tags if selected_tags else cand_df['tag'].tolist()[:2]
        ql = _build_specific_query(geo_info, fallback_tags)

    query = textwrap.dedent(ql).strip()
    print(f"✅ Generated Overpass query:\n{query}")
    return query



def extract_location(query: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Extract the location name from the query.
    """
    if client is None:
        raise ValueError("OpenAI client not initialized. Call initialize_models() first.")
    
    print(f"📍 Extracting location from query: '{query}'")
        
    prompt = (
        "You are an assistant with a very simple task.\n"
        "The user has a natural language query, and you need to extract the full location name from it.\n"
        "For example, if the user asks for San Francisco, you should return the most likely, most populous location, like San Francisco, CA, USA, not San Francisco, Mexico.\n"
        "Always return the full location name, not just a partial match.\n"
        f"User query:\n{query}\n\n"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
    print(f"✅ Extracted location: '{raw}'")
    return raw


def _extract_metadata_from_features(features: List[Dict]) -> Dict[str, Any]:
    """
    Extract metadata from GeoJSON features using pandas for fast analysis.
    Excludes geometry data to avoid memory overhead from complex polygons.
    
    Args:
        features: List of GeoJSON features
        
    Returns:
        Metadata dictionary with field analysis
    """
    if not features:
        return {
            "attributes": {},
            "extractedAt": time.time(),
            "totalFeatures": 0
        }
    
    print(f"[OSM-METADATA] Extracting metadata from {len(features)} features...")
    start_time = time.time()
    
    # Extract just the properties (no geometries) for memory efficiency
    properties_list = []
    for feature in features:
        props = feature.get("properties", {})
        if props:  # Only include features with properties
            properties_list.append(props)
    
    if not properties_list:
        return {
            "attributes": {},
            "extractedAt": time.time(),
            "totalFeatures": len(features)
        }
    
    # Create DataFrame from properties only (very fast, no geometry data)
    try:
        df = pd.DataFrame(properties_list)
        print(f"[OSM-METADATA] Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"[OSM-METADATA] Error creating DataFrame: {e}")
        return {
            "attributes": {},
            "extractedAt": time.time(),
            "totalFeatures": len(features)
        }
    
    attributes = {}
    
    # Process each column efficiently
    for column in df.columns:
        try:
            series = df[column]
            null_count = series.isnull().sum()
            non_null_series = series.dropna()
            
            if len(non_null_series) == 0:
                # All null column
                attributes[column] = {
                    "fieldType": "unknown",
                    "nullCount": null_count,
                    "uniqueCount": 0,
                    "uniqueValues": [],
                    "maxValue": None,
                    "minValue": None
                }
                continue
            
            # Determine field type and convert to appropriate type
            sample_value = non_null_series.iloc[0]
            
            if pd.api.types.is_numeric_dtype(non_null_series) or (
                isinstance(sample_value, str) and 
                sample_value.replace('.', '').replace('-', '').replace('+', '').isdigit()
            ):
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(non_null_series, errors='coerce')
                    numeric_series = numeric_series.dropna()
                    
                    if len(numeric_series) > 0:
                        # Numeric field
                        all_unique_values = numeric_series.unique().tolist()
                        
                        # Sample up to 1000 unique values for display, but keep accurate min/max
                        if len(all_unique_values) <= 1000:
                            sample_unique_values = sorted(all_unique_values)
                        else:
                            # Take a stratified sample: include min, max, and random middle values
                            min_val = float(numeric_series.min())
                            max_val = float(numeric_series.max())
                            
                            # Get a random sample of middle values (excluding min/max if they exist)
                            middle_values = [v for v in all_unique_values if v != min_val and v != max_val]
                            if len(middle_values) > 998:  # Leave room for min and max
                                middle_sample = pd.Series(middle_values).sample(n=998, random_state=42).tolist()
                            else:
                                middle_sample = middle_values
                            
                            # Combine min, max, and middle sample, then sort
                            sample_unique_values = sorted([min_val, max_val] + middle_sample)
                        
                        attributes[column] = {
                            "fieldType": "number",
                            "nullCount": int(null_count),
                            "uniqueCount": len(all_unique_values),
                            "uniqueValues": sample_unique_values,
                            "maxValue": float(numeric_series.max()),
                            "minValue": float(numeric_series.min())
                        }
                        continue
                except:
                    pass  # Fall through to string processing
            
            # String field processing
            unique_values = non_null_series.value_counts().head(1000).index.tolist()
            
            attributes[column] = {
                "fieldType": "string", 
                "nullCount": int(null_count),
                "uniqueCount": int(non_null_series.nunique()),
                "uniqueValues": unique_values,
                "maxValue": None,
                "minValue": None
            }
            
        except Exception as e:
            print(f"[OSM-METADATA] Error processing column {column}: {e}")
            attributes[column] = {
                "fieldType": "unknown",
                "nullCount": 0,
                "uniqueCount": 0,
                "uniqueValues": [],
                "maxValue": None,
                "minValue": None
            }
    
    processing_time = time.time() - start_time
    print(f"[OSM-METADATA] Metadata extraction completed in {processing_time:.3f}s for {len(attributes)} attributes")
    
    return {
        "attributes": attributes,
        "extractedAt": time.time(),
        "totalFeatures": len(features)
    }


def _convert_to_geojson(osm_data: bytes) -> str:
    """
    Convert raw OSM data to GeoJSON using the Node.js script.
    """
    print(f"[Step 4] Input: OSM data ({len(osm_data)} bytes) to convert to GeoJSON")
    
    # Correctly locate convert_to_geojson.js relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converter_script = os.path.join(script_dir, "convert_to_geojson.js")
    
    if not os.path.exists(converter_script):
        # Try to find it in the parent directory if it's not in the same directory
        alt_converter_script = os.path.join(os.path.dirname(script_dir), "convert_to_geojson.js")
        if os.path.exists(alt_converter_script):
            converter_script = alt_converter_script
        else:
            raise FileNotFoundError(f"Converter script not found at {converter_script} or {alt_converter_script}")

    try:
        import subprocess
        osm_data_str = osm_data.decode('utf-8')
        result = subprocess.run(
            ["node", converter_script],
            input=osm_data_str,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        geojson_output = result.stdout
        print(f"[Step 4] Output: GeoJSON data with {len(geojson_output)} characters.")
        print(f"[Step 4] GeoJSON output preview: {geojson_output[:500]}...")
        return geojson_output
        
    except subprocess.CalledProcessError as e:
        print(f"[Step 4] Error: {e.stderr.strip() if e.stderr else 'No stderr'}")
        print("[Step 4] Please ensure Node.js and osmtogeojson are installed globally:")
        print("npm install -g osmtogeojson")
        raise RuntimeError(f"GeoJSON conversion failed: {e.stderr.strip() if e.stderr else 'No stderr'}")
    except Exception as e:
        print(f"[Step 4] Unexpected error during GeoJSON conversion: {str(e)}")
        raise


def _get_default_styling_payload(layer_type: str, layer_name: str, user_name: str = "User") -> dict:
    """Get default styling payload based on layer type."""
    
    if layer_type == 'point':
        return {
            "name": layer_name,
            "created_by": user_name,
            "layerInUse": "default",
            "default": {
                "size": 6,
                "fillColor": "#0ea5e9",
                "strokeColor": "#FFFFFF",
                "strokeWidth": 2,
                "iconType": "circle",
                "labelBy": None
            },
            "categories": {
                "fieldChosen": None,
                "showing": 10,
                "swatchChoice": "vibrant",
                "size": 6,
                "opacity": 100,
                "colorMap": {}
            },
            "colorRange": {
                "fieldChosen": None,
                "minValue": 0,
                "maxValue": 100,
                "minColor": "#E3F2FD",
                "maxColor": "#1565C0",
                "strokeColor": "#000000",
                "strokeWidth": 1,
                "labelBy": None
            },
            "sizeRange": {
                "fieldChosen": None,
                "minValue": 0,
                "maxValue": 100,
                "minSize": 2,
                "maxSize": 20,
                "fillColor": "#0ea5e9",
                "strokeColor": "#FFFFFF",
                "strokeWidth": 2,
                "labelBy": None
            },
            "heatmap": {
                "intensity": 50,
                "radius": 30,
                "minColor": "#f6dc8a",
                "maxColor": "#C62828",
                "labelBy": None
            }
        }
    elif layer_type == 'polygon':
        return {
            "name": layer_name,
            "created_by": user_name,
            "layerInUse": "default",
            "default": {
                "fillColor": "#0ea5e9",
                "strokeColor": "#0ea5e9",
                "strokeWidth": 5,
                "opacity": 30,
                "labelBy": None
            },
            "categories": {
                "fieldChosen": None,
                "showing": 10,
                "swatchChoice": "vibrant",
                "opacity": 30,
                "strokeWidth": 5,
                "colorMap": {}
            },
            "colorRange": {
                "fieldChosen": None,
                "minValue": 0,
                "maxValue": 100,
                "minColor": "#a855f7",
                "maxColor": "#f97316",
                "fillOpacity": 0.3,
                "strokeWidth": 5,
                "labelBy": None
            }
        }
    elif layer_type == 'line':
        return {
            "name": layer_name,
            "created_by": user_name,
            "layerInUse": "default",
            "default": {
                "strokeColor": "#22c55e",
                "strokeWidth": 5,
                "strokeOpacity": 1,
                "strokeDashArray": None,
                "labelBy": None
            },
            "categories": {
                "fieldChosen": None,
                "showing": 10,
                "swatchChoice": "vibrant",
                "strokeWidth": 5,
                "strokeOpacity": 1,
                "colorMap": {}
            },
            "colorRange": {
                "fieldChosen": None,
                "minValue": 0,
                "maxValue": 100,
                "minColor": "#a855f7",
                "maxColor": "#f97316",
                "strokeWidth": 5,
                "strokeOpacity": 1,
                "labelBy": None
            }
        }
    else:
        # Default fallback (polygon style)
        return {
            "name": layer_name,
            "created_by": user_name,
            "layerInUse": "default",
            "default": {
                "fillColor": "#0ea5e9",
                "strokeColor": "#0ea5e9",
                "strokeWidth": 5,
                "opacity": 100,
                "labelBy": None
            }
        }


def _insert_default_styling(conn, project_id: str, parent_id: str, layer_type: str, layer_name: str, user_name: str = "User", metadata: Dict = None):
    """Insert default styling payload for a layer with optional metadata."""
    print(f"[OSM] Creating default styling for parent_id: {parent_id}, layer_type: {layer_type}")
    
    cur = conn.cursor()
    try:
        # Get the default styling payload
        default_payload = _get_default_styling_payload(layer_type, layer_name, user_name)
        
        # Insert into project_dataset_styling table with metadata
        cur.execute("""
            INSERT INTO project_dataset_styling (
                project_id, parent_id, payload, filters, metadata
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (project_id, parent_id) 
            DO UPDATE SET 
                payload = EXCLUDED.payload,
                filters = EXCLUDED.filters,
                metadata = EXCLUDED.metadata
        """, (
            project_id,
            parent_id,
            json.dumps(default_payload),
            json.dumps([]),  # Empty filters array
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        print(f"[OSM] Successfully inserted default styling for {parent_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"[OSM] Error inserting default styling: {e}")
        raise
    finally:
        cur.close()


def _set_jwt_claims_safely(conn, user_id: str) -> None:
    """Safely set JWT claims for Supabase RLS authentication."""
    cur = conn.cursor()
    
    # Ensure we're in autocommit mode before setting session variables
    original_autocommit = conn.autocommit
    conn.autocommit = True
    
    try:
        # Set JWT claims while in autocommit mode (session settings must be outside transactions)
        cur.execute(
            "SET request.jwt.claims TO %s",
            (json.dumps({"sub": user_id, "role": "authenticated"}),),
        )
        print(f"[OSM] Successfully set JWT claims for user: {user_id}")
    except Exception as e:
        print(f"[OSM] Error setting JWT claims: {e}")
        raise
    finally:
        # Restore original autocommit setting
        conn.autocommit = original_autocommit
        cur.close()


def _insert_features_optimized(
    conn,
    project_id: str,
    user_id: str,
    features: list,
    parent_id: str,
    layer_type: str,
    original_filename: str = "OSM_Query",
):
    """Insert OSM features into project_gis_data table."""
    if not features:
        return
    
    feature_count = len(features)
    print(f"[OSM] Inserting {feature_count} {layer_type} features")
    
    # Set JWT claims safely before any transaction
    _set_jwt_claims_safely(conn, user_id)
    
    # Use a regular cursor for better performance
    cur = conn.cursor()
    
    # Disable autocommit and start a single transaction for data insertion
    original_autocommit = conn.autocommit
    conn.autocommit = False
    
    try:
        element_type = layer_type
        
        # Prepare all data for batch insert
        batch_data = []
        for idx, feat in enumerate(features):
            element_id = f"{parent_id}_{idx}"
            properties = json.dumps(feat.get("properties", {}))
            payload = json.dumps({
                "name": feat.get("properties", {}).get("name", original_filename),
                "color": "#0096C7",
                "fillOpacity": 0.1,
                "strokeWidth": 1,
                "strokeOpacity": 1.0,
            })
            
            batch_data.append((
                project_id,
                element_type,
                json.dumps(feat["geometry"]),
                properties,
                payload,
                element_id,
                original_filename,
                parent_id,
                layer_type,
            ))
        
        # Insert in batches using execute_values
        execute_values(
            cur,
            """
            INSERT INTO project_gis_data (
                id, project_id, element_type, geom, properties, payload,
                element_id, source_type, original_filename, parent_id,
                layer_type, created_at, updated_at
            ) VALUES %s
            """,
            batch_data,
            template="""(
                gen_random_uuid(), %s, %s, ST_GeomFromGeoJSON(%s), %s, %s,
                %s, 'geojson_upload', %s, %s, %s, NOW(), NOW()
            )""",
            page_size=1000
        )
        
        # Commit the transaction
        conn.commit()
        print(f"[OSM] Successfully inserted {feature_count} features")
        
    except Exception as e:
        print(f"[OSM] Error in batch insert: {e}")
        if not conn.autocommit:
            conn.rollback()
        raise
    finally:
        # Restore original autocommit setting
        conn.autocommit = original_autocommit
        cur.close()


class DataLibraryInput(BaseModel):
    """Input schema for the DataLibraryTool."""
    query: str = Field(..., description="Natural language query for OpenStreetMap data (e.g., 'Show me all parks in Berlin'). NOTE: Uses advanced semantic search to find the most relevant OSM tags.")


class DataLibraryTool(Tool):
    """Tool for adding OpenStreetMap data to the map based on a natural language query using semantic search."""

    def __init__(self, gis_socket_url: str = None):
        """
        Initialize the DataLibraryTool.
        
        Args:
            gis_socket_url: URL for the GIS socket service (optional)
        """
        tool_description = "Adds OpenStreetMap data to the map based on a natural language query using semantic search for OSM tags. Converts the query to Overpass, fetches data, converts to GeoJSON, and displays it."
        
        super().__init__(
            name="data_library",
            description=tool_description,
            args_schema=DataLibraryInput,
            gis_socket_url=gis_socket_url
        )

    async def execute(self, query: str) -> str:
        """
        Execute the DataLibraryTool pipeline with Supabase upload.
        """
        print(f"\n=== DataLibraryTool: Executing with query: '{query}' ===")
        
        # Validate session context
        if not self.user_id or not self.project_id:
            return "Error: User ID and Project ID are required for OSM data upload."
        
        try:
            # Initialize models if not already done
            if model is None:
                initialize_models()
            
            # Generate Overpass query using semantic search
            print(f"\n🚀 Converting NL query to Overpass-QL: '{query}'")
            overpass_query = await nl_to_overpass(query)
            
            # Execute query against Overpass API
            print(f"\n🌐 Executing Overpass API query...")
            async with httpx.AsyncClient() as client_http:
                resp = await client_http.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": overpass_query},
                    timeout=httpx.Timeout(300.0, read=300.0, connect=60.0)
                )
                resp.raise_for_status()
                osm_data = resp.content
            
            if not osm_data:
                return "No data received from Overpass API."
            
            # Convert OSM data to GeoJSON
            geojson_result_str = _convert_to_geojson(osm_data)
            if not geojson_result_str:
                return "Failed to convert OSM data to GeoJSON."
            
            # Validate and process GeoJSON
            try:
                geojson_payload = json.loads(geojson_result_str)
            except json.JSONDecodeError as e:
                print(f"Invalid GeoJSON produced: {e}")
                return f"Error: Produced invalid GeoJSON. {str(e)}"

            if not geojson_payload.get('features'):
                return "No features found in OSM query results."

            print(f"[OSM] Processing {len(geojson_payload['features'])} features from OSM query")

            # Split features by geometry type
            points = []
            lines = []
            polygons = []
            
            for feature in geojson_payload['features']:
                geom_type = feature["geometry"]["type"]
                if geom_type in ["Point", "MultiPoint"]:
                    points.append(feature)
                elif geom_type in ["LineString", "MultiLineString"]:
                    lines.append(feature)
                elif geom_type in ["Polygon", "MultiPolygon"]:
                    polygons.append(feature)
            
            # Find the geometry type with the highest record count
            geometry_types = []
            if points:
                geometry_types.append({"features": points, "type": "point", "name": "points", "count": len(points)})
            if lines:
                geometry_types.append({"features": lines, "type": "line", "name": "lines", "count": len(lines)})
            if polygons:
                geometry_types.append({"features": polygons, "type": "polygon", "name": "polygons", "count": len(polygons)})
            
            if not geometry_types:
                return "No valid geometry features found in OSM query results."
            
            # Sort by count (descending) and take the highest one
            highest_count_type = max(geometry_types, key=lambda x: x["count"])
            
            # Upload only the highest count geometry type to Supabase
            uploaded_layers = []
            session_key = get_session_key(self.user_id, self.project_id, self.session_id)
            
            parent_id, used_concurrent = await self._upload_to_supabase(
                highest_count_type["features"], 
                highest_count_type["type"], 
                session_key, 
                query
            )
            uploaded_layers.append({
                "type": highest_count_type["name"], 
                "parent_id": parent_id, 
                "count": highest_count_type["count"]
            })
            
            # Send WebSocket trigger only if concurrent processing wasn't used
            if not used_concurrent:
                print(f"[OSM] 📡 Sending WebSocket trigger for sequential processing")
                await self._send_websocket_trigger(uploaded_layers, session_key)
            else:
                print(f"[OSM] 🚫 Skipping WebSocket trigger - concurrent processing handled it")
            
            # Create summary message
            layer_summary = f"{uploaded_layers[0]['count']} {uploaded_layers[0]['type']}"
            total_processed = len(points) + len(lines) + len(polygons)
            
            success_msg = f"Successfully processed query '{query}' and found {total_processed} total features. Uploaded the {layer_summary} (highest count geometry type) to project. Data should render momentarily."
            print(f"Success: {success_msg}")
            print("=== End DataLibraryTool execution ===\n")
            return success_msg

        except FileNotFoundError as e:
            error_msg = f"Error during OSM processing: {str(e)}. Ensure Node.js and osmtogeojson are installed and convert_to_geojson.js is in the tools directory."
            print(f"Error: {error_msg}")
            return error_msg
        except Exception as e:
            import traceback
            error_msg = f"Error processing query '{query}': {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            print("=== End DataLibraryTool execution with error ===\n")
            return error_msg
    
    async def _upload_to_supabase(self, features: list, layer_type: str, session_key: str, query: str) -> tuple[str, bool]:
        """Upload features to Supabase and return (parent_id, used_concurrent_processing). Uses concurrent processing for point layers."""
        parent_id = str(uuid.uuid4())
        
        # 🎯 POINT LAYER OPTIMIZATION: Use concurrent processing for point layers
        if layer_type == "point":
            print(f"[OSM-CONCURRENT] 🚀 Detected point layer - using concurrent processing for {len(features)} features")
            
            try:
                # Convert OSM features to GeoJSON FeatureCollection format
                geojson_features = []
                for feature in features:
                    geojson_feature = {
                        "type": "Feature",
                        "geometry": feature.get("geometry", {}),
                        "properties": feature.get("properties", {})
                    }
                    geojson_features.append(geojson_feature)
                
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": geojson_features
                }
                
                # Create a temporary file for the concurrent endpoint
                import tempfile
                import json
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp_file:
                    json.dump(geojson_data, tmp_file)
                    tmp_file_path = tmp_file.name
                
                # Call the concurrent point upload endpoint
                ogr_service_url = os.getenv("OGR_SERVICE_URL", "http://host.docker.internal:8002")
                
                # Prepare form data for the concurrent endpoint with internal service authentication
                form_data = {
                    'parent_id': parent_id,
                    'project_id': self.project_id,
                    'user_id': self.user_id,  # Required for internal service authentication
                    'layer_name': f"OSM_{query}",
                    'session_key': session_key  # Add session key for WebSocket notifications
                }
                
                # Prepare file data
                with open(tmp_file_path, 'rb') as f:
                    files = {'geojson_file': (f'{query}_points.geojson', f, 'application/json')}
                    
                    # Use internal service token for authentication
                    internal_service_token = os.getenv("INTERNAL_SERVICE_TOKEN")
                    if not internal_service_token:
                        raise Exception("INTERNAL_SERVICE_TOKEN environment variable not set")
                    
                    headers = {
                        'Authorization': f'Bearer {internal_service_token}'
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{ogr_service_url}/api/concurrent-point-upload",
                            data=form_data,
                            files=files,
                            headers=headers,
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            raise Exception(f"Concurrent upload failed: {response.status_code} - {response.text}")
                        
                        result = response.json()
                        print(f"[OSM-CONCURRENT] ✅ Concurrent processing completed in {result.get('total_processing_time', 0):.2f}s")
                        print(f"[OSM-CONCURRENT] Supabase: {result['supabase_result']['features_uploaded']} features")
                        print(f"[OSM-CONCURRENT] Parquet: {result['parquet_result']['file_size_bytes']} bytes")
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                print(f"[OSM-CONCURRENT] 🎉 Successfully used concurrent processing for point layer: {parent_id}")
                print(f"[OSM-CONCURRENT] 🚫 Skipping WebSocket trigger - concurrent endpoint will handle it")
                return parent_id, True  # Return success flag for concurrent processing
                
            except Exception as concurrent_error:
                print(f"[OSM-CONCURRENT] ❌ Concurrent processing failed: {concurrent_error}")
                print(f"[OSM-CONCURRENT] 🔄 Falling back to sequential processing...")
                # Fall through to sequential processing
        
        # 📊 SEQUENTIAL PROCESSING: For non-point layers or concurrent fallback
        print(f"[OSM-SEQUENTIAL] Using sequential processing for {layer_type} layer with {len(features)} features")
        
        try:
            conn = _get_connection()
            
            # Set JWT claims for RLS authentication
            _set_jwt_claims_safely(conn, self.user_id)
            
            # Insert features using optimized batch insert
            _insert_features_optimized(
                conn, self.project_id, self.user_id, features, parent_id, layer_type, f"{query}"
            )
            
            # Extract metadata from features for styling
            print(f"[OSM-SEQUENTIAL] Extracting metadata from {len(features)} features...")
            metadata_start_time = time.time()
            metadata = _extract_metadata_from_features(features)
            metadata_end_time = time.time()
            print(f"[OSM-SEQUENTIAL] Metadata extraction completed in {metadata_end_time - metadata_start_time:.2f} seconds")
            
            # Insert default styling with metadata
            _insert_default_styling(
                conn, self.project_id, parent_id, layer_type, f"{query}", "User", metadata
            )
            
            print(f"[OSM-SEQUENTIAL] Uploaded {len(features)} {layer_type} features with parent_id {parent_id}")
                
        except Exception as e:
            print(f"[OSM-SEQUENTIAL] Error uploading data: {e}")
            raise
        finally:
            if conn:
                conn.close()
        
        return parent_id, False  # Return False for sequential processing (needs WebSocket trigger)
    
    async def _send_websocket_trigger(self, layers: list, session_key: str):
        """Send WebSocket trigger to frontend."""
        try:
            # Create parent_ids mapping
            parent_ids = {}
            for layer in layers:
                layer_key = f"OSM_{layer['type']}_{layer['parent_id'][:8]}"
                parent_ids[layer_key] = layer['parent_id']
            
            load_layer_command = {
                "type": "load_layer",
                "payload": {
                    "parent_ids": parent_ids,
                    "project_id": self.project_id,
                    "source": "osm_query"
                }
            }
            
            print(f"[OSM] Sending load_layer command with parent_ids: {list(parent_ids.keys())}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{WEBSOCKET_SERVICE_URL}/api/send-message",
                    json={
                        "text": json.dumps(load_layer_command),
                        "session_key": session_key
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                
                print(f"[OSM] WebSocket response status: {response.status_code}")
                response.raise_for_status()
                print(f"[OSM] WebSocket load_layer command sent successfully")
                    
        except Exception as e:
            print(f"[OSM] Error sending WebSocket trigger: {e}")


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) != 2:
#         print("Usage: python add_osm.py 'Parks in Berlin'")
#         sys.exit(1)
    
#     query = sys.argv[1]
    
#     # Initialize models
#     initialize_models()
    
#     # Generate Overpass query
#     overpass_query = asyncio.run(nl_to_overpass(query))
    
#     # Execute query against Overpass API
#     async def run_query():
#         async with httpx.AsyncClient() as client_http:
#             resp = await client_http.post(
#                 "https://overpass-api.de/api/interpreter",
#                 data={"data": overpass_query},
#                 timeout=httpx.Timeout(300.0, read=300.0, connect=30.0)
#             )
#             resp.raise_for_status()
#             return resp.json()
    
#     result = asyncio.run(run_query())
    
#     print(f"\n📊 Found {len(result['elements'])} features")
#     result_json = json.dumps(result, indent=2)
#     if len(result_json) > 1000:
#         print(f"📄 First 1000 characters of result:\n{result_json[:1000]}...")
#     else:
#         print(f"📄 Result:\n{result_json}")