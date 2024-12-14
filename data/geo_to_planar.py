import geopandas as gpd

# Load GeoJSON file
geo_df = gpd.read_file('grid.geojson')

# Ensure the original CRS is set to WGS84 (EPSG:4326)
geo_df = geo_df.set_crs(epsg=4326)

# Reproject to UTM or local projection (e.g., EPSG:26971 for Chicago)
projected_df = geo_df.to_crs(epsg=26971)

# Access projected coordinates
for idx, row in projected_df.iterrows():
    print(row['geometry'])  # Now in planar coordinates

projected_df.to_file("grid_projected.geojson", driver="GeoJSON")

print(projected_df.crs.axis_info[0].unit_name)