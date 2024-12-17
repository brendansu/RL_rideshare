import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pickle

# Load CSV file
file_path = "chi_0610.csv"  # Replace with your file path
df = pd.read_csv(file_path, parse_dates=["Trip Start Timestamp", "Trip End Timestamp"])

# Process time data
df['Time Bin'] = df['Trip Start Timestamp'].dt.hour * 4 + df['Trip Start Timestamp'].dt.minute // 15

print("Unique Time Bins:", df['Time Bin'].unique())

# Optionally, ensure all time bins fall within the expected range (0–95)
assert df['Time Bin'].between(0, 95).all(), "Error: Some timestamps are outside the 15-minute bin range!"

# Show updated dataframe
print(df[['Trip Start Timestamp', 'Time Bin']].head())

# Process coordinate data
grid_map = gpd.read_file("grid_projected.geojson")
print("Grid CRS:", grid_map.crs)  # Confirm the CRS of the grid map

# Create GeoDataFrame for pickup points
pickup_gdf = gpd.GeoDataFrame(df,
                              geometry=gpd.points_from_xy(df["Pickup Centroid Longitude"], df["Pickup Centroid Latitude"]),
                              crs="EPSG:4326")  # Original CRS (WGS84)

# Create GeoDataFrame for dropoff points
dropoff_gdf = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["Dropoff Centroid Longitude"], df["Dropoff Centroid Latitude"]),
                               crs="EPSG:4326")

# Reproject both GeoDataFrames to match the grid CRS
pickup_gdf = pickup_gdf.to_crs(grid_map.crs)
dropoff_gdf = dropoff_gdf.to_crs(grid_map.crs)

# Confirm reprojected CRS
print("Pickup CRS:", pickup_gdf.crs)
print("Dropoff CRS:", dropoff_gdf.crs)

# Spatial join to assign pickup grid cells
pickup_gdf = gpd.sjoin(pickup_gdf, grid_map, how="left", predicate="intersects")
pickup_gdf = pickup_gdf.rename(columns={"id": "Pickup Grid ID"})

# Spatial join to assign dropoff grid cells
dropoff_gdf = gpd.sjoin(dropoff_gdf, grid_map, how="left", predicate="intersects")
dropoff_gdf = dropoff_gdf.rename(columns={"id": "Dropoff Grid ID"})

# Add grid IDs back to the original DataFrame
df["Pickup Grid ID"] = pickup_gdf["Pickup Grid ID"]
df["Dropoff Grid ID"] = dropoff_gdf["Dropoff Grid ID"]

# generating the final filtered dataframe
columns_to_keep = [
    "Time Bin", "Pickup Grid ID", "Dropoff Grid ID", "Fare", "Shared Trip Authorized"
]

df_filtered = pd.DataFrame(df[columns_to_keep])
df_filtered = df_filtered.dropna(subset=["Pickup Grid ID", "Dropoff Grid ID"])
df_filtered.to_csv('req_filtered.csv', index = False)