import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os
import re
import math


# Define the function to extract data from filenames
def extract_data_from_filenames(directory):
    data = []
    pattern = re.compile(r'heading=([-\d.]+)&.*&location=([-\d.]+),([-\d.]+)&')

    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            heading = float(match.group(1))
            location_y = float(match.group(2))
            location_x = float(match.group(3))
            data.append({'filename': filename, 'location_x': location_x, 'location_y': location_y, 'heading': heading})

    data.sort(key=lambda item: item['location_x'])
    return data


# Define the function to plot the data
def plot_data_files(data):
    x_coords = [item['location_x'] for item in data]
    y_coords = [item['location_y'] for item in data]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c='green', marker='o')

    for item in data:
        plt.annotate(f"H{item['heading']}", (item['location_x'], item['location_y']))

    plt.title('Locations from Filenames')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def convert_to_wgs84():
    # Load your Excel file into a pandas DataFrame
    df = pd.read_excel('TLV_Trees.xlsx')

    # Assuming your DataFrame has columns 'x' and 'y' for the coordinates
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.point_X, df.point_Y))

    # Set the current CRS to EPSG:2039 (Israeli Transverse Mercator)
    gdf.set_crs(epsg=2039, inplace=True)

    # Reproject/transform the GeoDataFrame to WGS84 (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)

    # Extract the transformed coordinates and add them to the DataFrame
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y

    # Save the DataFrame with the new WGS84 coordinates to a new Excel file
    df.to_excel('TLV_Trees_WGS.xlsx', index=False)


# Function to adjust coordinates based on angle
def adjust_coordinates(x, y, angle, distance=0.0001):
    if angle == 0:        # North
        y += distance
    elif angle == 60:     # North-East
        x += distance * math.cos(math.radians(60))
        y += distance * math.sin(math.radians(60))
    elif angle == 120:    # South-East
        x += distance * math.cos(math.radians(120))
        y += distance * math.sin(math.radians(120))
    elif angle == 180:    # South
        y -= distance
    elif angle == 240:    # South-West
        x += distance * math.cos(math.radians(240))
        y += distance * math.sin(math.radians(240))
    elif angle == 300:    # South-West
        x += distance * math.cos(math.radians(300))
        y += distance * math.sin(math.radians(300))
    return (x, y)


# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def match_point_to_image(points, images):
    # Initialize a list to store matched pairs
    matched_pairs = []

    # Iterate over points in the first list
    for point in points:
        min_distance = float('inf')
        matched_image = None

        # Iterate over points in the images list
        for image in images:
            adjusted_point = adjust_coordinates(image['location_x'], image['location_y'], image['heading'])
            distance = calculate_distance((point['location_x'], point['location_y']), (adjusted_point[0], adjusted_point[1]))

            # Check if this point is closer than previously found
            if distance < min_distance:
                min_distance = distance
                matched_image = image

        # Store the matched pair
        if matched_image and min_distance < 0.001:
            matched_pairs.append((point, matched_image))

    return matched_pairs


def plot_trees_points(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['y'], c='blue', marker='o')
    plt.title('WGS84 Coordinates')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


df = pd.read_excel('TLV_Trees_WGS.xlsx')
plot_trees_points(df)

coordinates = df[['x', 'y', 'Tree_name']].to_numpy()
points = []
for item in coordinates:
    points.append({'location_x': float(item[0]), 'location_y': float(item[1]), 'tree_name': item[2]})

points.sort(key=lambda item: item['location_x'])

for item in points:
    print(item)

data_images = extract_data_from_filenames('gsv tel aviv')
for item in data_images:
    print(item)
plot_data_files(data_images)

matched_pairs = match_point_to_image(points, data_images)
for item in matched_pairs:
    print(item)
print(len(matched_pairs))