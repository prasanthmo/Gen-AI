import pandas as pd
import networkx as nx

# Load the dataset
file_path = r'C:\Users\lahar\Downloads\airlines_flights_dataset.csv'
df = pd.read_csv(file_path)

# Step 1: Initialize a directed graph
G = nx.DiGraph()

# Step 2: Add edges to the graph (source -> destination)
for index, row in df.iterrows():
    G.add_edge(row['Source'], row['Destination'], airline=row['Airline Name'], flight_no=row['Flight No.'])

# Optional: Add additional flights if necessary
#G.add_edge('Boston', 'New York', airline='Delta Airlines', flight_no='DL123')
#G.add_edge('Boston', 'London', airline='British Airways', flight_no='BA456')
#G.add_edge('New York', 'Boston', airline='American Airlines', flight_no='AA789')

# Step 3: Get user input for the source and destination cities
source_city = input("Enter the source city: ").strip()
destination_city = input("Enter the destination city: ").strip()

# Step 4: Find the shortest path (in terms of minimum stops) using Breadth-First Search
try:
    shortest_path = nx.shortest_path(G, source=source_city, target=destination_city)
    print(f"\nThe shortest path (minimum stops) from {source_city} to {destination_city} is: {shortest_path}")
    print(f"Number of stops: {len(shortest_path) - 2} (excluding start and end cities)")
    
    # Show details for each leg of the trip
    print("\nFlight details for the shortest path:")
    for i in range(len(shortest_path) - 1):
        source = shortest_path[i]
        destination = shortest_path[i + 1]
        flight_info = G.get_edge_data(source, destination)
        print(f"Flight from {source} to {destination}: Airline = {flight_info['airline']}, Flight No. = {flight_info['flight_no']}")

except nx.NetworkXNoPath:
    print(f"No available path from {source_city} to {destination_city}.")
