import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "ENTER PATH INTERMEDIATE CSV - THE ONE CONTAINING ONLY " 
df = pd.read_csv(csv_path, sep=";", skiprows=1, header=None, names=['Activity Name', 'Activity ID', 'Previous Activity IDs'])

# Check required columns
assert df.columns[:3].tolist() == ['Activity Name', 'Activity ID', 'Previous Activity IDs'], "Incorrect column names"

# === 2. Create the directed graph ===
G = nx.DiGraph()

# Add nodes with labels
for _, row in df.iterrows():
    G.add_node(row['Activity ID'], label=row['Activity Name'])

# Add edges (precedence relationships)
for _, row in df.iterrows():
    current_id = row['Activity ID']
    prev_ids = str(row['Previous Activity IDs']).split(',')
    
    for prev_id in prev_ids:
        prev_id = prev_id.strip()
        if prev_id and prev_id != 'N/A':
            G.add_edge(prev_id, current_id)

# === 3. Visualize the DAG ===
plt.figure(figsize=(5, 5))
pos = nx.spring_layout(G)
# labels = nx.get_node_attributes(G, 'label')
labels = {node: node for node in G.nodes}
nx.draw(
    G, pos,
    with_labels=True,
    labels=labels,
    node_color="lightblue",
    edge_color="gray",
    node_size=200,
    font_size=9,
    font_weight="bold",
    arrows=True
)
plt.title("DAG Visualization")
plt.show()
