import os
import json

# Customize these
client_count = 4  #Change with the number of clients
client_prefix = "site"
startup_dir = "startup"
app_name = "app"

# Create startup folder
os.makedirs(startup_dir, exist_ok=True)

# Server config
server_config = {
    "name": "server",
    "org": "central",
    "role": "server",
    "app": app_name
}
with open(os.path.join(startup_dir, "fed_server.json"), "w") as f:
    json.dump(server_config, f, indent=2)

# Client configs
for i in range(client_count):
    client_name = f"{client_prefix}-{i+1}"
    client_config = {
        "name": client_name,
        "org": f"hospital_{chr(97+i)}",  # hospital_a, hospital_b, ...
        "role": "client",
        "app": app_name
    }
    with open(os.path.join(startup_dir, f"fed_client_{i}.json"), "w") as f:
        json.dump(client_config, f, indent=2)

print(f"Generated startup/ configs for {client_count} clients and 1 server.")

