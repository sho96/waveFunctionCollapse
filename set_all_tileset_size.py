import os
import json

tilesets_dir = "./tilesets"

tileset_names = os.listdir(tilesets_dir)

for tileset_name in tileset_names:
  with open(os.path.join(tilesets_dir, tileset_name, "tileset.json"), "r") as f:
    tileset = json.load(f)
  tileset["grid_size"] = [9 * 3, 16 * 3]

  with open(os.path.join(tilesets_dir, tileset_name, "tileset.json"), "w") as f:
    json.dump(tileset, f, indent=2) 
  