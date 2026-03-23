import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

DIRECTIONS = ("left", "down", "right", "up")
OFFSETS = {
    "left": (-1, 0),
    "down": (0, 1),
    "right": (1, 0),
    "up": (0, -1),
}
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "down": "up",
    "right": "left",
    "up": "down",
}
BASE_DIR = Path(__file__).resolve().parent
TILESETS_DIR = BASE_DIR / "tilesets"

# Optional overrides. Leave these as None to use the interactive prompts/defaults.
tileset_name = None  # Example: "knots"
start_generation_pos = None  # Example: (5, 5)
start_tile_name = None


class ContradictionError(ValueError):
    pass


@dataclass(frozen=True)
class Tile:
    name: str
    image: np.ndarray
    connections: Optional[tuple[bool, bool, bool, bool]]
    weight: float = 1.0


class ImageGrid:
    def __init__(self, cell_size, grid_size):
        self.cell_size = tuple(cell_size)
        self.grid_size = tuple(grid_size)
        height = self.cell_size[1] * self.grid_size[1]
        width = self.cell_size[0] * self.grid_size[0]
        self.whole_img = np.zeros((height, width, 3), dtype=np.float32)

    def clear(self):
        self.whole_img.fill(0)

    def fill(self, pos, img):
        real_origin = np.array(pos) * self.cell_size
        ending_point = (real_origin[0] + img.shape[1], real_origin[1] + img.shape[0])
        self.whole_img[real_origin[1]:ending_point[1], real_origin[0]:ending_point[0]] = img / 255.0

    def show(self, winname="grid", waitkey=0, ratio=1):
        height, width = self.whole_img.shape[:2]
        resized = cv2.resize(
            self.whole_img,
            dsize=(int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(winname, resized)
        cv2.waitKey(waitkey)


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def list_tilesets(tilesets_dir):
    if not tilesets_dir.exists():
        return []
    return sorted(
        path.name
        for path in tilesets_dir.iterdir()
        if path.is_dir() and (path / "tileset.json").exists()
    )


def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load tile image: {image_path}")
    return image


def parse_connections(connections):
    return tuple(bool(connections[direction]) for direction in DIRECTIONS)


def load_tiles(config, base_dir):
    tiles = []
    for tile_config in config["tiles"]:
        if "rotate" in tile_config:
            raise ValueError(
                "Runtime tile rotation is no longer supported. "
                "Pre-generate rotated images in the tileset instead."
            )
        image_path = base_dir / tile_config["image"]
        image = load_image(image_path)
        tiles.append(
            Tile(
                name=tile_config["name"],
                image=image,
                connections=(
                    parse_connections(tile_config["connections"])
                    if "connections" in tile_config
                    else None
                ),
                weight=float(tile_config.get("weight", 1.0)),
            )
        )
    return tiles


def prompt_user(prompt_text, default_value):
    prompt_suffix = f" [{default_value}]" if default_value not in (None, "") else ""
    try:
        user_input = input(f"{prompt_text}{prompt_suffix}: ").strip()
    except EOFError:
        user_input = ""
    return user_input or default_value


def prompt_for_tileset_name():
    available_tilesets = list_tilesets(TILESETS_DIR)
    if not available_tilesets:
        raise FileNotFoundError(f"No tilesets found in {TILESETS_DIR}")

    default_tileset = tileset_name or available_tilesets[0]
    while True:
        chosen_tileset = prompt_user(
            f"Tileset name (available: {', '.join(available_tilesets)})",
            default_tileset,
        )
        if chosen_tileset in available_tilesets:
            return chosen_tileset
        print(f"Unknown tileset '{chosen_tileset}'. Choose one of: {', '.join(available_tilesets)}")


def parse_start_position(raw_value):
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError("Start position must look like x,y")
    return int(parts[0]), int(parts[1])


def get_middle_position(grid_size):
    width, height = grid_size
    return width // 2, height // 2


def prompt_for_start_position(grid_size):
    if start_generation_pos is not None:
        return tuple(start_generation_pos)

    middle_pos = get_middle_position(grid_size)
    default_value = f"{middle_pos[0]},{middle_pos[1]}"
    while True:
        raw_value = prompt_user("Start generation position x,y", default_value)
        try:
            return parse_start_position(raw_value)
        except ValueError as error:
            print(f"Invalid start position '{raw_value}': {error}")


def build_adjacency_rules(tiles):
    adjacency_rules = []
    for source_tile in tiles:
        if source_tile.connections is None:
            raise ValueError("Cannot derive adjacency rules without tile connections")
        tile_rules = {}
        for direction_index, direction in enumerate(DIRECTIONS):
            expected_connection = source_tile.connections[direction_index]
            opposite_direction = OPPOSITE_DIRECTIONS[direction]
            opposite_index = DIRECTIONS.index(opposite_direction)
            allowed_neighbor_indices = [
                neighbor_index
                for neighbor_index, neighbor_tile in enumerate(tiles)
                if neighbor_tile.connections[opposite_index] == expected_connection
            ]
            tile_rules[direction] = allowed_neighbor_indices
        adjacency_rules.append(tile_rules)
    return adjacency_rules


def build_adjacency_rules_from_config(config, tiles):
    tile_name_to_index = {tile.name: index for index, tile in enumerate(tiles)}
    adjacency_rules = []

    for tile_config in config["tiles"]:
        if "neighbors" not in tile_config:
            raise ValueError(
                "Every tile must define 'neighbors' when using explicit adjacency-rule mode"
            )

        tile_rules = {}
        for direction in DIRECTIONS:
            neighbor_names = tile_config["neighbors"].get(direction, [])
            try:
                tile_rules[direction] = [tile_name_to_index[name] for name in neighbor_names]
            except KeyError as error:
                raise ValueError(
                    f"Unknown neighbor tile '{error.args[0]}' referenced by '{tile_config['name']}'"
                ) from error
        adjacency_rules.append(tile_rules)

    return adjacency_rules


def resolve_adjacency_rules(config, tiles):
    if all("neighbors" in tile_config for tile_config in config["tiles"]):
        return build_adjacency_rules_from_config(config, tiles)
    return build_adjacency_rules(tiles)


def validate_start_position(start_pos, grid_size):
    x, y = start_pos
    width, height = grid_size
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"start_generation_pos {start_pos} is outside the grid bounds {(width, height)}"
        )


def create_option_grid(grid_size, tile_count):
    width, height = grid_size
    all_options = list(range(tile_count))
    return [[all_options.copy() for _ in range(width)] for _ in range(height)]


def choose_weighted_tile(option_indices, tiles):
    weights = [max(tiles[index].weight, 0.0) for index in option_indices]
    if sum(weights) <= 0:
        return random.choice(option_indices)
    return random.choices(option_indices, weights=weights, k=1)[0]


def propagate(options, adjacency_rules, start_positions):
    height = len(options)
    width = len(options[0])
    queue = deque(start_positions)

    while queue:
        x, y = queue.popleft()
        current_options = options[y][x]

        for direction, (dx, dy) in OFFSETS.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            allowed_neighbor_options = set()
            for tile_index in current_options:
                allowed_neighbor_options.update(adjacency_rules[tile_index][direction])

            new_neighbor_options = [
                tile_index
                for tile_index in options[ny][nx]
                if tile_index in allowed_neighbor_options
            ]

            if not new_neighbor_options:
                raise ContradictionError(f"No valid tiles remain at position {(nx, ny)}")

            if len(new_neighbor_options) < len(options[ny][nx]):
                options[ny][nx] = new_neighbor_options
                queue.append((nx, ny))


def find_lowest_entropy_cell(options):
    best_pos = None
    best_option_count = None

    for y, row in enumerate(options):
        for x, cell_options in enumerate(row):
            option_count = len(cell_options)
            if option_count == 1:
                continue
            if best_option_count is None or option_count < best_option_count:
                best_option_count = option_count
                best_pos = (x, y)

    return best_pos


def render_collapsed_tiles(grid, options, tiles):
    grid.clear()
    for y, row in enumerate(options):
        for x, cell_options in enumerate(row):
            if len(cell_options) == 1:
                grid.fill((x, y), tiles[cell_options[0]].image)


def resolve_start_settings(config):
    configured_start_tile = config.get("start_tile")
    grid_size = tuple(config["grid_size"])

    resolved_start_pos = prompt_for_start_position(grid_size)
    resolved_start_tile = start_tile_name or configured_start_tile
    return tuple(resolved_start_pos), resolved_start_tile


def get_tileset_config_path():
    selected_tileset = prompt_for_tileset_name()
    return TILESETS_DIR / selected_tileset / "tileset.json"


def run_generation_attempt(config, tiles, adjacency_rules):
    grid_size = tuple(config["grid_size"])
    cell_size = tuple(config["cell_size"])
    display_ratio = config.get("display_ratio", 1)
    show_preview = config.get("show_preview", True)

    start_pos, initial_tile_name = resolve_start_settings(config)
    validate_start_position(start_pos, grid_size)

    options = create_option_grid(grid_size, len(tiles))
    if initial_tile_name is not None:
        tile_name_to_index = {tile.name: index for index, tile in enumerate(tiles)}
        if initial_tile_name not in tile_name_to_index:
            raise ValueError(f"Unknown start tile '{initial_tile_name}'")
        initial_tile_index = tile_name_to_index[initial_tile_name]
    else:
        initial_tile_index = choose_weighted_tile(options[start_pos[1]][start_pos[0]], tiles)
    options[start_pos[1]][start_pos[0]] = [initial_tile_index]
    propagate(options, adjacency_rules, [start_pos])

    grid = ImageGrid(cell_size, grid_size)

    total_cells = grid_size[0] * grid_size[1]
    for _ in tqdm(range(total_cells - 1)):
        next_pos = find_lowest_entropy_cell(options)
        if next_pos is None:
            break

        x, y = next_pos
        chosen_tile = choose_weighted_tile(options[y][x], tiles)
        options[y][x] = [chosen_tile]
        propagate(options, adjacency_rules, [next_pos])

        if show_preview:
            render_collapsed_tiles(grid, options, tiles)
            grid.show(waitkey=1, ratio=display_ratio)

    render_collapsed_tiles(grid, options, tiles)
    if show_preview:
        grid.show(ratio=display_ratio)


def main():
    config_path = get_tileset_config_path()
    config = load_config(config_path)
    tiles = load_tiles(config, config_path.parent)
    adjacency_rules = resolve_adjacency_rules(config, tiles)
    max_attempts = int(config.get("max_attempts", 10))

    for attempt in range(1, max_attempts + 1):
        try:
            run_generation_attempt(config, tiles, adjacency_rules)
            return
        except ContradictionError:
            if attempt == max_attempts:
                raise
            print(f"Generation contradicted on attempt {attempt}/{max_attempts}, retrying...")


if __name__ == "__main__":
    main()
