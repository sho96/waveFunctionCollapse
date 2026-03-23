import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import waveFunctionCollapse as w

DEFAULT_FPS = 12
DEFAULT_SCALE = 4
DEFAULT_MAX_ATTEMPTS = 30
DEFAULT_MAX_DIMENSION = 2048


def choose_weighted_tile(option_indices, tiles, rng):
    weights = [max(tiles[index].weight, 0.0) for index in option_indices]
    if sum(weights) <= 0:
        return rng.choice(option_indices)
    return rng.choices(option_indices, weights=weights, k=1)[0]


def get_base_frame_size(config):
    grid_width, grid_height = config["grid_size"]
    cell_width, cell_height = config["cell_size"]
    return grid_width * cell_width, grid_height * cell_height


def clamp_even(value):
    return max(2, value - (value % 2))


def resolve_output_size(config, requested_scale, max_dimension):
    base_width, base_height = get_base_frame_size(config)
    width = base_width * requested_scale
    height = base_height * requested_scale

    largest_dimension = max(width, height)
    if largest_dimension > max_dimension:
        shrink = max_dimension / float(largest_dimension)
        width = int(width * shrink)
        height = int(height * shrink)

    return clamp_even(width), clamp_even(height)


def make_frame(grid, output_size):
    frame = np.clip(grid.whole_img * 255.0, 0, 255).astype(np.uint8)
    interpolation = cv2.INTER_NEAREST
    if output_size[0] < frame.shape[1] or output_size[1] < frame.shape[0]:
        interpolation = cv2.INTER_AREA
    return cv2.resize(
        frame,
        dsize=output_size,
        interpolation=interpolation,
    )


def run_generation(config, tiles, adjacency_rules, seed, output_size, frame_callback=None):
    rng = random.Random(seed)
    grid_size = tuple(config["grid_size"])
    cell_size = tuple(config["cell_size"])
    start_pos = tuple(config.get("start_generation_pos", w.get_middle_position(grid_size)))
    start_tile_name = config.get("start_tile")

    w.validate_start_position(start_pos, grid_size)
    options = w.create_option_grid(grid_size, len(tiles))

    if start_tile_name is not None:
        tile_name_to_index = {tile.name: index for index, tile in enumerate(tiles)}
        initial_tile_index = tile_name_to_index[start_tile_name]
    else:
        initial_tile_index = choose_weighted_tile(options[start_pos[1]][start_pos[0]], tiles, rng)

    options[start_pos[1]][start_pos[0]] = [initial_tile_index]
    w.propagate(options, adjacency_rules, [start_pos])

    grid = w.ImageGrid(cell_size, grid_size)
    if frame_callback is not None:
        w.render_collapsed_tiles(grid, options, tiles)
        frame_callback(make_frame(grid, output_size))

    total_cells = grid_size[0] * grid_size[1]
    for _ in range(total_cells - 1):
        next_pos = w.find_lowest_entropy_cell(options)
        if next_pos is None:
            break

        x, y = next_pos
        chosen_tile = choose_weighted_tile(options[y][x], tiles, rng)
        options[y][x] = [chosen_tile]
        w.propagate(options, adjacency_rules, [next_pos])

        if frame_callback is not None:
            w.render_collapsed_tiles(grid, options, tiles)
            frame_callback(make_frame(grid, output_size))

    w.render_collapsed_tiles(grid, options, tiles)
    final_frame = make_frame(grid, output_size)
    if frame_callback is not None:
        frame_callback(final_frame)
    return final_frame


def find_successful_seed(config, tiles, adjacency_rules, output_size, base_seed, max_attempts):
    for attempt in range(max_attempts):
        seed = base_seed + attempt
        try:
            run_generation(config, tiles, adjacency_rules, seed, output_size)
            return seed
        except w.ContradictionError:
            continue
    raise RuntimeError(
        f"Could not complete generation after {max_attempts} attempts starting from seed {base_seed}"
    )


def open_video_writer(output_path, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def render_tileset_video(tileset_name, output_dir, fps, scale, max_attempts, base_seed, max_dimension):
    config_path = w.TILESETS_DIR / tileset_name / "tileset.json"
    config = w.load_config(config_path)
    tiles = w.load_tiles(config, config_path.parent)
    adjacency_rules = w.resolve_adjacency_rules(config, tiles)
    output_size = resolve_output_size(config, scale, max_dimension)

    seed = find_successful_seed(config, tiles, adjacency_rules, output_size, base_seed, max_attempts)
    print(f"[{tileset_name}] using seed {seed}")
    print(f"[{tileset_name}] frame size {output_size[0]}x{output_size[1]}")

    preview_grid = w.ImageGrid(tuple(config["cell_size"]), tuple(config["grid_size"]))
    preview_frame = make_frame(preview_grid, output_size)
    output_path = output_dir / f"{tileset_name}.mp4"
    writer = open_video_writer(output_path, (preview_frame.shape[1], preview_frame.shape[0]), fps)

    try:
        for _ in range(fps):
            writer.write(preview_frame)

        def write_frame(frame):
            writer.write(frame)

        final_frame = run_generation(config, tiles, adjacency_rules, seed, output_size, write_frame)
        for _ in range(fps * 2):
            writer.write(final_frame)
    finally:
        writer.release()

    print(f"[{tileset_name}] wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render generation videos for converted tilesets")
    parser.add_argument("tilesets", nargs="*", help="Optional tileset names to render")
    parser.add_argument("--output-dir", default="videos", help="Directory for output video files")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Video frames per second")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Nearest-neighbor upscale factor")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        help="Clamp the largest video dimension to this many pixels",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum contradiction retries per tileset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Each retry uses seed + attempt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tilesets = args.tilesets or w.list_tilesets(w.TILESETS_DIR)
    if not tilesets:
        raise FileNotFoundError(f"No tilesets found in {w.TILESETS_DIR}")

    for index, tileset_name in enumerate(tilesets):
        render_tileset_video(
            tileset_name=tileset_name,
            output_dir=output_dir,
            fps=args.fps,
            scale=args.scale,
            max_dimension=args.max_dimension,
            max_attempts=args.max_attempts,
            base_seed=args.seed + index * 1000,
        )


if __name__ == "__main__":
    main()
