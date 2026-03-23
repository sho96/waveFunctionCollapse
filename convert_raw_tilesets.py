import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2

DIRECTIONS = ("left", "down", "right", "up")


@dataclass(frozen=True)
class RawTile:
    name: str
    symmetry: str
    weight: float
    cardinality: int


def get_tile_symmetry(tile_element):
    symmetry = tile_element.attrib.get("symmetry", "X")
    if symmetry == "L":
        return 4, (lambda i: (i + 1) % 4), (lambda i: i + 1 if i % 2 == 0 else i - 1)
    if symmetry == "T":
        return 4, (lambda i: (i + 1) % 4), (lambda i: i if i % 2 == 0 else 4 - i)
    if symmetry == "I":
        return 2, (lambda i: 1 - i), (lambda i: i)
    if symmetry == "\\":
        return 2, (lambda i: 1 - i), (lambda i: 1 - i)
    if symmetry == "F":
        return 8, (lambda i: (i + 1) % 4 if i < 4 else 4 + ((i - 1) % 4)), (lambda i: i + 4 if i < 4 else i - 4)
    return 1, (lambda i: i), (lambda i: i)


def get_output_tileset_name(xml_path):
    return xml_path.stem.lower()


def get_variant_name(raw_tile, variant_index):
    if raw_tile.cardinality == 1:
        return raw_tile.name
    return f"{raw_tile.name}__{variant_index}"


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def reflect_image(image):
    return cv2.flip(image, 1)


def parse_neighbor_ref(ref_text):
    parts = ref_text.split()
    if len(parts) == 1:
        return parts[0], 0
    return " ".join(parts[:-1]), int(parts[-1])


def parse_subsets(root, raw_tiles):
    subsets_element = root.find("subsets")
    if subsets_element is None:
        return {}

    subsets = {}
    for subset_element in subsets_element.findall("subset"):
        subset_name = subset_element.attrib["name"]
        variant_names = []
        for tile_element in subset_element.findall("tile"):
            raw_tile = raw_tiles[tile_element.attrib["name"]]
            for variant_index in range(raw_tile.cardinality):
                variant_names.append(get_variant_name(raw_tile, variant_index))
        subsets[subset_name] = variant_names
    return subsets


def build_tiles_and_actions(root, raw_tileset_dir, output_tileset_dir, unique):
    tiles = []
    actions = []
    first_occurrence = {}
    raw_tiles = {}
    cell_size = None

    for tile_element in root.find("tiles").findall("tile"):
        name = tile_element.attrib["name"]
        cardinality, rotate_fn, reflect_fn = get_tile_symmetry(tile_element)
        raw_tile = RawTile(
            name=name,
            symmetry=tile_element.attrib.get("symmetry", "X"),
            weight=float(tile_element.attrib.get("weight", 1.0)),
            cardinality=cardinality,
        )
        raw_tiles[name] = raw_tile

        tile_start_index = len(actions)
        first_occurrence[name] = tile_start_index
        tile_actions = []

        for tile_index in range(cardinality):
            mapping = [0] * 8
            mapping[0] = tile_index
            mapping[1] = rotate_fn(tile_index)
            mapping[2] = rotate_fn(mapping[1])
            mapping[3] = rotate_fn(mapping[2])
            mapping[4] = reflect_fn(tile_index)
            mapping[5] = reflect_fn(mapping[1])
            mapping[6] = reflect_fn(mapping[2])
            mapping[7] = reflect_fn(mapping[3])
            mapping = [tile_start_index + value for value in mapping]
            tile_actions.append(mapping)
            actions.append(mapping)

        tile_images = []
        if unique:
            for tile_index in range(cardinality):
                source_path = raw_tileset_dir / f"{name} {tile_index}.png"
                image = load_image(source_path)
                tile_images.append(image)
        else:
            base_image = load_image(raw_tileset_dir / f"{name}.png")
            tile_images.append(base_image)
            for tile_index in range(1, cardinality):
                if tile_index <= 3:
                    tile_images.append(rotate_image(tile_images[tile_index - 1]))
                else:
                    tile_images.append(reflect_image(tile_images[tile_index - 4]))

        for tile_index, image in enumerate(tile_images):
            variant_name = get_variant_name(raw_tile, tile_index)
            output_filename = f"{variant_name}.png"
            output_path = output_tileset_dir / output_filename
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Could not write image: {output_path}")
            if cell_size is None:
                cell_size = (image.shape[1], image.shape[0])

            tiles.append(
                {
                    "index": tile_start_index + tile_index,
                    "name": variant_name,
                    "image": output_filename,
                    "weight": raw_tile.weight,
                }
            )

    return tiles, actions, first_occurrence, raw_tiles, cell_size


def build_dense_propagator(root, actions, first_occurrence):
    tile_count = len(actions)
    dense_propagator = [[[False for _ in range(tile_count)] for _ in range(tile_count)] for _ in range(4)]

    neighbors_element = root.find("neighbors")
    if neighbors_element is None:
        return dense_propagator

    for neighbor_element in neighbors_element.findall("neighbor"):
        left_name, left_variant = parse_neighbor_ref(neighbor_element.attrib["left"])
        right_name, right_variant = parse_neighbor_ref(neighbor_element.attrib["right"])

        left_index = actions[first_occurrence[left_name]][left_variant]
        left_down = actions[left_index][1]
        right_index = actions[first_occurrence[right_name]][right_variant]
        right_up = actions[right_index][1]

        dense_propagator[0][right_index][left_index] = True
        dense_propagator[0][actions[right_index][6]][actions[left_index][6]] = True
        dense_propagator[0][actions[left_index][4]][actions[right_index][4]] = True
        dense_propagator[0][actions[left_index][2]][actions[right_index][2]] = True

        dense_propagator[1][right_up][left_down] = True
        dense_propagator[1][actions[left_down][6]][actions[right_up][6]] = True
        dense_propagator[1][actions[right_up][4]][actions[left_down][4]] = True
        dense_propagator[1][actions[left_down][2]][actions[right_up][2]] = True

    for tile_b in range(tile_count):
        for tile_a in range(tile_count):
            dense_propagator[2][tile_b][tile_a] = dense_propagator[0][tile_a][tile_b]
            dense_propagator[3][tile_b][tile_a] = dense_propagator[1][tile_a][tile_b]

    return dense_propagator


def convert_tileset(xml_path, raw_dir, output_dir):
    root = ET.parse(xml_path).getroot()
    raw_tileset_dir = raw_dir / xml_path.stem
    output_tileset_dir = output_dir / get_output_tileset_name(xml_path)
    output_tileset_dir.mkdir(parents=True, exist_ok=True)

    unique = root.attrib.get("unique", "").lower() == "true"
    tiles, actions, first_occurrence, raw_tiles, cell_size = build_tiles_and_actions(
        root, raw_tileset_dir, output_tileset_dir, unique
    )
    dense_propagator = build_dense_propagator(root, actions, first_occurrence)
    tile_name_by_index = {tile["index"]: tile["name"] for tile in tiles}

    for tile in tiles:
        tile["neighbors"] = {
            direction: [
                tile_name_by_index[neighbor_index]
                for neighbor_index, allowed in enumerate(dense_propagator[direction_index][tile["index"]])
                if allowed
            ]
            for direction_index, direction in enumerate(DIRECTIONS)
        }
        del tile["index"]

    config = {
        "cell_size": [cell_size[0], cell_size[1]],
        "grid_size": [30, 30],
        "display_ratio": 5,
        "show_preview": True,
        "start_tile": None,
        "tiles": tiles,
    }

    subsets = parse_subsets(root, raw_tiles)
    if subsets:
        config["subsets"] = subsets

    config_path = output_tileset_dir / "tileset.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_tileset_dir


def find_xml_files(raw_dir):
    return sorted(raw_dir.glob("*.xml"))


def select_xml_files(raw_dir, requested_names):
    xml_files = find_xml_files(raw_dir)
    if not requested_names:
        return xml_files

    xml_by_key = {}
    for xml_path in xml_files:
        xml_by_key[xml_path.stem.lower()] = xml_path
        xml_by_key[xml_path.stem] = xml_path

    selected = []
    for name in requested_names:
        if name not in xml_by_key:
            raise ValueError(f"Unknown raw tileset '{name}'")
        selected.append(xml_by_key[name])
    return selected


def main():
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "tilesets-raw"
    output_dir = base_dir / "tilesets"

    parser = argparse.ArgumentParser(description="Convert tilesets-raw XML tilesets into tilesets/")
    parser.add_argument("tilesets", nargs="*", help="Optional raw tileset names to convert")
    args = parser.parse_args()

    xml_files = select_xml_files(raw_dir, args.tilesets)
    if not xml_files:
        raise FileNotFoundError(f"No XML tilesets found in {raw_dir}")

    for xml_path in xml_files:
        output_tileset_dir = convert_tileset(xml_path, raw_dir, output_dir)
        print(f"Converted {xml_path.stem} -> {output_tileset_dir.name}")


if __name__ == "__main__":
    main()
