# TODO / Project Understanding

## What this project is trying to do

This repository looks like a small prototype of a **Wave Function Collapse (WFC)-style tile generator**.

The goal is to:

- build a `30 x 30` grid of `16 x 16` image tiles
- choose tiles so neighboring tiles are visually compatible
- display the generation process live with OpenCV

In practice, the tile set currently represents a simple connection system:

- `blank1.png`: empty tile
- `cross1.png`: 4-way connection
- `right1.png` rotated into `up`, `right`, `down`, `left`: 3-way connection / T-junction variants

So the generated result is meant to look like a connected road, pipe, or path network assembled from these tiles.

## How the current script works

`waveFunctionCollapse.py` does the following:

1. Loads the tile images.
2. Builds rotated versions of one base directional tile.
3. Defines adjacency rules for each tile:
   - allowed neighbors for `up`
   - allowed neighbors for `right`
   - allowed neighbors for `down`
   - allowed neighbors for `left`
4. Starts every cell with all tile options available.
5. Forces the top-left cell to be a `cross`.
6. Repeatedly picks a cell with the fewest remaining options.
7. Randomly collapses that cell to one tile.
8. Restricts neighboring cells based on the chosen tile.
9. Draws the partial result after each step.

That means this is not just drawing a grid of sprites. It is trying to **generate a valid tiled pattern under local connection constraints**.

## Important caveat

This is closer to a **basic WFC prototype** than a full WFC implementation.

Reasons:

- constraint propagation only happens in a limited way
- there is no proper propagation queue
- there is no backtracking when contradictions happen
- a cell with zero options raises an exception instead of recovering

So the project intent is clearly WFC-style procedural generation, but the current script is still an early experiment.

## TODOs to make the project clearer and more complete

- Add a short `README.md` explaining the tile meanings, dependencies (`opencv-python`, `numpy`, `tqdm`), and how to run the script.
- Rename `imgGrid` and helper functions to more descriptive names.
- Replace the `exec(...)`-based neighbor update logic with normal Python code.
- Implement proper recursive/queued constraint propagation until the grid reaches a stable state.
- Add contradiction handling and backtracking instead of crashing on impossible states.
- Move tile definitions and adjacency rules into a data structure that is easier to edit and test.
- Save the final generated image to disk in addition to showing it in a window.
- Clean up unused assets (`blank.png`, `cross.png`, `cross2.png`, `right.png`, `right2.png`, `something0.png`, `something1.png`) or document their intended purpose.

## Short summary

My current understanding is:

> This project is trying to procedurally generate a grid-based connection pattern using WFC-like adjacency constraints, with OpenCV used to preview the generated result tile by tile.
