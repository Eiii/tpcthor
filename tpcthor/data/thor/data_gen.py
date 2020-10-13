from .mcs_env import McsEnv

import pickle
import random
import itertools
import lzma
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def convert_output(o, i):
    objs = o.object_list
    structs = o.structural_object_list
    """
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs).flatten()
    depth_mask = np.array(o.depth_mask_list[-1])
    camera_desc = [o.camera_aspect_ratio, o.camera_clipping_planes,
                   o.camera_field_of_view, o.position]
    """
    return ThorFrame(objs, structs, None, None, None)

def convert_obj_mask(mask, objs):
    color_map = {convert_color(o.color):i for i, o in enumerate(objs)}
    arr_mask = np.array(mask)
    out_mask = -np.ones(arr_mask.shape[0:2], dtype=np.int8)
    for x in range(arr_mask.shape[0]):
        for y in range(arr_mask.shape[1]):
            idx = color_map.get(tuple(arr_mask[x, y]), -1)
            out_mask[x, y] = idx
    return out_mask

def px_to_pos(px_arr, depth_arr, aspect_ratio, clip_planes, fov_deg, camera_pos):
    uv_arr = px_arr*[2/w for w in aspect_ratio]-1
    uv_arr[:, :, 1] *= -1
    depth_mix = depth_arr/255
    depth = clip_planes[0] + (clip_planes[1]-clip_planes[0])*depth_mix
    vfov = np.radians(fov_deg)
    hfov = np.radians(fov_deg*aspect_ratio[0]/aspect_ratio[1])
    tans = np.array([np.tan(x/2) for x in (hfov, vfov)])
    dir_arr = uv_arr * tans
    zs = np.ones((dir_arr.shape[0:2])+(1,))
    dir_arr = np.concatenate((dir_arr, zs), axis=-1)
    camera_offsets = dir_arr * np.expand_dims(depth, axis=-1)
    pos_to_list = lambda x: [x['x'], x['y'], x['z']]
    return camera_offsets + pos_to_list(camera_pos)

def convert_depth_mask(depth, camera_aspect_ratio, camera_clipping_planes,
                       camera_field_of_view, position):
    sz = depth.shape
    px_grid = np.stack(np.meshgrid(np.arange(sz[1]), np.arange(sz[0])), axis=-1)
    world_pos = px_to_pos(px_grid, depth, camera_aspect_ratio,
                          camera_clipping_planes, camera_field_of_view,
                          position)
    pts = world_pos.reshape(-1, world_pos.shape[-1])
    return pts

def convert_scenes(env, paths):
    for scene_path in paths:
        print(scene_path)
        out_path = scene_path.with_suffix('.pkl.xz')
        if out_path.exists():
            print(f'{out_path} exists, skipping')
            continue
        print(f'{scene_path} -> {out_path}')
        scene_output = [convert_output(o, i) for i, o in enumerate(env.run_scene(scene_path))]
        with lzma.open(out_path, 'wb') as fd:
            pickle.dump(scene_output, fd)

def output_scene(env, path):
    scene_output = [convert_output(o, i) for i, o in enumerate(env.run_scene(path))]
    with lzma.open('./output.pkl.xz', 'wb') as fd:
        pickle.dump(scene_output, fd)
    return scene_output

def convert_color(col):
    return (col['r'], col['g'], col['b'])

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/thor/scenes'))
    parser.add_argument('--filter', type=str, default='object_permanence')
    return parser

def main(data_path, filter):
    env = McsEnv('./data/thor', data_path, filter)
    print(len(env.all_scenes))
    scenes = list(env.all_scenes)
    random.shuffle(scenes)
    convert_scenes(env, scenes)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.data, args.filter)

