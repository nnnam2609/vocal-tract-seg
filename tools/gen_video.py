import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
except Exception:
    VideoFileClip = None
    AudioFileClip = None
    concatenate_videoclips = None
    concatenate_audioclips = None

try:
    import textgrid
except Exception:
    textgrid = None


DEFAULT_FPS = 1 / 0.01998


def parse_model_arg(value: str) -> Tuple[str, str, str]:
    """
    Parse --model argument of the form:
        name=DIR[:TEMPLATE]
    TEMPLATE can use {name} (full filename) and {stem} (name without extension).
    """
    if "=" not in value:
        raise ValueError("--model must be in the form name=DIR[:TEMPLATE]")
    name, rest = value.split("=", 1)
    if ":" in rest:
        dir_path, template = rest.split(":", 1)
    else:
        dir_path, template = rest, "eval_{stem}.png"
    return name.strip(), dir_path.strip(), template.strip()


def load_frames_list(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def resolve_frame_path(model_dir: str, template: str, frame_name: str) -> str:
    stem = os.path.splitext(os.path.basename(frame_name))[0]
    filename = template.format(name=frame_name, stem=stem)
    return os.path.join(model_dir, filename)


def read_image(path: str, fallback_size: Tuple[int, int]) -> np.ndarray:
    if not path or not os.path.exists(path):
        return make_placeholder(f"missing", fallback_size)
    img = cv2.imread(path)
    if img is None:
        return make_placeholder(f"bad image", fallback_size)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_placeholder(text: str, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def resize_to_tile(img: np.ndarray, tile_size: Tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_size
    return cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)


def label_tile(img: np.ndarray, label: str) -> np.ndarray:
    if not label:
        return img
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def compose_grid(tiles: List[np.ndarray], cols: int, tile_size: Tuple[int, int], pad: int) -> np.ndarray:
    if cols <= 0:
        cols = len(tiles)
    rows = int(np.ceil(len(tiles) / cols))
    tile_w, tile_h = tile_size
    grid_w = cols * tile_w + (cols - 1) * pad
    grid_h = rows * tile_h + (rows - 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y0 = r * (tile_h + pad)
        x0 = c * (tile_w + pad)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def write_video(frames: List[np.ndarray], output_path: str, fps: float, codec: str) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def merge_audio(video_path: str, audio_path: str, output_path: str) -> None:
    if VideoFileClip is None or AudioFileClip is None:
        raise RuntimeError("moviepy is required for audio merge")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final = video.set_audio(audio)
    final.write_videofile(output_path, codec="libx264", fps=video.fps)


def load_textgrid(path: str):
    if textgrid is None:
        raise RuntimeError("textgrid package not installed")
    tg = textgrid.TextGrid.fromFile(path)
    tiers_dict = {}
    for tier in tg.tiers:
        intervals = []
        for interval in tier:
            intervals.append((interval.minTime, interval.maxTime, interval.mark))
        tiers_dict[tier.name] = intervals
    return tiers_dict


def delete_silence(video_path: str, audio_path: str, textgrid_path: str, output_path: str) -> None:
    if VideoFileClip is None or AudioFileClip is None or concatenate_videoclips is None:
        raise RuntimeError("moviepy is required for silence removal")
    tg = load_textgrid(textgrid_path)
    non_silence_intervals = []
    silence_labels = ["#", ""]
    for sentence in tg.get("SentenceTier", []):
        start, end, label = sentence
        if label not in silence_labels:
            non_silence_intervals.append((start, end))
    if not non_silence_intervals:
        raise ValueError("No non-silence intervals found")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    video_segments = [video.subclip(start, end) for start, end in non_silence_intervals]
    audio_segments = [audio.subclip(start, end) for start, end in non_silence_intervals]
    final_video = concatenate_videoclips(video_segments).set_audio(concatenate_audioclips(audio_segments))
    final_video.write_videofile(output_path, codec="libx264", fps=video.fps)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison video across models from test frames")
    parser.add_argument("--frames", required=True, help="Path to list of frame filenames (one per line)")
    parser.add_argument("--model", action="append", required=True, help="Model spec name=DIR[:TEMPLATE]")
    parser.add_argument("--input-dir", default=None, help="Optional input image directory (raw test images)")
    parser.add_argument("--output", required=True, help="Output video path (.mp4 or .avi)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument("--tile-width", type=int, default=640, help="Tile width")
    parser.add_argument("--tile-height", type=int, default=480, help="Tile height")
    parser.add_argument("--cols", type=int, default=0, help="Number of columns in grid (0=auto)")
    parser.add_argument("--pad", type=int, default=10, help="Padding between tiles")
    parser.add_argument("--codec", default=None, help="FourCC codec (mp4v for mp4, MJPG for avi)")
    parser.add_argument("--audio", default=None, help="Optional audio file to merge")
    parser.add_argument("--with-audio", action="store_true", help="Write output with audio if --audio is set")
    parser.add_argument("--no-silence", default=None, help="Optional TextGrid to create a no-silence video")

    args = parser.parse_args()

    frames = load_frames_list(args.frames)
    tile_size = (args.tile_width, args.tile_height)

    models = []
    for model_arg in args.model:
        name, dir_path, template = parse_model_arg(model_arg)
        models.append((name, dir_path, template))

    cols = args.cols if args.cols > 0 else len(models) + (1 if args.input_dir else 0)

    frames_out = []
    for frame_name in frames:
        tiles = []
        if args.input_dir:
            input_path = os.path.join(args.input_dir, frame_name)
            img = read_image(input_path, tile_size)
            img = resize_to_tile(img, tile_size)
            tiles.append(label_tile(img, "input"))

        for name, dir_path, template in models:
            img_path = resolve_frame_path(dir_path, template, frame_name)
            img = read_image(img_path, tile_size)
            img = resize_to_tile(img, tile_size)
            tiles.append(label_tile(img, name))

        grid = compose_grid(tiles, cols, tile_size, args.pad)
        frames_out.append(grid)

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if args.codec:
        codec = args.codec
    else:
        codec = "mp4v" if ext == ".mp4" else "MJPG"

    write_video(frames_out, output_path, args.fps, codec)

    if args.audio and args.with_audio:
        output_audio = os.path.splitext(output_path)[0] + "_with_audio" + ext
        merge_audio(output_path, args.audio, output_audio)
        if args.no_silence:
            output_no_silence = os.path.splitext(output_path)[0] + "_no_silence" + ext
            delete_silence(output_audio, args.audio, args.no_silence, output_no_silence)


if __name__ == "__main__":
    main()
