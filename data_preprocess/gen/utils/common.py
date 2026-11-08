"""
Common utility functions
"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Optional, Union, Tuple, List


def get_image_count_category(image_count: int) -> str:
    """Return category directory name based on image count"""
    if image_count <= 3:
        return "1-3"
    elif image_count <= 5:
        return "4-5"
    elif image_count <= 7:
        return "6-7"
    else:
        return ">=8"


def load_generated_ids(
    task_dir: Path,
    split_type: str,
    image_count_category: str,
    sub_type: Optional[str] = None
) -> Set[str]:
    """
    Load the set of already-generated sample unique IDs
    
    Args:
        task_dir: task directory, e.g. final/customization
        split_type: "train" or "eval"
        image_count_category: image count category, e.g. "1-3"
        sub_type: subtype (optional)
    
    Returns:
        set of already-generated unique IDs
    """
    generated_ids = set()

    if sub_type:
        json_dir = task_dir / split_type / sub_type / image_count_category / "json"
    else:
        json_dir = task_dir / split_type / image_count_category / "json"
    if not json_dir.exists():
        return generated_ids

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                unique_id = data.get('unique_id')
                if unique_id:
                    generated_ids.add(unique_id)
        except Exception as e:
            print(f"Warning: failed to read JSON file {json_file}: {e}")
            continue

    return generated_ids


def load_all_generated_ids(
    task_dir: Path,
    split_type: str,
    sub_type: Optional[str] = None
) -> Set[str]:
    """
    Load the set of already-generated unique IDs for all categories under the split_type directory
    """
    generated_ids = set()

    if sub_type:
        base_dir = task_dir / split_type / sub_type
    else:
        base_dir = task_dir / split_type

    if not base_dir.exists():
        return generated_ids

    for category_dir in base_dir.glob("*"):
        if not category_dir.is_dir():
            continue

        json_dir = category_dir / "json"
        if not json_dir.exists():
            continue

        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    unique_id = data.get('unique_id')
                    if unique_id:
                        generated_ids.add(unique_id)
            except Exception as e:
                print(f"Warning: failed to read JSON file {json_file}: {e}")
                continue

    return generated_ids


# Control whether to use MD5 hash as unique_id
USE_MD5_HASH = False
SAVE_ORIGINAL_STRING = False


def save_sample_data(
    task_dir: Path,
    split_type: str,
    image_count_category: str,
    idx: int,
    unique_id: Union[str, Tuple[str, str]],
    json_data: Dict,
    image_files: Optional[Dict] = None,
    sub_type: Optional[str] = None
) -> bool:
    """
    Save sample data to the specified directory
    """
    try:
        if sub_type:
            data_dir = task_dir / split_type / sub_type / image_count_category / "data" / f"{idx:08d}"
            json_dir = task_dir / split_type / sub_type / image_count_category / "json"
        else:
            data_dir = task_dir / split_type / image_count_category / "data" / f"{idx:08d}"
            json_dir = task_dir / split_type / image_count_category / "json"

        data_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        # Match against the original full paths, never against filenames or
        # destinations written by an earlier iteration. Keep all occurrences:
        # an interleaved document can legitimately reuse the same source image.
        original_inputs = list(json_data.get('input_images', []))
        original_output = json_data.get('output_image')
        path_destinations = defaultdict(list)
        if image_files:
            import shutil
            for filename, source in image_files.items():
                dest_path = data_dir / filename

                if isinstance(source, Path):
                    if source.exists():
                        try:
                            if source.resolve() == dest_path.resolve():
                                pass
                            else:
                                shutil.copy2(source, dest_path)
                        except (OSError, ValueError):
                            if str(source) != str(dest_path):
                                shutil.copy2(source, dest_path)
                        path_destinations[str(source.resolve())].append(str(dest_path))
                else:
                    try:
                        if hasattr(source, 'save'):
                            source.save(dest_path, quality=95)
                        else:
                            import numpy as np
                            from PIL import Image
                            if isinstance(source, np.ndarray):
                                img = Image.fromarray(source)
                                img.save(dest_path, quality=95)
                            else:
                                raise ValueError(f"Cannot save image file {filename}, unsupported type: {type(source)}")
                    except Exception as e:
                        print(f"Warning: failed to save image file {filename}: {e}")

        if path_destinations:
            occurrences = defaultdict(int)
            mapped_inputs = []
            for original in original_inputs:
                key = str(Path(original).resolve())
                destinations = path_destinations.get(key)
                if destinations:
                    # The destination order follows image_files, as supplied by
                    # the sequence builder. Reuse one destination if only one
                    # physical copy was requested for repeated references.
                    position = min(occurrences[key], len(destinations) - 1)
                    mapped_inputs.append(destinations[position])
                    occurrences[key] += 1
                else:
                    mapped_inputs.append(original)
            if 'input_images' in json_data:
                json_data['input_images'] = mapped_inputs
            if original_output:
                destinations = path_destinations.get(str(Path(original_output).resolve()))
                if destinations:
                    # Callers append the target after the reference images.
                    json_data['output_image'] = destinations[-1]

        if isinstance(unique_id, tuple):
            md5_hash, original_string = unique_id
            json_data['unique_id'] = md5_hash
            if SAVE_ORIGINAL_STRING:
                json_data['original_unique_id'] = original_string
        else:
            json_data['unique_id'] = unique_id
            if not USE_MD5_HASH and SAVE_ORIGINAL_STRING:
                json_data['original_unique_id'] = unique_id

        json_data['idx'] = idx

        json_path = json_dir / f"{idx:08d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error: failed to save sample data idx={idx}, unique_id={unique_id}: {e}")
        return False


def get_combination_key(files: List[str]) -> str:
    """
    Generate a unique key for an image combination (for deduplication)
    """
    sorted_files = sorted(files)
    key_str = "|".join(sorted_files)
    return hashlib.md5(key_str.encode()).hexdigest()


def generate_unique_id(task: str, return_original: bool = False, **kwargs) -> Union[str, Tuple[str, str]]:
    """
    Generate a unique identification number
    """
    if task == "customization":
        combination_key = kwargs.get('combination_key', '')
        if combination_key:
            original_id = f"customization_{combination_key}"
        else:
            original_id = f"customization_{kwargs.get('idx', 'unknown')}"

    elif task == "illustration":
        source_file = kwargs.get('source_file', 'unknown')
        source_line = kwargs.get('source_line', -1)
        true_index = kwargs.get('true_index', 0)
        source_basename = Path(source_file).stem if source_file != 'unknown' else 'unknown'
        original_id = f"illustration_{source_basename}_{source_line:08d}_{true_index:02d}"

    elif task == "spatial":
        sub_type = kwargs.get('sub_type', 'unknown')
        source = kwargs.get('source', 'unknown')
        view_config = kwargs.get('view_config', 'unknown')
        if hasattr(source, '__fspath__') or hasattr(source, 'parts'):
            source = str(source)
        original_id = f"spatial_{sub_type}_{source}_{view_config}"

    elif task == "temporal":
        # process_sample passes the source manifest, its row, and clip index.
        # Preserve the published ID convention so resume/dedup stays compatible.
        yt_file = kwargs.get('yt_file')
        yt_line = kwargs.get('yt_line')
        part_idx = kwargs.get('part_idx')
        if yt_file and yt_line is not None and part_idx is not None:
            yt_line, part_idx = int(yt_line), int(part_idx)
            if yt_line < 0 or part_idx < 0:
                raise ValueError('Temporal source row and clip index must be nonnegative')
            original_id = f"temporal_{Path(yt_file).stem}_{yt_line:08d}_{part_idx:04d}"
        elif kwargs.get('video_path') and kwargs.get('frame_indices'):
            # Retain support for callers identifying a clip by video + frames.
            video_name = Path(kwargs['video_path']).stem
            frames_str = "_".join(str(i) for i in sorted(kwargs['frame_indices']))
            original_id = f"temporal_{video_name}_{frames_str}"
        else:
            raise ValueError('Temporal IDs require yt_file/yt_line/part_idx or video_path/frame_indices')
    else:
        original_id = f"{task}_{hashlib.md5(str(kwargs).encode()).hexdigest()[:8]}"

    if USE_MD5_HASH:
        md5_hash = hashlib.md5(original_id.encode()).hexdigest()
        if return_original:
            return (md5_hash, original_id)
        return md5_hash
    else:
        return original_id
