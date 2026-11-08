"""Data-construction regressions. Run: python -m unittest discover -s tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'data_preprocess/gen/utils/common.py'
spec = importlib.util.spec_from_file_location('preprocess_common', MODULE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class SaveSampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, name, color):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), color).save(p)
        return p

    def save(self, inputs, output, image_files):
        data = {'input_images': list(map(str, inputs)), 'output_image': str(output)}
        self.assertTrue(common.save_sample_data(self.root / 'final', 'eval', '>=8', 0, 'fixture', data, image_files))
        saved = json.loads((self.root / 'final/eval/>=8/json/00000000.json').read_text())
        self.assertEqual(data, saved)
        return saved

    def test_later_short_basename_does_not_rewrite_earlier_destination(self):
        inputs = [self.image(f'source/raw_{i}_unique.jpg', (i * 20, 0, 0)) for i in range(1, 10)]
        inputs.append(self.image('source/1.jpg', (0, 200, 0)))
        target = self.image('source/target.jpg', (0, 0, 200))
        files = {f'image_{i}.jpg': p for i, p in enumerate(inputs, 1)}
        files['image_output.jpg'] = target
        data = self.save(inputs, target, files)
        self.assertEqual([Path(p).name for p in data['input_images']], [f'image_{i}.jpg' for i in range(1, 11)])
        for src, dest in zip(inputs, data['input_images']):
            self.assertEqual(src.read_bytes(), Path(dest).read_bytes())

    def test_same_basename_in_different_directories_stays_distinct(self):
        first = self.image('a/frame.jpg', (200, 0, 0))
        second = self.image('b/frame.jpg', (0, 200, 0))
        target = self.image('c/frame.jpg', (0, 0, 200))
        data = self.save([first, second], target, {'image_1.jpg': first, 'image_2.jpg': second, 'image_output.jpg': target})
        self.assertEqual([Path(p).name for p in data['input_images']], ['image_1.jpg', 'image_2.jpg'])
        self.assertEqual(Path(data['output_image']).name, 'image_output.jpg')

    def test_repeated_source_preserves_sequence_positions_and_target(self):
        image = self.image('source/reused.jpg', (20, 40, 60))
        data = self.save([image, image], image, {'image_1.jpg': image, 'image_2.jpg': image, 'image_output.jpg': image})
        self.assertEqual([Path(p).name for p in data['input_images']], ['image_1.jpg', 'image_2.jpg'])
        self.assertEqual(Path(data['output_image']).name, 'image_output.jpg')

    def test_repeated_reference_can_share_a_single_requested_copy(self):
        image = self.image('source/reused.jpg', (20, 40, 60))
        target = self.image('source/target.jpg', (80, 40, 60))
        data = self.save([image, image], target, {'image_1.jpg': image, 'image_output.jpg': target})
        self.assertEqual(data['input_images'][0], data['input_images'][1])

    def test_temporal_pil_branch_keeps_eight_nine_and_ten_ordered_inputs(self):
        for count in (8, 9, 10):
            with self.subTest(count=count):
                data_dir = self.root / 'final/eval/>=8/data/00000000'
                inputs = [data_dir / f'image_{i}.jpg' for i in range(1, count + 1)]
                target = data_dir / 'image_output.jpg'
                files = {p.name: Image.new('RGB', (8, 8), (i * 20, 0, 0)) for i, p in enumerate(inputs, 1)}
                files[target.name] = Image.new('RGB', (8, 8), (0, 0, 200))
                data = self.save(inputs, target, files)
                self.assertEqual(data['input_images'], list(map(str, inputs)))
                self.assertTrue(all(Path(p).is_file() for p in data['input_images']))


class TemporalIdTests(unittest.TestCase):
    def test_current_caller_metadata_matches_published_id_convention(self):
        value = common.generate_unique_id('temporal', yt_file='/raw/youtube_ids_omnicorpus_yt-downloaded_1.json', yt_line=95329, part_idx=30)
        self.assertEqual(value, 'temporal_youtube_ids_omnicorpus_yt-downloaded_1_00095329_0030')

    def test_distinct_source_rows_and_clips_have_distinct_ids(self):
        ids = {common.generate_unique_id('temporal', yt_file=f'/raw/source_{file}.json', yt_line=row, part_idx=clip)
               for file in (1, 2) for row in (1, 2) for clip in (0, 1)}
        self.assertEqual(len(ids), 8)

    def test_legacy_video_and_frame_call_still_works(self):
        self.assertEqual(common.generate_unique_id('temporal', video_path='/raw/video.mp4', frame_indices=[5, 1, 3]), 'temporal_video_1_3_5')

    def test_incomplete_metadata_does_not_silently_share_default_id(self):
        with self.assertRaises(ValueError):
            common.generate_unique_id('temporal', yt_file='source.json')
        with self.assertRaises(ValueError):
            common.generate_unique_id('temporal', yt_file='source.json', yt_line=-1, part_idx=0)


if __name__ == '__main__':
    unittest.main()
