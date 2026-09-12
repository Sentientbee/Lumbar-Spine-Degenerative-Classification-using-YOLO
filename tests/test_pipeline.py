import os
import re
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from src.data.dataset_builder import (
    point_to_yolo_bbox,
    extract_multislice_roi,
    normalize_level_name,
    LEVELS,
    SEVERITIES,
)
from src.data.dicom_reader import apply_voi_lut, load_dicom_series
from src.models.severity_classifier import LumbarSeverityClassifier
from src.models.yolo_detector import SpineLevelDetector
from src.metrics.competition_loss import RSNALogLoss, compute_rsna_log_loss
from src.pipeline.predict_study import predict_study


class TestLumbarSpinePipeline(unittest.TestCase):

    def test_level_name_normalization(self):
        self.assertEqual(normalize_level_name("L1/L2"), "l1_l2")
        self.assertEqual(normalize_level_name("l5_s1"), "l5_s1")
        self.assertEqual(normalize_level_name("L3-L4"), "l3_l4")

    def test_point_to_yolo_bbox(self):
        xc, yc, w, h = point_to_yolo_bbox(100.0, 200.0, 500, 500, box_size=50)
        self.assertAlmostEqual(xc, 0.2, places=4)
        self.assertAlmostEqual(yc, 0.4, places=4)
        self.assertAlmostEqual(w, 0.1, places=4)
        self.assertAlmostEqual(h, 0.1, places=4)

    def test_point_to_yolo_bbox_zero_dimension_protection(self):
        # Must not raise ZeroDivisionError when width or height is 0
        xc, yc, w, h = point_to_yolo_bbox(10.0, 10.0, 0, 0, box_size=48)
        self.assertTrue(0.0 <= xc <= 1.0)
        self.assertTrue(0.0 <= yc <= 1.0)

    def test_extract_multislice_roi(self):
        depth, height, width = 15, 256, 256
        dummy_volume = np.random.randint(0, 255, size=(depth, height, width), dtype=np.uint8)

        roi = extract_multislice_roi(
            volume=dummy_volume,
            key_slice_idx=7,
            center_x=120,
            center_y=130,
            crop_size=128,
            num_slices=3
        )
        self.assertEqual(roi.shape, (3, 128, 128))
        self.assertTrue(0.0 <= roi.min() and roi.max() <= 1.0)

    def test_extract_multislice_roi_exact_slice_counts(self):
        depth, height, width = 10, 128, 128
        vol = np.random.randint(0, 255, size=(depth, height, width), dtype=np.uint8)

        for requested_slices in [1, 2, 3, 4, 5]:
            roi = extract_multislice_roi(vol, key_slice_idx=4, center_x=64, center_y=64, crop_size=64, num_slices=requested_slices)
            self.assertEqual(roi.shape[0], requested_slices, f"Failed for requested {requested_slices} slices; got {roi.shape[0]}")

    def test_apply_voi_lut(self):
        class DummyDcm:
            WindowCenter = 50.0
            WindowWidth = 100.0

        img = np.array([-100.0, 0.0, 50.0, 100.0, 200.0], dtype=np.float32)
        windowed = apply_voi_lut(img, DummyDcm())
        # Window: min = 50 - 50 = 0, max = 50 + 50 = 100
        self.assertEqual(float(windowed.min()), 0.0)
        self.assertEqual(float(windowed.max()), 100.0)

    def test_apply_voi_lut_invalid_window_width(self):
        class BadDcm:
            WindowCenter = 50.0
            WindowWidth = -10.0  # Invalid negative width

        img = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        # Must not raise ValueError: min must be <= max
        windowed = apply_voi_lut(img, BadDcm())
        self.assertEqual(windowed.shape, img.shape)

    def test_load_dicom_series_npy_scaling_and_ordering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create three .npy slices with float values in [0, 1]
            s1 = np.full((64, 64), 0.25, dtype=np.float32)
            s2 = np.full((64, 64), 0.50, dtype=np.float32)
            s3 = np.full((64, 64), 0.75, dtype=np.float32)

            np.save(os.path.join(tmp_dir, "slice_003.npy"), s3)
            np.save(os.path.join(tmp_dir, "slice_001.npy"), s1)
            np.save(os.path.join(tmp_dir, "slice_002.npy"), s2)

            volume, instance_nums = load_dicom_series(tmp_dir)

            self.assertEqual(volume.shape, (3, 64, 64))
            self.assertEqual(instance_nums, [1, 2, 3])
            # Verify float values were scaled to ~ [0, 255] and NOT truncated to 0
            self.assertGreater(int(volume[0, 0, 0]), 50)   # 0.25 * 255 ~ 63
            self.assertGreater(int(volume[1, 0, 0]), 100)  # 0.50 * 255 ~ 127
            self.assertGreater(int(volume[2, 0, 0]), 180)  # 0.75 * 255 ~ 191

    def test_spine_level_detector_simulation(self):
        detector = SpineLevelDetector(model_weights="yolo11s.pt", allow_simulation=True)
        dummy_slice = np.zeros((256, 256, 3), dtype=np.uint8)

        # Single slice inference
        dets = detector.predict_slice(dummy_slice)
        self.assertEqual(len(dets), 5)
        detected_levels = [d["level"] for d in dets]
        for lvl in LEVELS:
            self.assertIn(lvl, detected_levels)

        # Volume inference
        dummy_vol = np.zeros((8, 256, 256), dtype=np.uint8)
        vol_dets = detector.predict_volume(dummy_vol)
        self.assertEqual(len(vol_dets), 5)

    def test_lumbar_severity_classifier_forward(self):
        model = LumbarSeverityClassifier(backbone_name="resnet18", num_classes=3, pretrained=False)
        model.eval()

        dummy_input = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = model.predict_proba(dummy_input)

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(probs.shape, (2, 3))

        # Check probabilities sum to 1
        sums = probs.sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(2), rtol=1e-5)

    def test_rsna_log_loss_pytorch_and_numpy(self):
        criterion = RSNALogLoss()

        # Logits that predict class 0 strongly
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss = criterion(logits, targets)
        self.assertTrue(loss.item() >= 0.0)
        self.assertTrue(loss.item() < 0.1)  # Low loss on correct predictions

        # Numpy comparison
        probs = torch.softmax(logits, dim=-1).numpy()
        np_loss = compute_rsna_log_loss(targets.numpy(), probs)
        self.assertAlmostEqual(loss.item(), np_loss, places=4)

    def test_rsna_log_loss_empty_inputs(self):
        criterion = RSNALogLoss()
        empty_logits = torch.zeros((0, 3), dtype=torch.float32)
        empty_targets = torch.zeros((0,), dtype=torch.long)
        loss = criterion(empty_logits, empty_targets)
        self.assertEqual(loss.item(), 0.0)

        np_loss = compute_rsna_log_loss([], np.zeros((0, 3)))
        self.assertEqual(np_loss, 0.0)

    def test_lumbar_crop_dataset_and_augmentation(self):
        from src.data.crop_dataset import LumbarCropDataset
        from src.data.augmentations import MedicalMultiSliceAugmentations

        augmenter = MedicalMultiSliceAugmentations()
        ds = LumbarCropDataset(is_synthetic=True, num_synthetic_samples=10, transform=augmenter)
        self.assertEqual(len(ds), 10)

        x, y = ds[0]
        self.assertEqual(x.shape, (3, 128, 128))
        self.assertTrue(y.item() in [0, 1, 2])
        self.assertTrue(0.0 <= x.min() and x.max() <= 1.0)

    def test_train_and_evaluate_classifier_pipeline(self):
        from src.pipeline.train_severity_classifier import train_severity_classifier
        from src.pipeline.evaluate_classifier import evaluate_classifier

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1-epoch quick dry-run
            history = train_severity_classifier(
                demo=True,
                epochs=1,
                batch_size=8,
                output_dir=tmp_dir,
                device="cpu"
            )
            self.assertIn("best_weights_path", history)
            weights_path = history["best_weights_path"]
            self.assertTrue(os.path.exists(weights_path))

            # Evaluate the saved checkpoint
            eval_res = evaluate_classifier(
                weights_path=weights_path,
                num_test_samples=16,
                batch_size=8,
                device="cpu"
            )
            self.assertIn("accuracy", eval_res)
            self.assertIn("rsna_log_loss", eval_res)
            self.assertIn("confusion_matrix", eval_res)
            self.assertEqual(eval_res["total_samples"], 16)

    def test_predict_study_end_to_end(self):
        from demo.generate_sample_study import create_sample_study

        with tempfile.TemporaryDirectory() as tmp_dir:
            create_sample_study(tmp_dir, num_slices=6)
            report = predict_study(series_dir=tmp_dir, device="cpu")

            self.assertEqual(report["slices_count"], 6)
            self.assertEqual(len(report["detected_levels"]), 5)
            for lvl in LEVELS:
                self.assertIn(lvl, report["detected_levels"])
                lvl_info = report["detected_levels"][lvl]
                self.assertTrue(lvl_info["detected"])
                self.assertIn(lvl_info["predicted_severity"], SEVERITIES)
                probs = lvl_info["probabilities"]
                prob_sum = sum(probs.values())
                self.assertAlmostEqual(prob_sum, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
