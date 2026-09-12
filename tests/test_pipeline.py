import unittest
import numpy as np
import torch

from src.data.dataset_builder import (
    point_to_yolo_bbox,
    extract_multislice_roi,
    normalize_level_name,
    LEVELS,
    SEVERITIES,
)
from src.data.dicom_reader import apply_voi_lut
from src.models.severity_classifier import LumbarSeverityClassifier
from src.metrics.competition_loss import RSNALogLoss, compute_rsna_log_loss


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

    def test_apply_voi_lut(self):
        class DummyDcm:
            WindowCenter = 50.0
            WindowWidth = 100.0

        img = np.array([-100.0, 0.0, 50.0, 100.0, 200.0], dtype=np.float32)
        windowed = apply_voi_lut(img, DummyDcm())
        # Window: min = 50 - 50 = 0, max = 50 + 50 = 100
        self.assertEqual(float(windowed.min()), 0.0)
        self.assertEqual(float(windowed.max()), 100.0)

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


if __name__ == "__main__":
    unittest.main()
