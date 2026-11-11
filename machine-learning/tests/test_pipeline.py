"""
Unit test suite for machine-learning pipeline (Person 3).
Tests dataset parsing, NaN sanitization, normalization, model architectures,
masked loss invariants, evaluation metrics, overfit sanity check, and streaming inference.
"""
import numpy as np
import pytest
import torch

from src.dataset import CSINormalizer, CSIPoseDataset
from src.losses import MaskedPoseLoss
from src.metrics import compute_mpjpe, compute_pck, compute_presence_metrics
from src.models import CSIPoseNet, LinearMLPBaseline, MeanPoseBaseline, load_checkpoint, save_checkpoint
from src.predict import CSIPosePredictor, CSIStreamBuffer


class TestDatasetAndNormalizer:
    def test_normalizer_fit_transform_and_roundtrip(self, tmp_path):
        train_csi = np.random.randn(10, 20, 224).astype(np.float32) * 5.0 + 10.0
        # Inject zero-variance channel at index 0
        train_csi[:, :, 0] = 42.0

        norm = CSINormalizer().fit(train_csi)
        transformed = norm.transform(train_csi)
        assert np.isfinite(transformed).all()
        assert norm.std[0] == 1.0  # clamped against zero division

        norm_path = tmp_path / "norm.json"
        norm.save(norm_path)
        reloaded = CSINormalizer.load(norm_path)
        assert np.allclose(norm.mean, reloaded.mean)
        assert np.allclose(norm.std, reloaded.std)

    def test_dataset_sanitizes_empty_nans(self):
        csi = np.random.randn(2, 20, 224).astype(np.float32)
        kpts = np.full((2, 33, 2), np.nan, dtype=np.float32)
        vis = np.full((2, 33), np.nan, dtype=np.float32)
        mask = np.ones((2, 33), dtype=np.float32)
        presence = np.array([0.0, 0.0], dtype=np.float32)

        ds = CSIPoseDataset(csi, kpts, vis, mask, presence, ["empty", "empty"], ["s0", "s0"])
        sample = ds[0]
        assert torch.isfinite(sample["keypoints"]).all()
        assert torch.isfinite(sample["visibility"]).all()
        assert torch.all(sample["mask"] == 0.0)


class TestModelArchitecture:
    def test_csiposenet_forward_and_checkpoint_roundtrip(self, tmp_path):
        batch_size, seq_len, feat_dim = 4, 20, 224
        model = CSIPoseNet(input_features=feat_dim, gru_hidden_dim=64, num_keypoints=33)
        dummy_input = torch.randn(batch_size, seq_len, feat_dim)

        pres_logits, keypoints = model(dummy_input)
        assert pres_logits.shape == (batch_size,)
        assert keypoints.shape == (batch_size, 33, 2)
        assert torch.all((keypoints >= 0.0) & (keypoints <= 1.0))

        ckpt = tmp_path / "model.pt"
        cfg = {"model": {"name": "CSIPoseNet", "input_features": 224, "gru_hidden_dim": 64, "num_keypoints": 33}}
        save_checkpoint(model, ckpt, config=cfg, epoch=3, val_loss=0.123)
        loaded_model, loaded_cfg = load_checkpoint(ckpt)
        assert loaded_cfg["model"]["gru_hidden_dim"] == 64

        model.eval()
        with torch.no_grad():
            p1, k1 = model(dummy_input)
            p2, k2 = loaded_model(dummy_input)
        assert torch.allclose(p1, p2, atol=1e-5)
        assert torch.allclose(k1, k2, atol=1e-5)

    def test_baselines(self):
        # MeanPoseBaseline fit ignores empty samples
        kpts = np.array([np.full((33, 2), 99.0), np.full((33, 2), 0.5)], dtype=np.float32)
        presence = np.array([0.0, 1.0], dtype=np.float32)
        mask = np.ones((2, 33), dtype=np.float32)
        base = MeanPoseBaseline().fit(kpts, presence, mask)
        pres_out, pose_out = base.predict(np.zeros((3, 20, 224), dtype=np.float32))
        assert pres_out.shape == (3,)
        assert pose_out.shape == (3, 33, 2)
        assert np.allclose(pose_out[0], 0.5)

        # LinearMLPBaseline works even for batch_size=1
        mlp = LinearMLPBaseline(window_size=20, subcarriers=224, num_keypoints=33)
        mlp.train()
        pres_mlp, kpts_mlp = mlp(torch.randn(1, 20, 224))
        assert pres_mlp.shape == (1,)
        assert kpts_mlp.shape == (1, 33, 2)


class TestMaskedLoss:
    def test_empty_room_with_nan_keypoints_has_zero_pose_loss_and_finite_grads(self):
        loss_fn = MaskedPoseLoss(lambda_presence=1.0, lambda_pose=10.0)
        pred_kpts = torch.randn(2, 33, 2, requires_grad=True)
        presence_logits = torch.tensor([-5.0, -10.0], requires_grad=True)

        gt_presence = torch.tensor([0.0, 0.0])
        gt_keypoints = torch.full((2, 33, 2), float("nan"))
        visibility = torch.full((2, 33), float("nan"))
        mask = torch.zeros(2, 33)

        total_loss, l_pres, l_pose = loss_fn(
            presence_logits, pred_kpts, gt_presence, gt_keypoints, visibility, mask
        )
        assert l_pose.item() == 0.0
        assert torch.isfinite(total_loss)
        total_loss.backward()
        assert pred_kpts.grad is None or torch.isfinite(pred_kpts.grad).all()
        assert torch.isfinite(presence_logits.grad).all()

    def test_present_sample_computes_pose_loss(self):
        loss_fn = MaskedPoseLoss(lambda_presence=1.0, lambda_pose=1.0)
        presence_logits = torch.tensor([5.0])
        pred_keypoints = torch.zeros(1, 33, 2)
        gt_presence = torch.tensor([1.0])
        gt_keypoints = torch.ones(1, 33, 2)
        visibility = torch.ones(1, 33)
        mask = torch.ones(1, 33)

        total_loss, l_pres, l_pose = loss_fn(
            presence_logits, pred_keypoints, gt_presence, gt_keypoints, visibility, mask
        )
        assert l_pose.item() > 0.0
        assert torch.isclose(total_loss, l_pres + l_pose)

    def test_small_subset_overfit_sanity_check(self):
        """Step 4 technical check: model must be able to overfit a tiny synthetic batch."""
        torch.manual_seed(0)
        model = CSIPoseNet(input_features=224, gru_hidden_dim=32, gru_num_layers=1)
        loss_fn = MaskedPoseLoss(lambda_presence=1.0, lambda_pose=5.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        x = torch.randn(4, 20, 224)
        gt_pres = torch.tensor([0.0, 1.0, 1.0, 1.0])
        gt_kpts = torch.rand(4, 33, 2)
        gt_vis = torch.ones(4, 33)
        gt_mask = torch.ones(4, 33)
        gt_mask[0] = 0.0

        init_loss = None
        final_loss = None
        for _ in range(40):
            opt.zero_grad()
            p_log, p_kpts = model(x)
            loss, _, _ = loss_fn(p_log, p_kpts, gt_pres, gt_kpts, gt_vis, gt_mask)
            if init_loss is None:
                init_loss = loss.item()
            loss.backward()
            opt.step()
            final_loss = loss.item()

        assert final_loss < init_loss * 0.2


class TestMetrics:
    def test_metrics_calculation(self):
        pred_kps = np.zeros((2, 33, 2), dtype=np.float32)
        gt_kps = np.ones((2, 33, 2), dtype=np.float32) * 0.1
        mask = np.ones((2, 33), dtype=np.float32)
        presence = np.array([1.0, 0.0], dtype=np.float32)  # Only sample 0 should count

        mpjpe, per_joint, n_eval = compute_mpjpe(
            pred_kps, gt_kps, mask, presence, pixel_scale=(1000.0, 1000.0)
        )
        assert n_eval == 33
        assert abs(mpjpe - np.sqrt(100.0**2 + 100.0**2)) < 1e-3
        assert len(per_joint) == 33

        pck = compute_pck(pred_kps, gt_kps, mask, presence, thresholds=[0.10, 0.20])
        assert pck["pck@10"] == 0.0  # distance = 0.1414 > 0.10
        assert pck["pck@20"] == 100.0  # distance = 0.1414 <= 0.20

    def test_presence_metrics(self):
        preds = np.array([0.9, 0.1, 0.8, 0.7])
        targets = np.array([1.0, 0.0, 0.0, 1.0])
        res = compute_presence_metrics(preds, targets)
        assert res["accuracy"] == 0.75
        assert "f1" in res
        assert res["confusion_matrix"] == [[1, 1], [0, 2]]


class TestPredictorAndStreamBuffer:
    def test_predictor_error_states_and_empty_suppression(self, tmp_path):
        ckpt = tmp_path / "m.pt"
        norm_p = tmp_path / "n.json"
        model = CSIPoseNet(input_features=224, gru_hidden_dim=32, gru_num_layers=1)
        cfg = {"version": "1.0.0", "model": {"name": "CSIPoseNet", "input_features": 224, "gru_hidden_dim": 32, "gru_num_layers": 1, "num_keypoints": 33}, "data": {"window_size": 20, "num_subcarriers": 224, "frame_width": 1920, "frame_height": 1080}}
        save_checkpoint(model, ckpt, config=cfg)
        CSINormalizer().fit(np.random.randn(5, 20, 224).astype(np.float32)).save(norm_p)

        predictor = CSIPosePredictor(checkpoint_path=ckpt, normalizer_path=norm_p)

        # Missing input
        err_missing = predictor.predict_window(None)
        assert err_missing["status"] == "error" and err_missing["error_code"] == "MISSING_DATA"
        assert err_missing["keypoints"] == []

        # Wrong shape
        err_shape = predictor.predict_window(np.zeros((10, 224), dtype=np.float32))
        assert err_shape["status"] == "error" and err_shape["error_code"] == "INVALID_SHAPE"

        # NaN input
        bad_win = np.zeros((20, 224), dtype=np.float32)
        bad_win[0, 0] = np.nan
        err_nan = predictor.predict_window(bad_win)
        assert err_nan["status"] == "error" and err_nan["error_code"] == "NUMERICAL_INSTABILITY"

        # Stale timestamp
        err_stale = predictor.predict_window(
            np.zeros((20, 224), dtype=np.float32),
            timestamp=100.0,
            max_age_sec=5.0,
            current_time=110.0,
        )
        assert err_stale["status"] == "error" and err_stale["error_code"] == "STALE_INPUT"
        assert err_stale["person_present"] is False and err_stale["keypoints"] == []

    def test_buffer_push_and_step_emit(self, tmp_path):
        ckpt = tmp_path / "m2.pt"
        norm_p = tmp_path / "n2.json"
        model = CSIPoseNet(input_features=224, gru_hidden_dim=32, gru_num_layers=1)
        cfg = {"version": "1.0.0", "model": {"name": "CSIPoseNet", "input_features": 224, "gru_hidden_dim": 32, "gru_num_layers": 1, "num_keypoints": 33}, "data": {"window_size": 20, "num_subcarriers": 224, "frame_width": 1920, "frame_height": 1080}}
        save_checkpoint(model, ckpt, config=cfg)
        CSINormalizer().fit(np.random.randn(5, 20, 224).astype(np.float32)).save(norm_p)
        predictor = CSIPosePredictor(checkpoint_path=ckpt, normalizer_path=norm_p)

        buf = CSIStreamBuffer(predictor=predictor, window_size=20, step_size=2, feature_dim=224, max_age_sec=None)
        emitted_steps = []
        for idx in range(25):
            res = buf.push_frame(np.ones(224, dtype=np.float32) * idx, timestamp=float(idx))
            if res is not None:
                assert res["status"] == "success"
                emitted_steps.append(idx)

        # Warmup completes at idx=19 (20th frame), then step_size=2 -> 19, 21, 23
        assert emitted_steps == [19, 21, 23]
