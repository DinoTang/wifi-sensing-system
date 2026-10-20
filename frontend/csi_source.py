"""
Wi-Fi Sensing System - CSI Data Source & Waveform Generator
Mô phỏng và xử lý tín hiệu sóng CSI chuẩn bài báo khoa học (Walking, Falling, Sitting, Hand Swing)
với các mốc phân đoạn sự kiện (Start - End boundaries).
"""

import time
import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple


class CSIDataSource:
    """Nguồn cấp dữ liệu CSI hỗ trợ Real-time và các mẫu sự kiện chuẩn nghiên cứu."""

    def __init__(self, num_subcarriers: int = 64):
        self.num_subcarriers = num_subcarriers
        self.subcarrier_indices = list(range(-num_subcarriers // 2, num_subcarriers // 2))

        # Trạng thái hiện tại
        self.mode = "mock"
        self.packet_rate = 100  # 100Hz = 10ms mỗi gói (chuẩn thu thập CSI)
        self.time_ms = 0

        # Cửa sổ hiển thị thời gian thực (lưu 200 điểm ~ 4-5 giây)
        self.buffer_size = 150
        self.time_points: List[int] = list(range(0, self.buffer_size * 20, 20))
        self.signal_buffer: List[float] = [0.01 + random.gauss(0, 0.001) for _ in range(self.buffer_size)]

        # Phân đoạn sự kiện (Event Segmentation)
        self.is_event_active = False
        self.event_start_idx: int = -1
        self.event_end_idx: int = -1
        self.current_activity = "walking"  # walking, hand_swing, falling, sitting_down, empty

        # Cài đặt ngưỡng
        self.threshold = 0.008
        self.baseline_noise = 0.0015

        # Nhật ký sự kiện
        self.event_logs: List[Dict[str, Any]] = [
            {
                "time": time.strftime("%H:%M:%S"),
                "event": "Khởi động hệ thống phân tích tín hiệu CSI thời gian thực",
                "state": "STANDBY",
                "confidence": "100%",
            }
        ]

    def generate_benchmark_pattern(self, activity_type: str, num_points: int = 200) -> Tuple[List[int], List[float], int, int]:
        """
        Tạo dạng sóng CSI chuẩn bài báo khoa học (IEEE Wi-Fi Sensing)
        theo đúng mẫu thực tế: walking, hand_swing, falling, sitting_down.
        Trả về: (time_ms_list, csi_amplitude_list, start_idx, end_idx)
        """
        np.random.seed(int(time.time() * 1000) % 10000)
        time_ms = [i * 20 for i in range(num_points)]  # 0 đến 4000ms
        signal = np.zeros(num_points)

        # Baseline tĩnh ban đầu
        baseline = 0.01 + np.random.normal(0, 0.001, num_points)
        start_idx = 40   # tại ~800ms
        end_idx = 150    # tại ~3000ms

        if activity_type == "walking":
            # Đi bộ: Dao động tần số cao liên tục do bước chân và thân người chuyển động
            start_idx = 35
            end_idx = 155
            for i in range(num_points):
                if i < start_idx:
                    signal[i] = 0.012 + np.random.normal(0, 0.0008)
                elif i <= end_idx:
                    # Giai đoạn quá độ khởi đầu (transient start)
                    t_rel = (i - start_idx) / (end_idx - start_idx)
                    carrier = 0.006 * np.sin(2 * np.pi * 3.5 * t_rel * 4)
                    burst = 0.008 * np.sin(2 * np.pi * 9.0 * t_rel * 4) * np.sin(2 * np.pi * 1.2 * t_rel)
                    envelope = 0.005 + 0.004 * np.sin(np.pi * t_rel)
                    noise = np.random.normal(0, 0.0015)
                    signal[i] = carrier + burst + noise + envelope
                else:
                    signal[i] = -0.012 + np.random.normal(0, 0.0008)

        elif activity_type == "hand_swing":
            # Vung tay: Tín hiệu dâng cao dần thành một vòm (envelope) với gợn sóng nhỏ
            start_idx = 50
            end_idx = 140
            for i in range(num_points):
                if i < start_idx:
                    signal[i] = -0.005 + np.random.normal(0, 0.0008)
                elif i <= end_idx:
                    t_rel = (i - start_idx) / (end_idx - start_idx)
                    arch = 0.035 * np.sin(np.pi * t_rel)
                    ripples = 0.005 * np.sin(2 * np.pi * 5.0 * t_rel * 3) + np.random.normal(0, 0.002)
                    signal[i] = arch + ripples
                else:
                    signal[i] = 0.002 + np.random.normal(0, 0.0008)

        elif activity_type == "falling":
            # Ngã: Đột biến tín hiệu mạnh rồi giảm dốc sâu đột ngột
            start_idx = 45
            end_idx = 135
            for i in range(num_points):
                if i < start_idx:
                    signal[i] = 0.018 + np.random.normal(0, 0.0006)
                elif i <= end_idx:
                    t_rel = (i - start_idx) / (end_idx - start_idx)
                    if t_rel < 0.2:
                        # Lao xuống / hẫng
                        signal[i] = 0.015 + 0.008 * np.sin(2 * np.pi * 4 * t_rel) + np.random.normal(0, 0.001)
                    elif t_rel < 0.6:
                        # Rơi tự do & va đập mạnh
                        decay = np.exp(-3 * (t_rel - 0.2))
                        signal[i] = -0.015 * (t_rel / 0.6) + 0.008 * decay * np.sin(2 * np.pi * 8 * t_rel) + np.random.normal(0, 0.002)
                    else:
                        # Nằm yên trên sàn sau cú ngã
                        signal[i] = -0.008 + np.random.normal(0, 0.0008)
                else:
                    signal[i] = -0.009 + np.random.normal(0, 0.0006)

        elif activity_type == "sitting_down":
            # Ngồi xuống: Có bước sóng âm rồi đảo pha dương rồi bình ổn
            start_idx = 40
            end_idx = 130
            for i in range(num_points):
                if i < start_idx:
                    signal[i] = 0.014 + np.random.normal(0, 0.0008)
                elif i <= end_idx:
                    t_rel = (i - start_idx) / (end_idx - start_idx)
                    dip = -0.025 * np.exp(-((t_rel - 0.25) ** 2) / 0.02)
                    rebound = 0.032 * np.exp(-((t_rel - 0.6) ** 2) / 0.04)
                    wobble = 0.003 * np.sin(2 * np.pi * 6 * t_rel) + np.random.normal(0, 0.0015)
                    signal[i] = dip + rebound + wobble
                else:
                    signal[i] = -0.005 + np.random.normal(0, 0.0008)

        else:  # empty / tĩnh
            signal = [0.005 + random.gauss(0, 0.001) for _ in range(num_points)]
            start_idx = -1
            end_idx = -1

        return time_ms, [round(float(val), 5) for val in signal], start_idx, end_idx

    def step_realtime(self) -> Dict[str, Any]:
        """Cập nhật 1 bước thời gian thực và tự động phát hiện ranh giới Start / End."""
        self.time_ms += 50
        noise = random.gauss(0, 0.0008)

        # Tính giá trị mới dựa trên trạng thái
        if self.current_activity == "empty":
            val = 0.005 + noise
            self.is_event_active = False
            state_label = "Phòng trống"
            conf = 0.96
        elif self.current_activity == "walking":
            # Dao động chuyển động
            t_sec = self.time_ms / 1000.0
            val = 0.015 * math.sin(2 * math.pi * 3.5 * t_sec) + 0.008 * math.sin(2 * math.pi * 8.0 * t_sec) + random.gauss(0, 0.002)
            state_label = "Có người (Đi bộ / Walking)"
            conf = 0.98
        elif self.current_activity == "falling":
            t_sec = self.time_ms / 1000.0
            val = -0.012 + 0.008 * math.sin(2 * math.pi * 6.0 * t_sec) * math.exp(-(t_sec % 3)) + random.gauss(0, 0.0015)
            state_label = "CẢNH BÁO: Phát hiện ngã (Falling)!"
            conf = 0.94
        elif self.current_activity == "sitting_down":
            t_sec = self.time_ms / 1000.0
            val = 0.018 * math.sin(2 * math.pi * 1.5 * t_sec) + random.gauss(0, 0.0012)
            state_label = "Có người (Ngồi xuống / Sitting)"
            conf = 0.92
        else:  # hand_swing
            t_sec = self.time_ms / 1000.0
            val = 0.022 * math.sin(2 * math.pi * 2.2 * t_sec) + random.gauss(0, 0.0015)
            state_label = "Có người (Vung tay / Cử động)"
            conf = 0.95

        # Cập nhật buffer
        self.signal_buffer.append(round(val, 5))
        self.signal_buffer.pop(0)

        # Cập nhật time points
        self.time_points.append(self.time_ms)
        self.time_points.pop(0)

        # Tính năng lượng biến thiên gần đây (window 15 mẫu)
        recent_window = self.signal_buffer[-15:]
        variance = float(np.var(recent_window))

        # Tự động phát hiện Start / End dựa trên ngưỡng biến thiên
        if variance > 0.00004 and not self.is_event_active:
            self.is_event_active = True
            self.event_start_idx = len(self.signal_buffer) - 10
            ts = time.strftime("%H:%M:%S")
            self.event_logs.insert(0, {
                "time": ts,
                "event": f"START: Phát hiện khởi phát hành vi [{self.current_activity}]",
                "state": "START",
                "confidence": f"{conf*100:.1f}%",
            })
        elif variance <= 0.00002 and self.is_event_active:
            self.is_event_active = False
            self.event_end_idx = len(self.signal_buffer) - 5
            ts = time.strftime("%H:%M:%S")
            self.event_logs.insert(0, {
                "time": ts,
                "event": f"END: Kết thúc hành vi [{self.current_activity}]",
                "state": "END",
                "confidence": f"{conf*100:.1f}%",
            })

        if len(self.event_logs) > 30:
            self.event_logs = self.event_logs[:30]

        return {
            "time_ms": self.time_points,
            "signal": self.signal_buffer,
            "variance": variance * 10000,
            "is_event_active": self.is_event_active,
            "state_label": state_label,
            "confidence": conf,
            "activity": self.current_activity,
        }
