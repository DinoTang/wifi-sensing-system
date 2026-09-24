# Wi-Fi Sensing System

A Wi-Fi sensing system that uses **Channel State Information (CSI)** to detect human presence in an indoor environment.

The system collects CSI data from Wi-Fi signals, processes and extracts meaningful signal features, applies a detection algorithm, and visualizes the results through an application interface.

---

## 📌 Project Overview

Wi-Fi signals are affected by objects and people within the propagation environment. By analyzing changes in **Channel State Information (CSI)**, the system can detect whether a person is present in a monitored indoor environment.

The project consists of five main components:

- Hardware & CSI Data Collection
- Signal Processing
- Machine Learning
- Backend
- Frontend

---

## 🎯 Project Objectives

- Collect CSI data from Wi-Fi signals.
- Process and analyze raw CSI data.
- Visualize CSI signals.
- Extract meaningful features from CSI.
- Detect human presence in an indoor environment.
- Provide a user interface for monitoring CSI and detection results.
- Establish communication between all system components.

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐
│    Wi-Fi Access Point   │
└────────────┬────────────┘
             │
             │ Wi-Fi Signal
             ▼
┌─────────────────────────┐
│   CSI-capable Device    │
│         ESP32           │
└────────────┬────────────┘
             │
             │ Raw CSI
             ▼
┌─────────────────────────┐
│    Signal Processing    │
│                         │
│  • Amplitude            │
│  • Phase                │
│  • Filtering            │
│  • Normalization        │
│  • Feature Extraction   │
└────────────┬────────────┘
             │
             │ Processed CSI
             │ / Features
             ▼
┌─────────────────────────┐
│    Machine Learning     │
│                         │
│  Human Presence         │
│  Detection              │
└────────────┬────────────┘
             │
             │ Detection Result
             ▼
┌─────────────────────────┐
│        Backend          │
│                         │
│  REST API / WebSocket   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│        Frontend         │
│                         │
│  ┌───────────────────┐  │
│  │ CSI Visualization │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ Human Detection   │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

---

## 🔄 Data Flow

```text
Wi-Fi Signal
     │
     ▼
CSI-capable Device
     │
     ▼
Raw CSI Data
     │
     ▼
Signal Processing
     │
     ├── Amplitude
     ├── Phase
     ├── Filtering
     └── Normalization
     │
     ▼
Feature Extraction
     │
     ▼
Machine Learning
     │
     ▼
Human / No Human
     │
     ▼
Backend
     │
     ▼
Frontend
```

---

## 📂 Project Structure

```text
wifi-sensing-system/
│
├── README.md
├── .gitignore
│
├── hardware/
│   └── esp32-csi/
│
├── signal-processing/
│   ├── src/
│   │   ├── collector/
│   │   ├── processing/
│   │   └── features/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   └── tests/
│
├── machine-learning/
│   ├── data/
│   ├── src/
│   ├── models/
│   └── notebooks/
│
├── backend/
│   ├── app/
│   └── tests/
│
├── frontend/
│   ├── lib/
│   └── assets/
│
└── docs/
    ├── architecture/
    ├── data-contract/
    └── api/
```

---

## 👥 Team Responsibilities

| Module | Responsibility |
|---|---|
| **Hardware** | Wi-Fi configuration, ESP32 setup and CSI data collection |
| **Signal Processing** | CSI preprocessing, filtering, normalization and feature extraction |
| **Machine Learning** | Human presence detection using processed CSI features |
| **Backend** | System integration, API and real-time communication |
| **Frontend** | CSI visualization and human detection interface |

---

## 🔬 Signal Processing

The Signal Processing module transforms raw CSI data into processed signals and meaningful features.

### Processing Pipeline

```text
Raw CSI
   │
   ▼
Amplitude / Phase Extraction
   │
   ▼
Noise Filtering
   │
   ▼
Normalization
   │
   ▼
Feature Extraction
   │
   ▼
Processed CSI / Features
```

### Technologies

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

## 🤖 Machine Learning

The Machine Learning module uses processed CSI data to determine whether a person is present in the monitored environment.

### Input

Processed CSI features from the Signal Processing module.

### Output

```text
Human Detected
       or
No Human Detected
```

The project may use classical Machine Learning algorithms initially, with more advanced models considered depending on dataset size and experimental results.

Potential technologies:

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch

---

## 🌐 Backend

The Backend acts as the communication layer between the detection system and the frontend application.

### Responsibilities

- Receive processed data and detection results.
- Provide APIs for the frontend.
- Provide real-time data communication.
- Integrate system components.

### Planned Technologies

- Python
- FastAPI
- REST API
- WebSocket

---

## 📱 Frontend

The application provides two main screens.

### 1. CSI Visualization

Displays CSI signal information for monitoring and analysis.

### 2. Human Detection

Displays the current human presence detection result.

### Planned Technologies

- Flutter
- Dart

---

## 🔀 Git Workflow

The project uses Git for version control.

The `main` branch contains the stable version of the project.

Each team member works on a separate feature branch.

```text
                         ┌── feature/hardware
                         │
                         ├── feature/signal-processing
                         │
main ────────────────────┼── feature/machine-learning
                         │
                         ├── feature/backend
                         │
                         └── feature/frontend
```

### Branches

```text
main
feature/hardware
feature/signal-processing
feature/machine-learning
feature/backend
feature/frontend
```

### Development Workflow

```text
Create / Switch Branch
        │
        ▼
      Develop
        │
        ▼
       Test
        │
        ▼
      Commit
        │
        ▼
       Push
        │
        ▼
 Pull Request
        │
        ▼
   Code Review
        │
        ▼
 Merge into main
```

---

## 🛠️ Development Environment

### Hardware

- Wi-Fi Access Point
- ESP32 or another CSI-capable device

### Signal Processing

- Python 3
- NumPy
- SciPy
- Pandas
- Matplotlib
- Jupyter Notebook

### Machine Learning

- Python
- Scikit-learn
- PyTorch (if required)

### Backend

- Python
- FastAPI
- WebSocket

### Frontend

- Flutter
- Dart

### Version Control

- Git
- GitHub

---

## 📋 Development Status

| Component | Status |
|---|---|
| Project Structure | 🟢 Completed |
| Git Repository | 🟢 Completed |
| Git Branch Structure | 🟢 Completed |
| Hardware / CSI Collection | 🟡 In Progress |
| Signal Processing | 🟡 In Progress |
| Feature Extraction | 🟡 Planned |
| Machine Learning | 🟡 Planned |
| Backend | 🟡 Planned |
| Frontend | 🟡 Planned |
| System Integration | ⚪ Not Started |
| Testing | ⚪ Not Started |

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/<username>/wifi-sensing-system.git
cd wifi-sensing-system
```

### Check Available Branches

```bash
git branch -a
```

### Switch to Your Development Branch

Example:

```bash
git switch feature/signal-processing
```

---

## 📡 CSI Data

The system works with raw CSI information obtained from the CSI-capable device.

A raw CSI sample may contain information such as:

```json
{
  "timestamp": 123456.78,
  "rssi": -45,
  "channel": 6,
  "csi_real": [12, 15, 18, 20],
  "csi_imag": [-4, -6, -3, -5]
}
```

After signal processing, the data can be transformed into amplitude, phase, and other extracted features.

The exact data format will be finalized after the hardware and signal-processing modules are integrated.

---

## 📚 Documentation

Project documentation is stored in:

```text
docs/
├── architecture/
├── data-contract/
└── api/
```

### Architecture

Contains system architecture and module relationships.

### Data Contract

Defines the data formats exchanged between modules.

### API

Defines communication between Backend and other components.

---

## 🎯 Final Goal

Build a functional Wi-Fi sensing system capable of:

```text
Collect CSI
     ↓
Process CSI
     ↓
Extract Features
     ↓
Detect Human Presence
     ↓
Send Detection Result
     ↓
Display Results
```

The final system should provide both **CSI signal visualization** and **real-time human presence detection** through the application interface.

---

## 📜 License

This project is developed for academic and educational purposes.
