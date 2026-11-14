
# AI Bubble Pop Game

Real-time webcam-based interactive game powered by **MediaPipe Hand Landmarker**, OpenCV and Python.

Players use their hands and fingers to pop bubbles falling from the top of the screen. The system supports simultaneous hand detection, allowing multiple players to participate at the same time.

## Features

- Real-time hand detection using webcam.
- Multi-hand tracking (up to 6 hands).
- 21 hand landmarks detection.
- Finger tip tracking.
- CPU-based inference.
- Real-time FPS monitoring.
- Interactive bubble popping game (coming soon).

## Tech Stack

- Python 3.12
- MediaPipe
- OpenCV
- NumPy
- Pygame (planned)

## Project Structure

```text
ai-bubble-pop/
│
├── models/
│   ├── .gitkeep
│   └── hand_landmarker.task
│
├── src/
│   └── __init__.py
│
├── assets/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12
- Webcam
- Windows / Linux
- Minimum 8 GB RAM recommended
- NVIDIA GPU is not required

## Installation

### 1. Clone repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai-bubble-pop
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate virtual environment

For Git Bash:

```bash
source .venv/Scripts/activate
```

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download MediaPipe model

Create the models directory if it does not exist:

```bash
mkdir -p models
```

Download the pretrained Hand Landmarker model:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o models/hand_landmarker.task
```

The model will be saved inside the `models/` directory.

## Run the Project

Start the application:

```bash
python main.py
```

The webcam will open automatically.

Show your hands to the camera to test hand detection and tracking.

Press `Q` to exit.

## Hand Detection

MediaPipe Hand Landmarker detects 21 landmarks for each detected hand.

Important landmarks:

| Index | Finger |
|---|---|
| 4 | Thumb |
| 8 | Index Finger |
| 12 | Middle Finger |
| 16 | Ring Finger |
| 20 | Pinky |

The application is configured to detect up to 6 hands simultaneously.

## Development Roadmap

- [x] Initial project setup
- [x] MediaPipe integration
- [x] Webcam integration
- [x] Multi-hand detection
- [x] Hand landmark visualization
- [ ] Finger tracking improvements
- [ ] Bubble animation
- [ ] Collision detection
- [ ] Bubble popping effects
- [ ] Sound effects
- [ ] Score system
- [ ] Multiple player support
- [ ] Game levels and difficulty

## Notes

- Detection performance depends on lighting, camera quality and CPU performance.
- Keep your hands visible to the webcam.
- The application does not identify individual players yet.
- The pretrained model is downloaded separately and is not stored in Git.

## License

This project is developed for educational and experimental purposes.
