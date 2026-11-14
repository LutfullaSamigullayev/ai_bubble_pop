# AI Bubble Pop Game

Webkamera orqali ikkita qo'lni kuzatadigan va bosh barmoq bilan ko'rsatkich barmoqni qisish (pinch) orqali havo sharlarini yoradigan interaktiv o'yin.

## Asosiy imkoniyatlar

- MediaPipe `0.10.21` Legacy Hands API.
- `model_complexity=0` — CPU uchun yengil model.
- Bir vaqtning o'zida maksimal 2 ta qo'l.
- Har bir qo'l uchun 21 ta landmark.
- Shar faqat yangi pinch harakati shar tanasi ichida sodir bo'lsa yoriladi.
- Hajmli gradient, yorug'lik aks etishi, tugun va ipga ega sharlar.
- Kamera va AI alohida worker threadda ishlaydi.
- Pygame o'yin sikli 60 FPS maqsadida ishlaydi.
- Full-screen rejim.

## Talablar

- Python 3.12
- Windows 10/11 (Linux va macOS ham mos bo'lishi mumkin)
- Webkamera
- Tavsiya etilgan RAM: 8 GB yoki undan yuqori
- NVIDIA GPU talab qilinmaydi

## O'rnatish

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Kutubxonalarni o'rnatish:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Muhim: bu versiya MediaPipe `0.10.21` bilan tekshirilgan `mp.solutions.hands` API'sidan foydalanadi. `mp.tasks` API'si ishlatilmaydi.

## Avval kamerani alohida tekshirish

```bash
python hand_test_lite.py
```

`Q` tugmasi bilan testdan chiqing.

## O'yinni ishga tushirish

```bash
python main.py
```

- Bosh va ko'rsatkich barmoq uchini bir-biriga qisib sharni yoriting.
- Oddiy tegish sharni yormaydi.
- Har bir yorilgan shar 10 ball beradi.
- `ESC` yoki `Q` tugmasi o'yindan chiqaradi.

## Sozlamalar

`src/hand_tracker.py` faylida:

```python
MAX_HANDS = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.45
MIN_TRACKING_CONFIDENCE = 0.45
```

Pinch sezgirligi shu fayldagi `PinchTracker` klassida:

```python
PINCH_START_RATIO = 0.30
PINCH_RELEASE_RATIO = 0.46
```

- `PINCH_START_RATIO` kattaroq bo'lsa, qisish osonroq aniqlanadi, lekin noto'g'ri trigger ehtimoli ortadi.
- `PINCH_RELEASE_RATIO` start qiymatidan kattaroq bo'lishi kerak. Bu jitter'ni kamaytiruvchi hysteresis beradi.

## FPS haqida

- `O'YIN FPS` — Pygame animatsiya tezligi.
- `AI FPS` — kamera kadrlarini MediaPipe qayta ishlash tezligi.

Thread ishlatish AI hisoblashini tezlashtirmaydi, ammo inference o'yin animatsiyasini bloklamasligiga yordam beradi. AI FPS protsessor, kamera va yorug'lik sharoitiga bog'liq.

## Loyiha tuzilishi

```text
ai-bubble-pop/
├── assets/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── hand_tracker.py
├── hand_test_lite.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

MediaPipe `0.10.21` Legacy Hands API model og'irliklarini paketning o'zidan yuklaydi. `models/hand_landmarker.task` fayli bu versiyada kerak emas.
