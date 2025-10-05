# Lung Cancer Detector (Android + TensorFlow Lite)

A lightweight Android app that runs **on-device** to analyze chest images and surface lung-cancer findings using TensorFlow Lite.  
Works offline, keeps data private, and lets you swap models easily.

---

## ⚠️ Medical Disclaimer

> **This is a research/educational prototype and not a medical device. Do not use it for diagnosis or clinical decision-making.**

---

## 🚀 Features

- **On-device inference (offline):** No image upload or network calls.
- **Simple UI:** Pick from camera/gallery and get a prediction.
- **Drop-in model updates:** Replace `model.tflite` + `labels.txt` without code changes.
- **Optimized:** Supports TFLite quantization for speed and smaller size.

---

## 🛠 Requirements

- Android Studio (JDK 17)
- Android SDK 24+
- TensorFlow Lite runtime:

  ```groovy
  // app/build.gradle
  implementation "org.tensorflow:tensorflow-lite:2.14.0"
  ```

---

## ⚙️ Setup & Run

1. **Clone the repository**

   ```sh
   git clone https://github.com/USERNAME/lung-cancer-Innovative-Mobile-Cancer-Detection.git
   cd lung-cancer-Innovative-Mobile-Cancer-Detection
   ```

2. **Open in Android Studio**  
   Let Gradle sync.

3. **Add model files under `assets/`:**
   - `model.tflite`
   - `labels.txt` (one class name per line)

4. **Run ▶️** on a real device or emulator.

---

## 📁 Project Structure

```
app/                           # Android (Kotlin)
assets/
  ├─ model.tflite              # TFLite model (replace with yours)
  └─ labels.txt                # Class labels (one per line)
docs/
  └─ lung_cancer_classification.ipynb  # Training & TFLite export
build.gradle  settings.gradle  gradle/
README.md  LICENSE
```

---

## 🧩 Minimal Inference Snippet (Kotlin)

> Ensure input size & normalization match training exactly.

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

// Load TFLite from assets
val tflite = Interpreter(
    FileUtil.loadMappedFile(this, "model.tflite"),
    Interpreter.Options().apply { setNumThreads(4) } // tune for device
)

// Example input [1, 224, 224, 3] Float32, pre-normalized as in training
val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
val output = Array(1) { FloatArray(NUM_CLASSES) }

// Run inference
tflite.run(input, output)

// Top-1
val probs = output[0]
val bestIdx = probs.indices.maxBy { probs[it] } ?: 0
val confidence = probs[bestIdx]
```

---

## 🧪 Train & Export (Brief)

Use `docs/lung_cancer_classification.ipynb` to:

- Prepare data (resize, normalize, split, light augmentation)
- Fine-tune (e.g., MobileNet/EfficientNet-Lite)
- Evaluate (Accuracy, ROC-AUC, PR-AUC, confusion matrix)
- Export to TFLite (consider Dynamic/INT8 quantization)
- Copy `model.tflite` + `labels.txt` to `assets/`

---

## 🔐 Privacy

- Inference happens **entirely on device**.
- No telemetry or analytics by default.

---

## 📜 License

Open-source under MIT License (add/adjust the LICENSE file as needed).

---

**ملحوظة:** هذا المشروع للبحث والتعليم فقط، وليس جهازًا طبيًا.
