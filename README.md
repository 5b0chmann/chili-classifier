# 🌶️ Chili Classifier

Ein Deep-Learning-Projekt zur Klassifikation von roten und grünen Chilis mit TensorFlow, PyTorch und Transfer Learning.

## Projektübersicht

In diesem Projekt wurde ein eigener Bilddatensatz mit roten und grünen Chilis erstellt und für verschiedene Bildklassifikationsverfahren verwendet.

Umgesetzt wurden:

- Eigenes Bilddataset
- CNN von Scratch
- Transfer Learning mit MobileNetV2
- Transfer Learning mit ResNet18
- Fine-Tuning eines vortrainierten Modells
- Evaluation auf neuen, zuvor unbekannten Testbildern

## Datensatz

Der Datensatz besteht aus:

| Klasse | Anzahl Bilder |
|----------|----------:|
| Red Chili | 33 |
| Green Chili | 33 |
| Gesamt | 66 |

Ordnerstruktur:

```text
chili_dataset/
├── red_chili/
└── green_chili/
```

## Verwendete Technologien

- Python
- TensorFlow / Keras
- PyTorch
- MobileNetV2
- ResNet18
- Google Colab
- Google Drive
- Git / GitHub

---

# Aufgabe 2 – Dataset laden

Die Bilder wurden mit einer ImageFolder-Struktur organisiert und geladen.

Verwendete Vorverarbeitung:

- Resize auf 224 × 224 Pixel
- Umwandlung in Tensoren
- Normalisierung

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

---

# Aufgabe 3 – Scratch-CNN

Ein eigenes CNN wurde ohne vortrainierte Gewichte entwickelt und trainiert.

Architektur:

```text
Rescaling
↓
Conv2D (16)
↓
MaxPooling
↓
Conv2D (32)
↓
MaxPooling
↓
Conv2D (64)
↓
MaxPooling
↓
Dense
↓
Dropout
↓
Dense (Sigmoid)
```

## Overfitting

Das Modell erreichte:

- Training Accuracy ≈ 100 %
- Validation Accuracy ≈ 92 %

Der Validation Loss stieg gegen Ende des Trainings an.

Dies deutet auf leichtes Overfitting hin.

---

# Aufgabe 4 – Transfer Learning

Es wurde ein vortrainiertes ResNet18 verwendet.

Vorgehen:

1. Vortrainiertes ResNet18 geladen
2. Alle Features eingefroren
3. Klassifikationskopf ersetzt
4. Nur den Head trainiert

Neuer Klassifikationskopf:

```python
model.fc = nn.Linear(512, 2)
```

---

# Aufgabe 5 – Fine-Tuning

Nach dem Head-Training wurde Fine-Tuning durchgeführt.

Freigegeben wurden:

```text
layer4
fc
```

Alle früheren Schichten blieben eingefroren.

Für das Fine-Tuning wurde eine kleine Lernrate verwendet:

```python
lr = 1e-5
```

---

# Aufgabe 6 – Eigene Testbilder

Für die Evaluation wurden neue Bilder verwendet, die nicht Bestandteil des Trainingsdatensatzes waren.

## Testergebnisse

| Testbild | Ergebnis |
|-----------|-----------|
| green_test_01 | ✅ |
| green_test_02 | ❌ |
| green_test_03 | ✅ |
| green_test_04 | ✅ |
| red_test_01 | ✅ |
| red_test_02 | ❌ |
| red_test_03 | ✅ |
| red_test_04 | ✅ |

### Gesamtgenauigkeit

```text
6 von 8 korrekt
= 75 %
```

## Beobachtungen

Das Modell erzielte die besten Ergebnisse bei Bildern, die überwiegend rote oder grüne Chilis enthielten.

Schwierigkeiten entstanden insbesondere bei Bildern mit gleichzeitig roten und grünen Chilis.

Dadurch sank die Vorhersagesicherheit und es kam vereinzelt zu Fehlklassifikationen.

---

# Modell speichern

```python
model.save("chili_classifier.keras")
```

Modell laden:

```python
from tensorflow import keras

model = keras.models.load_model(
    "chili_classifier.keras"
)
```

---

# Projektstruktur

```text
chili-classifier/
│
├── Chili_Classifier.ipynb
├── README.md
├── requirements.txt
│
├── model/
│   └── chili_classifier.keras
│
├── images/
│   └── accuracy_plot.png
│
├── red_chili/
├── green_chili/
└── test_images/
```

---

# Fazit

Mit einem relativ kleinen Datensatz von 66 Bildern konnten mittels Transfer Learning und Fine-Tuning gute Ergebnisse erzielt werden.

Das Projekt demonstriert den vollständigen Workflow eines Computer-Vision-Projekts:

- Datenerfassung
- Datenvorbereitung
- Training
- Transfer Learning
- Fine-Tuning
- Evaluation auf neuen Testbildern

---

Erstellt als Lern- und Portfolio-Projekt mit TensorFlow, PyTorch und Google Colab.
