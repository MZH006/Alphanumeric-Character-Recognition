# Handwritten Digit Recognition with Pygame and PyTorch

This project is a digit classification system that lets you **draw digits in a Pygame canvas**, and a **PyTorch-based Convolutional Neural Network (CNN)** predicts which digit you drew.

It was trained on a combination of:
- The **EMNIST Digits dataset** (by NIST),
- And **my own handwritten digits**, collected using the Pygame interface in this project.

---

## Features

-  Pygame canvas for real-time digit drawing
-  Image preprocessing (centered, thresholded, normalized)
-  CNN-based digit classification using PyTorch
-  Trained with learning rate scheduling and fine-tuned for high accuracy
-  Easily extendable to alphanumeric (WIP – see below)

---

## How It Works

1. Run the program.
2. Draw a digit (0–9) in the black square.
3. Press `C` to clear or `Q` to quit.
4. The model classifies your drawing and displays the prediction at the top.

---

## Model Details

- CNN with two convolutional layers and max pooling
- Trained on [EMNIST Digits dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- Fine-tuned on self-drawn digits
- Achieves **~86% accuracy** on the personal test set

---

## Files

- `main.py`: Entry point for running the app
- `canvas.py`: Pygame drawing interface
- `train_model.py`: Model architecture and training script
- `fine_tune.py`: Custom dataset loader and fine-tuning logic
- `digit_model.pt`: Pre-trained model file
- `fine_tuned_digit_model.pt`: Fined tuned model with self-generated handwriting

---

## Alphanumeric Support – WIP 

I’m currently working on expanding this project to support **A–Z characters**, turning it into a full **alphanumeric character recognizer**.

---

## License and Credits

- EMNIST data by [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- All handwritten digit data used for fine-tuning was self-generated.
- This project is for **educational and demonstration purposes** only.

---

## Contributing

Ideas, improvements, or pull requests are welcome — feel free to fork this repo or open an issue!

---

## Author

**Muhammad Zaid Hashmi**

