A CNN-BiLSTM Approach using MobileNetV2

### Project Overview
This project implements a deep learning pipeline to detect real-life violence in video clips. By combining spatial feature extraction (CNN) with temporal sequence modeling (BiLSTM), the model can distinguish between normal human interactions and violent altercations with high precision.

### Technical Architecture
- **Feature Extractor**: MobileNetV2 (Pre-trained on ImageNet) used as a Time-Distributed layer.

- **Sequence Processor**: Bidirectional LSTM (128 units) to analyze motion in both forward and backward directions.

- **Input Pipeline**: Frame Sampling: 20 frames per video (Uniformly sampled).

Resolution: 128x128 pixels.

Normalization: Rescaled to [0,1].

### Performance Summary
Due to computational resource management, the model was trained on a curated subset of the Real Life Violence Situations Dataset.

### Metric	Result
Dataset Size	400 Videos (200 Violence / 200 Non-Violence)
Training Accuracy	96.1%
Validation Accuracy	91.2%
Loss Function	Binary Crossentropy
Optimizer	Adam (Learning Rate: 10^−4)

### Key Modifications for Hardware Efficiency
**Uniform Interval Sampling**: Instead of processing entire videos, the code divides each video into 20 equal segments, capturing the "story" of the clip without overwhelming the GPU RAM.

**Early Stopping**: Implemented to monitor val_loss and prevent overfitting, restoring the best weights once the model reaches peak performance.

**Time-Distributed CNN**: Efficiently handles the 5D input tensor (Batch,Frames,Height,Width,Channels).

### Future Improvements
Full Dataset Scaling: Transition to a DataGenerator pipeline to utilize all 2,000+ videos.

Attention Mechanisms: Incorporating "Attention" layers to help the model focus on specific objects (e.g., weapons or fast-moving limbs).

TFLite Conversion: Optimizing the model for real-time edge deployment on CCTV hardware.