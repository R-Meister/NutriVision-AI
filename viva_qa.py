"""
Viva Q&A and project documentation helper.
Run with: python viva_qa.py
"""
VIVA_QA = [
    {
        "q": "What is a CNN and how does it work?",
        "a": """A Convolutional Neural Network (CNN) is a deep learning model optimised for grid-structured data like images.
Key components:
  • Convolutional Layer: Applies learnable filters to extract local features (edges, textures)
  • ReLU: Introduces non-linearity (f(x) = max(0, x))
  • Batch Normalisation: Normalises activations for stable training
  • Max Pooling: Down-samples spatial dimensions, retains dominant features
  • Flatten: Converts 3D feature maps to 1D for fully-connected layers
  • Dense + Softmax: Produces class probabilities"""
    },
    {
        "q": "What is Transfer Learning?",
        "a": """Transfer Learning reuses a model trained on a large dataset (ImageNet) for a new, related task.
Workflow:
  1. Take MobileNetV2 pretrained on 1.28M ImageNet images
  2. Remove the top classification layer
  3. Freeze all base layers (preserve learned features)
  4. Add a new classification head for 12 food classes
  5. Train only the head (Phase 1 - Feature Extraction)
  6. Optionally unfreeze top layers for fine-tuning (Phase 2)"""
    },
    {
        "q": "What is Backpropagation?",
        "a": """Backpropagation is the algorithm for training neural networks.
  1. Forward pass: Compute predictions and loss
  2. Compute gradient of loss w.r.t. each weight using chain rule
  3. Update weights: w = w - lr × gradient
  In TensorFlow, tf.GradientTape computes gradients automatically.
  Adam optimiser applies adaptive per-parameter learning rates."""
    },
    {
        "q": "What is the difference between CNN and ANN in this project?",
        "a": """CNN: Image classifier
  Input: Raw pixels (224×224×3)
  Purpose: Classify food type from visual features
  Output: Probability over 12 food classes

ANN: Nutritional regressor
  Input: 13 numerical features (one-hot food class + normalised portion)
  Purpose: Predict calories, protein, carbs, fat
  Output: 4 continuous values
  
CNN answers: 'What food is this?'
ANN answers: 'How much nutrition does this portion contain?'"""
    },
    {
        "q": "Why is portion size estimation approximate?",
        "a": """A 2D RGB image provides no depth information. The same visual area
could represent 100g if the camera is close, or 1kg if far away.
Accurate estimation requires:
  • Depth sensor (LiDAR, Time-of-Flight)
  • Reference object of known size in frame
  • Stereo vision for 3D reconstruction
We use heuristic estimation (food pixel ratio × typical serving size)
and provide a manual slider for user correction."""
    },
    {
        "q": "What activation functions are used and why?",
        "a": """ReLU (Rectified Linear Unit): f(x) = max(0, x)
  Used in all hidden layers
  Advantages: No vanishing gradient, computationally cheap, sparse activations

Softmax: σ(z)_i = exp(z_i) / Σ exp(z_j)
  Used in output layer for multi-class classification
  Converts logits to probability distribution (sums to 1)

Linear (no activation): Used in ANN output for regression
  Allows prediction of any positive/negative value"""
    },
    {
        "q": "What is the calorie calculation formula?",
        "a": """Total Calories = (portion_grams / 100) × calories_per_100g

Example: 150g of Pizza
  = (150 / 100) × 266 kcal/100g
  = 1.5 × 266
  = 399 kcal

Same formula for protein, carbs, fat:
  Nutrient_g = (portion_g / 100) × nutrient_per_100g"""
    },
    {
        "q": "What dataset did you use?",
        "a": """Real photographs assembled from four public datasets:
  • Food-101 (7 classes): pizza, burger, french fries, samosa,
    fried rice, club sandwich, omelette
  • Indian Foods Dataset: biryani, dosa
  • Indian Food Images: idli
  • Fruits-360: apple, banana

Per class: 200 train / 40 validation / 47 held-out test images
(2,400 train, 480 validation, 564 test in total).

The official dataset splits are respected:
  • Food-101 validation split → our test set
  • Indian Foods / Indian Food Images / Fruits-360 test splits → our test set

The synthetic random-noise loader still exists, but it is ONLY for checking
that the training code runs. It is never used for reported accuracy."""
    },
    {
        "q": "How do you prevent overfitting?",
        "a": """Several techniques:
  1. Dropout (0.3–0.5): Randomly deactivates neurons during training
  2. Batch Normalisation: Regularises activations
  3. Data Augmentation: Increases training data diversity
  4. Early Stopping: Halts training when val_loss stops improving
  5. Learning Rate Decay: ReduceLROnPlateau callback
  6. L2 regularisation: Adam's weight decay
  7. Validation monitoring: Track val_accuracy vs train_accuracy gap"""
    },
    {
        "q": "What is MobileNetV2 and why is it suitable for this project?",
        "a": """MobileNetV2 is a lightweight CNN architecture by Google.
Key features:
  • Depthwise Separable Convolutions: 8-9× fewer operations than standard Conv
  • Inverted Residuals: Bottleneck blocks with linear activations
  • 3.4M parameters (vs VGG16's 138M)
  • Pre-trained on ImageNet (1.28M images, 1000 classes)

Why suitable:
  • Fast inference (CPU-compatible)
  • Food visual features transfer well from ImageNet
  • Frozen ImageNet features + a linear probe reached 91.1% top-1 and
    98.6% top-3 on our 562-image held-out real-photo test set"""
    },
]

if __name__ == "__main__":
    print("=" * 70)
    print("  VIVA Q&A — Food Recognition & Calorie Estimation Project")
    print("=" * 70)
    for i, qa in enumerate(VIVA_QA, 1):
        print(f"\nQ{i}: {qa['q']}")
        print(f"A: {qa['a']}")
        print("-" * 60)
