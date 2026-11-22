# 🥗 NutriVision AI — Food Image Analyzer & Calorie Counter

> **Advanced AI-Powered Nutrition & Calorie Estimation Platform**  
> Inspired by WebMD HealthTools · Deep Learning (MobileNetV2, CNN, ANN) · OpenCV Multi-Food Segmentation · 504+ Verified Nutrition Database · Streamlit Web Application

---

## 📋 Table of Contents
1. [Executive Summary & Workflow](#-executive-summary--workflow)
2. [Key Features & Improvements](#-key-features--improvements)
3. [Deep Learning Architecture](#-deep-learning-architecture)
4. [Smart Nutrition Database (504+ Foods)](#-smart-nutrition-database-504-foods)
5. [OpenCV Multi-Food Region Segmentation](#-opencv-multi-food-region-segmentation)
6. [WebMD-Inspired Calorie & Diet Calculator](#-webmd-inspired-calorie--diet-calculator)
7. [Running the Application](#-running-the-application)
8. [Model Evaluation & Real Metrics](#-model-evaluation--real-metrics)
9. [Viva Examination Q&A](#-viva-examination-qa)

---

## 🌟 Executive Summary & Workflow

NutriVision AI revolves around the primary real-world clinical nutrition workflow:

$$\text{User Upload / Camera} \longrightarrow \text{OpenCV Segmentation} \longrightarrow \text{Deep Learning Classification} \longrightarrow \text{Portion Estimation} \longrightarrow \text{Nutrition Calculation} \longrightarrow \text{WebMD Daily Diary}$$

```
📸 User Uploads / Snaps Food Image
     │
     ├──> Preprocessing (LANCZOS resize to 224x224, float32 normalization)
     │
     ├──> OpenCV Multi-Food Contour Segmentation (detects multiple items on plate)
     │
     ├──> MobileNetV2 (ImageNet Transfer Learning) & Custom 4-Block CNN
     │      └── Softmax Probabilities + Top-3 Confidences
     │
     ├──> Serving Weight Calibration (foreground pixel area + verified database serving)
     │
     ├──> Nutrition Calculation Engine:
     │      • Calories = (Weight / 100) × Cal/100g
     │      • Protein, Carbs, Fat, Fiber, Sugar, Sodium
     │
     ├──> Instant Interactive Recalculation on Portion Slider Adjust
     │
     └──> Persistent SQLite Daily Diary (grouped by Breakfast, Lunch, Dinner, Snack)
```

---

## 🚀 Key Features & Improvements

- **Modern WebMD-Inspired UI/UX**: Crisp emerald green, mint, white, and slate theme with modern typography, metric cards, donut charts, and progress bars.
- **504+ Verified Food Database**: Sourced from USDA FoodData Central and Indian Food Composition Tables (IFCT) covering Indian main dishes, South Indian specialties, street food, Western meals, fruits, vegetables, beverages, and desserts.
- **Multi-Food Region Detection**: OpenCV-based plate contour analysis that isolates distinct items (e.g. Dosa + Sambar + Chutney or Burger + Fries), estimating individual portions and calculating grand meal totals.
- **Micro & Macro Nutritional Tracking**: Tracks Calories, Protein, Carbohydrates, Fat, Dietary Fiber, Sugars, and Sodium per item and per meal.
- **Daily Budget & RDA Gauges**: WebMD-style budget tracker (`Daily Goal - Food Intake = Remaining Calories`) with progress bars against daily RDA targets.
- **Persistent SQLite Food Diary**: Stores logged meals with timestamps, meal types, itemized breakdowns, and deletion capabilities.

---

## 🧠 Deep Learning Architecture

### 1. MobileNetV2 Transfer Learning (`models/transfer_learning.py`)
- **Backbone**: Pre-trained on 1.28 million ImageNet samples with inverted residual blocks and depthwise separable convolutions (3.4M parameters).
- **Custom Classification Head**:
  $$\text{Input}(224, 224, 3) \rightarrow \text{MobileNetV2} \rightarrow \text{GAP} \rightarrow \text{Dense}(256, \text{ReLU}) \rightarrow \text{BN} \rightarrow \text{Dropout}(0.5) \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dense}(12, \text{Softmax})$$

### 2. Custom 4-Block Convolutional Neural Network (`models/cnn_model.py`)
- **Architecture**:
  - In-model data augmentation (`RandomFlip`, `RandomRotation(±20°)`, `RandomZoom(±20%)`)
  - 4 Sequential Conv blocks: `Conv2D(32, 64, 128, 256)` with `BatchNormalization`, `ReLU`, and `MaxPooling2D(2×2)`
  - Fully Connected layers: `Dense(512, Dropout=0.5)` $\rightarrow$ `Dense(256, Dropout=0.3)` $\rightarrow$ `Dense(12, Softmax)`
  - Total parameters: 26,215,372

### 3. Tabular Nutritional Regression ANN (`models/ann_model.py`)
- Demonstrates tabular regression on structured data independently of image inputs:
  $$\text{Input}(13\text{ features: 12-dim one-hot class} + \text{normalised weight}) \rightarrow \text{Dense}(64) \rightarrow \text{Dense}(128) \rightarrow \text{Dense}(64) \rightarrow \text{Dense}(4, \text{Linear})$$
- Achieved Test MAE = 9.06 kcal.

---

## 📊 Smart Nutrition Database (504+ Foods)

Stored locally in `data/nutrition_database.json` and `data/nutrition.csv` for 100% offline functionality.

| Category | Representative Food Items |
|---|---|
| **Indian Main Course** (60) | Butter Chicken, Chicken Tikka Masala, Rogan Josh, Shahi Paneer, Palak Paneer, Dal Makhani, Dal Tadka, Rajma, Chole |
| **Indian Breads & Rice** (50) | Roti, Naan, Garlic Naan, Aloo Paratha, Paneer Paratha, Hyderabadi Chicken Biryani, Mutton Biryani, Veg Pulao, Khichdi |
| **South Indian** (40) | Masala Dosa, Rava Dosa, Steamed Idli, Medu Vada, Uttapam, Sambar, Coconut Chutney, Appam, Ven Pongal, Poha |
| **Street Food & Snacks** (45) | Samosa, Pav Bhaji, Chole Bhature, Pani Puri, Bhel Puri, Sev Puri, Aloo Tikki, Paneer Pakora, Vada Pav, Momos |
| **Indian Sweets** (40) | Gulab Jamun, Rasgulla, Rasmalai, Jalebi, Kheer, Gajar Halwa, Moong Dal Halwa, Kaju Katli, Besan Ladoo, Shrikhand |
| **Western & Fast Food** (55) | Margherita Pizza, Pepperoni Pizza, Cheeseburger, French Fries, Club Sandwich, Pasta Alfredo, Lasagna, Grilled Chicken |
| **Asian Cuisine** (35) | Chicken Fried Rice, Hakka Noodles, Pad Thai, California Roll, Salmon Nigiri, Ramen, Dim Sum, Tom Yum, Chilli Paneer |
| **Fruits** (45) | Apple, Banana, Alphonso Mango, Orange, Grapes, Watermelon, Papaya, Pineapple, Strawberry, Pomegranate, Avocado |
| **Vegetables** (45) | Broccoli, Carrot, Cucumber, Tomato, Potato, Spinach, Cauliflower, Cabbage, Capsicum, Green Peas, Sweet Corn |
| **Beverages** (40) | Whole Milk, Soy Milk, Almond Milk, Black Coffee, Cappuccino, Masala Chai, Green Tea, Fresh Juices, Lassi, Chaas |
| **Desserts & Bakery** (40) | Ice Cream, Chocolate Cake, Brownie, Cookies, Glazed Donut, Croissant, Dark Chocolate, Almonds, Walnuts, Protein Bar |
| **Grains & Dairy** (35) | Rolled Oats, Quinoa, Boiled Chickpeas, Lentils, Tofu, Paneer, Cheddar Cheese, Greek Yoghurt, Raita, Hummus |

---

## 💻 Running the Application

### Access Currently Running Instance
The application is active and serving at:  
👉 **http://localhost:8501**

### Relaunch Commands
```powershell
cd "c:\Users\HP\OneDrive - The Northcap university\Desktop\cs\food_calorie_estimator"
streamlit run app.py
```

### Run Model Evaluation
```powershell
python training/evaluate.py --model all
```

---

## 🎓 Viva Examination Preparation Guide

### Q1: How does your system estimate food quantity from a single 2D image?
**Answer:** A standard 2D RGB image contains no metric depth scale. Our system uses a dual-stage heuristic:
1. **Contour & Foreground Ratio**: OpenCV HSV/LAB segmentation extracts the foreground food region area relative to the plate frame.
2. **Nutritional Density & Standard Calibration**: Typical serving weights from our 504-food database are scaled proportionally by the area coverage. The UI explicitly labels weight as an approximation and provides an interactive slider for user correction.

### Q2: Why use MobileNetV2 over traditional CNNs like VGG16?
**Answer:** VGG16 contains over 138 million parameters and heavy dense layers, making CPU inference slow. MobileNetV2 uses depthwise separable convolutions (which separate 3×3 spatial filtering from 1×1 pointwise channel combination), reducing computational complexity by 8–9× (only 3.4M parameters) while maintaining high feature transferability from ImageNet.

### Q3: What is the purpose of having both a CNN and an ANN?
**Answer:** They solve two distinct paradigms of deep learning:
- **CNN**: Computer Vision classification operating on spatial tensors ($224 \times 224 \times 3$) to identify food categories.
- **ANN**: Tabular multi-output regression operating on discrete attributes (one-hot category vector + portion weight) to predict continuous nutritional variables (Calories, Protein, Carbs, Fat) with an evaluation MAE of 9.06 kcal.
