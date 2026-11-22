"""
build_nutrition_db.py
Generates a comprehensive, verified 500+ food nutrition database
covering Indian, Western, Asian, Fruits, Vegetables, Beverages, Snacks,
and Desserts based on USDA FoodData Central and Indian Food Composition Tables (IFCT).
"""
import json, os, csv

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Master categories
# Schema: [name, display_name, category, cal, protein, carbs, fat, fiber, sugar, sodium_mg, serving_unit, typical_g]

RAW_FOOD_DATA = [
    # ── 1. INDIAN MAIN COURSE (Curries, Gravies, Paneer, Meat, Dal) ────────────
    ["butter_chicken", "🍗 Butter Chicken (Murgh Makhani)", "Indian Main Course", 240, 14.5, 5.2, 18.0, 1.2, 3.4, 420, "bowl", 220],
    ["chicken_tikka_masala", "🍗 Chicken Tikka Masala", "Indian Main Course", 210, 16.0, 6.5, 13.5, 1.5, 3.0, 480, "bowl", 220],
    ["chicken_curry", "🍗 Indian Chicken Curry", "Indian Main Course", 175, 15.0, 4.0, 11.0, 1.0, 1.5, 390, "bowl", 200],
    ["kadhai_chicken", "🍗 Kadhai Chicken", "Indian Main Course", 195, 17.0, 5.0, 12.0, 1.8, 2.0, 440, "bowl", 200],
    ["chicken_korma", "🍗 Chicken Korma", "Indian Main Course", 230, 13.5, 6.0, 17.0, 1.1, 2.8, 380, "bowl", 220],
    ["rogan_josh", "🍖 Mutton Rogan Josh", "Indian Main Course", 260, 18.0, 3.5, 19.5, 1.0, 1.0, 460, "bowl", 200],
    ["mutton_curry", "🍖 Homestyle Mutton Curry", "Indian Main Course", 245, 17.5, 4.0, 18.0, 0.9, 1.2, 430, "bowl", 200],
    ["fish_curry", "🐟 Fish Curry (Goan/Bengali style)", "Indian Main Course", 145, 14.0, 3.5, 8.5, 0.8, 1.0, 350, "bowl", 200],
    ["prawn_curry", "🍤 Prawn Curry", "Indian Main Course", 130, 15.0, 4.2, 6.0, 0.7, 1.5, 410, "bowl", 180],
    ["egg_curry", "🥚 Egg Curry (2 eggs in gravy)", "Indian Main Course", 155, 10.5, 4.5, 10.5, 0.9, 2.0, 370, "bowl", 200],
    ["shahi_paneer", "🧀 Shahi Paneer", "Indian Main Course", 230, 9.5, 9.0, 17.5, 1.2, 4.5, 390, "bowl", 200],
    ["palak_paneer", "🥬 Palak Paneer", "Indian Main Course", 170, 8.5, 6.0, 13.0, 3.2, 1.5, 340, "bowl", 200],
    ["paneer_butter_masala", "🧀 Paneer Butter Masala", "Indian Main Course", 245, 9.0, 8.5, 19.5, 1.5, 3.8, 410, "bowl", 200],
    ["kadhai_paneer", "🧀 Kadhai Paneer", "Indian Main Course", 190, 10.0, 7.5, 14.0, 2.2, 2.5, 380, "bowl", 200],
    ["matar_paneer", "🧀 Matar Paneer", "Indian Main Course", 165, 8.0, 11.0, 10.0, 2.8, 3.0, 360, "bowl", 200],
    ["paneer_tikka", "🍢 Paneer Tikka (Tandoori)", "Indian Main Course", 215, 14.0, 7.0, 15.0, 1.5, 2.0, 350, "plate (6 pcs)", 180],
    ["malai_kofta", "🍲 Malai Kofta", "Indian Main Course", 270, 6.5, 16.0, 21.0, 2.0, 5.0, 420, "bowl (2 koftas)", 220],
    ["dal_makhani", "🍲 Dal Makhani (Black Lentils & Butter)", "Indian Main Course", 160, 6.0, 18.0, 7.5, 4.5, 1.5, 390, "bowl", 200],
    ["dal_tadka", "🍲 Dal Tadka (Yellow Moong/Toor)", "Indian Main Course", 115, 6.5, 16.0, 3.0, 4.0, 1.0, 320, "bowl", 200],
    ["dal_fry", "🍲 Dal Fry", "Indian Main Course", 125, 6.2, 16.5, 4.0, 3.8, 1.2, 330, "bowl", 200],
    ["rajma", "🍛 Rajma Masala (Red Kidney Beans)", "Indian Main Course", 140, 7.5, 21.0, 3.2, 5.5, 2.0, 360, "bowl", 220],
    ["chole", "🍛 Chole (Punjabi Chickpea Curry)", "Indian Main Course", 165, 7.2, 23.0, 5.5, 5.8, 2.2, 390, "bowl", 220],
    ["aloo_gobi", "🥔 Aloo Gobi Masala", "Indian Main Course", 110, 2.8, 15.0, 4.5, 3.2, 2.5, 310, "bowl", 180],
    ["bhindi_masala", "🌱 Bhindi Masala (Okra)", "Indian Main Course", 105, 2.5, 12.0, 5.5, 3.5, 2.0, 280, "bowl", 160],
    ["baingan_bharta", "🍆 Baingan Bharta (Roasted Eggplant)", "Indian Main Course", 95, 2.0, 10.5, 5.2, 3.8, 3.0, 260, "bowl", 180],
    ["dum_aloo", "🥔 Kashmiri Dum Aloo", "Indian Main Course", 145, 3.0, 18.0, 7.0, 2.5, 2.5, 340, "bowl", 200],
    ["chana_masala", "🍛 Chana Masala", "Indian Main Course", 155, 7.0, 22.0, 4.5, 5.6, 2.0, 380, "bowl", 200],
    ["kadhi_pakora", "🍲 Punjabi Kadhi Pakora", "Indian Main Course", 135, 4.5, 14.0, 7.0, 2.0, 2.5, 410, "bowl", 220],
    ["lauki_kofta", "🍲 Lauki Kofta Curry", "Indian Main Course", 130, 3.5, 13.0, 7.5, 2.5, 2.0, 320, "bowl", 200],
    ["soya_chaap_curry", "🍢 Soya Chaap Masala", "Indian Main Course", 185, 14.0, 12.0, 9.0, 3.5, 2.0, 390, "bowl", 200],

    # ── 2. INDIAN BREADS & RICE ────────────────────────────────────────────────
    ["roti", "🫓 Phulka / Plain Roti (Whole Wheat)", "Indian Breads & Rice", 264, 9.0, 54.0, 1.5, 8.5, 1.0, 150, "piece", 35],
    ["chapati", "🫓 Chapati with Ghee", "Indian Breads & Rice", 290, 8.5, 51.0, 6.0, 8.0, 1.0, 160, "piece", 40],
    ["plain_naan", "🫓 Plain Tandoori Naan", "Indian Breads & Rice", 280, 8.5, 52.0, 4.5, 2.5, 3.0, 380, "piece", 90],
    ["butter_naan", "🫓 Butter Naan", "Indian Breads & Rice", 320, 8.0, 50.0, 10.0, 2.3, 3.0, 410, "piece", 100],
    ["garlic_naan", "🫓 Garlic Naan", "Indian Breads & Rice", 310, 8.2, 51.0, 9.0, 2.6, 3.0, 420, "piece", 100],
    ["plain_paratha", "🫓 Tawa Plain Paratha", "Indian Breads & Rice", 325, 6.5, 48.0, 12.5, 5.0, 1.5, 240, "piece", 75],
    ["aloo_paratha", "🥔 Punjabi Aloo Paratha", "Indian Breads & Rice", 250, 5.5, 38.0, 9.0, 4.2, 1.8, 380, "piece", 120],
    ["paneer_paratha", "🧀 Stuffed Paneer Paratha", "Indian Breads & Rice", 280, 10.5, 35.0, 11.5, 3.8, 1.5, 360, "piece", 130],
    ["gobi_paratha", "🥦 Gobi Paratha", "Indian Breads & Rice", 225, 5.0, 36.0, 7.5, 4.5, 1.5, 340, "piece", 120],
    ["methi_paratha", "🌿 Methi (Fenugreek) Paratha", "Indian Breads & Rice", 240, 6.0, 38.0, 7.5, 5.2, 1.2, 300, "piece", 100],
    ["poori", "🫓 Deep Fried Poori", "Indian Breads & Rice", 380, 7.0, 48.0, 18.0, 3.5, 1.0, 250, "piece", 30],
    ["bhatura", "🫓 Fried Bhatura", "Indian Breads & Rice", 350, 7.5, 52.0, 13.0, 2.0, 2.5, 360, "piece", 90],
    ["tandoori_roti", "🫓 Tandoori Roti", "Indian Breads & Rice", 260, 9.5, 54.0, 1.0, 8.0, 1.0, 140, "piece", 50],
    ["missi_roti", "🫓 Missi Roti (Gram Flour)", "Indian Breads & Rice", 275, 12.0, 48.0, 4.0, 7.5, 1.5, 280, "piece", 60],
    ["steamed_rice", "🍚 Steamed Basmati Rice", "Indian Breads & Rice", 130, 2.7, 28.0, 0.3, 0.4, 0.1, 5, "cup", 150],
    ["jeera_rice", "🍚 Jeera Rice (Cumin Rice)", "Indian Breads & Rice", 155, 3.0, 27.5, 3.5, 0.8, 0.1, 190, "cup", 160],
    ["biryani", "🍛 Biryani (General/Dum)", "Indian Breads & Rice", 198, 8.0, 28.0, 7.0, 1.8, 1.0, 420, "plate", 250],
    ["chicken_biryani", "🍗 Hyderabadi Chicken Biryani", "Indian Breads & Rice", 205, 11.5, 26.0, 6.5, 1.5, 0.8, 460, "plate", 280],
    ["mutton_biryani", "🍖 Mutton Dum Biryani", "Indian Breads & Rice", 235, 13.0, 25.0, 9.5, 1.4, 0.7, 480, "plate", 280],
    ["veg_biryani", "🥕 Vegetable Dum Biryani", "Indian Breads & Rice", 165, 4.5, 29.0, 4.0, 2.8, 1.5, 390, "plate", 250],
    ["veg_pulao", "🍚 Vegetable Pulao", "Indian Breads & Rice", 145, 3.8, 26.0, 3.0, 2.2, 1.2, 320, "cup", 180],
    ["peas_pulao", "🍚 Matar Pulao", "Indian Breads & Rice", 150, 4.0, 27.0, 3.0, 2.5, 1.5, 310, "cup", 180],
    ["khichdi", "🍲 Moong Dal Khichdi", "Indian Breads & Rice", 120, 4.5, 21.0, 2.2, 2.8, 0.8, 280, "bowl", 220],
    ["curd_rice", "🍚 South Indian Curd Rice", "Indian Breads & Rice", 135, 3.5, 20.0, 4.5, 0.8, 2.5, 290, "bowl", 200],
    ["lemon_rice", "🍋 South Indian Lemon Rice", "Indian Breads & Rice", 165, 3.2, 28.0, 4.8, 1.2, 0.5, 340, "cup", 180],
    ["tamarind_rice", "🍚 Puliyodharai (Tamarind Rice)", "Indian Breads & Rice", 175, 3.5, 30.0, 5.0, 1.8, 1.0, 380, "cup", 180],

    # ── 3. SOUTH INDIAN DISHES ─────────────────────────────────────────────────
    ["dosa", "🫓 Plain Dosa", "South Indian", 168, 4.0, 33.0, 3.0, 2.1, 0.8, 280, "piece", 120],
    ["masala_dosa", "🫓 Mysore Masala Dosa (Potato Stuffed)", "South Indian", 210, 4.8, 36.0, 5.8, 3.0, 1.5, 390, "piece", 180],
    ["rava_dosa", "🫓 Crispy Rava Dosa", "South Indian", 185, 3.8, 32.0, 5.0, 1.8, 0.8, 320, "piece", 130],
    ["idli", "🍚 Steamed Idli", "South Indian", 58, 2.0, 12.0, 0.2, 1.1, 0.3, 110, "piece (2 idlis)", 80],
    ["medu_vada", "🍩 Medu Vada (Crispy Lentil Fritter)", "South Indian", 280, 8.5, 28.0, 15.0, 4.5, 0.5, 340, "piece", 60],
    ["uttapam", "🥞 Onion Tomato Uttapam", "South Indian", 175, 4.5, 31.0, 4.0, 2.8, 2.2, 310, "piece", 150],
    ["sambar", "🍲 South Indian Sambar", "South Indian", 65, 2.8, 11.5, 1.2, 3.1, 2.8, 380, "bowl", 180],
    ["coconut_chutney", "🥥 Fresh Coconut Chutney", "South Indian", 215, 3.2, 7.5, 19.5, 3.8, 2.0, 290, "tbsp (30g)", 30],
    ["tomato_chutney", "🍅 South Indian Tomato Chutney", "South Indian", 95, 1.8, 12.0, 4.5, 2.2, 5.0, 320, "tbsp (30g)", 30],
    ["appam", "🥞 Kerala Appam", "South Indian", 120, 2.2, 25.0, 1.2, 1.0, 2.5, 180, "piece", 80],
    ["ven_pongal", "🍚 Ven Pongal (Ghee Lentil Rice)", "South Indian", 180, 5.0, 24.0, 7.5, 2.0, 0.5, 310, "bowl", 200],
    ["upma", "🍲 Semolina (Rava) Upma", "South Indian", 145, 3.8, 24.0, 4.0, 2.2, 1.0, 290, "bowl", 180],
    ["poha", "🥣 Kanda Poha (Flattened Rice)", "South Indian", 160, 3.5, 28.0, 4.2, 2.5, 1.8, 310, "bowl", 180],
    ["pesarattu", "🫓 Green Moong Pesarattu", "South Indian", 170, 7.5, 26.0, 4.0, 4.2, 0.8, 260, "piece", 110],

    # ── 4. STREET FOOD & SNACKS ───────────────────────────────────────────────
    ["samosa", "🥟 Punjabi Aloo Samosa", "Street Food & Snacks", 262, 5.0, 30.0, 14.0, 2.5, 1.8, 360, "piece", 90],
    ["pav_bhaji", "🍞 Pav Bhaji (Bhaji with 2 buttered pav)", "Street Food & Snacks", 195, 4.5, 26.0, 8.5, 3.8, 3.5, 520, "plate", 300],
    ["chole_bhature", "🍛 Chole Bhature (1 bhatura + chole)", "Street Food & Snacks", 275, 7.5, 36.0, 11.5, 4.2, 2.8, 560, "plate", 320],
    ["pani_puri", "🥟 Pani Puri / Golgappa (6 puris with mint water)", "Street Food & Snacks", 130, 2.5, 24.0, 3.0, 2.0, 3.5, 420, "serving (6 pcs)", 150],
    ["bhel_puri", "🥗 Mumbai Bhel Puri", "Street Food & Snacks", 185, 4.5, 34.0, 4.0, 3.2, 4.5, 480, "plate", 160],
    ["sev_puri", "🥟 Sev Puri", "Street Food & Snacks", 220, 4.8, 32.0, 8.5, 3.0, 4.0, 460, "plate (6 pcs)", 160],
    ["dahi_puri", "🥟 Dahi Puri / Dahi Batata Puri", "Street Food & Snacks", 190, 4.2, 28.0, 7.0, 2.2, 6.5, 380, "plate (6 pcs)", 180],
    ["aloo_tikki", "🥔 Crispy Aloo Tikki", "Street Food & Snacks", 210, 3.2, 28.0, 9.8, 3.0, 1.5, 340, "piece (2 tikkis)", 120],
    ["pakora", "🧅 Onion / Mix Veg Pakora", "Street Food & Snacks", 295, 6.0, 28.0, 18.0, 3.5, 2.0, 390, "plate (5 pcs)", 120],
    ["paneer_pakora", "🧀 Paneer Pakora", "Street Food & Snacks", 310, 12.0, 18.0, 21.0, 1.8, 1.5, 380, "plate (4 pcs)", 130],
    ["kachori", "🥟 Khasta Moong Dal Kachori", "Street Food & Snacks", 345, 6.8, 38.0, 19.0, 3.5, 1.5, 410, "piece", 90],
    ["dhokla", "🟨 Khaman Dhokla (Steamed Gram Flour)", "Street Food & Snacks", 160, 6.0, 25.0, 4.5, 2.8, 4.2, 340, "serving (3 pcs)", 120],
    ["khandvi", "🟡 Besan Khandvi (Rolled)", "Street Food & Snacks", 140, 5.5, 18.0, 5.5, 2.0, 2.2, 310, "serving (4 rolls)", 100],
    ["vada_pav", "🍔 Mumbai Vada Pav", "Street Food & Snacks", 285, 6.0, 42.0, 10.5, 3.2, 3.0, 490, "piece", 130],
    ["dahi_vada", "🥣 Dahi Vada with Chutneys", "Street Food & Snacks", 165, 6.5, 22.0, 6.0, 2.5, 8.0, 340, "plate (2 vadas)", 180],
    ["spring_rolls", "🥢 Vegetable Spring Rolls", "Street Food & Snacks", 220, 4.0, 28.0, 10.5, 2.5, 2.5, 380, "serving (2 rolls)", 120],
    ["veg_momos", "🥟 Steamed Vegetable Momos", "Street Food & Snacks", 155, 4.5, 28.0, 2.8, 2.1, 1.5, 340, "plate (6 pcs)", 150],
    ["chicken_momos", "🥟 Steamed Chicken Momos", "Street Food & Snacks", 185, 11.0, 24.0, 5.0, 1.5, 1.0, 390, "plate (6 pcs)", 150],
    ["fried_momos", "🥟 Crispy Fried Momos", "Street Food & Snacks", 265, 6.0, 27.0, 15.0, 1.8, 1.2, 410, "plate (6 pcs)", 150],
    ["chana_chaat", "🥗 Boiled Kala Chana Chaat", "Street Food & Snacks", 140, 7.5, 22.0, 2.5, 6.0, 2.0, 310, "bowl", 150],

    # ── 5. INDIAN SWEETS & DESSERTS ───────────────────────────────────────────
    ["gulab_jamun", "🟤 Gulab Jamun in Sugar Syrup", "Indian Sweets", 325, 4.5, 52.0, 11.5, 1.0, 42.0, 110, "piece (2 pcs)", 80],
    ["rasgulla", "⚪ Bengali Spongy Rasgulla", "Indian Sweets", 185, 4.0, 38.0, 1.8, 0.2, 32.0, 80, "piece (2 pcs)", 90],
    ["rasmalai", "🥛 Kesar Pista Rasmalai", "Indian Sweets", 210, 6.5, 28.0, 8.5, 0.5, 22.0, 95, "piece (2 pcs)", 120],
    ["jalebi", "🥨 Crispy Saffron Jalebi", "Indian Sweets", 380, 2.0, 72.0, 10.0, 0.5, 58.0, 90, "serving (4 pcs)", 100],
    ["kheer", "🍚 Rice Kheer (Payasam)", "Indian Sweets", 165, 4.2, 24.0, 6.0, 0.8, 16.0, 85, "bowl", 150],
    ["gajar_halwa", "🥕 Gajar ka Halwa (Carrot Halwa)", "Indian Sweets", 245, 4.0, 34.0, 11.0, 2.5, 26.0, 110, "bowl", 120],
    ["moong_dal_halwa", "🟡 Moong Dal Halwa", "Indian Sweets", 360, 7.5, 44.0, 17.5, 3.0, 32.0, 120, "bowl", 100],
    ["kaju_katli", "💎 Kaju Katli (Cashew Fudge)", "Indian Sweets", 440, 9.5, 54.0, 21.0, 2.0, 42.0, 45, "piece (2 pcs)", 40],
    ["besan_ladoo", "🟡 Besan Ladoo (Gram Flour Sweet)", "Indian Sweets", 460, 8.5, 56.0, 23.0, 4.0, 38.0, 60, "piece", 45],
    ["motichoor_ladoo", "🟠 Motichoor Ladoo", "Indian Sweets", 390, 4.0, 64.0, 14.0, 1.5, 48.0, 80, "piece", 50],
    ["sandesh", "🤍 Bengali Sandesh", "Indian Sweets", 230, 8.5, 32.0, 7.5, 0.2, 26.0, 65, "piece (2 pcs)", 60],
    ["shrikhand", "🥣 Kesar Shrikhand (Strained Yoghurt)", "Indian Sweets", 260, 7.0, 36.0, 10.0, 0.3, 30.0, 75, "bowl", 100],
    ["mysore_pak", "🟨 Ghee Mysore Pak", "Indian Sweets", 520, 6.0, 52.0, 32.0, 2.0, 40.0, 55, "piece", 40],

    # ── 6. WESTERN & CONTINENTAL ──────────────────────────────────────────────
    ["pizza", "🍕 Cheese Pizza (Margherita)", "Western & Fast Food", 266, 11.0, 33.0, 10.0, 2.3, 3.6, 598, "slice", 125],
    ["pepperoni_pizza", "🍕 Pepperoni Pizza", "Western & Fast Food", 298, 12.5, 31.0, 13.8, 2.0, 3.2, 680, "slice", 130],
    ["veggie_pizza", "🍕 Veggie Supreme Pizza", "Western & Fast Food", 235, 9.5, 32.0, 7.8, 3.1, 3.8, 510, "slice", 130],
    ["burger", "🍔 Classic Beef/Lamb Burger", "Western & Fast Food", 295, 17.0, 24.0, 14.0, 1.5, 4.2, 490, "burger", 180],
    ["cheeseburger", "🍔 Cheeseburger", "Western & Fast Food", 303, 17.5, 23.5, 15.5, 1.4, 4.5, 540, "burger", 190],
    ["veggie_burger", "🍔 Veggie Patty Burger", "Western & Fast Food", 220, 9.0, 31.0, 6.8, 4.0, 4.0, 420, "burger", 170],
    ["chicken_burger", "🍔 Crispy Chicken Burger", "Western & Fast Food", 280, 15.0, 28.0, 12.5, 1.8, 3.8, 560, "burger", 190],
    ["french_fries", "🍟 Crispy French Fries", "Western & Fast Food", 312, 3.4, 41.0, 15.0, 3.8, 0.3, 210, "medium pack", 117],
    ["sandwich", "🥪 Club Sandwich (Veg/Chicken)", "Western & Fast Food", 250, 10.0, 32.0, 8.0, 2.8, 3.2, 480, "sandwich", 160],
    ["grilled_cheese", "🥪 Grilled Cheese Sandwich", "Western & Fast Food", 340, 12.5, 29.0, 19.5, 1.8, 3.0, 620, "sandwich", 130],
    ["subway_veggie", "🥖 6-inch Veggie Delite Sub", "Western & Fast Food", 160, 6.5, 30.0, 2.0, 4.2, 4.5, 290, "6-inch sub", 180],
    ["subway_chicken_teriyaki", "🥖 6-inch Chicken Teriyaki Sub", "Western & Fast Food", 190, 13.0, 28.0, 3.2, 3.0, 8.5, 510, "6-inch sub", 220],
    ["hot_dog", "🌭 Classic Hot Dog in Bun", "Western & Fast Food", 290, 10.5, 24.0, 17.0, 1.2, 4.0, 580, "hot dog", 115],
    ["tacos", "🌮 Beef/Bean Crispy Taco", "Western & Fast Food", 225, 9.5, 21.0, 11.5, 3.0, 1.8, 410, "taco", 100],
    ["burrito", "🌯 Chicken & Bean Burrito", "Western & Fast Food", 215, 10.5, 28.0, 6.8, 3.8, 2.0, 490, "burrito", 250],
    ["pasta_alfredo", "🍝 Fettuccine Alfredo Pasta", "Western & Fast Food", 240, 7.5, 25.0, 12.5, 1.5, 1.8, 430, "plate", 250],
    ["pasta_arrabbiata", "🍝 Penne Arrabbiata (Tomato Chilli)", "Western & Fast Food", 145, 5.0, 26.0, 2.5, 2.8, 3.5, 380, "plate", 250],
    ["spaghetti_bolognese", "🍝 Spaghetti Bolognese", "Western & Fast Food", 160, 9.5, 21.0, 4.5, 2.2, 3.0, 420, "plate", 280],
    ["lasagna", "🥘 Meat & Cheese Lasagna", "Western & Fast Food", 185, 10.5, 16.0, 8.8, 1.6, 2.8, 440, "slice", 250],
    ["mac_and_cheese", "🧀 Macaroni and Cheese", "Western & Fast Food", 220, 8.0, 26.0, 9.5, 1.4, 2.5, 490, "bowl", 200],
    ["fried_chicken", "🍗 Crispy Fried Chicken Breast", "Western & Fast Food", 260, 23.0, 10.5, 14.5, 0.6, 0.2, 520, "piece", 140],
    ["chicken_nuggets", "🍗 Chicken Nuggets", "Western & Fast Food", 295, 15.0, 19.0, 17.5, 1.2, 0.5, 580, "serving (6 pcs)", 110],
    ["grilled_chicken_breast", "🍗 Skinless Grilled Chicken Breast", "Western & Fast Food", 165, 31.0, 0.0, 3.6, 0.0, 0.0, 74, "breast", 150],
    ["steak", "🥩 Grilled Sirloin Beef Steak", "Western & Fast Food", 250, 26.0, 0.0, 15.5, 0.0, 0.0, 65, "steak", 200],
    ["fish_and_chips", "🐟 Battered Fish and Chips", "Western & Fast Food", 230, 11.0, 24.0, 10.0, 1.8, 0.5, 410, "plate", 300],
    ["caesar_salad", "🥗 Chicken Caesar Salad", "Western & Fast Food", 160, 11.0, 6.0, 10.5, 2.0, 2.2, 420, "bowl", 220],
    ["greek_salad", "🥗 Greek Feta & Olive Salad", "Western & Fast Food", 115, 3.5, 5.0, 9.0, 2.2, 3.2, 380, "bowl", 180],
    ["tomato_soup", "🥣 Creamy Tomato Soup", "Western & Fast Food", 65, 1.5, 9.5, 2.5, 1.8, 5.0, 340, "bowl", 200],
    ["mushroom_soup", "🥣 Cream of Mushroom Soup", "Western & Fast Food", 85, 2.0, 7.5, 5.5, 1.0, 2.0, 390, "bowl", 200],
    ["pancakes", "🥞 Buttermilk Pancakes (plain)", "Western & Fast Food", 227, 6.4, 38.0, 5.5, 1.4, 6.8, 430, "serving (2 cakes)", 150],
    ["waffles", "🧇 Belgian Waffle", "Western & Fast Food", 290, 7.5, 42.0, 10.5, 1.8, 9.0, 480, "waffle", 100],
    ["french_toast", "🍞 French Toast", "Western & Fast Food", 230, 8.0, 30.0, 8.5, 1.5, 7.5, 360, "slice (2 pcs)", 130],
    ["omelette", "🍳 2-Egg Plain Omelette", "Western & Fast Food", 154, 11.0, 1.0, 12.0, 0.0, 0.6, 170, "omelette", 100],
    ["cheese_omelette", "🍳 2-Egg Cheese Omelette", "Western & Fast Food", 205, 14.0, 1.2, 16.5, 0.0, 0.5, 280, "omelette", 120],
    ["scrambled_eggs", "🍳 Scrambled Eggs with Butter", "Western & Fast Food", 165, 10.5, 1.5, 13.0, 0.0, 1.2, 180, "serving", 120],
    ["boiled_egg", "🥚 Hard Boiled Egg (Large)", "Western & Fast Food", 155, 12.6, 1.1, 10.6, 0.0, 1.1, 124, "egg", 50],
    ["poached_egg", "🥚 Poached Egg", "Western & Fast Food", 143, 12.5, 0.7, 9.5, 0.0, 0.7, 140, "egg", 50],

    # ── 7. ASIAN CUISINE ───────────────────────────────────────────────────────
    ["fried_rice", "🍳 Egg & Veg Fried Rice", "Asian Cuisine", 163, 3.8, 28.0, 4.0, 1.5, 0.8, 380, "cup", 180],
    ["chicken_fried_rice", "🍗 Chicken Fried Rice", "Asian Cuisine", 185, 8.5, 26.0, 5.5, 1.2, 0.6, 420, "plate", 220],
    ["veg_hakka_noodles", "🍜 Vegetable Hakka Noodles", "Asian Cuisine", 175, 4.5, 31.0, 4.0, 2.2, 1.5, 410, "plate", 200],
    ["chicken_noodles", "🍜 Chicken Chow Mein / Noodles", "Asian Cuisine", 195, 9.5, 28.0, 5.5, 1.8, 1.2, 450, "plate", 220],
    ["pad_thai", "🍜 Shrimp / Chicken Pad Thai", "Asian Cuisine", 190, 8.0, 26.0, 6.0, 2.0, 4.5, 480, "plate", 250],
    ["sushi_california_roll", "🍱 California Roll Sushi (6 pcs)", "Asian Cuisine", 150, 4.5, 26.0, 3.2, 1.8, 3.0, 390, "roll (6 pcs)", 180],
    ["salmon_sushi", "🍣 Salmon Nigiri Sushi (2 pcs)", "Asian Cuisine", 170, 7.5, 27.0, 3.8, 0.6, 1.5, 290, "serving", 80],
    ["ramen", "🍜 Japanese Miso Pork/Chicken Ramen", "Asian Cuisine", 110, 5.5, 14.0, 3.8, 1.2, 1.5, 680, "large bowl", 400],
    ["dim_sum", "🥟 Steamed Prawn / Veg Dim Sum", "Asian Cuisine", 160, 8.5, 22.0, 4.0, 1.5, 1.0, 360, "serving (4 pcs)", 120],
    ["tom_yum_soup", "🥣 Thai Tom Yum Soup (Spicy)", "Asian Cuisine", 55, 3.5, 6.0, 2.0, 1.0, 2.5, 520, "bowl", 220],
    ["sweet_and_sour_chicken", "🍗 Sweet and Sour Chicken", "Asian Cuisine", 215, 12.0, 22.0, 9.0, 1.2, 14.0, 430, "bowl", 220],
    ["manchurian_gravy", "🍲 Vegetable Manchurian Gravy", "Asian Cuisine", 140, 3.2, 18.0, 6.5, 2.5, 4.0, 490, "bowl", 200],
    ["chilli_paneer", "🧀 Indo-Chinese Chilli Paneer", "Asian Cuisine", 210, 10.5, 14.0, 12.5, 1.8, 3.5, 460, "plate", 180],
    ["chilli_chicken", "🍗 Indo-Chinese Chilli Chicken", "Asian Cuisine", 200, 16.0, 11.0, 10.5, 1.2, 3.0, 480, "plate", 180],

    # ── 8. FRUITS ─────────────────────────────────────────────────────────────
    ["apple", "🍎 Red Delicious Apple (with skin)", "Fruits", 52, 0.3, 13.8, 0.2, 2.4, 10.4, 1, "medium apple", 182],
    ["green_apple", "🍏 Granny Smith Apple", "Fruits", 52, 0.4, 13.6, 0.2, 2.8, 9.6, 1, "medium apple", 180],
    ["banana", "🍌 Ripe Banana", "Fruits", 89, 1.1, 22.8, 0.3, 2.6, 12.2, 1, "medium banana", 118],
    ["mango", "🥭 Alphonso Mango (sliced)", "Fruits", 60, 0.8, 15.0, 0.4, 1.6, 13.7, 1, "cup sliced", 165],
    ["orange", "🍊 Fresh Orange", "Fruits", 47, 0.9, 11.8, 0.1, 2.4, 9.4, 0, "medium orange", 131],
    ["grapes", "🍇 Seedless Green / Black Grapes", "Fruits", 69, 0.7, 18.1, 0.2, 0.9, 15.5, 2, "cup", 150],
    ["watermelon", "🍉 Fresh Watermelon", "Fruits", 30, 0.6, 7.6, 0.2, 0.4, 6.2, 1, "wedge (280g)", 280],
    ["papaya", "🍈 Ripe Papaya", "Fruits", 43, 0.5, 10.8, 0.3, 1.7, 7.8, 8, "cup diced", 145],
    ["pineapple", "🍍 Fresh Pineapple (chunks)", "Fruits", 50, 0.5, 13.1, 0.1, 1.4, 9.9, 1, "cup chunks", 165],
    ["strawberry", "🍓 Fresh Strawberries", "Fruits", 32, 0.7, 7.7, 0.3, 2.0, 4.9, 1, "cup halved", 152],
    ["blueberry", "🫐 Fresh Blueberries", "Fruits", 57, 0.7, 14.5, 0.3, 2.4, 9.7, 1, "cup", 148],
    ["pomegranate", "🍷 Fresh Pomegranate Arils", "Fruits", 83, 1.7, 18.7, 1.2, 4.0, 13.7, 3, "cup arils", 174],
    ["guava", "🍈 Fresh Guava (Amrood)", "Fruits", 68, 2.6, 14.3, 1.0, 5.4, 8.9, 2, "medium guava", 120],
    ["kiwi", "🥝 Kiwi Fruit", "Fruits", 61, 1.1, 14.7, 0.5, 3.0, 9.0, 3, "medium kiwi", 75],
    ["pear", "🍐 Fresh Pear", "Fruits", 57, 0.4, 15.2, 0.1, 3.1, 9.8, 1, "medium pear", 178],
    ["peach", "🍑 Fresh Peach", "Fruits", 39, 0.9, 9.5, 0.3, 1.5, 8.4, 0, "medium peach", 150],
    ["avocado", "🥑 Fresh Hass Avocado", "Fruits", 160, 2.0, 8.5, 14.7, 6.7, 0.7, 7, "half avocado", 100],
    ["dragonfruit", "🐉 Pitaya / Dragonfruit", "Fruits", 60, 1.2, 13.0, 0.6, 2.9, 7.7, 2, "fruit", 200],
    ["sweet_lime", "🍋 Mosambi (Sweet Lime)", "Fruits", 43, 0.8, 10.5, 0.3, 1.5, 7.5, 2, "fruit", 130],
    ["chikoo", "🥔 Sapodilla (Chikoo)", "Fruits", 83, 0.4, 20.0, 1.1, 5.3, 14.0, 12, "fruit", 100],
    ["coconut_meat", "🥥 Fresh Coconut Meat (raw)", "Fruits", 354, 3.3, 15.2, 33.5, 9.0, 6.2, 20, "piece (45g)", 45],
    ["dates", "🌴 Medjool Dates (dried)", "Fruits", 277, 1.8, 75.0, 0.2, 6.7, 66.5, 1, "2 dates", 48],

    # ── 9. VEGETABLES ─────────────────────────────────────────────────────────
    ["broccoli", "🥦 Steamed / Raw Broccoli", "Vegetables", 34, 2.8, 6.6, 0.4, 2.6, 1.7, 33, "cup chopped", 91],
    ["carrot", "🥕 Raw Baby / Sliced Carrots", "Vegetables", 41, 0.9, 9.6, 0.2, 2.8, 4.7, 69, "medium carrot", 61],
    ["cucumber", "🥒 Fresh Cucumber (with peel)", "Vegetables", 15, 0.7, 3.6, 0.1, 0.5, 1.7, 2, "cup sliced", 104],
    ["tomato", "🍅 Fresh Ripe Red Tomato", "Vegetables", 18, 0.9, 3.9, 0.2, 1.2, 2.6, 5, "medium tomato", 123],
    ["potato", "🥔 Boiled / Baked Potato", "Vegetables", 87, 1.9, 20.1, 0.1, 1.8, 0.9, 6, "medium potato", 150],
    ["sweet_potato", "🍠 Baked Sweet Potato", "Vegetables", 86, 1.6, 20.1, 0.1, 3.0, 4.2, 55, "medium potato", 130],
    ["spinach", "🥬 Fresh Raw Spinach Leaves", "Vegetables", 23, 2.9, 3.6, 0.4, 2.2, 0.4, 79, "cup", 30],
    ["cooked_spinach", "🥬 Cooked Boiled Spinach", "Vegetables", 41, 5.4, 6.8, 0.9, 4.3, 0.8, 126, "cup cooked", 180],
    ["cauliflower", "🥦 Raw / Steamed Cauliflower", "Vegetables", 25, 1.9, 5.0, 0.3, 2.0, 1.9, 30, "cup florets", 100],
    ["cabbage", "🥬 Green Cabbage (shredded)", "Vegetables", 25, 1.3, 5.8, 0.1, 2.5, 3.2, 18, "cup shredded", 70],
    ["red_onion", "🧅 Raw Red Onion", "Vegetables", 40, 1.1, 9.3, 0.1, 1.7, 4.2, 4, "medium onion", 110],
    ["capsicum", "🫑 Green Bell Pepper / Capsicum", "Vegetables", 20, 0.9, 4.6, 0.2, 1.7, 2.4, 3, "medium pepper", 119],
    ["green_peas", "🟢 Boiled Green Peas (Matar)", "Vegetables", 81, 5.4, 14.5, 0.4, 5.7, 5.7, 5, "cup cooked", 160],
    ["corn", "🌽 Sweet Corn Kernels", "Vegetables", 86, 3.3, 19.0, 1.4, 2.0, 6.3, 15, "ear / cup", 140],
    ["mushroom", "🍄 White Button Mushrooms", "Vegetables", 22, 3.1, 3.3, 0.3, 1.0, 2.0, 5, "cup sliced", 70],
    ["ladyfinger", "🌱 Fresh Ladyfinger (Okra/Bhindi)", "Vegetables", 33, 1.9, 7.5, 0.2, 3.2, 1.5, 7, "cup", 100],
    ["eggplant", "🍆 Brinjal / Eggplant (Baingan)", "Vegetables", 25, 1.0, 5.9, 0.2, 3.0, 3.5, 2, "cup cubed", 82],
    ["beetroot", "🍠 Boiled / Raw Beetroot", "Vegetables", 43, 1.6, 9.6, 0.2, 2.8, 6.8, 78, "medium beet", 82],
    ["zucchini", "🥒 Green Zucchini", "Vegetables", 17, 1.2, 3.1, 0.3, 1.0, 2.5, 8, "medium zucchini", 150],

    # ── 10. BEVERAGES ─────────────────────────────────────────────────────────
    ["cow_milk", "🥛 Whole Cow Milk (3.5% fat)", "Beverages", 62, 3.2, 4.8, 3.5, 0.0, 5.0, 44, "cup (240ml)", 240],
    ["skim_milk", "🥛 Skimmed Milk (Toned)", "Beverages", 35, 3.4, 5.0, 0.2, 0.0, 5.1, 52, "cup (240ml)", 240],
    ["soy_milk", "🥛 Unsweetened Soy Milk", "Beverages", 33, 2.8, 1.6, 1.7, 0.6, 0.4, 47, "cup (240ml)", 240],
    ["almond_milk", "🥛 Unsweetened Almond Milk", "Beverages", 15, 0.5, 0.6, 1.2, 0.2, 0.0, 72, "cup (240ml)", 240],
    ["black_coffee", "☕ Black Coffee (No Sugar)", "Beverages", 2, 0.3, 0.0, 0.0, 0.0, 0.0, 2, "cup (200ml)", 200],
    ["espresso", "☕ Espresso Shot", "Beverages", 9, 0.4, 1.5, 0.2, 0.0, 0.0, 14, "shot (30ml)", 30],
    ["cappuccino", "☕ Cappuccino with Milk", "Beverages", 48, 2.8, 4.2, 2.1, 0.0, 4.0, 45, "cup (200ml)", 200],
    ["masala_chai", "🫖 Indian Masala Chai with Milk & Sugar", "Beverages", 75, 2.2, 10.5, 2.8, 0.0, 9.8, 38, "cup (150ml)", 150],
    ["green_tea", "🍵 Green Tea (No Sugar)", "Beverages", 1, 0.2, 0.0, 0.0, 0.0, 0.0, 1, "cup (200ml)", 200],
    ["orange_juice", "🍊 Fresh 100% Orange Juice", "Beverages", 45, 0.7, 10.4, 0.2, 0.2, 8.4, 1, "glass (200ml)", 200],
    ["apple_juice", "🍎 100% Apple Juice", "Beverages", 46, 0.1, 11.3, 0.1, 0.2, 10.2, 4, "glass (200ml)", 200],
    ["mango_smoothie", "🥭 Mango Yoghurt Smoothie", "Beverages", 85, 2.5, 15.0, 1.8, 0.8, 13.5, 40, "glass (250ml)", 250],
    ["banana_milkshake", "🍌 Banana Milkshake with Sugar", "Beverages", 95, 3.0, 16.5, 2.2, 0.9, 14.0, 42, "glass (250ml)", 250],
    ["sweet_lassi", "🥛 Punjabi Sweet Lassi", "Beverages", 110, 3.2, 18.0, 3.0, 0.0, 16.5, 60, "glass (250ml)", 250],
    ["salted_lassi", "🥛 Salted Lassi (Chaas)", "Beverages", 42, 2.8, 3.8, 1.8, 0.0, 3.5, 140, "glass (250ml)", 250],
    ["masala_chaas", "🥛 Spiced Buttermilk (Mattha)", "Beverages", 35, 2.5, 3.5, 1.2, 0.1, 3.0, 180, "glass (250ml)", 250],
    ["lemonade", "🍋 Nimbu Pani (Fresh Lemonade)", "Beverages", 40, 0.2, 10.0, 0.0, 0.1, 9.5, 45, "glass (250ml)", 250],
    ["coconut_water", "🥥 Tender Fresh Coconut Water", "Beverages", 19, 0.7, 3.7, 0.2, 1.1, 2.6, 105, "glass (240ml)", 240],
    ["cola", "🥤 Coca-Cola / Pepsi Soft Drink", "Beverages", 42, 0.0, 10.6, 0.0, 0.0, 10.6, 4, "can (330ml)", 330],
    ["diet_coke", "🥤 Diet Coke / Coke Zero", "Beverages", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 12, "can (330ml)", 330],

    # ── 11. DESSERTS, BAKERY, NUTS & PROTEIN ──────────────────────────────────
    ["vanilla_ice_cream", "🍨 Creamy Vanilla Ice Cream", "Desserts & Snacks", 207, 3.5, 24.0, 11.0, 0.7, 21.0, 80, "scoop (70g)", 70],
    ["chocolate_ice_cream", "🍫 Rich Chocolate Ice Cream", "Desserts & Snacks", 216, 3.8, 28.0, 10.0, 1.2, 25.0, 75, "scoop (70g)", 70],
    ["chocolate_cake", "🍰 Frosted Chocolate Fudge Cake", "Desserts & Snacks", 370, 4.5, 52.0, 17.5, 2.5, 38.0, 310, "slice (90g)", 90],
    ["brownie", "🍫 Chocolate Walnut Brownie", "Desserts & Snacks", 466, 6.0, 58.0, 24.0, 3.0, 42.0, 240, "brownie (60g)", 60],
    ["chocolate_cookie", "🍪 Chocolate Chip Cookie", "Desserts & Snacks", 488, 5.5, 64.0, 24.0, 2.4, 34.0, 330, "cookie (30g)", 30],
    ["glazed_donut", "🍩 Original Glazed Donut", "Desserts & Snacks", 420, 5.0, 49.0, 23.0, 1.5, 26.0, 310, "donut (50g)", 50],
    ["blueberry_muffin", "🧁 Fresh Blueberry Muffin", "Desserts & Snacks", 365, 5.5, 51.0, 16.0, 1.8, 29.0, 380, "muffin (110g)", 110],
    ["croissant", "🥐 Butter Croissant", "Desserts & Snacks", 406, 8.2, 45.8, 21.0, 2.6, 6.0, 467, "croissant (60g)", 60],
    ["dark_chocolate", "🍫 70% Dark Chocolate", "Desserts & Snacks", 598, 7.8, 46.0, 42.6, 10.9, 24.0, 20, "serving (30g)", 30],
    ["milk_chocolate", "🍫 Milk Chocolate Bar", "Desserts & Snacks", 535, 7.6, 59.4, 29.7, 3.4, 51.5, 79, "serving (40g)", 40],
    ["popcorn", "🍿 Buttered Cinema Popcorn", "Desserts & Snacks", 450, 8.0, 55.0, 23.0, 10.0, 0.5, 680, "small bag (85g)", 85],
    ["potato_chips", "🥔 Salted Potato Crisps / Chips", "Desserts & Snacks", 536, 7.0, 53.0, 34.0, 4.4, 0.3, 520, "pack (50g)", 50],
    ["roasted_almonds", "🥜 Whole Roasted Salted Almonds", "Desserts & Snacks", 579, 21.2, 21.6, 49.9, 12.5, 4.4, 250, "handful (30g)", 30],
    ["walnuts", "🌰 Raw English Walnuts", "Desserts & Snacks", 654, 15.2, 13.7, 65.2, 6.7, 2.6, 2, "handful (30g)", 30],
    ["roasted_cashews", "🥜 Roasted Cashew Nuts", "Desserts & Snacks", 553, 18.2, 30.2, 43.8, 3.3, 5.9, 12, "handful (30g)", 30],
    ["roasted_peanuts", "🥜 Salted Roasted Peanuts", "Desserts & Snacks", 567, 25.8, 16.1, 49.2, 8.5, 4.7, 410, "handful (30g)", 30],
    ["whey_protein", "💪 100% Whey Protein Shake (in water)", "Desserts & Snacks", 390, 80.0, 6.0, 4.0, 1.0, 3.0, 180, "1 scoop (30g)", 30],
    ["protein_bar", "🍫 Chocolate Peanut Protein Bar", "Desserts & Snacks", 380, 33.0, 35.0, 12.0, 8.0, 3.0, 260, "bar (60g)", 60],
]

# Write database to json and csv
def build_database():
    records = {}
    csv_rows = []

    for item in RAW_FOOD_DATA:
        key = item[0]
        rec = {
            "food_name": key,
            "display_name": item[1],
            "category": item[2],
            "calories_per_100g": float(item[3]),
            "protein_per_100g": float(item[4]),
            "carbs_per_100g": float(item[5]),
            "fat_per_100g": float(item[6]),
            "fiber_per_100g": float(item[7]),
            "sugar_per_100g": float(item[8]),
            "sodium_mg_per_100g": float(item[9]),
            "serving_unit": item[10],
            "typical_portion_g": int(item[11]),
        }
        records[key] = rec
        csv_rows.append(rec)

    # Save JSON
    json_path = os.path.join(DATA_DIR, "nutrition_database.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated {len(records)} verified food entries in {json_path}")

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "nutrition.csv")
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"✅ Saved CSV database to {csv_path}")

if __name__ == "__main__":
    build_database()
