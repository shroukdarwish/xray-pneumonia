import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# عنوان الصفحة
st.title("🩻 X-ray Pneumonia Detector")
st.write("ارفع صورة أشعة صدر وسأخبرك إذا كانت تدل على التهاب رئوي أم لا 😷")

# تحميل الموديل
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("xray_model.h5")
    return model

model = load_model()

# دالة لتحضير الصورة قبل التنبؤ
def preprocess_image(image):
    image = image.resize((150, 150))  # نفس حجم الصور أثناء التدريب
    image = np.array(image) / 255.0   # تطبيع القيم
    image = np.expand_dims(image, axis=0)  # إضافة بعد batch
    return image

# واجهة رفع الصورة
uploaded_file = st.file_uploader("📸 ارفع صورة الأشعة هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="صورة الأشعة", use_column_width=True)
    
    st.write("🔍 جاري التحليل...")
    img = preprocess_image(image)
    
    prediction = model.predict(img)
    result = "🌡️ التهاب رئوي" if prediction[0][0] > 0.5 else "✅ طبيعي"
    
    st.subheader("النتيجة:")
    st.success(result)
