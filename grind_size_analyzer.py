import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("☕ 커피 원두 입자 크기 분석기")

uploaded_file = st.file_uploader("원두 분쇄 사진을 업로드하세요 (흰 배경)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드한 원두 사진', use_column_width=True)

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 10000:  # 노이즈 제거
            sizes.append(np.sqrt(area))

    if sizes:
        st.subheader("🎯 입자 크기 분석 결과")
        st.write(f"총 입자 수: {len(sizes)}")
        st.write(f"평균 입자 크기: {np.mean(sizes):.2f} px")
        st.write(f"표준편차: {np.std(sizes):.2f} px")

        fig, ax = plt.subplots()
        ax.hist(sizes, bins=20, color='brown')
        ax.set_title("입자 크기 분포 히스토그램")
        ax.set_xlabel("크기(px)")
        ax.set_ylabel("빈도수")
        st.pyplot(fig)
    else:
        st.warning("입자가 제대로 인식되지 않았습니다. 흰 배경과 고해상도 사진을 사용해주세요.")
