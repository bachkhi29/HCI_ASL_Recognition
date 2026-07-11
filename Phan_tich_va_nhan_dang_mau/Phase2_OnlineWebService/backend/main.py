from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from utils import normalize_landmarks

app = FastAPI()

# Mở cổng CORS để Frontend nhận diện không bị chặn cổng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HandData(BaseModel):
    landmarks: list


# TẢI HỆ THỐNG AI PHÂN CẤP VÀO RAM
MODEL_DIR = "models"
print("[*] Đang nạp hệ thống toán học và AI Phân cấp...")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "svm_scaler.pkl"))
    super_model = joblib.load(os.path.join(MODEL_DIR, "super_model_svm.pkl"))

    sub_models = {}
    for i in range(1, 8):
        model_name = f"sub_model_svm_G{i}.pkl"
        sub_models[f"G{i}"] = joblib.load(os.path.join(MODEL_DIR, model_name))
    print("[+] KẾT NỐI ĐỒNG BỘ: Đã nạp thành công trọn bộ 9 file mô hình!")
except Exception as e:
    print(f"[-] LỖI KHÔNG NẠP ĐƯỢC MÔ HÌNH: {e}")


@app.post("/api/predict")
async def predict(data: HandData):
    try:
        # 1. Chuẩn hóa bằng toán học Exact Math chuẩn của ông
        norm_data = normalize_landmarks(data.landmarks)

        # 2. Đưa qua bộ Scaler huấn luyện
        scaled_data = scaler.transform([norm_data])

        # 3. Dự đoán tầng tổng chỉ huy (Group)
        predicted_group = super_model.predict(scaled_data)[0]

        # 4. Dự đoán tầng chuyên gia (Chốt chữ cái)
        final_prediction = sub_models[predicted_group].predict(scaled_data)[0]

        return {
            "status": "success",
            "group": predicted_group,
            "prediction": final_prediction
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)