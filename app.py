from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, os

# แก้ไข **file** เป็น __file__ (ถูกต้องแล้วจากรอบก่อน)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

# แก้ไข 1: เปลี่ยน **name** เป็น __name__
app = Flask(__name__, template_folder="templates")

@app.get("/health")
def health():
    # แก้ไข 2: เพิ่ม Indentation (ย่อหน้า)
    return {"ok": True, "features": FEATURES}

@app.post("/predict")
def predict():
    # แก้ไข 3: เพิ่ม Indentation (ย่อหน้า) และจัดระเบียบ logic
    data = request.get_json(silent=True) or {}
    x = data.get("features")
    
    if not x or len(x) != len(FEATURES):
        return jsonify(error=f"Expected 8 features: {FEATURES}"), 400
    
    X = np.array([x], dtype=float)
    yhat = model.predict(X).tolist()[0]
    return jsonify(prediction=yhat)

@app.route("/", methods=["GET", "POST"])
def main():
    # แก้ไข 4: เพิ่ม Indentation (ย่อหน้า)
    if request.method == "GET":
        return render_template("main.html")
    
    # ส่วนของ POST
    try:
        vals = [float(request.form.get(f)) for f in FEATURES]
        pred = model.predict(np.array([vals])).tolist()[0]
        return render_template("main.html", prediction=pred)
    except Exception as e:
        return render_template("main.html", error=str(e))

# แก้ไข 5: เปลี่ยน ** เป็น __ และย่อหน้าบรรทัด app.run
if __name__ == "__main__":
    app.run(debug=True)