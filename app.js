const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cors = require("cors");

const { initializeApp } = require("firebase/app");
const {
  getFirestore,
  doc,
  setDoc,
  collection,
  getDocs,
} = require("firebase/firestore");

const firebaseConfig = {
  apiKey: "AIzaSyD0XBYDXPGYBuiBROviMbmlkhkcCmaiZwQ",
  authDomain: "submissionmlgc-muhdastinf.firebaseapp.com",
  projectId: "submissionmlgc-muhdastinf",
  storageBucket: "submissionmlgc-muhdastinf.appspot.com",
  messagingSenderId: "52240791632",
  appId: "1:52240791632:web:3a4051e5d3d5c1560e7955",
  measurementId: "G-CXKTSD7GBY",
};

const firebaseApp = initializeApp(firebaseConfig);
const firestore = getFirestore(firebaseApp);

const app = express();
app.use(cors());

const upload = multer({
  limits: {
    fileSize: 1000000,
  },
}).single("image");

let model;
async function loadModel() {
  model = await tf.loadGraphModel(
    "https://storage.googleapis.com/ml-dastin-dicoding/ml-dicoding-dastin/submissions-model/model.json"
  );
  console.log("Model loaded successfully");
}
loadModel();

app.post("/predict", (req, res, next) => {
  upload(req, res, async function (err) {
    if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }

    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "No image uploaded",
      });
    }

    try {
      const imageBuffer = req.file.buffer;
      const tensor = tf.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat();

      const prediction = model.predict(tensor);
      const result = prediction.dataSync()[0] > 0.5 ? "Cancer" : "Non-cancer";
      const suggestion =
        result === "Cancer"
          ? "Segera periksa ke dokter!"
          : "Tidak ditemukan penyakit. Tetap jaga kesehatan!";

      const id = Math.random().toString(36).substr(2, 9);
      const createdAt = new Date().toISOString();

      await setDoc(doc(firestore, "predictions", id), {
        id: id,
        result: result,
        suggestion: suggestion,
        createdAt: createdAt,
      });

      res.status(201).json({
        status: "success",
        message: "Model is predicted successfully",
        data: {
          id: id,
          result: result,
          suggestion: suggestion,
          createdAt: createdAt,
        },
      });
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }
  });
});

app.get("/predict/histories", async (req, res) => {
  try {
    const querySnapshot = await getDocs(collection(firestore, "predictions"));
    const histories = [];

    querySnapshot.forEach((doc) => {
      const data = doc.data();
      histories.push({
        id: data.id,
        history: {
          result: data.result,
          createdAt: data.createdAt,
          suggestion: data.suggestion,
          id: data.id,
        },
      });
    });

    res.json({
      status: "success",
      data: histories,
    });
  } catch (error) {
    console.error("Error retrieving prediction history:", error);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan dalam mengambil riwayat prediksi",
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
