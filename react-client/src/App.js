import React, { useState } from "react";

function App() {
  const [video, setVideo] = useState(null);
  const [referenceImages, setReferenceImages] = useState([]);
  const [target, setTarget] = useState("cars");
  const [logOutput, setLogOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [outputImages, setOutputImages] = useState([]);

  const handleSubmit = async () => {
    if (!video || referenceImages.length === 0) {
      alert("Please upload a video and at least one reference image.");
      return;
    }

    setLoading(true);
    setLogOutput("");
    setOutputImages([]); // Clear previous images

    const formData = new FormData();
    formData.append("video", video);
    referenceImages.forEach((img) => {
      formData.append("reference_images", img);
    });
    formData.append("target", target);

    const res = await fetch("http://localhost:5000/process", {
      method: "POST",
      body: formData,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      setLogOutput((prev) => prev + decoder.decode(value));
    }

    // Fetch output images after processing
    const imageRes = await fetch("http://localhost:5000/results");
    const imageList = await imageRes.json();
    setOutputImages(imageList);

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Smart Object Matching Tool</h2>

      <div style={styles.formGroup}>
        <label>Upload Video:</label>
        <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files[0])} />
      </div>

      <div style={styles.formGroup}>
        <label>Upload Reference Images:</label>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => setReferenceImages(Array.from(e.target.files))}
        />
      </div>

      <div style={styles.formGroup}>
        <label>Search For:</label>
        <select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="cars">Cars</option>
          <option value="bikes">Bikes</option>
          <option value="humans">Humans</option>
        </select>
      </div>

      <button onClick={handleSubmit} disabled={loading} style={styles.button}>
        {loading ? "Processing..." : "Start"}
      </button>

      <pre style={styles.logBox}>{logOutput}</pre>

      {outputImages.length > 0 && (
        <div style={styles.imageGrid}>
          {outputImages.map((img, idx) => (
            <div key={idx} style={styles.imageCard}>
              <img
                src={`http://localhost:5000/output/${img}`}
                alt={`match ${idx}`}
                style={styles.image}
              />
              <p style={styles.caption}>{img}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "2rem",
    fontFamily: "sans-serif",
    maxWidth: "900px",
    margin: "auto",
  },
  heading: {
    textAlign: "center",
    marginBottom: "1rem",
  },
  formGroup: {
    marginBottom: "1.5rem",
    display: "flex",
    flexDirection: "column",
  },
  button: {
    padding: "0.75rem 1.5rem",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    cursor: "pointer",
    borderRadius: "4px",
    fontSize: "1rem",
  },
  logBox: {
    marginTop: "2rem",
    padding: "1rem",
    backgroundColor: "#f1f1f1",
    height: "300px",
    overflowY: "auto",
    whiteSpace: "pre-wrap",
    fontSize: "0.9rem",
    border: "1px solid #ccc",
  },
  imageGrid: {
    marginTop: "2rem",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
    gap: "1rem",
  },
  imageCard: {
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    borderRadius: "8px",
    overflow: "hidden",
    backgroundColor: "#fff",
    textAlign: "center",
    padding: "0.5rem",
  },
  image: {
    width: "100%",
    height: "auto",
    borderRadius: "4px",
  },
  caption: {
    fontSize: "0.85rem",
    marginTop: "0.5rem",
    color: "#333",
  },
};

export default App;
