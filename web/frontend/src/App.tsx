/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Container,
  Divider,
  FormControl,
  FormControlLabel,
  Paper,
  Radio,
  RadioGroup,
  Stack,
  Typography,
} from "@mui/material";
import ModelComparison from "./components/ModelComparison";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 10 * 1024 * 1024;
type ModelId = "cnn" | "cnn_leakage_aware" | "svm";
type ModelInfo = {
  name: string;
  available: boolean;
  status: string;
  classes: number;
  image_size: number;
};
type ModelRegistry = { models: Record<ModelId, ModelInfo> };
type SpeciesPrediction = { species: string; confidence: number };
type CNNPrediction = {
  model_name?: "cnn" | "cnn_leakage_aware";
  species: string;
  confidence: number;
  top_predictions: SpeciesPrediction[];
  timing?: Record<string, number>;
};
type SVMPrediction = { model: "svm"; species: string; decision_score?: number };
type Prediction = CNNPrediction | SVMPrediction;

const isCNNPrediction = (prediction: Prediction): prediction is CNNPrediction =>
  "confidence" in prediction;

function App() {
  const [models, setModels] = useState<ModelRegistry["models"] | null>(null);
  const [modelsError, setModelsError] = useState("");
  const [model, setModel] = useState<ModelId>("cnn");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchModels() {
      try {
        const response = await fetch(`${API_URL}/models`, {
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error("Unable to retrieve model availability.");
        const data: ModelRegistry = await response.json();
        setModels(data.models);
        setModelsError("");
      } catch (err) {
        if (controller.signal.aborted) return;
        setModelsError(
          err instanceof Error ? err.message : "Cannot connect to the API.",
        );
      }
    }
    void fetchModels();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    event.target.value = "";
    setPrediction(null);
    setShowDetails(false);
    setError("");
    if (!["image/jpeg", "image/png"].includes(selected.type)) {
      setFile(null);
      setError("Please select a JPEG or PNG image.");
      return;
    }
    if (selected.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("The image must be 10 MB or smaller.");
      return;
    }
    setFile(selected);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    if (!models?.[model]?.available) {
      setError("This model is not available for predictions yet.");
      return;
    }
    setLoading(true);
    setError("");
    setPrediction(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const endpoint =
        model === "svm"
          ? "/predict/svm"
          : `/predict?model_name=${encodeURIComponent(model)}`;
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const detail =
          data && typeof data === "object" && "detail" in data
            ? String(data.detail)
            : "Unable to classify this image.";
        throw new Error(detail);
      }
      if (!data || typeof data !== "object" || !("species" in data)) {
        throw new Error("The API returned an unexpected prediction response.");
      }
      setPrediction(data as Prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const modelAvailable = models?.[model]?.available ?? false;

  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "#f7f5f0",
        color: "#222",
        py: { xs: 4, md: 7 },
      }}
    >
      <Container maxWidth="lg">
        <Stack spacing={4}>
          <Stack
            direction="row"
            sx={{justifyContent:"space-between",
            alignItems:"center"}}
          >
            <Typography sx={{fontWeight:700}}>Wildlife Classifier</Typography>
            <Typography variant="body2" color="text.secondary">
              WildlifeReID-10K
            </Typography>
          </Stack>
          <Box>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                fontSize: { xs: "2.1rem", md: "3.1rem" },
                letterSpacing: "-0.04em",
              }}
            >
              Identifying an animal species
            </Typography>
            <Typography color="text.secondary" sx={{ mt: 2, maxWidth: 650 }}>
              Upload a wildlife photo and choose a classification model to
              identify which of the 22 species it most closely matches.
            </Typography>
          </Box>

          <Paper
            variant="outlined"
            sx={{ p: { xs: 2, md: 3 }, borderRadius: 2, boxShadow: "none" }}
          >
            <Stack spacing={3}>
              <Box>
                <Typography  sx={{ mb: 1, fontWeight:600 }}>
                  Classification model
                </Typography>
                {modelsError && (
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    {modelsError}
                  </Alert>
                )}
                <FormControl>
                  <RadioGroup
                    value={model}
                    onChange={(event) => {
                      setModel(event.target.value as ModelId);
                      setPrediction(null);
                      setError("");
                      setShowDetails(false);
                    }}
                  >
                    {(["cnn", "cnn_leakage_aware", "svm"] as const).map((id) => (
                      <FormControlLabel
                        key={id}
                        value={id}
                        disabled={!models?.[id]?.available}
                        control={<Radio />}
                        label={
                          <Stack
                            direction="row"
                            sx={{alignItems:"center"}}
                            spacing={1}
                          >
                            <Typography>
                              {models?.[id]?.name ??
                                (id === "cnn"
                                  ? "Original CNN"
                                  : id === "cnn_leakage_aware"
                                    ? "Leakage-Aware CNN"
                                    : "HOG + SVM (Traditional ML)")}
                            </Typography>
                            <Chip
                              size="small"
                              variant="outlined"
                              label={
                                models?.[id]?.available
                                  ? "Available"
                                  : "Pending"
                              }
                            />
                          </Stack>
                        }
                      />
                    ))}
                  </RadioGroup>
                </FormControl>
                {!models && !modelsError && (
                  <Typography variant="caption" color="text.secondary">
                    Checking model availability…
                  </Typography>
                )}
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mt: 1 }}
                >
                  The SVM will be enabled after a compatible model is verified.
                </Typography>
              </Box>
              <Divider />
              <Button
                component="label"
                variant="text"
                sx={{
                  minHeight: 160,
                  border: "1px dashed #aaa",
                  borderRadius: 1.5,
                  color: "#333",
                  textTransform: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.5,
                }}
              >
                <Typography sx={{fontWeight:600}}>
                  Choose a wildlife photo
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  JPEG or PNG · Maximum 10 MB
                </Typography>
                <input
                  hidden
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleFileChange}
                />
              </Button>
              {preview && (
                <Box>
                  <Box
                    component="img"
                    src={preview}
                    alt="Selected wildlife"
                    sx={{
                      width: "100%",
                      maxHeight: 380,
                      objectFit: "contain",
                      borderRadius: 1,
                      bgcolor: "#efede7",
                    }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {file?.name}
                  </Typography>
                </Box>
              )}
              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={!file || loading || !modelAvailable}
                sx={{
                  alignSelf: "flex-start",
                  px: 3,
                  py: 1.2,
                  borderRadius: 1.5,
                  textTransform: "none",
                  bgcolor: "#222",
                  boxShadow: "none",
                  "&:hover": { bgcolor: "#000", boxShadow: "none" },
                }}
              >
                {loading ? (
                  <>
                    <CircularProgress
                      size={17}
                      color="inherit"
                      sx={{ mr: 1 }}
                    />
                    Identifying species…
                  </>
                ) : (
                  "Identify species"
                )}
              </Button>
              {error && <Alert severity="error">{error}</Alert>}
            </Stack>
          </Paper>

          {prediction && (
            <Box>
              <Divider sx={{ mb: 3 }} />
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                  gap: 4,
                }}
              >
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    YOUR IMAGE
                  </Typography>
                  {preview && (
                    <Box
                      component="img"
                      src={preview}
                      alt="Uploaded wildlife"
                      sx={{
                        width: "100%",
                        mt: 1.5,
                        aspectRatio: "4 / 3",
                        objectFit: "cover",
                        borderRadius: 1,
                      }}
                    />
                  )}
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    RESULT · {models?.[model]?.name ?? model.toUpperCase()}
                  </Typography>
                  <Typography
                    variant="h4"
                    sx={{
                      mt: 1.5,
                      fontWeight: 700,
                      textTransform: "capitalize",
                    }}
                  >
                    {prediction.species}
                  </Typography>
                  {isCNNPrediction(prediction) ? (
                    <>
                      <Typography color="text.secondary" sx={{ mt: 0.5 }}>
                        {(prediction.confidence * 100).toFixed(1)}% CNN
                        confidence
                      </Typography>
                      <Typography  sx={{ mt: 3, mb: 2, fontWeight:600 }}>
                        Top 3 predictions
                      </Typography>
                      <Stack spacing={2}>
                        {(prediction.top_predictions ?? []).map(
                          (item, index) => (
                            <Box key={`${item.species}-${index}`}>
                              <Stack
                                direction="row"
                                
                                sx={{ mb: 0.5, justifyContent:"space-between" }}
                              >
                                <Typography variant="body2">
                                  {item.species}
                                </Typography>
                                <Typography variant="body2">
                                  {(item.confidence * 100).toFixed(1)}%
                                </Typography>
                              </Stack>
                              <Box
                                sx={{
                                  height: 6,
                                  bgcolor: "#e7e5df",
                                  borderRadius: 1,
                                  overflow: "hidden",
                                }}
                              >
                                <Box
                                  sx={{
                                    width: `${Math.min(
                                      100,
                                      Math.max(0, item.confidence * 100),
                                    )}%`,
                                    height: "100%",
                                    bgcolor:
                                      index === 0 ? "#454b40" : "#aaa99f",
                                  }}
                                />
                              </Box>
                            </Box>
                          ),
                        )}
                      </Stack>
                    </>
                  ) : (
                    <Typography color="text.secondary" sx={{ mt: 1 }}>
                      SVM prediction (probability confidence unavailable).
                    </Typography>
                  )}
                  <Alert severity="info" sx={{ mt: 3 }}>
                    Predictions may be incorrect. Confidence does not guarantee
                    accuracy.
                  </Alert>
                </Box>
              </Box>
              <Box sx={{ mt: 4 }}>
                <Button
                  variant="text"
                  onClick={() => setShowDetails(!showDetails)}
                  sx={{
                    px: 0,
                    color: "#222",
                    textTransform: "none",
                    fontWeight: 600,
                  }}
                >
                  {showDetails
                    ? "Hide research details"
                    : "View research details"}
                </Button>
                <Collapse in={showDetails}>
                  <Stack
                    spacing={1}
                    sx={{ mt: 2, pt: 2, borderTop: "1px solid #ddd" }}
                  >
                    <Stack direction="row" sx={{justifyContent:"space-between"}}>
                      <Typography>Model</Typography>
                      <Typography color="text.secondary">
                        {models?.[model]?.name ?? model}
                      </Typography>
                    </Stack>
                    <Stack direction="row" sx={{justifyContent:"space-between"}}>
                      <Typography>Dataset</Typography>
                      <Typography color="text.secondary">
                        WildlifeReID-10K
                      </Typography>
                    </Stack>
                    <Stack direction="row" sx={{justifyContent:"space-between"}}>
                      <Typography>Species classes</Typography>
                      <Typography color="text.secondary">22</Typography>
                    </Stack>
                    <Stack direction="row" sx={{justifyContent:"space-between"}}>
                      <Typography>API</Typography>
                      <Typography color="text.secondary">REST</Typography>
                    </Stack>
                  </Stack>
                </Collapse>
              </Box>
            </Box>
          )}
          <ModelComparison />
          <Typography variant="caption" color="text.secondary">
            Big data research project · Wildlife species classification
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
