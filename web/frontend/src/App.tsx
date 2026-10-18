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
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import ModelComparison from "./components/ModelComparison";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 10 * 1024 * 1024;

type ModelId = "cnn" | "cnn_leakage_aware" | "svm_clean" | "svm_original";

type ModelInfo = {
  name: string;
  available: boolean;
  status: string;
  classes: number;
  image_size: number;
};

type ModelRegistry = {
  models: Record<ModelId, ModelInfo>;
};

type SpeciesPrediction = {
  species: string;
  confidence: number;
};

type CNNPrediction = {
  model_name?: "cnn" | "cnn_leakage_aware";
  species: string;
  confidence: number;
  top_predictions: SpeciesPrediction[];
  timing?: Record<string, number>;
};

type SVMPrediction = {
  model: "svm";
  model_name: "svm_clean" | "svm_original";
  species: string;
  decision_score: number | null;
  timing: {
    total_ms: number;
  };
};

type Prediction = CNNPrediction | SVMPrediction;

const isCNNPrediction = (prediction: Prediction): prediction is CNNPrediction =>
  "confidence" in prediction;

const palette = {
  forest: "#14251B",
  forestLight: "#243B2C",
  moss: "#526B4F",
  sage: "#C7DB9B",
  ivory: "#F5F2E9",
  white: "#FFFFFF",
  muted: "#68776A",
  border: "#E2E6DC",
};

const modelOptions: {
  id: ModelId;
  title: string;
  subtitle: string;
  number: string;
}[] = [
  {
    id: "cnn",
    title: "Original CNN",
    subtitle: "Deep learning · Original model",
    number: "01",
  },
  {
    id: "cnn_leakage_aware",
    title: "Leakage-Aware CNN",
    subtitle: "Deep learning · Alternative model",
    number: "02",
  },
  {
    id: "svm_clean",
    title: "Clean-Split HOG + SVM",
    subtitle: "Traditional ML · Clean split",
    number: "03",
  },
  {
    id: "svm_original",
    title: "Other-Split HOG + SVM",
    subtitle: "Traditional ML · Alternative split",
    number: "04",
  },
];

const formatSpecies = (name: string) => name.replace(/[_-]/g, " ");

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

        if (!response.ok) {
          throw new Error("Unable to retrieve model availability.");
        }

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

  const resetPrediction = () => {
    setPrediction(null);
    setError("");
    setShowDetails(false);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (!selected) return;

    event.target.value = "";
    resetPrediction();

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
      const endpoint = model.startsWith("svm_")
        ? `/predict/svm?model_name=${encodeURIComponent(model)}`
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
        bgcolor: palette.ivory,
        color: palette.forest,
      }}
    >
      {/* Navigation */}
      <Box
        sx={{
          bgcolor: palette.forest,
          borderBottom: "1px solid #354839",
        }}
      >
        <Container maxWidth="lg">
          <Stack
            direction="row"
            sx={{
              py: 2.5,
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Stack direction="row" sx={{ alignItems: "center" }} spacing={1.5}>
              <Box
                sx={{
                  width: 38,
                  height: 38,
                  bgcolor: palette.sage,
                  color: palette.forest,
                  borderRadius: 2,
                  display: "grid",
                  placeItems: "center",
                  fontWeight: 900,
                  fontSize: 20,
                }}
              >
                W
              </Box>

              <Box>
                <Typography
                  sx={{
                    color: palette.white,
                    fontWeight: 800,
                    letterSpacing: "0.08em",
                    lineHeight: 1.2,
                  }}
                >
                  WILDTRACE
                  <Box component="span" sx={{ color: palette.sage }}>
                    {" "}
                    AI
                  </Box>
                </Typography>

                <Typography
                  sx={{
                    color: "#B8C6B7",
                    fontSize: 10,
                    letterSpacing: "0.18em",
                  }}
                >
                  WILDLIFE INTELLIGENCE LAB
                </Typography>
              </Box>
            </Stack>

            <Chip
              label="RESEARCH PROTOTYPE"
              size="small"
              sx={{
                bgcolor: "#2B4131",
                color: palette.sage,
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.08em",
                display: { xs: "none", sm: "flex" },
              }}
            />
          </Stack>
        </Container>
      </Box>

      {/* Hero */}
      <Box
        sx={{
          bgcolor: palette.forest,
          color: palette.white,
          position: "relative",
          overflow: "hidden",
          py: { xs: 7, md: 10 },
          backgroundImage:
            "radial-gradient(circle at 90% 20%, #36533C 0%, transparent 40%)",
        }}
      >
        <Container maxWidth="lg">
          <Box sx={{ maxWidth: 760, position: "relative" }}>
            <Typography
              sx={{
                color: palette.sage,
                fontSize: 12,
                fontWeight: 700,
                letterSpacing: "0.22em",
                mb: 2,
              }}
            >
              AI-POWERED SPECIES RECOGNITION
            </Typography>

            <Typography
              component="h1"
              sx={{
                fontSize: { xs: "2.7rem", md: "4.5rem" },
                fontWeight: 800,
                letterSpacing: "-0.055em",
                lineHeight: 1.08,
                mb: 3,
              }}
            >
              Discover wildlife.
              <Box
                component="span"
                sx={{ color: palette.sage, display: "block" }}
              >
                Through intelligence.
              </Box>
            </Typography>

            <Typography
              sx={{
                color: "#C4D0C2",
                fontSize: { xs: 15, md: 17 },
                maxWidth: 600,
                lineHeight: 1.8,
              }}
            >
              Explore wildlife species classification using neural networks and
              traditional machine learning. Upload an image to discover which of
              22 species a model predicts.
            </Typography>

            <Stack direction="row" sx={{ mt: 4, flexWrap: "wrap", gap: 1.5 }}>
              {[
                "22 species classes",
                "4 live classification models",
                "WildlifeReID-10K",
              ].map((item) => (
                <Chip
                  key={item}
                  label={item}
                  sx={{
                    bgcolor: "#29412F",
                    color: "#E0E9DA",
                    border: "1px solid #46604A",
                  }}
                />
              ))}
            </Stack>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: { xs: 5, md: 8 } }}>
        <Stack spacing={6}>
          {/* Workspace introduction */}
          <Box>
            <Typography
              sx={{
                color: palette.moss,
                fontSize: 12,
                fontWeight: 800,
                letterSpacing: "0.18em",
                mb: 1,
              }}
            >
              CLASSIFICATION WORKSPACE
            </Typography>

            <Typography
              variant="h4"
              sx={{
                fontWeight: 800,
                letterSpacing: "-0.04em",
                mb: 1,
              }}
            >
              Identify a species
            </Typography>

            <Typography sx={{ color: palette.muted }}>
              Select a model, upload your image and explore the result.
            </Typography>
          </Box>

          {/* Model selection */}
          <Box>
            <Typography sx={{ fontWeight: 800, mb: 2 }}>
              01 / Choose a classification model
            </Typography>

            {modelsError && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                {modelsError}
              </Alert>
            )}

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "1fr",
                  md: "repeat(2, 1fr)",
                  lg: "repeat(4, 1fr)",
                },
                gap: 2,
              }}
            >
              {modelOptions.map((option) => {
                const available = models?.[option.id]?.available ?? false;
                const selected = model === option.id;

                return (
                  <Paper
                    key={option.id}
                    component="button"
                    type="button"
                    disabled={!available || loading}
                    onClick={() => {
                      setModel(option.id);
                      resetPrediction();
                    }}
                    elevation={0}
                    sx={{
                      p: 3,
                      textAlign: "left",
                      fontFamily: "inherit",
                      cursor: available ? "pointer" : "not-allowed",
                      borderRadius: 3,
                      bgcolor: selected ? palette.forest : palette.white,
                      color: selected ? palette.white : palette.forest,
                      border: selected
                        ? `2px solid ${palette.forest}`
                        : `2px solid ${palette.border}`,
                      opacity: available ? 1 : 0.62,
                      transition: "all 0.2s ease",
                      "&:hover": available
                        ? {
                            borderColor: palette.moss,
                            transform: "translateY(-2px)",
                          }
                        : {},
                    }}
                  >
                    <Stack spacing={2}>
                      <Stack
                        direction="row"
                        sx={{
                          alignItems: "center",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography
                          sx={{
                            color: selected ? palette.sage : palette.moss,
                            fontWeight: 800,
                            fontSize: 13,
                          }}
                        >
                          {option.number}
                        </Typography>

                        <Chip
                          label={available ? "LIVE" : "RESULTS ONLY"}
                          size="small"
                          sx={{
                            bgcolor: selected ? "#37513C" : "#EDF1E9",
                            color: selected ? palette.sage : palette.moss,
                            fontWeight: 800,
                            fontSize: 10,
                          }}
                        />
                      </Stack>

                      <Box>
                        <Typography
                          sx={{
                            fontWeight: 800,
                            fontSize: 19,
                            mb: 0.5,
                          }}
                        >
                          {option.title}
                        </Typography>

                        <Typography
                          sx={{
                            color: selected ? "#BFCDBD" : palette.muted,
                            fontSize: 13,
                          }}
                        >
                          {option.subtitle}
                        </Typography>
                      </Box>

                      <Typography
                        sx={{
                          fontSize: 12,
                          color: selected ? palette.sage : palette.moss,
                          fontWeight: 700,
                        }}
                      >
                        {available
                          ? selected
                            ? "● Selected"
                            : "Select model →"
                          : "Live inference unavailable"}
                      </Typography>
                    </Stack>
                  </Paper>
                );
              })}
            </Box>

            {!models && !modelsError && (
              <Typography
                variant="caption"
                sx={{ color: palette.muted, mt: 1, display: "block" }}
              >
                Checking model availability…
              </Typography>
            )}
          </Box>

          {/* Upload */}
          <Box>
            <Typography sx={{ fontWeight: 800, mb: 2 }}>
              02 / Upload a wildlife image
            </Typography>

            <Paper
              elevation={0}
              sx={{
                p: { xs: 2, md: 4 },
                borderRadius: 4,
                border: `1px solid ${palette.border}`,
                bgcolor: palette.white,
              }}
            >
              <Stack spacing={3}>
                <Button
                  component="label"
                  variant="text"
                  disabled={loading}
                  sx={{
                    minHeight: preview ? 100 : 230,
                    border: `2px dashed ${palette.border}`,
                    borderRadius: 3,
                    bgcolor: "#FAFBF7",
                    color: palette.forest,
                    textTransform: "none",
                    display: "flex",
                    flexDirection: "column",
                    gap: 1,
                    "&:hover": {
                      bgcolor: "#F1F5EA",
                      borderColor: palette.moss,
                    },
                  }}
                >
                  <Box
                    sx={{
                      width: 54,
                      height: 54,
                      borderRadius: 3,
                      bgcolor: "#E7EEDC",
                      display: "grid",
                      placeItems: "center",
                      fontSize: 26,
                    }}
                  >
                    ↑
                  </Box>

                  <Typography sx={{ fontWeight: 800 }}>
                    {file
                      ? "Choose a different image"
                      : "Choose a wildlife photograph"}
                  </Typography>

                  <Typography sx={{ color: palette.muted, fontSize: 13 }}>
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
                        maxHeight: 420,
                        objectFit: "contain",
                        borderRadius: 3,
                        bgcolor: "#F0F2EA",
                      }}
                    />

                    <Typography
                      sx={{
                        mt: 1,
                        color: palette.muted,
                        fontSize: 12,
                        overflowWrap: "anywhere",
                      }}
                    >
                      {file?.name}
                    </Typography>
                  </Box>
                )}

                <Button
                  variant="contained"
                  onClick={handleSubmit}
                  disabled={!file || loading || !modelAvailable}
                  sx={{
                    alignSelf: { xs: "stretch", sm: "flex-start" },
                    bgcolor: palette.forest,
                    color: palette.white,
                    px: 4,
                    py: 1.5,
                    borderRadius: 2,
                    fontWeight: 800,
                    textTransform: "none",
                    boxShadow: "none",
                    "&:hover": {
                      bgcolor: palette.forestLight,
                      boxShadow: "none",
                    },
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
                    "Identify species →"
                  )}
                </Button>

                {error && <Alert severity="error">{error}</Alert>}
              </Stack>
            </Paper>
          </Box>

          {/* Prediction */}
          {prediction && (
            <Box>
              <Typography sx={{ fontWeight: 800, mb: 2 }}>
                03 / Classification result
              </Typography>

              <Paper
                elevation={0}
                sx={{
                  overflow: "hidden",
                  borderRadius: 4,
                  border: `1px solid ${palette.border}`,
                }}
              >
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: {
                      xs: "1fr",
                      md: "1fr 1fr",
                    },
                  }}
                >
                  <Box
                    sx={{
                      bgcolor: "#E8EDE4",
                      minHeight: 330,
                    }}
                  >
                    {preview && (
                      <Box
                        component="img"
                        src={preview}
                        alt="Classified wildlife"
                        sx={{
                          width: "100%",
                          height: "100%",
                          minHeight: 330,
                          maxHeight: 500,
                          objectFit: "contain",
                          display: "block",
                        }}
                      />
                    )}
                  </Box>

                  <Box sx={{ p: { xs: 3, md: 4 } }}>
                    <Typography
                      sx={{
                        color: palette.moss,
                        fontWeight: 800,
                        fontSize: 11,
                        letterSpacing: "0.16em",
                        mb: 2,
                      }}
                    >
                      PREDICTED SPECIES
                    </Typography>

                    <Typography
                      variant="h3"
                      sx={{
                        fontWeight: 800,
                        textTransform: "capitalize",
                        letterSpacing: "-0.04em",
                        overflowWrap: "anywhere",
                      }}
                    >
                      {formatSpecies(prediction.species)}
                    </Typography>

                    {isCNNPrediction(prediction) ? (
                      <>
                        <Typography
                          sx={{
                            color: palette.moss,
                            fontWeight: 800,
                            fontSize: 24,
                            mt: 1,
                          }}
                        >
                          {(prediction.confidence * 100).toFixed(1)}%
                        </Typography>

                        <Typography
                          sx={{
                            color: palette.muted,
                            fontSize: 12,
                          }}
                        >
                          Model confidence
                        </Typography>

                        <Typography sx={{ fontWeight: 800, mt: 4, mb: 2 }}>
                          Top predictions
                        </Typography>

                        <Stack spacing={2}>
                          {(prediction.top_predictions ?? []).map(
                            (item, index) => (
                              <Box key={`${item.species}-${index}`}>
                                <Stack
                                  direction="row"
                                  sx={{
                                    mb: 0.7,
                                    justifyContent: "space-between",
                                  }}
                                >
                                  <Typography
                                    sx={{
                                      fontSize: 13,
                                      textTransform: "capitalize",
                                    }}
                                  >
                                    {formatSpecies(item.species)}
                                  </Typography>

                                  <Typography
                                    sx={{
                                      fontSize: 13,
                                      fontWeight: 800,
                                    }}
                                  >
                                    {(item.confidence * 100).toFixed(1)}%
                                  </Typography>
                                </Stack>

                                <Box
                                  sx={{
                                    height: 8,
                                    bgcolor: "#E9EEE5",
                                    borderRadius: 4,
                                    overflow: "hidden",
                                  }}
                                >
                                  <Box
                                    sx={{
                                      height: "100%",
                                      width: `${Math.min(
                                        100,
                                        Math.max(0, item.confidence * 100),
                                      )}%`,
                                      bgcolor:
                                        index === 0 ? palette.moss : "#AFC2A1",
                                      borderRadius: 4,
                                    }}
                                  />
                                </Box>
                              </Box>
                            ),
                          )}
                        </Stack>
                      </>
                    ) : (
                      <Stack spacing={2} sx={{ mt: 3 }}>
                        <Box>
                          <Typography
                            sx={{
                              color: palette.muted,
                              fontSize: 12,
                              mb: 0.5,
                            }}
                          >
                            SVM decision score
                          </Typography>

                          <Typography
                            sx={{
                              color: palette.moss,
                              fontWeight: 800,
                              fontSize: 24,
                            }}
                          >
                            {prediction.decision_score == null
                              ? "Unavailable"
                              : prediction.decision_score.toFixed(4)}
                          </Typography>
                        </Box>

                        <Box>
                          <Typography
                            sx={{
                              color: palette.muted,
                              fontSize: 12,
                              mb: 0.5,
                            }}
                          >
                            Inference time
                          </Typography>

                          <Typography sx={{ fontWeight: 800, fontSize: 18 }}>
                            {prediction.timing.total_ms.toFixed(2)} ms
                          </Typography>
                        </Box>

                        <Typography
                          sx={{
                            color: palette.muted,
                            fontSize: 12,
                            lineHeight: 1.7,
                          }}
                        >
                          The decision score is an SVM classification margin,
                          not a probability or confidence percentage.
                        </Typography>
                      </Stack>
                    )}

                    <Alert severity="info" sx={{ mt: 4 }}>
                      This is a model prediction, not a confirmed species
                      identification.
                    </Alert>
                  </Box>
                </Box>

                <Box
                  sx={{
                    p: 3,
                    borderTop: `1px solid ${palette.border}`,
                    bgcolor: palette.white,
                  }}
                >
                  <Button
                    onClick={() => setShowDetails(!showDetails)}
                    sx={{
                      color: palette.forest,
                      fontWeight: 800,
                      textTransform: "none",
                    }}
                  >
                    {showDetails
                      ? "Hide research details ↑"
                      : "View research details ↓"}
                  </Button>

                  <Collapse in={showDetails}>
                    <Stack spacing={2} sx={{ mt: 2 }}>
                      {[
                        ["Model", models?.[model]?.name ?? model],
                        ["Dataset", "WildlifeReID-10K"],
                        [
                          "Species classes",
                          String(models?.[model]?.classes ?? 22),
                        ],
                        ["API", "FastAPI REST"],
                      ].map(([label, value]) => (
                        <Stack
                          key={label}
                          direction="row"
                          sx={{ justifyContent: "space-between", gap: 2 }}
                        >
                          <Typography sx={{ fontSize: 13, fontWeight: 700 }}>
                            {label}
                          </Typography>
                          <Typography
                            sx={{
                              fontSize: 13,
                              color: palette.muted,
                              textAlign: "right",
                            }}
                          >
                            {value}
                          </Typography>
                        </Stack>
                      ))}
                    </Stack>
                  </Collapse>
                </Box>
              </Paper>
            </Box>
          )}

          {/* Existing comparison component */}
          <ModelComparison />

          {/* Footer */}
          <Box
            sx={{
              pt: 3,
              borderTop: `1px solid ${palette.border}`,
            }}
          >
            <Stack
              direction={{ xs: "column", sm: "row" }}
              sx={{ justifyContent: "space-between" }}
              spacing={1}
            >
              <Typography sx={{ fontSize: 12, color: palette.muted }}>
                © WildTrace AI · Wildlife species classification research
              </Typography>

              <Typography sx={{ fontSize: 12, color: palette.muted }}>
                Powered by React + FastAPI
              </Typography>
            </Stack>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
