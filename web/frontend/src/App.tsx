import { useState } from "react";
import {
  Box,
  Button,
  Collapse,
  Container,
  Divider,
  Paper,
  Stack,
  Typography,
  Alert,
} from "@mui/material";

type SpeciesPrediction = {
  species: string;
  confidence: number;
};

type Prediction = {
  species: string;
  confidence: number;
  top_predictions: SpeciesPrediction[];
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showDetails, setShowDetails] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setPrediction(null);
    setError("");
    setShowDetails(false);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      /* if (!response.ok) {
        throw new Error("Prediction failed");
      } */

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(errorData?.detail || "Unable to classify this image.");
      }

      const result: Prediction = await response.json();

      setPrediction(result);

      const data: Prediction = await response.json();
      setPrediction(data);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Something went wrong.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        minWidth: { xs: "100%", lg: "120%" },
        backgroundColor: "#f7f5f0",
        color: "#222",
        py: { xs: 4, md: 7 },
        alignSelf: "center",
      }}
    >
      <Container sx={{ minWidth: "90%" }}>
        <Stack spacing={5}>
          <Box
            component="header"
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography
              variant="body1"
              sx={{ letterSpacing: "-0.02em", fontWeight: 700 }}
            >
              Wildlife Classifier
            </Typography>

            <Typography variant="body2" color="text.secondary">
              WildlifeReID-10K
            </Typography>
          </Box>

          <Box sx={{ maxWidth: "100%" }}>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                fontSize: { xs: "2.2rem", md: "3.3rem" },
                lineHeight: 1.05,
                letterSpacing: "-0.04em",
              }}
            >
              Identifying an animal species
            </Typography>

            <Typography
              variant="body1"
              color="text.secondary"
              sx={{
                mt: 2,
                fontSize: "1.05rem",
                lineHeight: 1.7,
                maxWidth: "100%",
              }}
            >
              Upload a wildlife photo and we’ll identify which of the 22 species
              it most closely matches.
            </Typography>
          </Box>

          <Paper
            variant="outlined"
            sx={{
              p: { xs: 2, md: 3 },
              borderRadius: 2,
              backgroundColor: "#fff",
              boxShadow: "none",
            }}
          >
            <Stack spacing={3}>
              <Button
                component="label"
                variant="text"
                sx={{
                  minHeight: 180,
                  border: "1px dashed #aaa",
                  borderRadius: 1.5,
                  color: "#333",
                  textTransform: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.5,
                  "&:hover": {
                    backgroundColor: "#faf9f6",
                    borderColor: "#777",
                  },
                }}
              >
                <Typography sx={{ fontWeight: 600 }}>
                  Drop a wildlife photo here
                </Typography>

                <Typography variant="body2" color="text.secondary">
                  or choose one from your computer
                </Typography>

                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mt: 1 }}
                >
                  JPG or PNG
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
                      display: "block",
                      borderRadius: 1,
                      backgroundColor: "#efede7",
                    }}
                  />

                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 1, display: "block" }}
                  >
                    {file?.name}
                  </Typography>
                </Box>
              )}

              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={!file || loading}
                sx={{
                  alignSelf: "flex-start",
                  px: 3,
                  py: 1.2,
                  borderRadius: 1.5,
                  textTransform: "none",
                  backgroundColor: "#222",
                  boxShadow: "none",
                  "&:hover": {
                    backgroundColor: "#000",
                    boxShadow: "none",
                  },
                }}
              >
                {loading ? "Identifying species..." : "Identify species"}
              </Button>

              {error && <Alert severity="error">{error}</Alert>}
            </Stack>
          </Paper>

          {prediction && (
            <Box>
              <Divider sx={{ mb: 4 }} />

              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "1fr",
                    md: "1fr 1fr",
                  },
                  gap: 4,
                  alignItems: "start",
                }}
              >
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    YOUR IMAGE
                  </Typography>

                  <Box
                    component="img"
                    src={preview ?? ""}
                    alt="Uploaded wildlife"
                    sx={{
                      width: "100%",
                      mt: 1.5,
                      aspectRatio: "4 / 3",
                      objectFit: "cover",
                      borderRadius: 1,
                    }}
                  />
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    RESULT
                  </Typography>

                  <Typography
                    variant="h4"
                    sx={{
                      mt: 1.5,
                      fontWeight: 700,
                      letterSpacing: "-0.03em",
                    }}
                  >
                    {prediction.species}
                  </Typography>

                  <Typography
                    variant="body1"
                    color="text.secondary"
                    sx={{ mt: 0.5 }}
                  >
                    {(prediction.confidence * 100).toFixed(1)}% confidence
                  </Typography>

                  <Box sx={{ mt: 3 }}>
                    <Typography
                      variant="subtitle2"
                      sx={{
                        fontWeight: 600,
                        mb: 2,
                      }}
                    >
                      Top 3 predictions
                    </Typography>

                    <Stack spacing={2}>
                      {prediction.top_predictions.map((item, index) => (
                        <Box key={`${item.species}-${index}`}>
                          <Stack
                            direction="row"
                            sx={{ mb: 0.5, justifyContent: "space-between" }}
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
                                width: `${item.confidence * 100}%`,
                                height: "100%",
                                bgcolor: index === 0 ? "#454b40" : "#aaa99f",
                                borderRadius: 1,
                              }}
                            />
                          </Box>
                        </Box>
                      ))}
                    </Stack>
                  </Box>
                </Box>
              </Box>

              <Box sx={{ mt: 5 }}>
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
                  <Box
                    sx={{
                      mt: 2,
                      pt: 2,
                      borderTop: "1px solid #ddd",
                    }}
                  >
                    <Stack spacing={1.5}>
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="body2">Model</Typography>
                        <Typography variant="body2" color="text.secondary">
                          CNN
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="body2">Dataset</Typography>
                        <Typography variant="body2" color="text.secondary">
                          WildlifeReID-10K
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="body2">Species classes</Typography>
                        <Typography variant="body2" color="text.secondary">
                          22
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <Typography variant="body2">API</Typography>
                        <Typography variant="body2" color="text.secondary">
                          REST
                        </Typography>
                      </Box>
                    </Stack>
                  </Box>
                </Collapse>
              </Box>
            </Box>
          )}

          <Typography variant="caption" color="text.secondary" sx={{ pt: 3 }}>
            Big data research project · Wildlife species classification
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
