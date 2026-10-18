
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";

type Metric = {
  label: string;
<<<<<<< HEAD
  original: number;
  clean: number;
};

const metrics: Metric[] = [
  { label: "Accuracy", original: 83.34, clean: 80.87 },
  { label: "Macro F1-score", original: 82.11, clean: 79.98 },
  { label: "Macro precision", original: 81.20, clean: 78.23 },
  { label: "Macro recall", original: 83.39, clean: 82.65 },
];

const palette = {
  forest: "#14251B",
  moss: "#526B4F",
  sage: "#C7DB9B",
  muted: "#68776A",
  border: "#E2E6DC",
};

function MetricValue({ value }: { value: number }) {
  return (
    <Stack spacing={0.75}>
      <Typography sx={{ fontWeight: 800, fontSize: 15 }}>
=======
  cnn: number | null;
  svm: number | null;
};

const metrics: Metric[] = [
  { label: "Accuracy", cnn: 84.0, svm: 80.87 }, //  { label: "Accuracy", cnn: null, svm: 83.34 },
  { label: "Macro F1-score", cnn: null, svm: 79.98 },
  { label: "Macro precision", cnn: null, svm: 78.23 },
  { label: "Macro recall", cnn: null, svm: 82.65 },
];

function MetricValue({
  value,
}: {
  value: number | null;
}) {
  if (value === null) {
    return (
      <Typography variant="body2" color="text.secondary">
        Not verified
      </Typography>
    );
  }

  return (
    <Stack spacing={0.5}>
      <Typography variant="body2" sx={{fontWeight:600}}>
>>>>>>> data
        {value.toFixed(2)}%
      </Typography>

      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
<<<<<<< HEAD
          height: 8,
          borderRadius: 4,
          bgcolor: "#E9EEE5",
          "& .MuiLinearProgress-bar": {
            bgcolor: palette.moss,
            borderRadius: 4,
          },
=======
          height: 7,
          borderRadius: 2,
>>>>>>> data
        }}
      />
    </Stack>
  );
}

export default function ModelComparison() {
  return (
<<<<<<< HEAD
    <Card
      variant="outlined"
      sx={{
        borderRadius: 4,
        borderColor: palette.border,
        boxShadow: "none",
      }}
    >
      <CardContent sx={{ p: { xs: 3, md: 4 } }}>
        <Stack spacing={3}>
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
              RESEARCH & EVALUATION
            </Typography>

            <Typography
              variant="h5"
              sx={{
                color: palette.forest,
                fontWeight: 800,
              }}
            >
=======
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      <CardContent sx={{ p: { xs: 2, md: 3 } }}>
        <Stack spacing={3}>
          <Box>
            <Typography variant="h5" sx={{fontWeight:700}}>
>>>>>>> data
              Model Performance Comparison
            </Typography>

            <Typography
<<<<<<< HEAD
              sx={{
                color: palette.muted,
                mt: 1,
                lineHeight: 1.7,
              }}
            >
              Explore previously reported evaluation results
              for wildlife species classification.
            </Typography>
          </Box>

          <Stack direction="row" sx={{flexWrap:"wrap", gap:1}}>
            <Chip
              label="2 CNN models · Live"
              size="small"
              sx={{
                bgcolor: "#EDF3E6",
                color: palette.forest,
                fontWeight: 700,
              }}
            />

            <Chip
              label="2 SVM models · Live"
              size="small"
              sx={{
                bgcolor: "#EDF3E6",
                color: palette.forest,
                fontWeight: 700,
              }}
=======
              variant="body2"
              color="text.secondary"
              sx={{ mt: 1 }}
            >
              Previously reported evaluation results for
              wildlife species classification.
            </Typography>
          </Box>

          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
          >
            <Chip
              label="CNN: Live inference available"
              color="success"
              variant="outlined"
            />

            <Chip
              label="SVM: Evaluation results only"
              variant="outlined"
>>>>>>> data
            />
          </Stack>

          <Divider />

<<<<<<< HEAD
          <Typography
            sx={{
              fontWeight: 800,
              color: palette.forest,
              fontSize: 17,
            }}
          >
            Previously reported HOG + SVM results
          </Typography>

=======
>>>>>>> data
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                md: "1fr 1fr",
              },
<<<<<<< HEAD
              gap: 2,
            }}
          >
            <Box
              sx={{
                p: 2.5,
                borderRadius: 3,
                bgcolor: "#F6F8F3",
              }}
            >
              <Typography sx={{ fontWeight: 800 }}>
                Original / Experiment Split
              </Typography>

              <Typography
                sx={{
                  color: palette.muted,
                  fontSize: 13,
                  mt: 1,
                }}
              >
                30,561 test images
              </Typography>

              <Chip
                label="Not identity-disjoint"
                size="small"
                variant="outlined"
                sx={{ mt: 2 }}
              />
            </Box>

            <Box
              sx={{
                p: 2.5,
                borderRadius: 3,
                bgcolor: "#F6F8F3",
              }}
            >
              <Typography sx={{ fontWeight: 800 }}>
                Clean / Leakage-Aware Split
              </Typography>

              <Typography
                sx={{
                  color: palette.muted,
                  fontSize: 13,
                  mt: 1,
                }}
              >
                23,178 test images
              </Typography>

              <Chip
                label="Identity-disjoint"
                size="small"
                variant="outlined"
                sx={{ mt: 2 }}
=======
              gap: 3,
            }}
          >
            <Box>
              <Typography
                variant="subtitle1"
                sx={{fontWeight:700}}
                gutterBottom
              >
                CNN
              </Typography>

              <Typography
                variant="body2"
                color="text.secondary"
                gutterBottom
              >
                Convolutional Neural Network
              </Typography>

              <Chip
                size="small"
                label="Provisional evaluation"
                sx={{ mt: 1 }}
              />
            </Box>

            <Box>
              <Typography
                variant="subtitle1"
                sx={{fontWeight:700}}
                gutterBottom
              >
                HOG + SVM
              </Typography>

              <Typography
                variant="body2"
                color="text.secondary"
                gutterBottom
              >
                Histogram of Oriented Gradients
                with Support Vector Machine
              </Typography>

              <Chip
                size="small"
label="Original split · 30,561 images"
                sx={{ mt: 1 }}
>>>>>>> data
              />
            </Box>
          </Box>

          <Divider />

          {metrics.map((metric) => (
            <Box key={metric.label}>
              <Typography
<<<<<<< HEAD
                sx={{
                  fontWeight: 800,
                  color: palette.forest,
                  mb: 1.5,
                }}
=======
                variant="subtitle2"
                sx={{ mb: 1.5,fontWeight:700 }}
>>>>>>> data
              >
                {metric.label}
              </Typography>

              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "1fr",
                    sm: "1fr 1fr",
                  },
<<<<<<< HEAD
                  gap: 3,
=======
                  gap: 2,
>>>>>>> data
                }}
              >
                <Box>
                  <Typography
<<<<<<< HEAD
                    sx={{
                      color: palette.muted,
                      fontSize: 12,
                      mb: 1,
                    }}
                  >
                    Original split
                  </Typography>

                  <MetricValue value={metric.original} />
=======
                    variant="caption"
                    color="text.secondary"
                  >
                    CNN
                  </Typography>
                  <MetricValue value={metric.cnn} />
>>>>>>> data
                </Box>

                <Box>
                  <Typography
<<<<<<< HEAD
                    sx={{
                      color: palette.muted,
                      fontSize: 12,
                      mb: 1,
                    }}
                  >
                    Clean split
                  </Typography>

                  <MetricValue value={metric.clean} />
=======
                    variant="caption"
                    color="text.secondary"
                  >
                    HOG + SVM
                  </Typography>
                  <MetricValue value={metric.svm} />
>>>>>>> data
                </Box>
              </Box>
            </Box>
          ))}

          <Divider />

<<<<<<< HEAD
          <Box>
            <Typography
              sx={{
                fontWeight: 800,
                color: palette.forest,
                mb: 1,
              }}
            >
              CNN evaluation
            </Typography>

            <Typography
              sx={{
                color: palette.muted,
                fontSize: 14,
                lineHeight: 1.8,
              }}
            >
              Both CNN models support live predictions.
              Their evaluation metrics have not yet been
              verified for display in this comparison.
            </Typography>
          </Box>

          <Box
            sx={{
              p: 2.5,
              bgcolor: "#F6F8F3",
              borderRadius: 3,
              border: `1px solid ${palette.border}`,
            }}
          >
            <Typography
              sx={{
                fontWeight: 800,
                color: palette.forest,
                mb: 1,
              }}
            >
              Important research notes
            </Typography>

            <Typography
              sx={{
                color: palette.muted,
                fontSize: 13,
                lineHeight: 1.9,
              }}
            >
              The SVM metrics shown above are historical
              results reported for the respective dataset
              splits. They have not yet been independently
              verified against the newly supplied
              scikit-learn model files.
            </Typography>

            <Typography
              sx={{
                color: palette.muted,
                fontSize: 13,
                lineHeight: 1.9,
                mt: 1.5,
              }}
            >
              All four models support live inference.
              Direct performance comparisons between CNN
              and SVM require evaluation on the same test
              partition using consistent metrics.
            </Typography>
          </Box>
=======
          <Typography variant="body2" color="text.secondary">
  SVM results were reported using the
  original/experiment split containing 30,561 test images.
  This split was not identity-disjoint.

  CNN evaluation metrics are awaiting verification.
  Direct model comparisons require evaluation on the
  same test partition.

  Live predictions are available for both CNN models.
  SVM inference is currently unavailable in this environment.
</Typography>
>>>>>>> data
        </Stack>
      </CardContent>
    </Card>
  );
}
