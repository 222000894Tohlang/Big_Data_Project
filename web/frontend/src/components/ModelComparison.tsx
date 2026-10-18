
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
        {value.toFixed(2)}%
      </Typography>

      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
          height: 8,
          borderRadius: 4,
          bgcolor: "#E9EEE5",
          "& .MuiLinearProgress-bar": {
            bgcolor: palette.moss,
            borderRadius: 4,
          },
        }}
      />
    </Stack>
  );
}

export default function ModelComparison() {
  return (
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
              Model Performance Comparison
            </Typography>

            <Typography
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
            />
          </Stack>

          <Divider />

          <Typography
            sx={{
              fontWeight: 800,
              color: palette.forest,
              fontSize: 17,
            }}
          >
            Previously reported HOG + SVM results
          </Typography>

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                md: "1fr 1fr",
              },
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
              />
            </Box>
          </Box>

          <Divider />

          {metrics.map((metric) => (
            <Box key={metric.label}>
              <Typography
                sx={{
                  fontWeight: 800,
                  color: palette.forest,
                  mb: 1.5,
                }}
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
                  gap: 3,
                }}
              >
                <Box>
                  <Typography
                    sx={{
                      color: palette.muted,
                      fontSize: 12,
                      mb: 1,
                    }}
                  >
                    Original split
                  </Typography>

                  <MetricValue value={metric.original} />
                </Box>

                <Box>
                  <Typography
                    sx={{
                      color: palette.muted,
                      fontSize: 12,
                      mb: 1,
                    }}
                  >
                    Clean split
                  </Typography>

                  <MetricValue value={metric.clean} />
                </Box>
              </Box>
            </Box>
          ))}

          <Divider />

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
        </Stack>
      </CardContent>
    </Card>
  );
}
