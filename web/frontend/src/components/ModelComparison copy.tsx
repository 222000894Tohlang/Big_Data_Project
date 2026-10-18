
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

const colors = {
  forest: "#14251B",
  moss: "#526B4F",
  sage: "#C7DB9B",
  ivory: "#F5F2E9",
  muted: "#68776A",
  border: "#E2E6DC",
};

const svmMetrics = [
  { label: "Accuracy", value: 83.34 },
  { label: "Macro F1-score", value: 82.11 },
  { label: "Macro precision", value: 81.2 },
  { label: "Macro recall", value: 83.39 },
];

function MetricBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <Box>
      <Stack
        direction="row"
        
        sx={{ mb: 1 , justifyContent:"space-between",
        alignItems:"center"}}
      >
        <Typography
          sx={{
            color: colors.forest,
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          {label}
        </Typography>

        <Typography
          sx={{
            color: colors.forest,
            fontSize: 14,
            fontWeight: 800,
          }}
        >
          {value.toFixed(2)}%
        </Typography>
      </Stack>

      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
          height: 9,
          borderRadius: 5,
          bgcolor: "#E7ECE2",
          "& .MuiLinearProgress-bar": {
            bgcolor: colors.moss,
            borderRadius: 5,
          },
        }}
      />
    </Box>
  );
}

export default function ModelComparison() {
  return (
    <Box component="section">
      <Box sx={{ mb: 3 }}>
        <Typography
          sx={{
            color: colors.moss,
            fontSize: 12,
            fontWeight: 800,
            letterSpacing: "0.18em",
            mb: 1,
          }}
        >
          RESEARCH & EVALUATION
        </Typography>

        <Typography
          variant="h4"
          sx={{
            color: colors.forest,
            fontWeight: 800,
            letterSpacing: "-0.04em",
            mb: 1,
          }}
        >
          Model intelligence
        </Typography>

        <Typography sx={{ color: colors.muted }}>
          Explore the classification approaches and available
          experimental results.
        </Typography>
      </Box>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: {
            xs: "1fr",
            md: "repeat(3, 1fr)",
          },
          gap: 2,
          mb: 3,
        }}
      >
        {[
          {
            number: "01",
            name: "Original CNN",
            method: "Convolutional Neural Network",
            status: "Live inference",
            active: true,
          },
          {
            number: "02",
            name: "Leakage-Aware CNN",
            method: "Convolutional Neural Network",
            status: "Live inference",
            active: true,
          },
          {
            number: "03",
            name: "HOG + SVM",
            method: "Traditional machine learning",
            status: "Evaluation only",
            active: false,
          },
        ].map((item) => (
          <Card
            key={item.number}
            elevation={0}
            sx={{
              borderRadius: 3,
              border: `1px solid ${colors.border}`,
              bgcolor: "#FFFFFF",
              height: "100%",
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Stack spacing={2}>
                <Stack
                  direction="row"
                  sx={{alignItems:"center",
                  justifyContent:"space-between"}}
                >
                  <Typography
                    sx={{
                      color: colors.moss,
                      fontWeight: 800,
                      fontSize: 12,
                    }}
                  >
                    MODEL {item.number}
                  </Typography>

                  <Chip
                    size="small"
                    label={item.status}
                    sx={{
                      bgcolor: item.active
                        ? "#E8F0E0"
                        : "#F1EEE7",
                      color: item.active
                        ? colors.moss
                        : colors.muted,
                      fontSize: 10,
                      fontWeight: 700,
                    }}
                  />
                </Stack>

                <Box>
                  <Typography
                    sx={{
                      fontWeight: 800,
                      fontSize: 19,
                      color: colors.forest,
                      mb: 0.5,
                    }}
                  >
                    {item.name}
                  </Typography>

                  <Typography
                    sx={{
                      color: colors.muted,
                      fontSize: 13,
                    }}
                  >
                    {item.method}
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        ))}
      </Box>

      <Card
        elevation={0}
        sx={{
          borderRadius: 4,
          overflow: "hidden",
          border: `1px solid ${colors.border}`,
        }}
      >
        <Box
          sx={{
            bgcolor: colors.forest,
            color: "#FFFFFF",
            p: { xs: 3, md: 4 },
          }}
        >
          <Typography
            sx={{
              color: colors.sage,
              fontWeight: 800,
              fontSize: 11,
              letterSpacing: "0.16em",
              mb: 1.5,
            }}
          >
            REPORTED EXPERIMENTAL PERFORMANCE
          </Typography>

          <Typography
            variant="h5"
            sx={{
              fontWeight: 800,
              letterSpacing: "-0.03em",
              mb: 1,
            }}
          >
            HOG + Support Vector Machine
          </Typography>

          <Typography
            sx={{
              color: "#C4D0C2",
              fontSize: 13,
              lineHeight: 1.7,
            }}
          >
            Results reported by Person 3 for the original
            experiment split of the wildlife dataset.
          </Typography>

          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
            sx={{ mt: 3 }}
          >
            <Box
              sx={{
                bgcolor: "#2B4231",
                borderRadius: 3,
                p: 3,
                flex: 1,
              }}
            >
              <Typography
                sx={{
                  color: colors.sage,
                  fontSize: 12,
                  mb: 1,
                }}
              >
                CLASSIFICATION ACCURACY
              </Typography>

              <Typography
                sx={{
                  fontSize: { xs: 36, md: 44 },
                  fontWeight: 800,
                  letterSpacing: "-0.05em",
                }}
              >
                83.34%
              </Typography>
            </Box>

            <Box
              sx={{
                bgcolor: "#2B4231",
                borderRadius: 3,
                p: 3,
                flex: 1,
              }}
            >
              <Typography
                sx={{
                  color: colors.sage,
                  fontSize: 12,
                  mb: 1,
                }}
              >
                MACRO F1-SCORE
              </Typography>

              <Typography
                sx={{
                  fontSize: { xs: 36, md: 44 },
                  fontWeight: 800,
                  letterSpacing: "-0.05em",
                }}
              >
                82.11%
              </Typography>
            </Box>
          </Stack>
        </Box>

        <CardContent sx={{ p: { xs: 3, md: 4 } }}>
          <Stack spacing={3}>
            <Box>
              <Typography
                sx={{
                  fontWeight: 800,
                  fontSize: 17,
                  color: colors.forest,
                  mb: 0.5,
                }}
              >
                Evaluation metrics
              </Typography>

              <Typography
                sx={{
                  color: colors.muted,
                  fontSize: 13,
                }}
              >
                Original split · 30,561 test images ·
                22 species classes
              </Typography>
            </Box>

            <Stack spacing={2.5}>
              {svmMetrics.map((metric) => (
                <MetricBar
                  key={metric.label}
                  label={metric.label}
                  value={metric.value}
                />
              ))}
            </Stack>

            <Divider />

            <Box>
              <Typography
                sx={{
                  fontWeight: 800,
                  color: colors.forest,
                  mb: 1,
                }}
              >
                Evaluation considerations
              </Typography>

              <Typography
                sx={{
                  color: colors.muted,
                  fontSize: 13,
                  lineHeight: 1.9,
                }}
              >
                These SVM results were reported using the
                original experiment split. This split was
                not identity-disjoint, so the reported
                scores should not be interpreted as
                identity-independent generalisation
                performance.
              </Typography>

              <Typography
                sx={{
                  color: colors.muted,
                  fontSize: 13,
                  lineHeight: 1.9,
                  mt: 1.5,
                }}
              >
                The CNN models support live image
                classification. Their evaluation metrics
                are not displayed here because comparable
                results have not yet been verified on the
                same held-out test partition.
              </Typography>
            </Box>

            <Box
              sx={{
                bgcolor: colors.ivory,
                borderRadius: 3,
                p: 2.5,
                borderLeft: `4px solid ${colors.moss}`,
              }}
            >
              <Typography
                sx={{
                  color: colors.forest,
                  fontWeight: 800,
                  fontSize: 13,
                  mb: 0.5,
                }}
              >
                Integration status
              </Typography>

              <Typography
                sx={{
                  color: colors.muted,
                  fontSize: 13,
                  lineHeight: 1.7,
                }}
              >
                Both CNN models are available for live
                inference through FastAPI. The trained SVM
                uses NVIDIA RAPIDS cuML and is currently
                presented through its reported evaluation
                results rather than live predictions.
              </Typography>
            </Box>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
