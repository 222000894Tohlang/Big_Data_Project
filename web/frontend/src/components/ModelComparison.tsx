
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
  cnn: number | null;
  svm: number | null;
};

const metrics: Metric[] = [
  { label: "Accuracy", cnn: 84.0, svm: 80.87 },
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
        {value.toFixed(2)}%
      </Typography>

      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
          height: 7,
          borderRadius: 2,
        }}
      />
    </Stack>
  );
}

export default function ModelComparison() {
  return (
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      <CardContent sx={{ p: { xs: 2, md: 3 } }}>
        <Stack spacing={3}>
          <Box>
            <Typography variant="h5" sx={{fontWeight:700}}>
              Model Performance Comparison
            </Typography>

            <Typography
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
            />
          </Stack>

          <Divider />

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                md: "1fr 1fr",
              },
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
                label="Held-out evaluation"
                sx={{ mt: 1 }}
              />
            </Box>
          </Box>

          <Divider />

          {metrics.map((metric) => (
            <Box key={metric.label}>
              <Typography
                variant="subtitle2"
                sx={{ mb: 1.5,fontWeight:700 }}
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
                  gap: 2,
                }}
              >
                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                  >
                    CNN
                  </Typography>
                  <MetricValue value={metric.cnn} />
                </Box>

                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                  >
                    HOG + SVM
                  </Typography>
                  <MetricValue value={metric.svm} />
                </Box>
              </Box>
            </Box>
          ))}

          <Divider />

          <Typography
            variant="body2"
            color="text.secondary"
          >
            CNN accuracy (84.0%) is an approximate
            previously reported result and requires
            verification. SVM metrics come from
            Person 3's evaluation of 23,178 test images.
            A direct comparison requires confirmation
            that both models were evaluated on the
            same held-out test split.
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}
