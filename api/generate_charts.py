from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------

performance = pd.read_csv(
    RESULTS_DIR / "performance_results.csv"
)

robustness = pd.read_csv(
    RESULTS_DIR / "image_robustness_results.csv"
)

quality = pd.read_csv(
    RESULTS_DIR / "image_quality_results.csv"
)


# --------------------------------------------------
# Helper: Save chart
# --------------------------------------------------

def save_chart(filename):
    plt.tight_layout()

    output_path = CHARTS_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Chart 1: HTTP Response Time Trend
# --------------------------------------------------

def plot_response_time_trend():

    plt.figure(figsize=(10, 5))

    plt.plot(
        performance["request_number"],
        performance["response_time_ms"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="HTTP response time",
    )

    average = performance["response_time_ms"].mean()

    plt.axhline(
        average,
        linestyle="--",
        label=f"Mean: {average:.2f} ms",
    )

    plt.title("HTTP Response Time Across 50 Requests")
    plt.xlabel("Request Number")
    plt.ylabel("Response Time (ms)")

    plt.legend()
    plt.grid(alpha=0.3)

    save_chart("response_time_trend.png")


# --------------------------------------------------
# Chart 2: Average Processing Times
# --------------------------------------------------

def plot_processing_time_comparison():

    measurements = {
        "Preprocessing": performance[
            "preprocessing_ms"
        ].mean(),

        "CNN Inference": performance[
            "inference_ms"
        ].mean(),

        "Total Processing": performance[
            "processing_ms"
        ].mean(),

        "HTTP Response": performance[
            "response_time_ms"
        ].mean(),
    }

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        measurements.keys(),
        measurements.values(),
    )

    plt.title("Average Prediction Processing Times")
    plt.ylabel("Time (ms)")

    plt.bar_label(
        bars,
        fmt="%.2f ms",
        padding=3,
    )

    plt.grid(axis="y", alpha=0.3)

    save_chart("processing_time_comparison.png")


# --------------------------------------------------
# Chart 3: Resolution vs Response Time
# --------------------------------------------------

def plot_resolution_response_time():

    data = robustness.copy()

    data["resolution"] = (
        data["width"].astype(str)
        + "×"
        + data["height"].astype(str)
    )

    pivot = data.pivot_table(
        index="resolution",
        columns="format",
        values="response_time_ms",
        aggfunc="mean",
    )

    resolution_order = [
        "64×64",
        "128×128",
        "256×256",
        "512×512",
        "1024×1024",
        "2048×2048",
    ]

    pivot = pivot.reindex(resolution_order)

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
        width=0.75,
    )

    ax.set_title(
        "Image Resolution vs API Response Time"
    )

    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Response Time (ms)")

    plt.xticks(rotation=30)
    plt.legend(title="Image Format")
    plt.grid(axis="y", alpha=0.3)

    save_chart("resolution_response_time.png")


# --------------------------------------------------
# Chart 4: Image Quality vs Confidence
# --------------------------------------------------

def plot_image_quality_confidence():

    data = quality.copy()

    data["confidence_percent"] = (
        data["confidence"] * 100
    )

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        data["variation"],
        data["confidence_percent"],
    )

    plt.title(
        "CNN Prediction Confidence Under Image Variations"
    )

    plt.xlabel("Confidence (%)")
    plt.ylabel("Image Variation")

    plt.xlim(0, 105)

    plt.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3,
    )

    plt.grid(axis="x", alpha=0.3)

    save_chart("image_quality_confidence.png")


# --------------------------------------------------
# Chart 5: Response Time Distribution
# --------------------------------------------------

def plot_response_time_distribution():

    plt.figure(figsize=(9, 5))

    plt.hist(
        performance["response_time_ms"],
        bins=12,
        edgecolor="black",
        alpha=0.8,
    )

    plt.title(
        "Distribution of HTTP Response Times"
    )

    plt.xlabel("Response Time (ms)")
    plt.ylabel("Number of Requests")

    plt.grid(axis="y", alpha=0.3)

    save_chart("response_time_distribution.png")


# --------------------------------------------------
# Generate all charts
# --------------------------------------------------

def main():

    print("\nGenerating Wildlife CNN Evaluation Charts\n")

    plot_response_time_trend()

    plot_processing_time_comparison()

    plot_resolution_response_time()

    plot_image_quality_confidence()

    plot_response_time_distribution()

    print("\nAll charts generated successfully!")


if __name__ == "__main__":
    main()