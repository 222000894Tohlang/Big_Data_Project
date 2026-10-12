import csv
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def export_results(filename: str, results: list[dict]):
    """
    Export evaluation results to a CSV file.
    """

    if not results:
        raise ValueError("No results available to export.")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / filename

    fieldnames = list(
        dict.fromkeys(
            key
            for result in results
            for key in result
        )
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults exported to: {output_path}")