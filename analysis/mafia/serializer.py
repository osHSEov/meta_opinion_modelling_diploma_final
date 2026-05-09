import json


def save_slice_result(
    output_path,
    formulas,
    metrics
):

    data = {
        "formulas": formulas,
        "metrics": metrics
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )