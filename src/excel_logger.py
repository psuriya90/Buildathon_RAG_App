from pathlib import Path
from datetime import datetime

import pandas as pd


EXCEL_PATH = Path(
    "logs/qa_history.xlsx"
)


def log_question(
    question,
    answer,
    source
):

    EXCEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_record = pd.DataFrame([
        {
            "Timestamp": datetime.now(),
            "Question": question,
            "Answer": answer,
            "Source": source
        }
    ])

    if EXCEL_PATH.exists():

        existing = pd.read_excel(
            EXCEL_PATH
        )

        result = pd.concat(
            [
                existing,
                new_record
            ],
            ignore_index=True
        )

    else:

        result = new_record

    result.to_excel(
        EXCEL_PATH,
        index=False
    )