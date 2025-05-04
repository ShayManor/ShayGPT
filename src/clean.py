import json
from pathlib import Path
from typing import Union, List

import pandas as pd


def full_clean(paths: Union[str, List[str]]):
    def clean(s: str):
        s = s.strip()
        s = " ".join(s.split())
        return s
    if isinstance(paths, List):
        paths = [str(paths)]
    texts = []

    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")

        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        raw = obj.get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if not raw:
                        continue
                    c = clean(raw)
                    if c:
                        texts.append(c)

        elif p.suffix in [".json", ".csv"]:
            if p.suffix == ".json":
                df = pd.read_json(str(p))
            else:
                df = pd.read_csv(str(p))
            if "text" not in df.columns:
                raise ValueError(f"{p} has no 'text' column")
            for raw in df["text"].dropna().astype(str):
                c = clean(raw)
                if c:
                    texts.append(c)


if __name__ == '__main__':
    full_clean("data/AI_Human.csv")
