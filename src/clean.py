import pandas as pd


def full_clean(csv):
    def clean(s: str):
        s = s.strip()
        s = " ".join(s.split())
        return s

    df = pd.read_csv(csv)
    texts = df["text"].dropna().tolist()
    texts = [clean(text) for text in texts if len(text) > 0]
    return texts


if __name__ == '__main__':
    full_clean("data/AI_Human.csv")
