import pandas as pd
import re

# Read Sheet1
df = pd.read_csv("hatespeech_vs_neutral.xlsx - Sheet1.csv", encoding="utf-8")

# Display first few rows
print(df.head())

print("yes")

def clean_text(text):
    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # 2. Remove emojis (using Unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002500-\U00002BEF"  # chinese characters
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # 3. Remove hashtags and mentions
    text = re.sub(r"#\S+", "", text)  # hashtags
    text = re.sub(r"@\S+", "", text)  # mentions

    # 4. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)
df.to_csv("cleaned_dataset.csv", index=False, encoding="utf-8-sig")
print(df[["text", "clean_text", "Class"]].head())



df_cleaned = pd.read_csv("cleaned_dataset.csv")
df_cleaned.head()

### checking imbalance of classes ###

# Check class distribution
print(df_cleaned["Class"].value_counts())

# Optionally, show proportions
print(df_cleaned["Class"].value_counts(normalize=True))