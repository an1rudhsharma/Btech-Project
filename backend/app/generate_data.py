"""Generate sample datasets for demo and testing."""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_churn_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic customer churn dataset."""
    np.random.seed(seed)

    price = np.random.uniform(50, 200, n)
    marketing_spend = np.random.uniform(1000, 15000, n)
    num_features = np.random.randint(1, 15, n)
    usage = np.random.uniform(5, 100, n)
    tenure = np.random.randint(1, 72, n)
    satisfaction = np.random.uniform(1, 5, n)

    # Churn logic: higher price + lower usage + fewer features -> more churn
    churn_score = (
        0.3 * (price - 50) / 150
        - 0.2 * (usage - 5) / 95
        - 0.15 * (num_features - 1) / 14
        - 0.2 * (tenure - 1) / 71
        - 0.15 * (satisfaction - 1) / 4
        + 0.1 * (1 - marketing_spend / 15000)
        + np.random.normal(0, 0.1, n)
    )
    churn = (churn_score > 0.15).astype(int)

    return pd.DataFrame({
        "price": np.round(price, 2),
        "marketing_spend": np.round(marketing_spend, 2),
        "num_features": num_features,
        "usage": np.round(usage, 2),
        "tenure": tenure,
        "satisfaction": np.round(satisfaction, 2),
        "churn": churn,
    })


def generate_marketing_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a marketing campaign performance dataset."""
    np.random.seed(seed)

    marketing_spend = np.random.uniform(500, 20000, n)
    impressions = marketing_spend * np.random.uniform(1.5, 4.0, n)
    clicks = impressions * np.random.uniform(0.01, 0.08, n)
    price = np.random.uniform(30, 200, n)

    # Conversion logic
    base_conv = 0.02 + 0.03 * (marketing_spend / 20000)
    price_effect = -0.01 * (price - 100) / 100
    noise = np.random.normal(0, 0.005, n)
    conversion_rate = np.clip(base_conv + price_effect + noise, 0.001, 0.15)

    return pd.DataFrame({
        "marketing_spend": np.round(marketing_spend, 2),
        "impressions": np.round(impressions).astype(int),
        "clicks": np.round(clicks).astype(int),
        "price": np.round(price, 2),
        "conversion_rate": np.round(conversion_rate, 4),
    })


def generate_pricing_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate a pricing and demand dataset."""
    np.random.seed(seed)

    price = np.random.uniform(20, 300, n)
    marketing_spend = np.random.uniform(1000, 15000, n)
    usage = np.random.uniform(10, 100, n)

    # Demand logic: inversely proportional to price, boosted by marketing
    base_demand = 500 - 1.5 * price
    marketing_boost = 0.02 * marketing_spend
    usage_signal = 0.5 * usage
    noise = np.random.normal(0, 20, n)
    demand = np.clip(base_demand + marketing_boost + usage_signal + noise, 10, 1000)

    return pd.DataFrame({
        "price": np.round(price, 2),
        "marketing_spend": np.round(marketing_spend, 2),
        "usage": np.round(usage, 2),
        "demand": np.round(demand, 2),
    })


def generate_sentiment_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a customer review sentiment dataset."""
    np.random.seed(seed)

    positive_templates = [
        "Great product, love the features and quality",
        "Excellent service, very satisfied with the purchase",
        "Amazing value for money, highly recommend",
        "The new features are fantastic, makes my life easier",
        "Best product I've used, worth every penny",
        "Outstanding quality and great customer support",
        "Very happy with this purchase, exceeded expectations",
        "Incredible product, the updates keep making it better",
    ]

    negative_templates = [
        "Too expensive for what you get, not worth the price",
        "Poor quality, disappointed with the product",
        "Terrible customer service, will not buy again",
        "The product keeps breaking, waste of money",
        "Overpriced and underdelivers, very frustrated",
        "Missing basic features that competitors have",
        "Very slow and buggy, regret this purchase",
        "Not worth the price increase, considering alternatives",
    ]

    texts = []
    sentiments = []

    for _ in range(n):
        if np.random.random() > 0.45:
            texts.append(np.random.choice(positive_templates) + f" {np.random.choice(['!', '.', '!!'])}")
            sentiments.append("positive")
        else:
            texts.append(np.random.choice(negative_templates) + f" {np.random.choice(['!', '.', '...'])}")
            sentiments.append("negative")

    return pd.DataFrame({
        "text": texts,
        "sentiment": sentiments,
    })


def create_all_samples(output_dir: Path):
    """Generate and save all sample datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sample_churn.csv": generate_churn_dataset(),
        "sample_marketing.csv": generate_marketing_dataset(),
        "sample_pricing.csv": generate_pricing_dataset(),
        "sample_sentiment.csv": generate_sentiment_dataset(),
    }

    for filename, df in datasets.items():
        df.to_csv(output_dir / filename, index=False)
        print(f"Generated {filename}: {df.shape}")


if __name__ == "__main__":
    create_all_samples(Path(__file__).parent / "data")
