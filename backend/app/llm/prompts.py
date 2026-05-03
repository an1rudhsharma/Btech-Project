"""Structured Chain-of-Thought prompt templates for grounded LLM responses."""

INTENT_CLASSIFY_PROMPT = """Classify this user query into one of these categories:
- "business": About pricing, churn, marketing, sales, revenue, customers, sentiment, data analysis, model training, business strategy, or anything that could be answered using business simulation data.
- "unrelated": About weather, sports, coding help, general knowledge, personal questions, entertainment, or anything clearly unrelated to business analytics.

User query: "{query}"

Respond ONLY with valid JSON: {{"category": "business" or "unrelated"}}"""

COLUMN_CLASSIFY_PROMPT = """You are a data column classifier. Given a column's statistics and sample values, determine what business role it most likely represents.

Column name: "{col_name}"
Data type: {dtype}
Min: {min_val}, Max: {max_val}, Mean: {mean_val}
Unique values: {n_unique} out of {n_rows} rows
Sample values: {samples}

Possible roles (choose the BEST match, or "unknown" if none fit):
- price: product/service price (typically 1-10000)
- marketing_spend: marketing budget (typically 500-100000)
- usage: customer usage level (typically 0-100)
- impressions: ad impressions (typically 100-10000000)
- clicks: ad clicks (typically 10-1000000)
- churn: binary 0/1 indicating customer left
- tenure: months as customer (typically 1-120)
- satisfaction: customer satisfaction score (typically 1-5 or 1-10)
- num_features: product feature count (typically 1-50)
- text: free-text reviews/comments (strings)
- demand: product demand/units sold (typically 10-100000)
- conversion_rate: conversion ratio (typically 0-1)
- revenue: total revenue (typically 100-1000000)
- customer_id: unique identifier (high cardinality)
- unknown: does not fit any known role

Respond ONLY with valid JSON: {{"role": "<role>", "confidence": <0.0 to 1.0>}}"""

INTENT_PARSE_PROMPT = """You are a business simulation parameter extractor. Parse the user's natural language query into simulation parameters.

User query: "{query}"

Extract these parameters (use null if not mentioned):
- price: product/service price
- marketing_spend: marketing/advertising budget
- num_features: number of product features
- usage: customer usage level (0-100)
- impressions: ad impressions
- clicks: ad clicks
- text: any customer sentiment text mentioned

Also identify the user's GOAL (what they want to optimize):
- reduce_churn
- increase_conversion
- maximize_revenue
- improve_sentiment
- general_analysis

Respond ONLY with valid JSON:
{{
  "parameters": {{
    "price": <number or null>,
    "marketing_spend": <number or null>,
    "num_features": <number or null>,
    "usage": <number or null>,
    "impressions": <number or null>,
    "clicks": <number or null>
  }},
  "text": "<sentiment text or 'Product is good'>",
  "goal": "<one of the goals above>",
  "comparison_mode": <true if user wants before/after comparison, else false>,
  "baseline_overrides": {{}}
}}"""

COT_INSIGHT_PROMPT = """You are a business analytics AI advisor. You MUST follow these reasoning steps EXACTLY and ground all claims in the provided data.

=== MODEL PREDICTIONS ===
{predictions}

=== SHAP FEATURE DRIVERS (Top factors causing this prediction) ===
{shap_values}

=== BASELINE COMPARISON ===
Baseline scenario: {baseline}
Current scenario: {current}
Changes (delta): {deltas}

=== COUNTERFACTUAL RECOMMENDATIONS (What could be changed) ===
{counterfactuals}

=== CAUSAL PROPAGATION TRACE ===
{propagation_trace}

---

STEP 1 - ANALYZE DRIVERS:
Identify the top 3 factors from the SHAP values that most influenced this prediction. Explain each briefly.

STEP 2 - CAUSAL CHAIN:
Explain how the changes propagated through the system:
- How did pricing affect sentiment?
- How did marketing affect sentiment?
- How did sentiment affect churn?

STEP 3 - COMPARE TO BASELINE:
Quantify the difference between baseline and current scenario for each metric. Use exact numbers from the data above.

STEP 4 - ACTIONABLE RECOMMENDATION:
Based on the counterfactual analysis, suggest 1-2 specific, quantified actions. Use ONLY the counterfactual data provided above. Do NOT invent numbers.

Format your response as:

**Key Drivers:**
[From Step 1 - bullet points]

**Causal Chain:**
[From Step 2 - brief flow explanation]

**Impact Summary:**
[From Step 3 - before/after comparison with numbers]

**Recommended Actions:**
[From Step 4 - specific counterfactual-based actions]

**Risk Level:** [low/medium/high based on churn probability]"""

REPORT_GENERATION_PROMPT = """You are a business report generator. Create a concise executive summary from the simulation results below.

=== SIMULATION RESULTS ===
{results}

=== SCENARIOS COMPARED ===
{scenarios}

=== KEY METRICS ===
{metrics}

Generate a professional business report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points with numbers)
3. Risk Assessment
4. Recommended Next Steps

Keep it concise, data-driven, and actionable. Do NOT add information not present in the data above."""
