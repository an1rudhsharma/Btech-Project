"""ML Sync - auto-inject training summaries into the RAG vector DB."""

from app.rag.embeddings import embed_batch
from app.db.knowledge import insert_chunks, create_document, update_document


async def sync_model_to_knowledge_base(
    user_id: str,
    model_name: str,
    metrics: dict,
    shap_values: dict = None,
    feature_importances: dict = None,
    dataset_info: str = "",
):
    """
    After training a model, generate a text summary and embed it into pgvector.
    This allows the chat to reference model results via RAG.
    """
    summary_parts = []

    # Model overview
    summary_parts.append(
        f"Machine Learning Model: {model_name.upper()} prediction model.\n"
        f"Training completed successfully.\n"
        f"Dataset: {dataset_info}"
    )

    # Metrics
    if metrics:
        metrics_text = f"Model performance metrics for {model_name}:\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_text += f"- {key}: {value:.4f}\n"
            else:
                metrics_text += f"- {key}: {value}\n"
        summary_parts.append(metrics_text)

    # Feature importances
    if feature_importances:
        fi_text = f"Feature importance for {model_name} model (what drives predictions):\n"
        sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, importance in sorted_features[:10]:
            fi_text += f"- {feat}: {importance:.1%} importance\n"
        summary_parts.append(fi_text)

    # SHAP values
    if shap_values:
        shap_text = f"SHAP analysis for {model_name} model (directional impact of each feature):\n"
        for feat, val in list(shap_values.items())[:10]:
            direction = "increases" if val > 0 else "decreases"
            shap_text += f"- {feat}: {direction} {model_name} probability by {abs(val):.3f}\n"
        summary_parts.append(shap_text)

    # Create document record
    full_text = "\n\n".join(summary_parts)
    doc_filename = f"_ml_model_{model_name}_summary.txt"

    doc = await create_document(
        user_id=user_id,
        filename=doc_filename,
        file_type="ml_summary",
        file_size=len(full_text.encode()),
        metadata={"model_name": model_name, "auto_generated": True, "queryable": False},
    )

    # Embed and store chunks
    chunks = summary_parts  # Each part is a natural chunk
    embeddings = embed_batch(chunks)

    chunk_records = [
        {
            "document_id": doc["id"],
            "user_id": user_id,
            "content": chunk,
            "embedding": emb,
            "chunk_index": i,
            "metadata": {"filename": doc_filename, "file_type": "ml_summary", "model_name": model_name},
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    await insert_chunks(chunk_records)
    await update_document(doc["id"], {"status": "ready", "chunk_count": len(chunks)})

    return {"synced": True, "model": model_name, "chunks": len(chunks)}
