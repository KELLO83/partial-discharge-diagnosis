from __future__ import annotations

from prpd_similarity_retrieval.backend_adapter import FeatureSimilarityCaseRetrievalAdapter

from service.backend.app.models.mock_adapters import (
    MockTimeSeriesInferenceAdapter,
    MockVisionInferenceAdapter,
    MockVlmInferenceAdapter,
)
from service.backend.app.models.model_runtime import MockModelAdapters, build_service_model_runtime
from service.backend.app.rag import PgvectorRagRetrievalAdapter, build_llm_rag_reporter


similar_case_adapter = FeatureSimilarityCaseRetrievalAdapter()
rag_adapter = PgvectorRagRetrievalAdapter()
model_runtime = build_service_model_runtime(
    MockModelAdapters(
        time_series=MockTimeSeriesInferenceAdapter(),
        vision=MockVisionInferenceAdapter(),
        vlm=MockVlmInferenceAdapter(),
    )
)
time_series_adapter = model_runtime.time_series_adapter
vision_adapter = model_runtime.vision_adapter
vlm_adapter = model_runtime.vlm_adapter
_, llm_rag_status = build_llm_rag_reporter(model_runtime.vlm_adapter)
