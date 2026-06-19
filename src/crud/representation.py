from __future__ import annotations

import datetime
import logging
import time
from contextlib import suppress
from typing import Any

from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, exceptions, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.embedding_client import embedding_client
from src.schemas import ResolvedConfiguration
from src.telemetry.events import EmbeddingCallPurpose
from src.telemetry.logging import accumulate_metric
from src.utils.formatting import format_datetime_utc
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)


def _observation_text(obs: ExplicitObservation | DeductiveObservation) -> str:
    """Return the canonical text payload for an explicit or deductive observation."""
    return obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content


def _normalized_observation(
    obs: ExplicitObservation | DeductiveObservation,
) -> ExplicitObservation | DeductiveObservation:
    """Return an observation with its persisted/embed text normalized."""
    text = _observation_text(obs).strip()
    if isinstance(obs, DeductiveObservation):
        return obs.model_copy(update={"conclusion": text})
    return obs.model_copy(update={"content": text})


class RepresentationManager:
    """Unified manager for representation and document queries."""

    def __init__(
        self,
        workspace_name: str,
        *,
        observer: str,
        observed: str,
    ) -> None:
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed

    async def save_representation(
        self,
        representation: Representation,
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        """
        Save Representation objects to the collection as a set of documents.

        Args:
            representation: Representation object
            message_ids: Message ID range to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created

        Returns:
            The number of *new documents saved*
        """

        new_documents = 0

        if not representation.deductive and not representation.explicit:
            logger.debug("No observations to save")
            return new_documents

        all_observations = [
            _normalized_observation(obs)
            for obs in representation.deductive + representation.explicit
            if _observation_text(obs).strip()
        ]
        if not all_observations:
            logger.debug("No non-empty observations to save")
            return new_documents

        # Batch embed all observations
        batch_embed_start = time.perf_counter()

        observation_texts = [_observation_text(obs) for obs in all_observations]
        try:
            with embedding_call_purpose(
                EmbeddingCallPurpose.CREATE_OBSERVATIONS.value,
                workspace_name=self.workspace_name,
                parent_category="representation",
            ):
                embeddings = await embedding_client.simple_batch_embed(
                    observation_texts
                )
        except ValueError as e:
            raise exceptions.ValidationException(
                "Observation content exceeds maximum token limit of "
                + f"{settings.EMBEDDING.MAX_INPUT_TOKENS}."
            ) from e

        batch_embed_duration = (time.perf_counter() - batch_embed_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "embed_new_observations",
            batch_embed_duration,
            "ms",
        )

        # Batch create document objects
        create_document_start = time.perf_counter()
        async with tracked_db("representation_manager.save_representation") as db:
            new_documents = await self._save_representation_internal(
                db,
                all_observations,
                embeddings,
                message_ids,
                session_name,
                message_created_at,
                message_level_configuration,
            )

        create_document_duration = (time.perf_counter() - create_document_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "save_new_observations",
            create_document_duration,
            "ms",
        )

        return new_documents

    async def _save_representation_internal(
        self,
        db: AsyncSession,
        all_observations: list[ExplicitObservation | DeductiveObservation],
        embeddings: list[list[float]],
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        # get_or_create_collection already handles IntegrityError with rollback and a retry
        collection = await crud.get_or_create_collection(
            db,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
        )

        # Prepare all documents for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs, embedding in zip(all_observations, embeddings, strict=True):
            # NOTE: will add additional levels of reasoning in the future
            if isinstance(obs, DeductiveObservation):
                obs_level = "deductive"
                obs_content = obs.conclusion
                obs_premises = obs.premises
            else:
                obs_level = "explicit"
                obs_content = obs.content
                obs_premises = None

            metadata: schemas.DocumentMetadata = schemas.DocumentMetadata(
                message_ids=message_ids,
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(
                    content=obs_content,
                    session_name=session_name,
                    level=obs_level,
                    metadata=metadata,
                    embedding=embedding,
                )
            )

        # Use bulk creation with optional duplicate detection
        accepted_documents = await crud.create_documents(
            db,
            documents_to_create,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            deduplicate=settings.DERIVER.DEDUPLICATE,
        )

        if message_level_configuration.dream.enabled:
            try:
                await check_and_schedule_dream(db, collection)
            except Exception as e:
                logger.warning(f"Failed to check dream scheduling: {e}")

        return len(accepted_documents)

    async def get_working_representation(
        self,
        *,
        db: AsyncSession | None = None,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        parent_category: str | None = None,
        embedding_purpose: EmbeddingCallPurpose = EmbeddingCallPurpose.SEARCH_MEMORY,
    ) -> Representation:
        """
        Get working representation with flexible query options.

        Args:
            db: Optional database session. If provided, uses it directly;
                otherwise creates a new session via tracked_db.
            session_name: Optional session to filter by
            include_semantic_query: Query for semantic search
            embedding: Pre-computed embedding for the semantic query.
            semantic_search_top_k: Number of semantic results
            semantic_search_max_distance: Maximum distance for semantic search
            include_most_derived: Include most derived observations
            max_observations: Maximum total observations to return
            parent_category: Optional workflow attribution forwarded to the
                fallback embedding call when the caller didn't pre-compute
                an embedding (or pre-compute failed).
            embedding_purpose: Embedding call_purpose tag to use on the
                fallback embed when no pre-computed embedding was supplied.
                Defaults to SEARCH_MEMORY; callers whose route-level
                precompute uses a more specific purpose (e.g.
                SESSION_CONTEXT_SEARCH) should pass that here so the
                fallback path lands in the same analytics bucket.

        Returns:
            Representation combining various query strategies
        """
        if include_semantic_query and embedding is None:
            # Best-effort precompute when caller didn't supply one (or their
            # precompute was suppressed). The purpose is parameterized so
            # this fallback shows up in the same telemetry bucket as the
            # successful path — see embedding_purpose docstring above.
            with (
                suppress(Exception),
                embedding_call_purpose(
                    embedding_purpose.value,
                    workspace_name=self.workspace_name,
                    parent_category=parent_category,
                ),
            ):
                embedding = await embedding_client.embed(include_semantic_query)

        if db is not None:
            return await self._get_working_representation_internal(
                db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                include_most_derived=include_most_derived,
                max_observations=max_observations,
            )

        async with tracked_db(
            "representation_manager.get_working_representation", read_only=True
        ) as new_db:
            return await self._get_working_representation_internal(
                new_db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                include_most_derived=include_most_derived,
                max_observations=max_observations,
            )

    # Private helper methods

    async def _get_working_representation_internal(
        self,
        db: AsyncSession,
        *,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    ) -> Representation:
        """Internal implementation of get_working_representation."""
        total = max_observations

        # Calculate how many observations to get from each source
        semantic_observations = (
            min(
                max(
                    0,
                    semantic_search_top_k
                    if semantic_search_top_k is not None
                    else total // 3,
                ),
                total,
            )
            if include_semantic_query
            else 0
        )

        if include_semantic_query and include_most_derived:
            # three-way blend: both semantic and derived requested
            top_observations = min(max(0, total // 3), total - semantic_observations)
        elif include_most_derived:
            # two-way blend: only derived requested
            top_observations = min(max(0, total // 2), total - semantic_observations)
        else:
            # no derived observations requested
            top_observations = 0

        # remaining observations are recent
        recent_observations = total - semantic_observations - top_observations

        representation = Representation()

        # Collect docs from all sources for post-merge dedup
        all_docs: list[models.Document] = []

        # Get semantic observations if requested
        if include_semantic_query:
            semantic_docs = await self._query_documents_semantic(
                db,
                query=include_semantic_query,
                top_k=semantic_observations,
                max_distance=semantic_search_max_distance,
                embedding=embedding,
            )
            all_docs.extend(semantic_docs)

        # Get most derived observations if requested
        if include_most_derived:
            derived_docs = await self._query_documents_most_derived(
                db, top_k=top_observations
            )
            all_docs.extend(derived_docs)

        # Get recent observations
        recent_docs = await self._query_documents_recent(
            db, top_k=recent_observations, session_name=session_name
        )
        all_docs.extend(recent_docs)

        # Post-merge: deduplicate by embedding cosine distance, then
        # exact-content dedup, then apply size-based budget.
        all_docs = self._dedup_docs_by_embedding(all_docs)
        representation = Representation.from_documents(all_docs)
        representation.deduplicate_semantic()
        representation.truncate_to_budget(
            max_chars=settings.DERIVER.WORKING_REPRESENTATION_MAX_CHARS
        )

        return representation

    def _dedup_docs_by_embedding(
        self, docs: list[models.Document]
    ) -> list[models.Document]:
        """Remove embedding-near-duplicate documents, keeping the more informative one.

        Compares every pair of documents.  If their cosine embedding distance
        is below the configured threshold, the shorter one (less informative)
        is dropped.  Documents without embeddings are kept as-is.

        This runs in-memory after all query sources are merged, so it catches
        duplicates that come from different sources (semantic, derived, recent).
        """
        import json
        import numpy as np

        if not docs or len(docs) < 2:
            return docs

        threshold = settings.DERIVER.WORKING_REPRESENTATION_DEDUP_DISTANCE

        # Parse embeddings — pgvector raw SQL returns strings, ORM returns lists
        parsed: list[tuple[models.Document, list[float] | None]] = []
        for d in docs:
            emb = d.embedding
            if emb is None:
                parsed.append((d, None))
            elif isinstance(emb, str):
                try:
                    parsed.append((d, json.loads(emb)))
                except (json.JSONDecodeError, ValueError):
                    parsed.append((d, None))
            elif isinstance(emb, (list, np.ndarray)):
                parsed.append((d, list(emb)))
            else:
                parsed.append((d, None))

        with_emb = [(d, e) for d, e in parsed if e is not None]
        without_emb = [d for d, e in parsed if e is None]

        if len(with_emb) < 2:
            # Dedup by content only
            seen: set[str] = set()
            result: list[models.Document] = []
            for d in docs:
                key = (d.content or "").strip().lower()
                if key not in seen:
                    seen.add(key)
                    result.append(d)
            return result

        # Compute pairwise cosine distances using numpy
        vectors = np.array([e for _, e in with_emb], dtype=np.float32)
        # Normalize for cosine: cos_dist = 1 - (a·b / |a||b|)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms
        # Pairwise cosine similarity matrix
        sim_matrix = normalized @ normalized.T
        # Convert to distance
        dist_matrix = 1.0 - sim_matrix

        n = len(with_emb)
        drop_indices: set[int] = set()

        for i in range(n):
            if i in drop_indices:
                continue
            for j in range(i + 1, n):
                if j in drop_indices:
                    continue
                if dist_matrix[i, j] < threshold:
                    # Keep the longer (more informative) one
                    len_i = len(with_emb[i][0].content or "")
                    len_j = len(with_emb[j][0].content or "")
                    if len_j >= len_i:
                        drop_indices.add(i)
                    else:
                        drop_indices.add(j)

        kept = [d for idx, (d, _) in enumerate(with_emb) if idx not in drop_indices]

        # Merge back with content dedup to avoid exact dupes between sources
        seen_content: set[str] = set()
        result: list[models.Document] = []
        for d in without_emb + kept:
            key = (d.content or "").strip().lower()
            if key not in seen_content:
                seen_content.add(key)
                result.append(d)
        return result

    async def _query_documents_semantic(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float | None = None,
        level: str | None = None,
        embedding: list[float] | None = None,
    ) -> list[models.Document]:
        """Query documents by semantic similarity."""
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    top_k,
                    max_distance,
                    embedding=embedding,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=query,
                    max_distance=max_distance,
                    top_k=top_k,
                    embedding=embedding,
                )
                db.expunge_all()
                return list(documents)

        except Exception as e:
            logger.error(f"Error getting relevant observations: {e}")
            return []

    async def _query_documents_recent(
        self, db: AsyncSession, top_k: int, session_name: str | None = None
    ) -> list[models.Document]:
        """Query most recent documents, excluding embedding-near-duplicates.

        For each document, excludes it if there exists another document in the
        same collection with cosine embedding distance < DEDUP_DISTANCE that is
        either longer or equally long (more informative).  This collapses
        semantically equivalent observations (e.g. "juan applies validation"
        vs "Applies validation before implementing") using the embeddings that
        the deriver already computed.

        Documents without embeddings are included as-is (pending sync).
        """
        dedup_distance = settings.DERIVER.WORKING_REPRESENTATION_DEDUP_DISTANCE

        session_filter = (
            "AND d.session_name = :session_name " if session_name else ""
        )
        session_param = {"session_name": session_name} if session_name else {}

        # Use raw SQL for pgvector <=> operator (not available in ORM)
        raw_sql = sa_text(f"""
            SELECT d.id, d.internal_metadata, d.content, d.level,
                   d.times_derived, d.embedding, d.source_ids,
                   d.created_at, d.observer, d.observed,
                   d.workspace_name, d.session_name, d.deleted_at,
                   d.sync_state, d.last_sync_at, d.sync_attempts
            FROM documents d
            WHERE d.workspace_name = :workspace_name
              AND d.observer = :observer
              AND d.observed = :observed
              AND d.deleted_at IS NULL
              {session_filter}
              AND (
                  d.embedding IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM documents d2
                      WHERE d2.workspace_name = d.workspace_name
                        AND d2.observer = d.observer
                        AND d2.observed = d.observed
                        AND d2.deleted_at IS NULL
                        AND d2.id != d.id
                        AND d2.embedding IS NOT NULL
                        AND (d2.embedding <=> d.embedding) < :dedup_dist
                        AND LENGTH(d2.content) >= LENGTH(d.content)
                  )
              )
            ORDER BY d.created_at DESC
            LIMIT :limit
        """)

        params = {
            "workspace_name": self.workspace_name,
            "observer": self.observer,
            "observed": self.observed,
            "dedup_dist": dedup_distance,
            "limit": top_k,
            **session_param,
        }

        result = await db.execute(raw_sql, params)
        rows = result.fetchall()
        # Map raw rows back to Document ORM objects
        documents = []
        for row in rows:
            doc = models.Document()
            doc.id = row[0]
            doc.internal_metadata = row[1]
            doc.content = row[2]
            doc.level = row[3]
            doc.times_derived = row[4]
            doc.embedding = row[5]
            doc.source_ids = row[6]
            doc.created_at = row[7]
            doc.observer = row[8]
            doc.observed = row[9]
            doc.workspace_name = row[10]
            doc.session_name = row[11]
            doc.deleted_at = row[12]
            doc.sync_state = row[13]
            doc.last_sync_at = row[14]
            doc.sync_attempts = row[15]
            documents.append(doc)
        db.expunge_all()
        return documents

    async def _query_documents_most_derived(
        self, db: AsyncSession, top_k: int
    ) -> list[models.Document]:
        """Query most derived documents, excluding embedding-near-duplicates.

        Ties in times_derived are broken by created_at descending (most recent
        first), then by id for deterministic ordering within the same batch
        timestamp.
        """
        dedup_distance = settings.DERIVER.WORKING_REPRESENTATION_DEDUP_DISTANCE

        raw_sql = sa_text("""
            SELECT d.id, d.internal_metadata, d.content, d.level,
                   d.times_derived, d.embedding, d.source_ids,
                   d.created_at, d.observer, d.observed,
                   d.workspace_name, d.session_name, d.deleted_at,
                   d.sync_state, d.last_sync_at, d.sync_attempts
            FROM documents d
            WHERE d.workspace_name = :workspace_name
              AND d.observer = :observer
              AND d.observed = :observed
              AND d.deleted_at IS NULL
              AND (
                  d.embedding IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM documents d2
                      WHERE d2.workspace_name = d.workspace_name
                        AND d2.observer = d.observer
                        AND d2.observed = d.observed
                        AND d2.deleted_at IS NULL
                        AND d2.id != d.id
                        AND d2.embedding IS NOT NULL
                        AND (d2.embedding <=> d.embedding) < :dedup_dist
                        AND LENGTH(d2.content) >= LENGTH(d.content)
                  )
              )
            ORDER BY d.times_derived DESC, d.created_at DESC, d.id
            LIMIT :limit
        """)

        params = {
            "workspace_name": self.workspace_name,
            "observer": self.observer,
            "observed": self.observed,
            "dedup_dist": dedup_distance,
            "limit": top_k,
        }

        result = await db.execute(raw_sql, params)
        rows = result.fetchall()
        documents = []
        for row in rows:
            doc = models.Document()
            doc.id = row[0]
            doc.internal_metadata = row[1]
            doc.content = row[2]
            doc.level = row[3]
            doc.times_derived = row[4]
            doc.embedding = row[5]
            doc.source_ids = row[6]
            doc.created_at = row[7]
            doc.observer = row[8]
            doc.observed = row[9]
            doc.workspace_name = row[10]
            doc.session_name = row[11]
            doc.deleted_at = row[12]
            doc.sync_state = row[13]
            doc.last_sync_at = row[14]
            doc.sync_attempts = row[15]
            documents.append(doc)
        db.expunge_all()
        return list(documents)

    async def _get_observations_internal(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float,
        level: str | None,
    ) -> list[models.Document]:
        """Internal method that does the actual observation retrieval."""
        return await self._query_documents_semantic(
            db, query, top_k, max_distance, level
        )

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        count: int,
        max_distance: float | None = None,
        embedding: list[float] | None = None,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=query,
            max_distance=max_distance,
            top_k=count,
            filters=self._build_filter_conditions(level),
            embedding=embedding,
        )

        # Sort by creation time
        docs_sorted: list[models.Document] = sorted(
            list(documents), key=lambda x: x.created_at, reverse=True
        )
        return docs_sorted

    def _build_filter_conditions(
        self,
        level: str | None = None,
    ) -> dict[str, Any]:
        """
        Build filter conditions for document queries.

        Returns a flat dict of key-value pairs for vector store filtering.
        """
        filters: dict[str, Any] = {}

        if level:
            filters["level"] = level

        return filters


# Module-level functions for backward compatibility and convenience


async def get_working_representation(
    workspace_name: str,
    *,
    db: AsyncSession | None = None,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    embedding: list[float] | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    include_most_derived: bool = False,
    max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    parent_category: str | None = None,
    embedding_purpose: EmbeddingCallPurpose = EmbeddingCallPurpose.SEARCH_MEMORY,
) -> Representation:
    """
    Get raw working representation data from the relevant document collection.

    This is a convenience function that creates a RepresentationManager and calls
    get_working_representation on it.

    Args:
        db: Optional database session. If provided, uses it directly;
            otherwise creates a new session via tracked_db.
        embedding: Pre-computed embedding for the semantic query.
        parent_category: Workflow attribution forwarded to the fallback
            embedding call when no pre-computed embedding was supplied.
        embedding_purpose: Embedding call_purpose for the fallback embed;
            callers should match it to whatever purpose their route-level
            precompute used so failure/retry paths stay in the same bucket.
    """
    manager = RepresentationManager(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    return await manager.get_working_representation(
        db=db,
        session_name=session_name,
        include_semantic_query=include_semantic_query,
        embedding=embedding,
        semantic_search_top_k=semantic_search_top_k,
        semantic_search_max_distance=semantic_search_max_distance,
        include_most_derived=include_most_derived,
        max_observations=max_observations,
        parent_category=parent_category,
        embedding_purpose=embedding_purpose,
    )
