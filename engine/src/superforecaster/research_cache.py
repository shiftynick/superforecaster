"""On-disk cache of research bundles, one JSON file per question.

A failed forecast pass costs real search credits if it throws away the research
that was already paid for. This cache persists each :class:`ResearchBundle` to
``<dir>/<question_id>.json`` the moment it is assembled — before the forecast
LLM call — so a later crash cannot lose it, and a future multi-pass ensemble can
share one bundle across passes instead of re-researching per pass.

Freshness is controlled by the caller: :meth:`load` returns ``None`` for a bundle
whose ``retrieved_at`` is older than ``max_age`` (stale reuse would forecast on
yesterday's news). Stale entries are left on disk — they are harmless history and
are overwritten on the next :meth:`save` for that question.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from .research import ResearchBundle


class ResearchCache:
    """A directory of cached research bundles keyed by question id.

    Args:
        dir: Directory holding the ``<question_id>.json`` files. Created on first
            :meth:`save`.
    """

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this cache reads from and writes to."""
        return self._dir

    def _path_for(self, question_id: int) -> Path:
        """Return the cache-file path for one question id."""
        return self._dir / f"{question_id}.json"

    def save(self, question_id: int, bundle: ResearchBundle) -> None:
        """Write ``bundle`` to disk for ``question_id``, overwriting any prior file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path_for(question_id).write_text(bundle.model_dump_json(), encoding="utf-8")

    def load(
        self,
        question_id: int,
        *,
        max_age: timedelta,
        now: datetime | None = None,
    ) -> ResearchBundle | None:
        """Return the cached bundle for ``question_id`` if present and fresh.

        Returns ``None`` when there is no cache file, when the file cannot be
        parsed into a :class:`ResearchBundle`, or when the bundle's
        ``retrieved_at`` is older than ``max_age`` relative to ``now`` (defaults
        to the current UTC time). A stale or corrupt file is left in place; the
        next :meth:`save` overwrites it.
        """
        path = self._path_for(question_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None
        try:
            bundle = ResearchBundle.model_validate_json(raw)
        except ValidationError:
            return None

        moment = now if now is not None else datetime.now(UTC)
        retrieved_at = bundle.retrieved_at
        # A naive timestamp (no tzinfo) is treated as UTC to stay comparable.
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=UTC)
        if moment - retrieved_at > max_age:
            return None
        return bundle
