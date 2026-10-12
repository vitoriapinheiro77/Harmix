import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from django.utils import timezone


logger = logging.getLogger('music.pipeline.diagnostics')

MAX_DIAGNOSTIC_EVENTS = 250


def _peak_memory_mb():
    """Return process peak RSS without adding a runtime dependency."""
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == 'darwin':
            value /= 1024 * 1024
        else:
            value /= 1024
        return round(value, 2)
    except (ImportError, OSError, ValueError):
        return None


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class PipelineDiagnostics:
    """Collect structured, bounded diagnostics for one pipeline execution."""

    def __init__(self, project_id, on_change=None):
        self.project_id = project_id
        self.on_change = on_change
        self.run_id = uuid.uuid4().hex
        self.started_at = None
        self.completed_at = None
        self.status = 'not_started'
        self.error = None
        self.events = []
        self.stages = []
        self._wall_started = None
        self._wall_completed = None
        self._stage_stack = []

    @property
    def active(self):
        return self._wall_started is not None and self.completed_at is None

    def start(self, details=None):
        if self._wall_started is not None:
            return
        self._wall_started = time.perf_counter()
        self.started_at = timezone.now().isoformat()
        self.status = 'running'
        self.event('pipeline_started', details=details or {})
        self._notify()

    def event(self, event, *, stage=None, details=None, level='info'):
        if self._wall_started is None:
            return
        payload = {
            'timestamp': timezone.now().isoformat(),
            'elapsed_seconds': round(time.perf_counter() - self._wall_started, 4),
            'event': event,
            'stage': stage,
            'level': level,
            'details': _json_safe(details or {}),
        }
        self.events.append(payload)
        if len(self.events) > MAX_DIAGNOSTIC_EVENTS:
            self.events = self.events[-MAX_DIAGNOSTIC_EVENTS:]

        log_fn = getattr(logger, level, logger.info)
        log_fn(
            'pipeline_event=%s',
            json.dumps(
                {'project_id': self.project_id, 'run_id': self.run_id, **payload},
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

    @contextmanager
    def stage(self, name, details=None):
        metrics = {}
        parent_stage = self._stage_stack[-1] if self._stage_stack else None
        self._stage_stack.append(name)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        memory_started = _peak_memory_mb()
        self.event('stage_started', stage=name, details=details or {})
        try:
            yield metrics
        except Exception as exc:
            stage_result = self._stage_result(
                name,
                'failed',
                wall_started,
                cpu_started,
                memory_started,
                metrics,
                parent_stage=parent_stage,
                error=exc,
            )
            self._stage_stack.pop()
            self.stages.append(stage_result)
            self.event(
                'stage_failed',
                stage=name,
                details=stage_result,
                level='error',
            )
            self._notify()
            raise
        else:
            stage_result = self._stage_result(
                name,
                'completed',
                wall_started,
                cpu_started,
                memory_started,
                metrics,
                parent_stage=parent_stage,
            )
            self._stage_stack.pop()
            self.stages.append(stage_result)
            self.event('stage_completed', stage=name, details=stage_result)
            self._notify()

    def finish(self, status, error=None):
        if self._wall_started is None or self.completed_at is not None:
            return
        self.status = status
        self.error = (
            {'type': type(error).__name__, 'message': str(error)}
            if error is not None
            else None
        )
        self.event(
            'pipeline_failed' if error else 'pipeline_completed',
            details={'status': status, 'error': self.error},
            level='error' if error else 'info',
        )
        self._wall_completed = time.perf_counter()
        self.completed_at = timezone.now().isoformat()
        self._notify()

    def as_dict(self):
        total_duration = (
            round((self._wall_completed or time.perf_counter()) - self._wall_started, 4)
            if self._wall_started is not None
            else 0.0
        )
        return {
            'schema_version': 1,
            'run_id': self.run_id,
            'project_id': self.project_id,
            'status': self.status,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'total_duration_seconds': total_duration,
            'peak_memory_mb': _peak_memory_mb(),
            'error': deepcopy(self.error),
            'stages': deepcopy(self.stages),
            'events': deepcopy(self.events),
        }

    def _stage_result(
        self,
        name,
        status,
        wall_started,
        cpu_started,
        memory_started,
        metrics,
        parent_stage=None,
        error=None,
    ):
        memory_finished = _peak_memory_mb()
        result = {
            'name': name,
            'status': status,
            'parent_stage': parent_stage,
            'duration_seconds': round(time.perf_counter() - wall_started, 4),
            'cpu_seconds': round(time.process_time() - cpu_started, 4),
            'peak_memory_mb_before': memory_started,
            'peak_memory_mb_after': memory_finished,
            'details': _json_safe(metrics),
        }
        if error is not None:
            result['error'] = {
                'type': type(error).__name__,
                'message': str(error),
            }
        return result

    def _notify(self):
        if not self.on_change:
            return
        try:
            self.on_change(self.as_dict())
        except Exception:
            logger.exception(
                'Nao foi possivel persistir diagnosticos da pipeline do projeto %s.',
                self.project_id,
            )
