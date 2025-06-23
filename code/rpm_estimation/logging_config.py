"""
Structured logging configuration for RPM estimation pipeline.

Provides JSON-formatted logs with error categorization and context tracking.
"""

import logging
import json
from datetime import datetime
from typing import Optional, Any, Dict
from pathlib import Path


class ProcessingError:
    """Error type categorization for structured logging."""
    RECOVERABLE = "recoverable"    # Missing sensor, continue processing
    FATAL = "fatal"                # Time index mismatch, abort
    QUALITY = "quality"            # Quality threshold exceeded
    IO = "io"                      # File I/O errors
    CONFIG = "config"              # Configuration errors
    VALIDATION = "validation"      # Data validation errors


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON objects with timestamp, level, context, and metadata.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add optional context fields
        optional_fields = [
            'experiment', 'session', 'sensor', 'error_type',
            'processing_step', 'duration_ms', 'sample_count'
        ]
        
        for field in optional_fields:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    log_obj[field] = value
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, default=str)


class ContextLogger:
    """
    Logger wrapper that maintains processing context.
    
    Automatically adds experiment, session, and sensor context to all log messages.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set persistent context fields."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context fields."""
        self.context.clear()
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal logging with context."""
        extra = dict(self.context)
        extra.update(kwargs)
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def log_timing(self, operation: str, duration_ms: float, **kwargs):
        """Log operation timing."""
        self.info(f"{operation} completed", duration_ms=duration_ms, **kwargs)
    
    def log_quality_issue(self, issue: str, severity: str = "warning", **kwargs):
        """Log data quality issues."""
        kwargs['error_type'] = ProcessingError.QUALITY
        if severity == "error":
            self.error(f"Quality issue: {issue}", **kwargs)
        else:
            self.warning(f"Quality issue: {issue}", **kwargs)


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    log_format: str = "json",
    console: bool = True
) -> ContextLogger:
    """
    Set up structured logging configuration.
    
    Args:
        log_file: Optional file path for log output
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        console: Whether to also log to console
        
    Returns:
        ContextLogger instance
    """
    # Create logger
    logger = logging.getLogger("rpm_estimation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return ContextLogger(logger)


def get_logger(name: Optional[str] = None) -> ContextLogger:
    """
    Get a context logger instance.
    
    Args:
        name: Optional logger name (defaults to rpm_estimation)
        
    Returns:
        ContextLogger instance
    """
    logger_name = f"rpm_estimation.{name}" if name else "rpm_estimation"
    logger = logging.getLogger(logger_name)
    return ContextLogger(logger)


# Example usage functions
def log_processing_start(logger: ContextLogger, experiment: str, session: str):
    """Log the start of experiment processing."""
    logger.set_context(experiment=experiment, session=session)
    logger.info(
        f"Starting processing for {experiment} ({session})",
        processing_step="initialization"
    )


def log_sensor_error(logger: ContextLogger, sensor: str, error: Exception, 
                    recoverable: bool = True):
    """Log sensor-specific errors with appropriate categorization."""
    logger.error(
        f"Error processing sensor {sensor}: {str(error)}",
        sensor=sensor,
        error_type=ProcessingError.RECOVERABLE if recoverable else ProcessingError.FATAL,
        processing_step="sensor_loading"
    )


def log_quality_summary(logger: ContextLogger, sensor: str, 
                       total_windows: int, clipped_windows: int):
    """Log quality assessment summary."""
    clipping_pct = 100.0 * clipped_windows / total_windows if total_windows > 0 else 0
    
    logger.info(
        f"Quality assessment complete for {sensor}",
        sensor=sensor,
        total_windows=total_windows,
        clipped_windows=clipped_windows,
        clipping_percentage=round(clipping_pct, 2),
        processing_step="quality_assessment"
    )