# core/validation.py
# -*- coding: utf-8 -*-
"""
Validation framework for OpenBench Wizard.

Provides real-time validation with blocking error popups and auto-focus.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from PySide6.QtWidgets import QWidget, QMessageBox


@dataclass
class ValidationError:
    """Single validation error."""
    field_name: str
    message: str
    page_id: str
    widget: Optional[QWidget] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation result containing validity status and errors."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)


class FieldValidator:
    """Field validator with static methods for common validation rules."""

    @staticmethod
    def required(
        value: Any,
        field_name: str,
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that a field is not empty.

        Args:
            value: The value to validate
            field_name: Name of the field for error reporting
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error

        Returns:
            ValidationError if invalid, None if valid
        """
        if value is None:
            return ValidationError(field_name, message, page_id, widget)

        if isinstance(value, str) and not value.strip():
            return ValidationError(field_name, message, page_id, widget)

        return None

    @staticmethod
    def path_exists(
        path: str,
        field_name: str,
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that a path exists.

        Args:
            path: Path to validate
            field_name: Name of the field for error reporting
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error

        Returns:
            ValidationError if invalid, None if valid
        """
        import os

        # Empty path is OK (optional field)
        if not path or not path.strip():
            return None

        if not os.path.exists(path):
            full_message = f"{message}: {path}"
            return ValidationError(field_name, full_message, page_id, widget)

        return None

    @staticmethod
    def number_range(
        value: float,
        min_val: float,
        max_val: float,
        field_name: str,
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that a number is within a range.

        Args:
            value: Number to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            field_name: Name of the field
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error

        Returns:
            ValidationError if invalid, None if valid
        """
        if value < min_val or value > max_val:
            return ValidationError(field_name, message, page_id, widget)
        return None

    @staticmethod
    def min_max(
        min_value: float,
        max_value: float,
        field_name: str,
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that min value is less than or equal to max value.

        Args:
            min_value: Minimum value
            max_value: Maximum value
            field_name: Name of the field
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error (usually the min field)

        Returns:
            ValidationError if min > max, None if valid
        """
        if min_value > max_value:
            return ValidationError(field_name, message, page_id, widget)
        return None

    @staticmethod
    def at_least_one(
        values: List[Any],
        field_names: List[str],
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that at least one of the values is non-empty.

        Args:
            values: List of values to check
            field_names: Names of the fields
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error

        Returns:
            ValidationError if all empty, None if at least one is valid
        """
        for value in values:
            if value is not None:
                if isinstance(value, str) and value.strip():
                    return None
                elif not isinstance(value, str) and value:
                    return None

        combined_name = "/".join(field_names)
        return ValidationError(combined_name, message, page_id, widget)

    @staticmethod
    def selection_required(
        selection: Dict[str, bool],
        field_name: str,
        message: str,
        page_id: str = "",
        widget: QWidget = None
    ) -> Optional[ValidationError]:
        """
        Validate that at least one item is selected (True).

        Args:
            selection: Dict mapping item names to selected status
            field_name: Name of the field
            message: Error message if validation fails
            page_id: Page ID for error context
            widget: Widget to focus on error

        Returns:
            ValidationError if none selected, None if at least one selected
        """
        if not selection:
            return ValidationError(field_name, message, page_id, widget)

        has_selection = any(v for v in selection.values())
        if not has_selection:
            return ValidationError(field_name, message, page_id, widget)

        return None


class ValidationManager:
    """
    Validation manager that coordinates validation flow.

    Handles showing errors sequentially and focusing on error widgets.
    """

    def __init__(self, parent_widget: QWidget = None):
        """
        Initialize validation manager.

        Args:
            parent_widget: Parent widget for message boxes
        """
        self._parent = parent_widget

    def show_error_and_focus(self, error: ValidationError, allow_skip: bool = True) -> bool:
        """
        Show error message and optionally allow user to skip validation.

        Args:
            error: ValidationError to display
            allow_skip: If True, show option to continue anyway

        Returns:
            True if user chose to continue anyway, False to stay on current page
        """
        if allow_skip:
            # Show warning with option to continue
            reply = QMessageBox.warning(
                self._parent,
                "Validation Warning",
                f"{error.message}\n\nDo you want to continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            user_wants_to_continue = (reply == QMessageBox.Yes)
        else:
            # Just show warning, no option to skip
            QMessageBox.warning(
                self._parent,
                "Validation Error",
                error.message
            )
            user_wants_to_continue = False

        if error.widget is not None and not user_wants_to_continue:
            error.widget.setFocus()

        return user_wants_to_continue

    def validate_and_show_errors(self, errors: List[ValidationError], allow_skip: bool = True) -> bool:
        """
        Process errors, showing the first one and optionally allowing skip.

        Args:
            errors: List of validation errors
            allow_skip: If True, allow user to continue despite errors

        Returns:
            True if no errors or user chose to continue, False otherwise
        """
        if not errors:
            return True

        # Show first error with option to skip
        return self.show_error_and_focus(errors[0], allow_skip=allow_skip)
