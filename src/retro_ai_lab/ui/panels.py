"""Collapsible panels that make up the main window.

Each :class:`Panel` has a header with a child-friendly title, an
expand/collapse button and a "What's this?" button, and a content area. The
six panels (DESIGN.md §10 and §11) are defined in :data:`PANELS`; their content is
filled in by later tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class PanelSpec:
    key: str
    title: str
    blurb: str  # one plain-English line shown while the panel is empty


WHATS_THIS = "What's this?"

PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("game", "Game", "This is where you will see the game as the AI plays it."),
    PanelSpec(
        "activity", "What the AI is doing", "The AI's current move and what it is trying to do."
    ),
    PanelSpec(
        "rewards", "Rewards", "Points the AI earns or loses, and the sliders to change them."
    ),
    PanelSpec("progress", "Progress", "How far the AI has got and how often it succeeds."),
    PanelSpec("learning", "Learning", "How the AI is getting better over time."),
    PanelSpec("compare", "Compare AIs", "Save your AIs by name and see which one does best."),
)


class Panel(QFrame):
    """A titled, collapsible box."""

    toggled = Signal(bool)  # emitted with the new expanded state
    help_requested = Signal(str)  # emitted with the panel key when "What's this?" is pressed

    def __init__(self, spec: PanelSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._expanded = True
        self.setObjectName(f"panel_{spec.key}")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._button = QToolButton(self)
        self._button.setObjectName(f"panel_{spec.key}_toggle")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.ArrowType.DownArrow)
        self._button.setText(spec.title)
        self._button.setCheckable(True)
        self._button.setChecked(True)
        self._button.setAutoRaise(True)
        self._button.clicked.connect(self._on_clicked)

        self._help = QToolButton(self)
        self._help.setObjectName(f"panel_{spec.key}_help")
        self._help.setText(WHATS_THIS)
        self._help.setToolTip(f"Explain the {spec.title} panel in plain words")
        self._help.setAutoRaise(True)
        self._help.clicked.connect(lambda _=False: self.help_requested.emit(self._spec.key))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._button)
        header.addStretch(1)
        header.addWidget(self._help)

        self._content = QWidget(self)
        self._content.setObjectName(f"panel_{spec.key}_content")
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        self._placeholder = QLabel(spec.blurb, self._content)
        self._placeholder.setObjectName(f"panel_{spec.key}_placeholder")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self._placeholder)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.addLayout(header)
        outer.addWidget(self._content, 1)

    # ------------------------------------------------------------- API

    @property
    def spec(self) -> PanelSpec:
        return self._spec

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def title(self) -> str:
        return self._spec.title

    @property
    def expanded(self) -> bool:
        return self._expanded

    @property
    def content(self) -> QWidget:
        """The area later tasks fill with real widgets."""
        return self._content

    def set_body(self, widget: QWidget) -> None:
        """Replace the placeholder with ``widget``."""
        layout = self._content.layout()
        assert layout is not None
        if self._placeholder is not None:
            layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None  # type: ignore[assignment]
        widget.setParent(self._content)
        layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._button.setChecked(expanded)
        self._button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding if expanded else QSizePolicy.Policy.Maximum,
        )
        self.toggled.emit(expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    @property
    def toggle_button(self) -> QToolButton:
        return self._button

    @property
    def help_button(self) -> QToolButton:
        return self._help

    def _on_clicked(self, checked: bool) -> None:
        self.set_expanded(checked)


__all__ = ["PANELS", "WHATS_THIS", "Panel", "PanelSpec"]
