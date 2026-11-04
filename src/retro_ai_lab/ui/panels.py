"""Collapsible panels that make up the main window.

Each :class:`Panel` has a header with a child-friendly title, an
expand/collapse button and a "What's this?" button, and a content area. The
six panels (DESIGN.md §10 and §11) are defined in :data:`PANELS`; their content is
filled in by later tasks.

The header is a widget of its own (T-881): the main window makes it the dock's
title bar, so a panel shows its name once, and wires its "pop out" and "hide"
buttons to the dock. A panel without a dock keeps the header at its top.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

NAME_COLUMN_WIDTH = 280  # the first column holds the words a person reads (T-856)


def name_column(tree: QTreeWidget, width: int = NAME_COLUMN_WIDTH) -> None:
    """Make a table's first column wide and draggable, with the last taking the rest.

    A ``Stretch`` first column cannot be dragged and shares the spare width with
    the stretched last column, so it came out narrow and cut the names short.
    """
    header = tree.header()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
    header.resizeSection(0, width)
    header.setStretchLastSection(True)


@dataclass(frozen=True, slots=True)
class PanelSpec:
    key: str
    title: str
    blurb: str  # one plain-English line shown while the panel is empty


WHATS_THIS = "What's this?"
POP_OUT = "Pop this panel out into its own window"
DOCK_BACK = "Put this panel back in the main window"
HIDE_PANEL = "Hide this panel (bring it back from the Panels menu)"

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
    pop_out_requested = Signal()  # the header's pop-out button, for the dock to act on
    hide_requested = Signal()  # the header's hide button, for the dock to act on

    def __init__(self, spec: PanelSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._expanded = True
        self.setObjectName(f"panel_{spec.key}")
        self.setProperty("panel", spec.key)  # the stylesheet colours each panel by its key
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._header = QWidget(self)
        self._header.setObjectName(f"panel_{spec.key}_header")
        self._header.setProperty("panel", spec.key)
        self._header.setProperty("panelHeader", True)
        self._header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._button = QToolButton(self._header)
        self._button.setObjectName(f"panel_{spec.key}_toggle")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.ArrowType.DownArrow)
        self._button.setText(spec.title)
        self._button.setCheckable(True)
        self._button.setChecked(True)
        self._button.setAutoRaise(True)
        self._button.clicked.connect(self._on_clicked)

        self._help = QToolButton(self._header)
        self._help.setObjectName(f"panel_{spec.key}_help")
        self._help.setText(WHATS_THIS)
        self._help.setToolTip(f"Explain the {spec.title} panel in plain words")
        self._help.setAutoRaise(True)
        self._help.clicked.connect(lambda _=False: self.help_requested.emit(self._spec.key))

        self._pop_out = QToolButton(self._header)
        self._pop_out.setObjectName(f"panel_{spec.key}_pop_out")
        self._pop_out.setAutoRaise(True)
        self._pop_out.setText("⧉")
        self._pop_out.setToolTip(POP_OUT)
        self._pop_out.setVisible(False)
        self._pop_out.clicked.connect(lambda _=False: self.pop_out_requested.emit())
        self._hide = QToolButton(self._header)
        self._hide.setObjectName(f"panel_{spec.key}_hide")
        self._hide.setAutoRaise(True)
        self._hide.setText("✕")
        self._hide.setToolTip(HIDE_PANEL)
        self._hide.setVisible(False)
        self._hide.clicked.connect(lambda _=False: self.hide_requested.emit())

        header = QHBoxLayout(self._header)
        header.setContentsMargins(4, 2, 4, 2)
        header.addWidget(self._button)
        header.addStretch(1)
        header.addWidget(self._help)
        header.addWidget(self._pop_out)
        header.addWidget(self._hide)

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
        outer.addWidget(self._header)
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
    def header(self) -> QWidget:
        """The title row: collapse, "What's this?", and (in a dock) pop out and hide."""
        return self._header

    @property
    def pop_out_button(self) -> QToolButton:
        return self._pop_out

    @property
    def hide_button(self) -> QToolButton:
        return self._hide

    def show_dock_buttons(self) -> None:
        """Show pop out and hide; the dock that holds the panel connects them."""
        self._pop_out.setVisible(True)
        self._hide.setVisible(True)

    def set_floating(self, floating: bool) -> None:
        """Make the pop-out button say what pressing it will do."""
        self._pop_out.setToolTip(DOCK_BACK if floating else POP_OUT)
        self._pop_out.setText("⇲" if floating else "⧉")

    @property
    def help_button(self) -> QToolButton:
        return self._help

    def _on_clicked(self, checked: bool) -> None:
        self.set_expanded(checked)


__all__ = [
    "DOCK_BACK",
    "HIDE_PANEL",
    "NAME_COLUMN_WIDTH",
    "PANELS",
    "POP_OUT",
    "WHATS_THIS",
    "Panel",
    "PanelSpec",
    "name_column",
]
