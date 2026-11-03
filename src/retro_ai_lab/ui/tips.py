"""The "What's this?" window: one tip at a time, in simple or grown-up words.

One :class:`TipDialog` is shared by every panel. It lists every tip, remembers
which kind of words were chosen for as long as the app runs, and lets the reader
hop between related tips.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.education import TIPS, tip_for

SIMPLE_WORDS = "Use grown-up words"
SEE_ALSO = "See also:"


class TipDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tip_dialog")
        self.setWindowTitle("What's this?")
        self.setModal(False)
        self.resize(560, 360)
        self._key: str | None = None
        self._advanced = False
        self._updating = False

        self._topics = QListWidget(self)
        self._topics.setObjectName("tip_topics")
        for tip in TIPS.values():
            item = QListWidgetItem(tip.title)
            item.setData(Qt.ItemDataRole.UserRole, tip.key)
            self._topics.addItem(item)
        self._topics.setMaximumWidth(200)
        self._topics.currentItemChanged.connect(self._on_topic_chosen)

        self._title = QLabel("", self)
        self._title.setObjectName("tip_title")
        self._title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._title.setWordWrap(True)
        self._body = QLabel("", self)
        self._body.setObjectName("tip_body")
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._level = QCheckBox(SIMPLE_WORDS, self)
        self._level.setObjectName("tip_level")
        self._level.toggled.connect(self.set_advanced)
        self._see_also_label = QLabel(SEE_ALSO, self)
        self._see_also_label.setObjectName("tip_see_also_label")
        self._see_also = QHBoxLayout()
        self._see_also.setContentsMargins(0, 0, 0, 0)
        self._see_also.addWidget(self._see_also_label)
        self._see_also.addStretch(1)
        self._see_also_buttons: list[QPushButton] = []

        right = QVBoxLayout()
        right.addWidget(self._title)
        right.addWidget(self._body, 1)
        right.addLayout(self._see_also)
        right.addWidget(self._level)
        layout = QHBoxLayout(self)
        layout.addWidget(self._topics)
        layout.addLayout(right, 1)

    # ------------------------------------------------------------- API

    @property
    def current_key(self) -> str | None:
        return self._key

    @property
    def advanced(self) -> bool:
        return self._advanced

    @property
    def title_text(self) -> str:
        return self._title.text()

    @property
    def body_text(self) -> str:
        return self._body.text()

    def see_also_keys(self) -> tuple[str, ...]:
        return tuple(str(b.property("tip_key")) for b in self._see_also_buttons)

    def topic_keys(self) -> tuple[str, ...]:
        return tuple(
            str(self._topics.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._topics.count())
        )

    def show_tip(self, key: str) -> None:
        """Show ``key`` (raising for an unknown one) and bring the window up."""
        tip = tip_for(key)
        self._key = tip.key
        self._updating = True
        try:
            for i in range(self._topics.count()):
                if self._topics.item(i).data(Qt.ItemDataRole.UserRole) == tip.key:
                    self._topics.setCurrentRow(i)
                    break
        finally:
            self._updating = False
        self._render()
        self.show()
        self.raise_()
        self.activateWindow()

    def set_advanced(self, advanced: bool) -> None:
        advanced = bool(advanced)
        if advanced != self._advanced:
            self._advanced = advanced
            self._level.setChecked(advanced)
        self._render()

    def follow_see_also(self, index: int) -> None:
        self.show_tip(self.see_also_keys()[index])

    # ------------------------------------------------------------- internals

    def _render(self) -> None:
        if self._key is None:
            return
        tip = TIPS[self._key]
        self._title.setText(tip.title)
        self._body.setText(tip.text(self._advanced))
        for button in self._see_also_buttons:
            self._see_also.removeWidget(button)
            button.deleteLater()
        self._see_also_buttons.clear()
        for position, other in enumerate(tip.see_also):
            button = QPushButton(TIPS[other].title, self)
            button.setObjectName(f"tip_see_also_{position}")
            button.setProperty("tip_key", other)
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=other: self.show_tip(k))
            self._see_also.insertWidget(1 + position, button)
            self._see_also_buttons.append(button)
        self._see_also_label.setVisible(bool(tip.see_also))

    def _on_topic_chosen(self, current: QListWidgetItem | None, _: object = None) -> None:
        if self._updating or current is None:
            return
        self.show_tip(str(current.data(Qt.ItemDataRole.UserRole)))


__all__ = ["SEE_ALSO", "SIMPLE_WORDS", "TipDialog"]
