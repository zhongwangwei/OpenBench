from openbench.config.user_settings import load_user_settings, save_user_settings
from openbench.gui.localization import CHINESE, ENGLISH, LanguageManager, translate_text


def test_language_switch_is_reversible_and_persistent(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    save_user_settings({"reference_root": "/data/reference"}, tmp_path)
    manager = LanguageManager(user_dir=tmp_path)
    root = QWidget()
    layout = QVBoxLayout(root)
    title = QLabel("General Settings")
    next_button = QPushButton("Next")
    count = QLabel("Selected: 2 / 5")
    layout.addWidget(title)
    layout.addWidget(next_button)
    layout.addWidget(count)

    manager.set_language(CHINESE)
    manager.apply(root)
    assert (title.text(), next_button.text(), count.text()) == ("常规设置", "下一步", "已选择：2 / 5")
    assert load_user_settings(tmp_path) == {"gui_language": CHINESE, "reference_root": "/data/reference"}

    count.setText("Selected: 3 / 5")
    manager.apply(root)
    assert count.text() == "已选择：3 / 5"

    manager.set_language(ENGLISH)
    manager.apply(root)
    assert (title.text(), next_button.text(), count.text()) == ("General Settings", "Next", "Selected: 3 / 5")
    assert LanguageManager(user_dir=tmp_path).language == ENGLISH
    assert translate_text("Step 3 of 12", CHINESE) == "第 3 步，共 12 步"


def test_show_event_respects_i18n_skip(qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QPushButton

    manager = LanguageManager(persist=False)
    manager.language = CHINESE
    button = QPushButton("New")
    button.setProperty("i18n_skip", True)

    manager.eventFilter(button, QEvent(QEvent.Show))

    assert button.text() == "New"


def test_main_window_language_button_updates_existing_pages(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLabel

    old_manager = getattr(qapp, "_openbench_language_manager", None)
    if old_manager is not None:
        qapp.removeEventFilter(old_manager)
    manager = LanguageManager(persist=False)
    qapp._openbench_language_manager = manager
    qapp.installEventFilter(manager)

    from openbench.gui.main_window import MainWindow

    window = None
    try:
        window = MainWindow()
        from PySide6.QtCore import Qt

        assert window.btn_language.text() == "中文"
        assert window.btn_language.parent() is window.title_label.parent()
        assert window.title_label.alignment() & Qt.AlignHCenter
        assert not window.logo_label.pixmap().isNull()
        assert not window.windowIcon().isNull()
        assert window.copyright_label.text() == (
            "Copyright: CoLM LSM Development Team, School of Atmospheric Sciences, SYSU"
        )
        window.btn_language.click()
        assert window.windowTitle() == "OpenBench NML 配置向导"
        assert window.btn_language.text() == "English"
        assert window.btn_about.text() == "关于"
        assert window.copyright_label.text() == "版权所有：CoLM陆面模式开发团队，中山大学大气科学学院"
        assert window.pages["registry"].tabs.tabBar().expanding() is False
        assert "alignment: left" in window.pages["registry"].tabs.styleSheet()
        assert window.btn_next.text() == "下一步"
        assert window.nav_list.item(0).text() == "运行环境"

        monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)
        window.btn_about.click()
        about = next(dialog for dialog in window.findChildren(QDialog) if dialog.windowTitle() == "关于 OpenBench")
        about_text = "\n".join(label.text() for label in about.findChildren(QLabel))
        assert "版权所有：CoLM陆面模式开发团队，中山大学大气科学学院" in about_text
        assert "开发与维护\nCoLM陆面模式开发团队" in about_text
        assert "联系人\n魏忠旺" in about_text

        window.btn_language.click()
        assert window.windowTitle() == "OpenBench NML Wizard"
        assert window.btn_next.text() == "Next"
    finally:
        qapp.removeEventFilter(manager)
        if window is not None:
            window.close()
            window.deleteLater()
        qapp.processEvents()
        if old_manager is None:
            del qapp._openbench_language_manager
        else:
            qapp._openbench_language_manager = old_manager
            qapp.installEventFilter(old_manager)
