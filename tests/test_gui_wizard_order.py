from openbench.gui.controller import WizardController


def test_gui_wizard_selects_evaluation_variables_before_ref_and_sim_data():
    controller = WizardController()

    visible = controller.get_visible_pages()

    assert visible.index("evaluation_items") < visible.index("ref_data") < visible.index("sim_data")


def test_next_navigation_follows_cli_config_order_through_data_pages():
    controller = WizardController()
    controller.current_page = "registry"

    assert controller.next_page() == "evaluation_items"
    assert controller.go_next() is True
    assert controller.current_page == "evaluation_items"
    assert controller.next_page() == "ref_data"
    assert controller.go_next() is True
    assert controller.current_page == "ref_data"
    assert controller.next_page() == "sim_data"
