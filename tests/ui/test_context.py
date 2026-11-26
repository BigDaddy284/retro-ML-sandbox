def test_context_owns_the_reward_service(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert context.reward_store.folder == tmp_path / "rewards"
    assert context.rewards.store is context.reward_store
    assert context.rewards.profile is None


def test_context_owns_the_goal_service(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert context.goal_store.folder == tmp_path / "goals"
    assert context.goals.store is context.goal_store
    assert context.goals.goal is None


def test_context_owns_the_saved_ai_service(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert context.saved_ai_store.folder == tmp_path / "saved_ais"
    assert context.saved_ais.store is context.saved_ai_store
    assert not context.saved_ais.enabled and context.saved_ais.names() == ()


def test_context_names_the_folder_save_states_are_kept_in(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """T-1068: beside the brains and the ways, not in the temp folder."""
    from retro_ai_lab.emulator.bizhawk import BizHawkSimulator
    from retro_ai_lab.ui.context import STATES_FOLDER, AppContext

    context = AppContext(data_dir=tmp_path)
    assert STATES_FOLDER == "states"
    with BizHawkSimulator() as sim:
        context.connection.connect("bizhawk", port=sim.port)
        context.connection.call(lambda a: a.save_state("a"))
        saves = [r for r in sim.requests if r.get("cmd") == "save"]
        assert [r.get("dir") for r in saves] == [str(tmp_path / "states")]
        assert (tmp_path / "states").is_dir()
        context.connection.disconnect()
