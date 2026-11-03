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
