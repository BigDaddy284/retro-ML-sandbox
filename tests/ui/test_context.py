def test_context_owns_the_reward_service(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert context.reward_store.folder == tmp_path / "rewards"
    assert context.rewards.store is context.reward_store
    assert context.rewards.profile is None
