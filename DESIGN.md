# RETRO AI LAB
## An Emulator-Based Machine Learning Sandbox

**Project:** Retro AI Lab  
**Application:** Standalone Windows desktop software  
**Primary audience:** Children and beginners, with advanced research capabilities  
**Core principle:** Teach AI through rewards, experimentation and gameplay.

---

# 1. Project Vision

Retro AI Lab is a standalone machine learning sandbox that allows children to train artificial intelligence to play retro video games.

The application connects to external game emulators, observes gameplay, identifies game information and learns how to interpret actions, events, objectives and progression.

Children interact with the AI through a friendly graphical interface.

They can:
- Set overall goals.
- Adjust rewards and penalties.
- Teach the AI what behaviours are desirable.
- Watch the AI learn through repeated gameplay.
- Optionally demonstrate actions themselves.
- Compare different AI personalities and learning strategies.
- Experiment with different machine learning algorithms.
- Save and share their trained AI.

The application handles the technical complexity behind the scenes.

Children should never need to write code, manipulate hexadecimal values or understand memory addresses to use the sandbox.

**The child decides what the AI should achieve. The AI learns how to achieve it.**

# 2. Fundamental Design Decisions

The following requirements are established:

1. Reward-based learning is the foundation.
2. Objectives, individual behaviours and optional demonstrations are supported.
3. No pre-built game profiles are required.
4. The sandbox must discover and interpret unfamiliar games itself.
5. Machine learning should progress from simple to advanced.
6. The main interface displays gameplay, AI decisions, rewards and learning progress together.
7. Both live and background training are supported.
8. Children can use categories, individual reward controls, AI personalities and suggested reward values.
9. Emulator support must be modular and independent of a specific emulator.
10. Overall goals can be broken into smaller objectives.
11. Independent, guided and manual operating modes are available.
12. The application should eventually support education, research, competitions and community-created extensions.
13. Claude should build the application autonomously, one task at a time, with mandatory testing and approval before progressing.

# 3. Game Understanding Engine

This is one of the most important and technically challenging parts of the application.

The sandbox must be capable of approaching a game it has never encountered before.

It must not depend on developers manually creating a game profile for every title.

Instead, it should contain its own Game Understanding Engine.

This engine observes games, investigates their behaviour and builds an understanding of their internal mechanics.

## 3.1 Level 1 – Game Observation

The application should:
- Read available emulator memory.
- Observe the game screen.
- Identify controller inputs.
- Detect changes in memory.
- Track player movement.
- Observe menus and game transitions.
- Identify changes between frames.

## 3.2 Level 2 – Game Interpretation

The application attempts to identify:
- Player health.
- Lives.
- Score.
- Experience.
- Inventory.
- Player position.
- Collectibles.
- Enemies.
- Game objectives.
- Quest progression.
- Mission completion.
- Character progression.

It should combine memory analysis, screen recognition, text recognition and observed gameplay.

## 3.3 Level 3 – Game Understanding

The application builds relationships between events.

For example:

- The player defeats an enemy.
- The player receives experience.
- Experience reaches a threshold.
- The character levels up.
- New abilities become available.

It should understand that these events are related rather than treating them as unrelated memory changes.

## 3.4 Active Discovery

The application should experiment with the game to improve its understanding.

For example:

1. Observe the game in its current state.
2. Make a controlled change, such as pressing a button.
3. Observe what happens.
4. Compare memory and screen changes.
5. Repeat the experiment.
6. Identify consistent relationships.
7. Assign a confidence score to the discovery.

Where supported, emulator save states should be used to repeat experiments safely.

The application should also request user assistance when it cannot confidently identify an event.

## 3.5 Confidence and Verification

The application must distinguish between confirmed information and guesses.

Example:

| Discovery | Confidence |
|---|---|
| Player score | Confirmed |
| Player health | High |
| Experience points | High |
| Quest progression | Medium |
| Hidden objective | Unknown |

Children should be able to inspect suggested discoveries and confirm or correct them.

The system must never pretend it understands something it has not verified.

## 3.6 No Pre-Built Game Profiles

There should be no requirement for a manually maintained collection of game profiles.

The sandbox creates and saves its own learned understanding as it explores games.

Its discoveries should be reusable, editable and exportable.

The application should be capable of improving its understanding over time.

# 4. Universal Game Objective System

This is the central feature that separates Retro AI Lab from basic game-playing AI experiments.

The application must understand that games contain meaningful objectives, not simply button presses.

It should support:

- Individual actions.
- Achievements.
- Collection.
- Exploration.
- Character progression.
- Quest progression.
- Story milestones.
- Mission objectives.
- Racing objectives.
- Survival objectives.
- Completion objectives.

## 4.1 Overall Goals

Children can select or create an overall goal.

Examples:

- Become Pokémon Champion.
- Complete a Zelda adventure.
- Complete every GoldenEye mission.
- Win a racing championship.
- Discover everything in a game.

The AI should attempt to break these larger goals into smaller achievable objectives where it has sufficient information.

## 4.2 Individual Objectives

Each objective should have:
- A child-friendly description.
- A detection method.
- A reward value.
- An optional penalty.
- A completion status.
- A confidence level.
- An enabled/disabled setting.

## 4.3 Objective Relationships

Objectives should support dependencies.

For example:

Become Pokémon Champion:

1. Obtain a starter Pokémon.
2. Build a suitable team.
3. Train Pokémon.
4. Earn Gym badges.
5. Defeat the Elite Four.
6. Become Champion.

The software should identify these relationships where possible.

It should also allow children to edit the objective structure.

# 5. Reward and Penalty System

Rewards are the primary way children teach their AI.

Every detected action or achievement that can be meaningfully evaluated should have an adjustable reward.

The interface should use simple positive and negative controls.

## 5.1 Reward Controls

Each behaviour should have:
- Positive reward (+).
- Negative reward (-).
- Current reward value.
- Enable/disable toggle.
- Suggested reward value.

Children should be able to increase or decrease rewards without seeing the underlying mathematics.

## 5.2 Reward Categories

The interface should organise discovered actions into categories.

Suggested categories:

- Exploration.
- Collection.
- Combat.
- Survival.
- Progression.
- Quests and Story.
- Efficiency.
- Completion.

Categories should expand to show individual actions.

## 5.3 AI Personalities

Provide starting configurations:

- Explorer.
- Collector.
- Completionist.
- Speedrunner.
- Battler.
- Survivor.
- Balanced Player.

These are reward configurations rather than completely separate AI models.

Children can change every setting.

## 5.4 Automatic Reward Suggestions

The software can recommend reward values based on objective importance and relationships.

For example, completing a major mission should generally carry more significance than collecting an ordinary item.

Recommendations must remain editable.

## 5.5 Reward Exploitation

The application must identify situations where an AI appears to exploit its reward system.

For example:

A Pokémon AI might learn to repeatedly perform an action that gives experience but never progresses towards becoming Champion.

The application should flag this behaviour and explain why it might be happening.

# 6. Machine Learning Engine

The learning engine must be modular.

Different games and learning problems require different approaches.

## 6.1 Progressive Learning

Start with relatively simple learning algorithms.

**Beginner:**
- Reward-based learning.
- Simple state/action learning.
- Tabular Q-learning for suitable small environments.

**Intermediate:**
- Deep Q-Networks.
- Neural networks.
- Experience replay.
- More complex state recognition.

**Advanced:**
- Hierarchical reinforcement learning.
- Multiple learning strategies.
- More sophisticated neural networks.
- Custom learning environments.

The application should select appropriate algorithms automatically by default.

Advanced users can experiment with different approaches.

## 6.2 Hierarchical Learning

The AI should eventually learn smaller skills that contribute towards larger objectives.

For example:

- Learn how to navigate.
- Learn how to collect items.
- Learn how to win battles.
- Learn how to complete missions.
- Combine these abilities to complete an entire game.

This allows the application to approach complicated games as collections of smaller learning problems.

## 6.3 Learning From Demonstrations

Demonstration-based learning should be optional.

Children can play the game themselves and show the AI useful behaviours.

The software can record:
- Inputs.
- Observations.
- Resulting actions.
- Changes in game state.
- Associated rewards.

These demonstrations can help initialise learning or guide the AI.

Children must never be required to demonstrate gameplay.

# 7. Emulator Integration

The application must not be tied to one emulator.

Use an adapter architecture.

Each adapter should expose a common interface for:

- Connecting to an emulator.
- Reading available memory.
- Capturing game screens.
- Sending controller inputs.
- Advancing frames.
- Controlling game speed.
- Managing save states where supported.
- Reporting emulator capabilities.

Initial development should target practical retro emulators, including:

- BizHawk / EmuHawk.
- VisualBoyAdvance / VBA-M.

Future integrations should include other emulators where suitable interfaces are available.

The application should detect which capabilities are available rather than assuming every emulator supports every feature.

# 8. Training Modes

Children should have three operating modes.

## Independent Mode

The AI:
- Controls the game.
- Makes its own decisions.
- Experiments with different actions.
- Receives rewards and penalties.
- Learns independently.

## Guided Mode

The AI plays independently, but children can:
- Change rewards.
- Redirect objectives.
- Pause training.
- Offer suggestions.
- Intervene when necessary.

## Manual Mode

Children control the game.

They can demonstrate actions and allow the AI to observe and learn from their decisions.

Demonstration is optional and can be combined with the other modes.

# 9. Training Environment

Support both live gameplay and background training.

## Live Training

Children watch the AI play in real time.

They can see its decisions and how rewards affect its behaviour.

## Background Training

The AI practises at higher speeds without requiring the child to watch every attempt.

Results can be reviewed afterwards.

## Checkpoint Training

Where emulator capabilities permit, the application should use save states to practise specific sections repeatedly.

This is particularly useful for:
- Difficult platforming sections.
- Boss battles.
- Racing circuits.
- Difficult missions.
- Repeated objectives.

# 10. Main Application Interface

The interface should be colourful, modern and approachable.

It should combine:

**Game Window**
- Live emulator screen.
- Current game state.
- AI controller status.

**AI Activity**
- Current action.
- Previous actions.
- Current objective.
- AI decisions.

**Rewards**
- Earned points.
- Penalties.
- Reward breakdown.
- Configurable actions.

**Progress**
- Overall objective completion.
- Individual achievements.
- Training attempts.
- Success rate.
- Performance trends.

**Learning**
- Learning progress.
- AI behaviour changes.
- Identified improvements.
- Suggested experiments.

All major information should be visible together, with the ability to expand individual sections.

# 11. Experiments and Comparison

Children should be able to save different AI configurations.

For example:

**AI A:**
- Prioritises exploration.
- Collects items.
- Avoids unnecessary battles.

**AI B:**
- Prioritises combat.
- Gains experience.
- Focuses on progression.

Both can be trained on the same game.

The application should compare:
- Rewards earned.
- Objectives completed.
- Time taken.
- Number of attempts.
- Successful outcomes.
- Different learned behaviours.

Children should be encouraged to experiment and discover why different reward configurations produce different results.

# 12. Educational Features

The application should teach machine learning through practical experimentation.

Topics should include:

- Artificial intelligence.
- Machine learning.
- Reinforcement learning.
- Rewards and penalties.
- Neural networks.
- Training and experimentation.
- Decision-making.
- Exploration versus exploitation.
- Reward exploitation.
- Hierarchical learning.

The application should explain technical concepts using simple language.

Advanced explanations should be available for older children and adults.

# 13. Competition and Sharing

The long-term platform should allow children to share trained AI configurations and compete.

Potential features:

- Share AI projects.
- Compare trained AI models.
- Compete on the same game.
- Compare completion times.
- Compare achievement totals.
- Share experiments.
- Community-created learning environments.

Competition should use consistent game versions, settings and starting conditions wherever possible.

Imported projects and models must be handled safely.

# 14. Technical Architecture

Suggested technology:

**Desktop Application**
- Python.
- PySide6 / Qt.

**Machine Learning**
- PyTorch.
- Gymnasium-compatible environments.
- Stable-Baselines3 or custom algorithms.

**Game Observation**
- Emulator APIs.
- Lua and C# interfaces where available.
- Memory analysis.
- Screen capture.
- Image recognition.
- OCR.

**Game Understanding**
- Event detection.
- State recognition.
- Objective discovery.
- Relationship detection.
- Confidence scoring.
- Learned game knowledge storage.

**Data Storage**
- SQLite.
- JSON.
- Model checkpoints.
- Training histories.

The application should be modular, allowing individual components to be replaced without rebuilding the entire system.

# 15. Safety and Privacy

- Local-first processing.
- No requirement for internet connectivity.
- No supplied game ROMs.
- Users provide legally obtained games.
- No modification of original ROM files.
- Safe handling of imported projects.
- No unrestricted execution of third-party scripts.
- No unnecessary collection of personal information.

# 16. Development Roadmap

Development should proceed in independently testable milestones.

**Phase 1 – Foundation**
- Windows desktop application.
- Main interface.
- Emulator adapter architecture.
- Connect to one emulator.
- Read game information.
- Capture screen.
- Send controller inputs.

**Phase 2 – Game Observation**
- Memory discovery.
- Screen observation.
- Input recording.
- Change detection.
- Save-state integration.

**Phase 3 – Game Understanding**
- Identify possible game variables.
- Recognise game events.
- Identify achievements.
- Detect progression.
- Build objective relationships.
- Confidence and verification system.

**Phase 4 – Reward System**
- Reward and penalty controls.
- Categories.
- AI personalities.
- Suggested reward values.
- Overall goals.
- Individual objectives.

**Phase 5 – Basic Learning**
- Implement reinforcement learning.
- Train through repeated gameplay.
- Reward integration.
- Basic action selection.
- Save and reload learning progress.

**Phase 6 – Training Interface**
- Live training.
- Background training.
- Guided and manual modes.
- Learning graphs.
- Training history.

**Phase 7 – Advanced Learning**
- Neural networks.
- DQN.
- Hierarchical reinforcement learning.
- Optional demonstration learning.
- More sophisticated game understanding.

**Phase 8 – Expansion**
- Additional emulator adapters.
- Experiment comparison.
- Community sharing.
- AI competitions.
- Advanced research mode.

# 17. First Working Demonstration

The first milestone should prove that the sandbox can independently examine an unfamiliar game and learn to play it.

Do not start by attempting to complete a complicated Pokémon adventure.

Use a relatively simple retro game first.

The first demonstration must:

1. Connect to an emulator.
2. Observe the game.
3. Identify available controller inputs.
4. Examine accessible game memory.
5. Identify useful game state information.
6. Allow the child to configure rewards.
7. Train an AI to achieve a simple objective.
8. Show its decisions and learning progress.
9. Allow reward values to be changed.
10. Demonstrate that changing rewards can change AI behaviour.

Once this works, increase the complexity of the learning task.

A racing game is a good early test because its objectives are relatively easy to identify.

Pokémon should be an important later test because it introduces collection, progression, battles and story objectives.

GoldenEye 007 provides a useful advanced test involving 3D movement, mission objectives, combat and exploration.

# 18. Autonomous Development and Strict Quality Assurance

## 18.1 Development Philosophy

The project must be developed autonomously by Claude, with minimal user intervention.

Claude is responsible for:
- Planning each development task.
- Understanding the existing codebase.
- Implementing one task at a time.
- Creating its own rigorous test plans.
- Running tests and analysing failures.
- Fixing problems before proceeding.
- Maintaining documentation.
- Managing Git branches and checkpoints.
- Reporting blockers rather than silently working around them.

The user should not need to supervise every individual coding action.

Claude must work methodically, prioritising reliability and correctness over development speed.

## 18.2 One Task at a Time

Claude must never attempt to implement multiple development tasks simultaneously.

Every task must have:

1. A clearly defined objective.
2. A precise scope.
3. Defined acceptance criteria.
4. A list of expected files or components affected.
5. A test plan.
6. A completion report.

Claude must complete, test and approve the current task before starting another.

Tasks should be small enough that failures can be isolated and corrected without destabilising unrelated components.

## 18.3 Mandatory Test-First Development

Before implementing each task, Claude must create a comprehensive test plan.

The plan must cover all relevant areas, including:

- Expected functionality.
- Normal operating conditions.
- Invalid inputs.
- Unexpected inputs.
- Edge cases.
- Failure handling.
- Integration with existing components.
- Regression testing.
- Performance where relevant.
- Data integrity where relevant.

Claude must write or identify the necessary automated tests before considering the task complete.

Tests should be repeatable and independently executable.

## 18.4 Strict 100% Pass Requirement

A task is not complete until every mandatory test passes.

**Required result: 100% pass rate.**

The following are not acceptable:

- Skipped tests.
- Ignored failures.
- Disabled failing assertions.
- Weakening tests to make implementation pass.
- Hiding errors.
- Marking unfinished functionality as complete.
- Moving forward with known regressions.

If a test fails, Claude must:

1. Investigate the cause.
2. Identify whether the issue is in the implementation or test.
3. Correct the underlying problem.
4. Run the affected tests again.
5. Run the complete relevant regression suite.
6. Confirm the task passes before proceeding.

A test may only be excluded if Claude documents why it is genuinely inapplicable and records that exclusion explicitly. Exclusions must never be used to conceal a failure.

Where a requirement cannot be tested automatically, Claude must create and complete a documented manual verification procedure.

A 100% pass rate is only meaningful if the tests properly cover the acceptance criteria. Claude must not treat a weak test suite as proof of completion.

## 18.5 Git Branch and Checkpoint Strategy

Each completed task must be preserved in Git.

Recommended workflow:

1. Start from the latest approved development branch.
2. Create a dedicated task branch.
3. Implement only the assigned task.
4. Run the complete test plan.
5. Fix all failures.
6. Run regression tests.
7. Commit the completed task.
8. Record the commit hash and test results.
9. Create the next task branch from the approved checkpoint.
10. Begin the next task.

Never build the next task on top of an untested or failing implementation.

If a task fails repeatedly, Claude must stop and report the blocker rather than allowing the problem to spread into later tasks.

## 18.6 Autonomous Development Loop

Claude should follow this process continuously:

**PLAN → TEST PLAN → IMPLEMENT → TEST → FIX → RETEST → REGRESSION TEST → COMMIT → NEXT TASK**

No stage may be skipped.

Claude should maintain a central development tracker showing:

- Task ID.
- Task description.
- Current status.
- Acceptance criteria.
- Test count.
- Passed tests.
- Failed tests.
- Excluded tests, with justification.
- Commit hash.
- Outstanding issues.

## 18.7 Stop Conditions

Claude must stop development and request user input if:

- A requirement is ambiguous and a wrong decision could affect architecture.
- A major architectural decision is required that has not been authorised.
- An external dependency prevents reliable testing.
- A critical security issue is discovered.
- The existing codebase contains an unexpected blocker.
- The implementation cannot meet its acceptance criteria.

Claude must explain the issue, its impact and the recommended solution.

It must not silently change the project direction to get around a blocker.

## 18.8 Independent Quality Assurance

Claude must not simply mark its own work successful because the code compiles.

Each task must be checked against its original acceptance criteria.

Where practical, tests should verify actual behaviour rather than merely checking that functions or classes exist.

Claude must also check that:

- Existing functionality still works.
- New functionality integrates correctly.
- No unrelated functionality has been damaged.
- Documentation matches the actual implementation.
- The application can still be built and launched.

## 18.9 Overnight and Long-Running Development

The development process should be suitable for unattended, extended coding sessions.

Claude should:

- Work through the approved task list sequentially.
- Save progress frequently.
- Commit every fully tested task.
- Maintain a persistent progress log.
- Resume from the last approved checkpoint after interruption.
- Never assume an interrupted task passed its tests.
- Continue automatically only when all requirements are satisfied.

At the end of a development session, Claude must provide a concise report covering completed tasks, commits, test results, current task and any blockers.

## 18.10 Permanent Development Documents

At the beginning of development, Claude must create:

**DEVELOPMENT_RULES.md**
- Permanent development instructions.
- One-task-at-a-time requirement.
- Testing requirements.
- Git workflow.
- Stop conditions.

**TASK_TRACKER.md**
- Complete development task list.
- Current task.
- Completed tasks.
- Test results.
- Commit references.
- Outstanding blockers.

**TESTING_STRATEGY.md**
- Overall testing architecture.
- Unit testing.
- Integration testing.
- Emulator testing.
- Regression testing.
- Manual verification requirements.
- Quality gates.

These documents must be maintained throughout development and updated whenever the project changes.

## 18.11 Definition of Done

A task is only considered complete when:

- Its objective has been achieved.
- All acceptance criteria are satisfied.
- Its mandatory test plan passes 100%.
- Relevant regression tests pass.
- No known critical defects remain.
- Documentation has been updated.
- Changes have been committed to Git.
- The checkpoint is suitable for building upon.

**The next task must never begin until the current task meets every condition above.**

# 19. Definition of Project Success

Retro AI Lab succeeds when a child can:

1. Open an emulator and load a game.
2. Connect Retro AI Lab.
3. Let the software investigate the game.
4. Select an overall goal.
5. Adjust positive and negative rewards.
6. Press Train.
7. Watch the AI learn.
8. Experiment with different rewards.
9. See how its AI changes its behaviour.
10. Save and continue their experiment.

The child should not need to understand programming or hexadecimal to accomplish any of this.

**The ultimate ambition is to give children a genuine machine learning laboratory disguised as a fun, approachable gaming sandbox.**

It should be simple on the surface, technically capable underneath and expandable enough to become a serious AI experimentation platform.

The application should be built carefully, incrementally and autonomously, with every development task independently verified before the next begins.