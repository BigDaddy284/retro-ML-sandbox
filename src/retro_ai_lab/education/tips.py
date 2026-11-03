"""Plain-words explanations for every panel and every idea DESIGN.md §12 lists.

Each :class:`Tip` has two tellings of the same thing: ``simple`` for a child of
about eight, and ``advanced`` for older children and adults. The texts are data,
not UI, so they can be shown in a dialog, a tooltip or a printed sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError


@dataclass(frozen=True, slots=True)
class Tip:
    key: str
    title: str
    simple: str
    advanced: str
    see_also: tuple[str, ...] = ()

    def text(self, advanced: bool) -> str:
        return self.advanced if advanced else self.simple


_PANEL_TIPS: tuple[Tip, ...] = (
    Tip(
        "panel.game",
        "The Game window",
        "This is the game. You can play it yourself with the buttons or the keyboard, "
        "and when the AI is learning you watch it play here instead. The clock shows "
        "how long the game has been running.",
        "The Game panel shows frames captured from the emulator and sends controller "
        "input to it. In manual mode your presses go to the game; during independent "
        "training the AI holds the controller and the pad is locked so the two don't "
        "fight. The frame rate shown is the capture rate, not the emulator's speed.",
        ("artificial_intelligence", "training"),
    ),
    Tip(
        "panel.activity",
        "What the AI is doing",
        "The AI starts by looking at the game very closely. It watches the numbers "
        "inside the game change and works out which ones matter, like lives, points "
        "and speed. Then it tells you what it is pressing and why.",
        "The understanding engine samples game memory every frame, keeps statistics "
        "per byte, and proposes variables (score, lives, position, flags, counters) "
        "with evidence and a confidence level. It links buttons to effects with "
        "save-state experiments and detects events such as progress, setbacks and "
        "resets. During training this panel also shows each decision: whether the AI "
        "explored a new move or took its best known one.",
        ("decision_making", "exploration", "machine_learning"),
    ),
    Tip(
        "panel.rewards",
        "Rewards",
        "Rewards are how you tell the AI what you want. Good things earn points and "
        "bad things lose points. Move the sliders to change what counts, and the AI "
        "will change how it plays.",
        "A reward profile maps detected events (score gain, setback, progress, reset, "
        "flags) to numeric rewards. The learner maximises the discounted sum of these, "
        "so their relative sizes shape the policy more than their absolute values. The "
        "warning line reports suspected reward exploitation: a reward earned far more "
        "often than the goal advances.",
        ("rewards", "reward_exploitation"),
    ),
    Tip(
        "panel.progress",
        "Progress",
        "A goal is a list of steps, like reaching a checkpoint or finishing a lap. The "
        "AI suggests steps it has noticed and you can add your own. This panel shows "
        "which steps are done and how often the AI reaches them.",
        "Objectives are triggered by detected events, can depend on each other, and "
        "carry a completion reward. The statistics come from the learning curve: tries, "
        "the share that completed at least one objective, and the trend of recent "
        "rewards against earlier ones. Ordering steps builds a simple hierarchy of "
        "sub-goals.",
        ("hierarchical_learning", "training"),
    ),
    Tip(
        "panel.learning",
        "Learning",
        "Here you train the AI. Pick how it should play, press Start and watch the "
        "graph. Each dot is one go. When the dots climb, the AI is getting better. "
        "The lines below say how its habits changed and what to try next.",
        "Training runs episodes: the game is restored to a checkpoint, the agent acts "
        "every few frames, and the episode ends on a reset, a cap on steps or Stop. The "
        "graph plots reward per episode with a rolling average; the trend compares the "
        "latest window with the one before. Behaviour changes compare the greedy policy "
        "now with the policy a window ago. The brain is saved per game.",
        ("reinforcement_learning", "training", "exploration"),
    ),
)

_TOPIC_TIPS: tuple[Tip, ...] = (
    Tip(
        "artificial_intelligence",
        "Artificial intelligence",
        "Artificial intelligence, or AI, is a computer program that can work things out "
        "for itself instead of being told every single step. Here, the AI learns to "
        "play a game nobody explained to it.",
        "AI is the broad field of making machines perform tasks that need some form of "
        "judgement: perceiving, reasoning, planning, learning. This sandbox uses a "
        "narrow, learning-based form: it has no knowledge of any game built in, and "
        "everything it knows about a game it discovers by observing memory and acting.",
        ("machine_learning",),
    ),
    Tip(
        "machine_learning",
        "Machine learning",
        "Machine learning means getting better at something by practising, not by being "
        "given the rules. The AI tries things, sees what happens, and remembers what "
        "worked.",
        "Machine learning fits a model to experience rather than hand-written rules. "
        "The three classic families are supervised (learn from labelled examples), "
        "unsupervised (find structure in data) and reinforcement learning (learn from "
        "rewards by acting). Retro AI Lab is a reinforcement learning sandbox.",
        ("reinforcement_learning", "neural_networks"),
    ),
    Tip(
        "reinforcement_learning",
        "Reinforcement learning",
        "Reinforcement learning is learning from rewards, a bit like training a puppy. "
        "Do something good, get a treat. Do something bad, no treat. After lots of "
        "tries the AI learns which moves get the most treats.",
        "An agent observes a state, picks an action, receives a reward and a new state, "
        "and aims to maximise the total discounted reward over time. The agent here is "
        "tabular Q-learning: it keeps a table of how good each move is in each "
        "situation and nudges the entries towards what it actually got, plus what it "
        "expects next.",
        ("rewards", "exploration", "decision_making"),
    ),
    Tip(
        "rewards",
        "Rewards and penalties",
        "A reward is points for doing something good, like scoring. A penalty is losing "
        "points for something bad, like crashing. The AI only cares about the points "
        "you give it, so choose them carefully!",
        "Rewards are the only signal the agent optimises, so the profile is the "
        "specification of the task. Sparse rewards make learning slow; dense shaping "
        "rewards speed it up but can be gamed. Penalties are just negative rewards; "
        "their size relative to positive rewards decides how cautious the policy is.",
        ("reward_exploitation", "reinforcement_learning"),
    ),
    Tip(
        "neural_networks",
        "Neural networks",
        "A neural network is a kind of brain made of maths. It has lots of tiny parts "
        "that each pass on a little signal, and together they can spot patterns. The "
        "AI here uses a simple table instead, but bigger AIs use networks.",
        "A neural network is a stack of layers of weighted sums with non-linear "
        "activations, trained by gradient descent. In reinforcement learning it can "
        "replace the Q-table so the agent generalises across similar states. The "
        "current learner is tabular; a network-based learner is a planned extension "
        "once the state space outgrows the table.",
        ("machine_learning",),
    ),
    Tip(
        "training",
        "Training and experimentation",
        "Training is practice. Every go, the AI starts the game again, plays, and "
        "remembers what happened. Experimenting means changing something, like a "
        "reward, and seeing whether the AI plays differently.",
        "Training is a sequence of episodes from a checkpoint with learning switched "
        "on. Experimentation is the scientific loop around it: change one thing (a "
        "reward, a personality, a goal), train, and compare the learning curves. Fast "
        "mode runs many episodes a second; live mode shows each move.",
        ("panel.learning", "rewards"),
    ),
    Tip(
        "decision_making",
        "Decision-making",
        "Every few frames the AI has to decide which buttons to press. It looks at the "
        "situation, checks which move earned the most points before, and usually picks "
        "that one.",
        "At each step the agent encodes the observed variables into a discrete state, "
        "looks up the estimated value of every action in that state, and acts greedily "
        "with probability 1 - epsilon or randomly otherwise. The activity panel reports "
        "which of the two it did and the value it expected.",
        ("exploration", "reinforcement_learning"),
    ),
    Tip(
        "exploration",
        "Exploring versus using what you know",
        "If the AI only ever did what worked before, it would never find anything "
        "better. So sometimes it tries a random move just to see. That's exploring. "
        "The rest of the time it uses what it knows.",
        "Exploration versus exploitation is the central trade-off in reinforcement "
        "learning. The agent uses epsilon-greedy action selection: epsilon starts high "
        "and decays per episode towards a floor, so early goes explore widely and later "
        "goes mostly exploit. The personalities change this schedule.",
        ("decision_making", "training"),
    ),
    Tip(
        "reward_exploitation",
        "Reward exploitation",
        "Sometimes the AI finds a sneaky way to get points without really playing "
        "well, like driving in circles to collect the same bonus. That's reward "
        "exploitation. If you see it, change the rewards so the trick stops paying.",
        "Reward exploitation (reward hacking) happens when the reward profile and the "
        "intended task diverge: the agent optimises the proxy. The detector flags a "
        "reward that fires far more often than the goal advances. Fixes are to lower or "
        "disable the farmed reward, reward progress instead of the proxy, or add a "
        "penalty for the loop.",
        ("rewards", "panel.rewards"),
    ),
    Tip(
        "hierarchical_learning",
        "Learning in steps",
        "Big goals are easier in small steps. Finish a lap? First reach the first "
        "checkpoint, then the next. The AI works on one step at a time, and each step "
        "it finishes earns a reward.",
        "Hierarchical learning decomposes a long-horizon task into sub-goals with their "
        "own rewards, so the agent gets feedback long before the final goal. In this "
        "sandbox the goal list with dependencies is the hierarchy: the next ready "
        "objective is the current sub-goal and its completion reward is added to the "
        "episode's return.",
        ("panel.progress", "rewards"),
    ),
)

TIPS: dict[str, Tip] = {t.key: t for t in (*_PANEL_TIPS, *_TOPIC_TIPS)}
TOPIC_KEYS: tuple[str, ...] = tuple(t.key for t in _TOPIC_TIPS)
PANEL_TIP: dict[str, str] = {t.key.removeprefix("panel."): t.key for t in _PANEL_TIPS}


def tip_for(key: str) -> Tip:
    try:
        return TIPS[key]
    except KeyError:
        raise ConfigurationError(f"There is no tip called {key!r}.") from None


__all__ = ["PANEL_TIP", "TIPS", "TOPIC_KEYS", "Tip", "tip_for"]
