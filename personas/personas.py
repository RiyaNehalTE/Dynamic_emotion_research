"""
personas.py
-----------
20 psychological persona profiles for emotional trajectory conditioning.

Each persona has:
  - id          : unique integer 0-19
  - name        : short identifier
  - description : full psychological profile (fed to RoBERTa)
  - group       : psychological category
  - volatility  : expected drift level (low/medium/high)

These personas condition HOW emotion evolves over time,
not WHAT emotion is at each point.
"""

PERSONAS = [
    # ── SUPPRESSIVE GROUP (low drift expected) ─────────────────────────────────
    {
        "id":          0,
        "name":        "Stoic Regulator",
        "description": "Remains emotionally stable and calm. Rarely reacts intensely. Suppresses emotional responses and maintains composure under all circumstances.",
        "group":       "suppressive",
        "volatility":  "low",
    },
    {
        "id":          1,
        "name":        "Calm Reflective",
        "description": "Processes emotions slowly and thoughtfully. Responds in a measured and deliberate way. Takes time before reacting emotionally.",
        "group":       "suppressive",
        "volatility":  "low",
    },
    {
        "id":          2,
        "name":        "Detached Observer",
        "description": "Observes situations from a distance without personal emotional involvement. Analytical and objective. Rarely feels strong emotions.",
        "group":       "suppressive",
        "volatility":  "low",
    },
    {
        "id":          3,
        "name":        "Rational Analyzer",
        "description": "Approaches everything through logic and reason. Emotions are secondary to analysis. Prefers curiosity over emotional reactions.",
        "group":       "suppressive",
        "volatility":  "low",
    },

    # ── REACTIVE GROUP (high drift expected) ───────────────────────────────────
    {
        "id":          4,
        "name":        "Emotionally Volatile",
        "description": "Experiences rapid and intense emotional swings. Reacts strongly to events. Emotions shift quickly from one extreme to another.",
        "group":       "reactive",
        "volatility":  "high",
    },
    {
        "id":          5,
        "name":        "Highly Sensitive",
        "description": "Feels emotions deeply and intensely. Small events trigger strong emotional responses. Highly empathetic and emotionally aware.",
        "group":       "reactive",
        "volatility":  "high",
    },
    {
        "id":          6,
        "name":        "Easily Overwhelmed",
        "description": "Becomes anxious and stressed quickly. Struggles to manage multiple emotions simultaneously. Prone to emotional flooding.",
        "group":       "reactive",
        "volatility":  "high",
    },
    {
        "id":          7,
        "name":        "Impulsive Reactor",
        "description": "Acts and feels on impulse without reflection. Emotions drive behavior immediately. Quick to react, slow to regulate.",
        "group":       "reactive",
        "volatility":  "high",
    },

    # ── POSITIVE GROUP (upward trajectory expected) ────────────────────────────
    {
        "id":          8,
        "name":        "Optimistic Interpreter",
        "description": "Finds positive meaning in all situations. Reframes negative events as opportunities. Maintains hopeful outlook consistently.",
        "group":       "positive",
        "volatility":  "medium",
    },
    {
        "id":          9,
        "name":        "Gratitude Focused",
        "description": "Centers attention on what is good and appreciated. Experiences frequent gratitude and contentment. Finds joy in small things.",
        "group":       "positive",
        "volatility":  "medium",
    },
    {
        "id":          10,
        "name":        "Hope Oriented",
        "description": "Maintains strong belief in positive future outcomes. Motivated by anticipation of good things. Resilient in the face of setbacks.",
        "group":       "positive",
        "volatility":  "medium",
    },
    {
        "id":          11,
        "name":        "Reward Seeker",
        "description": "Driven by excitement and positive reinforcement. Seeks pleasure and stimulation. Emotions peak around anticipated rewards.",
        "group":       "positive",
        "volatility":  "medium",
    },

    # ── NEGATIVE GROUP (downward trajectory expected) ──────────────────────────
    {
        "id":          12,
        "name":        "Negative Anticipator",
        "description": "Expects negative outcomes in most situations. Prone to worry and anxiety about the future. Interprets ambiguous events negatively.",
        "group":       "negative",
        "volatility":  "medium",
    },
    {
        "id":          13,
        "name":        "Critical Evaluator",
        "description": "Judges situations and people harshly. Focuses on flaws and failures. Experiences frequent disappointment and frustration.",
        "group":       "negative",
        "volatility":  "medium",
    },
    {
        "id":          14,
        "name":        "Self Demanding",
        "description": "Sets impossibly high standards for self. Experiences anxiety about performance. Self-critical and prone to disappointment.",
        "group":       "negative",
        "volatility":  "medium",
    },
    {
        "id":          15,
        "name":        "Threat Aware",
        "description": "Constantly monitors environment for threats and risks. Interprets neutral events as potentially dangerous. Anxiety-driven perception.",
        "group":       "negative",
        "volatility":  "medium",
    },

    # ── BALANCED GROUP (moderate drift expected) ───────────────────────────────
    {
        "id":          16,
        "name":        "Emotionally Balanced",
        "description": "Maintains moderate emotional reactions. Recovers quickly from negative events. Neither suppresses nor amplifies emotions.",
        "group":       "balanced",
        "volatility":  "medium",
    },
    {
        "id":          17,
        "name":        "Empathy Driven",
        "description": "Feels others emotions deeply. Emotional state influenced strongly by those around them. Highly attuned to social emotional cues.",
        "group":       "balanced",
        "volatility":  "medium",
    },
    {
        "id":          18,
        "name":        "Socially Sensitive",
        "description": "Emotions shaped by social context and group dynamics. Adapts emotional expression to environment. Seeks emotional harmony.",
        "group":       "balanced",
        "volatility":  "medium",
    },
    {
        "id":          19,
        "name":        "Cautious Processor",
        "description": "Hesitant and careful in emotional expression. Takes time to identify and express feelings. Avoids emotional extremes deliberately.",
        "group":       "balanced",
        "volatility":  "medium",
    },
]

# ── Helper lookups ─────────────────────────────────────────────────────────────
PERSONA_BY_ID   = {p["id"]:   p for p in PERSONAS}
PERSONA_BY_NAME = {p["name"]: p for p in PERSONAS}

# Group mappings
SUPPRESSIVE_IDS = [p["id"] for p in PERSONAS if p["group"] == "suppressive"]
REACTIVE_IDS    = [p["id"] for p in PERSONAS if p["group"] == "reactive"]
POSITIVE_IDS    = [p["id"] for p in PERSONAS if p["group"] == "positive"]
NEGATIVE_IDS    = [p["id"] for p in PERSONAS if p["group"] == "negative"]
BALANCED_IDS    = [p["id"] for p in PERSONAS if p["group"] == "balanced"]

HIGH_VOLATILITY_IDS = [p["id"] for p in PERSONAS if p["volatility"] == "high"]
LOW_VOLATILITY_IDS  = [p["id"] for p in PERSONAS if p["volatility"] == "low"]

def get_persona_text(persona_id: int) -> str:
    """Returns the description text for a given persona id."""
    return PERSONA_BY_ID[persona_id]["description"]

def get_persona_group(persona_id: int) -> str:
    """Returns the group for a given persona id."""
    return PERSONA_BY_ID[persona_id]["group"]

def get_persona_volatility(persona_id: int) -> str:
    """Returns expected volatility level for a given persona id."""
    return PERSONA_BY_ID[persona_id]["volatility"]

if __name__ == "__main__":
    print("=== PERSONA SUMMARY ===")
    print(f"Total personas: {len(PERSONAS)}")
    print()
    for group in ["suppressive", "reactive", "positive", "negative", "balanced"]:
        group_personas = [p for p in PERSONAS if p["group"] == group]
        print(f"{group.upper()} ({len(group_personas)} personas):")
        for p in group_personas:
            print(f"  [{p['id']:2d}] {p['name']:<22} volatility={p['volatility']}")
    print()
    print(f"High volatility IDs : {HIGH_VOLATILITY_IDS}")
    print(f"Low volatility IDs  : {LOW_VOLATILITY_IDS}")
