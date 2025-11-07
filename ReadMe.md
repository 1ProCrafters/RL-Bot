# Minecraft PvP Bot - Reinforcement Learning with Evolutionary Training

An advanced AI system that learns to play Minecraft PvP combat through a combination of Deep Q-Learning and Genetic Algorithms.

## 🧠 How It Works

### Overview

This system trains AI bots to fight in Minecraft using two complementary approaches:

1. **Deep Q-Learning (DQN)** - The bot learns moment-to-moment combat decisions
2. **Evolutionary Algorithm** - The population evolves over generations to find optimal strategies

### Deep Q-Learning Network

#### State Representation (10 features)

The bot observes the game through these inputs:

- **Health** (0-1): Bot's current health normalized
- **Food** (0-1): Bot's hunger level normalized
- **Distance** (0-1): Distance to opponent normalized (0 = very close, 1 = far)
- **Angle** (-1 to 1): Relative angle to opponent (-1 = behind, 0 = front, 1 = behind other side)
- **Velocity X/Z** (-10 to 10): Bot's movement speed in X and Z directions
- **Opponent Health** (0-1): Opponent's health normalized
- **Is Sprinting** (0 or 1): Whether the bot is currently sprinting
- **Can Attack** (always 1): Whether attack is available (simplified)
- **Distance Change** (-1 to 1): Whether bot is getting closer or farther from opponent

#### Neural Network Architecture

```
Input Layer (10 neurons)
    ↓
Hidden Layer 1 (128 neurons, ReLU activation)
    ↓
Hidden Layer 2 (128 neurons, ReLU activation)
    ↓
Output Layer (9 neurons, one per action)
```

The network uses:

- **He initialization** for better gradient flow
- **ReLU activation** for non-linearity
- **Q-value outputs** representing expected future reward for each action

#### Action Space (9 actions)

The bot can choose from:

1. **Attack** - Swing weapon at opponent
2. **Forward** - Move forward
3. **Backward** - Move backward
4. **Strafe Left** - Move left
5. **Strafe Right** - Move right
6. **Jump** - Jump in place
7. **Sprint** - Sprint forward
8. **Forward Left** - Diagonal movement
9. **Forward Right** - Diagonal movement

### Learning Process

#### 1. Experience Collection

- Bot performs action → observes result → stores experience
- Each experience contains: `(state, action, reward, next_state, done)`
- Experiences stored in replay memory (max 50,000 experiences)

#### 2. Reward System

The bot learns through these rewards:

**Positive Rewards:**

- Kill opponent: **+200**
- Hit opponent: **+20**
- Deal damage: **+3 per HP**
- Stay at optimal distance (2-5 blocks): **+1 per step**

**Negative Rewards:**

- Die: **-200**
- Take damage: **-3 per HP**
- Too far from opponent (>10 blocks): **-0.5 per step**

**Fitness Formula (for evolution):**

```javascript
fitness = kills * 100 - deaths * 50 + damageDealt * 2 - damageTaken * 1;
```

This can be customized in `TRAINING_CONFIG.fitnessFormula`

#### 3. Training Loop

Every 4 steps:

1. Sample 32 random experiences from memory
2. For each experience:
   - Calculate target Q-value: `reward + 0.99 * max(Q(next_state))`
   - Update network weights to minimize: `|current_Q - target_Q|`
3. Gradually reduce exploration (epsilon decay)

#### 4. Target Network

- A separate "target" network provides stable Q-value estimates
- Updated every 5 episodes to the current network's weights
- Prevents training instability from "chasing a moving target"

### Evolutionary Training

Every **N episodes** (configurable via `evolutionInterval`):

1. **Fitness Evaluation**

   - Each bot's performance is scored using the fitness formula
   - Accounts for kills, deaths, damage dealt/taken

2. **Selection**

   - The bot with the highest fitness is selected as the "champion"
   - This represents the best strategy found so far

3. **Mutation**

   - All other bots copy the champion's neural network
   - Random mutations are applied to weights and biases
   - Mutation rate: 10% of weights changed (configurable)
   - Mutation strength: ±20% of current value (configurable)

4. **New Generation**
   - The mutated bots form a new generation
   - They continue learning via Q-learning
   - Process repeats, combining learned behavior with genetic diversity

### Why This Hybrid Approach?

**Q-Learning Alone:**

- ✅ Learns precise moment-to-moment tactics
- ❌ Can get stuck in local optima
- ❌ Slow to discover novel strategies

**Evolution Alone:**

- ✅ Explores diverse strategies
- ❌ Requires many episodes to learn fine control
- ❌ Loses learned behavior between generations

**Combined (Q-Learning + Evolution):**

- ✅ Quick learning of basic tactics (Q-learning)
- ✅ Exploration of novel strategies (evolution)
- ✅ Preservation of best strategies (best agent selection)
- ✅ Continuous refinement through both mechanisms

## 📊 Training Metrics

### What to Watch

**Epsilon (ε)**: Exploration rate

- Starts at 1.0 (100% random actions)
- Decays to 0.05 (5% random, 95% learned)
- Shows how much the bot relies on learned strategy vs exploration

**Average Reward**:

- Should trend upward over episodes
- Negative → Positive indicates learning is working
- Plateau indicates convergence or local optimum

**K/D Ratio**:

- Starts around 0.5-1.0 (random fighting)
- Should approach or exceed 1.0 as bot learns
- Higher ratios indicate dominant strategies

**Episode Length**:

- Initially long (bots miss attacks, poor positioning)
- Should decrease as bots learn to fight efficiently
- Very short = quick decisive combat

**Fitness Score**:

- Composite metric of overall performance
- Should increase with each generation
- Best fitness tracked across all generations

### Convergence Signs

The bot is learning when you see:

1. Epsilon decreasing steadily
2. Average reward trending upward
3. More consistent episode outcomes
4. Shorter, more decisive combats
5. Higher fitness scores in new generations

## 🎮 Arena System

### Scaling for Multiple Bots

The arena system automatically scales:

- **Single Arena**: 2 bots (1v1)
- **Multiple Arenas**: N bots → N/2 arenas
- **Arena Spacing**: 30 blocks apart (configurable)
- **Minimum Spawn Distance**: 8 blocks (prevents sweeping damage)

### Anti-Sweeping Mechanics

To prevent sweeping edge damage affecting multiple bots:

1. Bots spawn with minimum 8-block separation
2. Each pair fights in its own isolated arena
3. Barrier walls prevent cross-arena interference
4. Arenas are spaced 30 blocks apart

### Arena Structure (per arena)

```Markdown
Size: 15x15 blocks
Floor: Stone
Walls: Barrier blocks (5 blocks high)
Spawn Height: Y=70
Combat Area: Y=65-70
```

## ⚙️ Configuration

### Key Training Parameters

```javascript
// How often to save the model
saveInterval: 10; // Every 10 episodes

// Evolution settings
enableEvolution: true;
evolutionInterval: 20; // Evolve every 20 episodes
mutationRate: 0.1; // 10% of weights mutated
mutationStrength: 0.2; // ±20% change to mutated weights

// Fitness formula (customize this!)
fitnessFormula: "kills * 100 - deaths * 50 + damageDealt * 2 - damageTaken * 1";

// Learning parameters
learningRate: 0.01;
discountFactor: 0.99; // Future reward importance
explorationRate: 1.0; // Start with full exploration
explorationDecay: 0.9995; // Gradual reduction
explorationMin: 0.05; // Never stop exploring completely

// Arena scaling
botsPerArena: 2; // Must be 2 for PvP
arenaSpacing: 30; // Blocks between arenas
minDistance: 8; // Min spawn distance (anti-sweep)
```

### Customizing Fitness

The fitness formula uses these variables:

- `kills` - Number of kills
- `deaths` - Number of deaths
- `damageDealt` - Total HP damage dealt
- `damageTaken` - Total HP damage taken
- `avgReward` - Average reward per episode

**Example formulas:**

Aggressive fighter:

```javascript
fitnessFormula: "kills * 200 + damageDealt * 5 - deaths * 100";
```

Defensive survivor:

```javascript
fitnessFormula: "kills * 50 - deaths * 200 - damageTaken * 3";
```

Balanced:

```javascript
fitnessFormula: "kills * 100 - deaths * 50 + damageDealt - damageTaken";
```

## 🚀 Usage

### Basic Training

```bash
# Start fresh training
node bot.js

# Continue from saved model
node bot.js --continue

# View help
node bot.js --help
```

### What Happens During Training

1. **Initialization** (Episode 0)

   - Operator bot connects and sets up arenas
   - Fighter bots connect and spawn
   - Neural networks initialized or loaded

2. **Episode Loop** (Episodes 1-N)

   - Bots spawn with minimum distance
   - Combat until death or timeout
   - Rewards accumulated
   - Q-network updated every 4 steps
   - Target network updated every 5 episodes

3. **Evolution** (Every 20 episodes)

   - Fitness calculated for all bots
   - Best bot selected
   - Other bots mutate from best
   - New generation begins

4. **Saving** (Every 10 episodes)
   - Best agent's network saved
   - Training history preserved
   - Backup created every 100 episodes

## 📁 Model Files

### Saved Data Structure

```json
{
  "agent": {
    "qNetwork": {
      /* weights and biases */
    },
    "targetNetwork": {
      /* weights and biases */
    },
    "epsilon": 0.234,
    "stateSize": 10,
    "actionSize": 9
  },
  "totalEpisodes": 150,
  "generation": 7,
  "bestFitness": 2847.3,
  "episodeRewards": [
    /* last 1000 */
  ],
  "episodeKills": [
    /* last 1000 */
  ],
  "episodeLengths": [
    /* last 1000 */
  ]
}
```

### File Locations

- **Main model**: `./models/pvp_model.json`
- **Backups**: `./models/pvp_model_ep100_gen5.json`

## 🔬 Expected Behavior

### Early Training (Episodes 1-50)

- Random, chaotic movements
- Many missed attacks
- Similar win rates between bots
- High epsilon (0.9-1.0)
- Negative average rewards

### Mid Training (Episodes 50-200)

- Basic attack patterns emerge
- Better positioning
- Epsilon around 0.3-0.6
- Rewards approaching zero
- Clear strategy differences between generations

### Late Training (Episodes 200+)

- Refined combat tactics
- Consistent attack timing
- Good distance management
- Epsilon around 0.05-0.2
- Positive average rewards
- Dominant strategies evolve

### Signs of Convergence

- Fitness plateau across generations
- Consistent episode outcomes
- Low epsilon with good performance
- Similar strategies across population

## 🛠️ Requirements

- Minecraft server (1.19.4 or compatible with mineflayer version)
- Operator bot must have OP permissions
- Node.js with mineflayer, mineflayer-pathfinder

## 📈 Performance Tips

1. **Faster Learning**: Increase `learningRate` to 0.02-0.05
2. **More Exploration**: Reduce `explorationDecay` to 0.999
3. **Aggressive Evolution**: Reduce `evolutionInterval` to 10-15
4. **Stable Training**: Increase `minMemorySize` to 200+
5. **Multiple Bots**: Increase `numBots` for diverse strategies

## 🎯 Advanced: Understanding the Math

### Q-Learning Update Rule

```Math
Q(s,a) ← Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
```

Where:

- `α` = learning rate (0.01)
- `r` = immediate reward
- `γ` = discount factor (0.99)
- `s,a` = current state-action
- `s'` = next state

### Epsilon-Greedy Policy

```Math
action = random()      if rand() < ε
         argmax(Q(s))  otherwise
```

Balances exploration (trying new actions) with exploitation (using best known actions).

### Mutation Operation

```Math
new_weight = old_weight + random(-1, 1) * strength
             if random() < mutation_rate
```

Introduces variation while preserving most learned behavior.

---

## 📝 Notes

- Model improves gradually; expect 200+ episodes for good results
- Evolution helps escape local optima
- Save interval of 10 balances safety with disk usage
- Sweeping damage prevented by arena spacing and spawn distance
- Each bot has its own agent during evolution for diversity

---
