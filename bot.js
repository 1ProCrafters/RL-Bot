const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock, GoalXZ, GoalY, GoalInvert, GoalFollow } = goals;
const fs = require('fs');
const path = require('path');

// ==================== CONFIGURATION ====================
const SERVER_CONFIG = {
    host: 'localhost',
    port: 25565,
    version: '1.19.4' // Adjust to your server version
};

const TRAINING_CONFIG = {
    numBots: 50,
    connectDelay: 10,
    episodesPerSession: 1000,
    maxStepsPerEpisode: 600,
    learningRate: 0.01,
    discountFactor: 0.99,
    explorationRate: 1.0,
    explorationDecay: 0.9995,
    explorationMin: 0.05,

    // Rewards
    rewardKill: 200,
    rewardHit: 20,
    rewardDamageDealt: 3,
    rewardDeath: -200,
    rewardDamageTaken: -3,
    rewardProximity: 1, // Reward for being closer to opponent
    rewardFarAway: -0.5, // Penalty for being too far

    // Training
    saveInterval: 5, // Save model every N episodes
    modelPath: './models/pvp_model.json',
    logInterval: 1, // Log stats every N episodes
    targetNetworkUpdateInterval: 5,
    batchSize: 32,
    minMemorySize: 50,

    // Combat
    attackCooldown: 500, // ms between attacks
    optimalDistance: 3.5, // Optimal attack distance
};

const ARENA_CONFIG = {
    center: { x: 0, y: 65, z: 0 },
    size: 15,
    respawnY: 70
};

// ==================== NEURAL NETWORK ====================
class SimpleQNetwork {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        this.hiddenSize = 128;
        this.weights1 = this.randomMatrix(stateSize, this.hiddenSize);
        this.bias1 = this.randomArray(this.hiddenSize);
        this.weights2 = this.randomMatrix(this.hiddenSize, this.hiddenSize);
        this.bias2 = this.randomArray(this.hiddenSize);
        this.weights3 = this.randomMatrix(this.hiddenSize, actionSize);
        this.bias3 = this.randomArray(actionSize);
    }

    randomMatrix(rows, cols) {
        const scale = Math.sqrt(2.0 / rows); // He initialization
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() => (Math.random() - 0.5) * 2 * scale)
        );
    }

    randomArray(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.01);
    }

    relu(x) {
        return Math.max(0, x);
    }

    forward(state) {
        // First hidden layer
        const hidden1 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        // Second hidden layer
        const hidden2 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden1[j] * this.weights2[j][i];
            }
            return this.relu(sum);
        });

        // Output layer
        const output = Array(this.actionSize).fill(0).map((_, i) => {
            let sum = this.bias3[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden2[j] * this.weights3[j][i];
            }
            return sum;
        });

        return output;
    }

    update(state, action, target, learningRate) {
        // Forward pass
        const hidden1 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        const hidden2 = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden1[j] * this.weights2[j][i];
            }
            return this.relu(sum);
        });

        const output = this.forward(state);

        // Compute error
        const outputError = output[action] - target;

        // Update output layer (layer 3)
        for (let i = 0; i < this.hiddenSize; i++) {
            this.weights3[i][action] -= learningRate * outputError * hidden2[i];
        }
        this.bias3[action] -= learningRate * outputError;

        // Backpropagate to hidden layers (simplified)
        for (let i = 0; i < this.hiddenSize; i++) {
            const hidden2Error = outputError * this.weights3[i][action];
            if (hidden2[i] > 0) {
                for (let j = 0; j < this.hiddenSize; j++) {
                    this.weights2[j][i] -= learningRate * hidden2Error * hidden1[j] * 0.1;
                }
            }
        }
    }

    copy() {
        const newNetwork = new SimpleQNetwork(this.stateSize, this.actionSize);
        newNetwork.weights1 = this.weights1.map(row => [...row]);
        newNetwork.bias1 = [...this.bias1];
        newNetwork.weights2 = this.weights2.map(row => [...row]);
        newNetwork.bias2 = [...this.bias2];
        newNetwork.weights3 = this.weights3.map(row => [...row]);
        newNetwork.bias3 = [...this.bias3];
        return newNetwork;
    }

    save() {
        return {
            stateSize: this.stateSize,
            actionSize: this.actionSize,
            hiddenSize: this.hiddenSize,
            weights1: this.weights1,
            bias1: this.bias1,
            weights2: this.weights2,
            bias2: this.bias2,
            weights3: this.weights3,
            bias3: this.bias3
        };
    }

    load(data) {
        this.stateSize = data.stateSize;
        this.actionSize = data.actionSize;
        this.hiddenSize = data.hiddenSize;
        this.weights1 = data.weights1;
        this.bias1 = data.bias1;
        this.weights2 = data.weights2;
        this.bias2 = data.bias2;
        this.weights3 = data.weights3;
        this.bias3 = data.bias3;
    }

    static fromData(data) {
        const network = new SimpleQNetwork(data.stateSize, data.actionSize);
        network.load(data);
        return network;
    }
}

// ==================== RL AGENT ====================
class RLAgent {
    constructor(config) {
        this.config = config;

        // Enhanced state: [health, food, distance, angle, velocity_x, velocity_z, 
        //                  opponent_health, is_sprinting, can_attack, opponent_distance_change]
        this.stateSize = 10;

        // Actions: [attack, forward, backward, strafe_left, strafe_right, jump, sprint, forward_left, forward_right]
        this.actionSize = 9;
        this.actions = ['attack', 'forward', 'backward', 'left', 'right', 'jump', 'sprint', 'forward_left', 'forward_right'];

        this.qNetwork = new SimpleQNetwork(this.stateSize, this.actionSize);
        this.targetNetwork = this.qNetwork.copy();

        this.epsilon = config.explorationRate;
        this.memory = [];
        this.maxMemorySize = 50000;
    }

    getState(bot, opponent, lastDistance) {
        if (!bot.entity || !opponent || !opponent.position) {
            return Array(this.stateSize).fill(0);
        }

        const dx = opponent.position.x - bot.entity.position.x;
        const dy = opponent.position.y - bot.entity.position.y;
        const dz = opponent.position.z - bot.entity.position.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Calculate angle to opponent
        const angle = Math.atan2(dz, dx);
        const botYaw = bot.entity.yaw;
        let relativeAngle = angle - botYaw;
        while (relativeAngle > Math.PI) relativeAngle -= 2 * Math.PI;
        while (relativeAngle < -Math.PI) relativeAngle += 2 * Math.PI;

        const distanceChange = lastDistance ? (distance - lastDistance) : 0;

        return [
            bot.health / 20, // Normalized health
            bot.food / 20, // Normalized food
            Math.min(distance / 20, 1), // Normalized distance
            relativeAngle / Math.PI, // Normalized angle
            (bot.entity.velocity?.x || 0) * 10, // Velocity x
            (bot.entity.velocity?.z || 0) * 10, // Velocity z
            opponent.health ? opponent.health / 20 : 0, // Opponent health
            bot.getControlState?.('sprint') ? 1 : 0, // Is sprinting
            1, // Can attack (simplified)
            Math.tanh(distanceChange) // Distance change
        ];
    }

    selectAction(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        }

        const qValues = this.qNetwork.forward(state);
        return qValues.indexOf(Math.max(...qValues));
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
        if (this.memory.length > this.maxMemorySize) {
            this.memory.shift();
        }
    }

    learn() {
        if (this.memory.length < this.config.minMemorySize) return;

        // Sample random batch
        const batchSize = Math.min(this.config.batchSize, this.memory.length);
        const batch = [];
        const usedIndices = new Set();

        while (batch.length < batchSize) {
            const idx = Math.floor(Math.random() * this.memory.length);
            if (!usedIndices.has(idx)) {
                usedIndices.add(idx);
                batch.push(this.memory[idx]);
            }
        }

        // Train on batch
        let totalLoss = 0;
        for (const { state, action, reward, nextState, done } of batch) {
            let target = reward;
            if (!done) {
                const nextQValues = this.targetNetwork.forward(nextState);
                target += this.config.discountFactor * Math.max(...nextQValues);
            }

            const currentQ = this.qNetwork.forward(state)[action];
            totalLoss += Math.abs(currentQ - target);

            this.qNetwork.update(state, action, target, this.config.learningRate);
        }

        // Decay exploration
        this.epsilon = Math.max(this.config.explorationMin, this.epsilon * this.config.explorationDecay);

        return totalLoss / batchSize;
    }

    updateTargetNetwork() {
        this.targetNetwork = this.qNetwork.copy();
    }

    save() {
        return {
            qNetwork: this.qNetwork.save(),
            targetNetwork: this.targetNetwork.save(),
            epsilon: this.epsilon,
            stateSize: this.stateSize,
            actionSize: this.actionSize,
            actions: this.actions
        };
    }

    load(data) {
        this.qNetwork = SimpleQNetwork.fromData(data.qNetwork);
        this.targetNetwork = SimpleQNetwork.fromData(data.targetNetwork);
        this.epsilon = data.epsilon;
        this.stateSize = data.stateSize;
        this.actionSize = data.actionSize;
        this.actions = data.actions;
    }

    static fromData(data, config) {
        const agent = new RLAgent(config);
        agent.load(data);
        return agent;
    }
}

// ==================== PVP BOT ====================
class PvPBot {
    constructor(name, config, agent) {
        this.name = name;
        this.config = config;
        this.agent = agent;
        this.bot = null;
        this.opponent = null;
        this.currentState = null;
        this.currentAction = null;
        this.episodeReward = 0;
        this.episodeSteps = 0;
        this.lastHealth = 20;
        this.lastOpponentHealth = 20;
        this.lastDistance = null;
        this.kills = 0;
        this.deaths = 0;
        this.hits = 0;
        this.lastAttackTime = 0;
        this.totalDamageDealt = 0;
        this.totalDamageTaken = 0;
    }

    async connect() {
        this.bot = mineflayer.createBot({
            ...SERVER_CONFIG,
            username: this.name
        });

        this.bot.loadPlugin(pathfinder);

        return new Promise((resolve, reject) => {
            this.bot.once('spawn', () => {
                console.log(`${this.name} connected!`);
                this.setupEventHandlers();
                resolve();
            });

            this.bot.once('error', reject);
            this.bot.once('kicked', (reason) => reject(new Error(reason)));
        });
    }

    setupEventHandlers() {
        this.bot.on('health', () => {
            if (this.lastHealth > this.bot.health) {
                const damage = this.lastHealth - this.bot.health;
                this.totalDamageTaken += damage;

                // Immediate negative reward for taking damage
                if (this.currentState && this.currentAction !== null) {
                    const reward = this.config.rewardDamageTaken * damage;
                    this.episodeReward += reward;
                }
            }
            this.lastHealth = this.bot.health;
        });

        this.bot.on('death', () => {
            console.log(`💀 ${this.name} died!`);
            this.deaths++;

            if (this.currentState && this.currentAction !== null) {
                const nextState = Array(this.agent.stateSize).fill(0);
                this.agent.remember(this.currentState, this.currentAction, this.config.rewardDeath, nextState, true);
                this.episodeReward += this.config.rewardDeath;
            }

            setTimeout(() => this.respawn(), 2000);
        });

        // Auto-equip best weapon
        this.bot.on('spawn', () => {
            setTimeout(() => this.equipBestWeapon(), 1000);
        });
    }

    equipBestWeapon() {
        const weapons = this.bot.inventory.items().filter(item =>
            item.name.includes('sword') || item.name.includes('axe')
        );

        if (weapons.length > 0) {
            const weaponPriority = ['diamond', 'iron', 'stone', 'wooden', 'golden'];
            weapons.sort((a, b) => {
                const aPriority = weaponPriority.findIndex(p => a.name.includes(p));
                const bPriority = weaponPriority.findIndex(p => b.name.includes(p));
                return aPriority - bPriority;
            });

            this.bot.equip(weapons[0], 'hand').catch(() => { });
        }
    }

    async respawn() {
        this.bot.chat(`/tp ${this.name} ${ARENA_CONFIG.center.x} ${ARENA_CONFIG.respawnY} ${ARENA_CONFIG.center.z}`);
        this.lastHealth = 20;
    }

    setOpponent(opponent) {
        this.opponent = opponent;
    }

    async step() {
        if (!this.opponent || !this.bot.entity) return;

        const opponentEntity = this.bot.players[this.opponent.name]?.entity;
        if (!opponentEntity) return;

        // Track opponent health for damage dealt
        if (this.lastOpponentHealth > opponentEntity.health) {
            const damage = this.lastOpponentHealth - opponentEntity.health;
            this.totalDamageDealt += damage;
            const reward = this.config.rewardDamageDealt * damage;
            this.episodeReward += reward;
        }
        this.lastOpponentHealth = opponentEntity.health;

        // Get current state
        const state = this.agent.getState(this.bot, opponentEntity, this.lastDistance);
        const distance = Math.sqrt(
            Math.pow(opponentEntity.position.x - this.bot.entity.position.x, 2) +
            Math.pow(opponentEntity.position.z - this.bot.entity.position.z, 2)
        );

        // Distance-based reward
        if (this.currentState) {
            let distanceReward = 0;
            if (distance < this.config.optimalDistance * 1.5 && distance > 2) {
                distanceReward = this.config.rewardProximity;
            } else if (distance > 10) {
                distanceReward = this.config.rewardFarAway;
            }
            this.episodeReward += distanceReward;
        }

        // Select action
        const action = this.agent.selectAction(state);

        // Store experience from previous action
        if (this.currentState !== null && this.currentAction !== null) {
            this.agent.remember(this.currentState, this.currentAction, 0, state, false);
        }

        this.currentState = state;
        this.currentAction = action;
        this.episodeSteps++;
        this.lastDistance = distance;

        // Execute action
        this.executeAction(action, opponentEntity, distance);

        // Learn periodically
        if (this.episodeSteps % 4 === 0) {
            this.agent.learn();
        }
    }

    executeAction(action, opponentEntity, distance) {
        const actionName = this.agent.actions[action];

        if (!this.bot.entity) return;

        // Always look at opponent
        this.bot.lookAt(opponentEntity.position.offset(0, opponentEntity.height * 0.9, 0));

        const now = Date.now();

        switch (actionName) {
            case 'attack':
                if (distance < 4 && (now - this.lastAttackTime) > this.config.attackCooldown) {
                    this.bot.attack(opponentEntity);
                    this.lastAttackTime = now;
                    this.hits++;

                    // Check for kill
                    setTimeout(() => {
                        if (!opponentEntity.isValid || opponentEntity.health <= 0) {
                            this.episodeReward += this.config.rewardKill;
                            this.kills++;
                            console.log(`⚔️  ${this.name} killed ${this.opponent.name}! (+${this.config.rewardKill} reward)`);
                        } else if (this.lastOpponentHealth > opponentEntity.health) {
                            // Hit registered
                            this.episodeReward += this.config.rewardHit;
                        }
                    }, 100);
                }
                break;

            case 'forward':
                this.bot.setControlState('forward', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;

            case 'backward':
                this.bot.setControlState('back', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;

            case 'left':
                this.bot.setControlState('left', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;

            case 'right':
                this.bot.setControlState('right', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;

            case 'jump':
                this.bot.setControlState('jump', true);
                setTimeout(() => this.bot.setControlState('jump', false), 100);
                break;

            case 'sprint':
                this.bot.setControlState('sprint', true);
                setTimeout(() => this.bot.setControlState('sprint', false), 200);
                break;

            case 'forward_left':
                this.bot.setControlState('forward', true);
                this.bot.setControlState('left', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;

            case 'forward_right':
                this.bot.setControlState('forward', true);
                this.bot.setControlState('right', true);
                setTimeout(() => this.bot.clearControlStates(), 150);
                break;
        }
    }

    resetEpisode() {
        this.episodeReward = 0;
        this.episodeSteps = 0;
        this.currentState = null;
        this.currentAction = null;
        this.lastHealth = 20;
        this.lastOpponentHealth = 20;
        this.lastDistance = null;
        this.lastAttackTime = 0;
    }

    getStats() {
        return {
            name: this.name,
            kills: this.kills,
            deaths: this.deaths,
            hits: this.hits,
            kd: this.deaths > 0 ? (this.kills / this.deaths).toFixed(2) : this.kills,
            epsilon: this.agent.epsilon.toFixed(4),
            lastReward: this.episodeReward.toFixed(1),
            damageDealt: this.totalDamageDealt.toFixed(1),
            damageTaken: this.totalDamageTaken.toFixed(1)
        };
    }
}

// ==================== OPERATOR BOT ====================
class OperatorBot {
    constructor() {
        this.bot = null;
    }

    async connect() {
        this.bot = mineflayer.createBot({
            ...SERVER_CONFIG,
            username: 'Operator'
        });

        return new Promise((resolve, reject) => {
            this.bot.once('spawn', () => {
                console.log('Operator connected!');
                resolve();
            });

            this.bot.once('error', reject);
            this.bot.once('kicked', (reason) => reject(new Error(reason)));
        });
    }

    async setupArena() {
        console.log('Setting up arena...');

        const { x, y, z } = ARENA_CONFIG.center;
        const size = ARENA_CONFIG.size;

        // Clear area
        await this.executeCommand(`/fill ${x - size} ${y - 5} ${z - size} ${x + size} ${y + 20} ${z + size} air`);

        // Create floor
        await this.executeCommand(`/fill ${x - size} ${y - 1} ${z - size} ${x + size} ${y - 1} ${z + size} stone`);

        // Create walls
        await this.executeCommand(`/fill ${x - size} ${y} ${z - size} ${x - size} ${y + 5} ${z + size} barrier`);
        await this.executeCommand(`/fill ${x + size} ${y} ${z - size} ${x + size} ${y + 5} ${z + size} barrier`);
        await this.executeCommand(`/fill ${x - size} ${y} ${z - size} ${x + size} ${y + 5} ${z - size} barrier`);
        await this.executeCommand(`/fill ${x - size} ${y} ${z + size} ${x + size} ${y + 5} ${z + size} barrier`);

        // Set time and weather
        await this.executeCommand('/time set day');
        await this.executeCommand('/weather clear');
        await this.executeCommand('/gamerule doDaylightCycle false');
        await this.executeCommand('/gamerule doMobSpawning false');

        console.log('Arena setup complete!');
    }

    async giveEquipment(playerName) {
        await this.executeCommand(`/clear ${playerName}`);
        await this.executeCommand(`/give ${playerName} diamond_sword 1`);
        await this.executeCommand(`/give ${playerName} golden_apple 64`);
        await this.executeCommand(`/give ${playerName} diamond_helmet{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_chestplate{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_leggings{Enchantments:[{id:"protection",lvl:4}]} 1`);
        await this.executeCommand(`/give ${playerName} diamond_boots{Enchantments:[{id:"protection",lvl:4}]} 1`);
    }

    async healPlayer(playerName) {
        await this.executeCommand(`/effect clear ${playerName}`);
        await this.executeCommand(`/effect give ${playerName} instant_health 1 10`);
        await this.executeCommand(`/effect give ${playerName} saturation 1 10`);
    }

    async teleportPlayer(playerName, x, y, z) {
        await this.executeCommand(`/tp ${playerName} ${x} ${y} ${z}`);
    }

    async executeCommand(command) {
        return new Promise((resolve) => {
            this.bot.chat(command);
            setTimeout(resolve, 50);
        });
    }

    disconnect() {
        if (this.bot) {
            this.bot.quit();
        }
    }
}

// ==================== TRAINING SYSTEM ====================
class TrainingSystem {
    constructor(loadModel = false) {
        this.operator = new OperatorBot();
        this.agent = loadModel ? this.loadAgent() : new RLAgent(TRAINING_CONFIG);
        this.bots = [];
        this.episode = 0;
        this.totalEpisodes = 0;
        this.startTime = Date.now();
        this.episodeRewards = [];
        this.episodeKills = [];
        this.episodeLengths = [];
    }

    loadAgent() {
        const modelPath = TRAINING_CONFIG.modelPath;
        if (fs.existsSync(modelPath)) {
            try {
                console.log(`Loading model from ${modelPath}...`);
                const data = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
                const agent = RLAgent.fromData(data.agent, TRAINING_CONFIG);
                this.totalEpisodes = data.totalEpisodes || 0;
                this.episodeRewards = data.episodeRewards || [];
                this.episodeKills = data.episodeKills || [];
                this.episodeLengths = data.episodeLengths || [];
                console.log(`✅ Model loaded! Resuming from episode ${this.totalEpisodes}`);
                console.log(`Current epsilon: ${agent.epsilon.toFixed(4)}`);
                return agent;
            } catch (error) {
                console.error('❌ Error loading model:', error);
                console.log('Starting fresh instead.');
                return new RLAgent(TRAINING_CONFIG);
            }
        } else {
            console.log('No saved model found. Starting fresh.');
            return new RLAgent(TRAINING_CONFIG);
        }
    }

    saveAgent() {
        try {
            const modelPath = TRAINING_CONFIG.modelPath;
            const dir = path.dirname(modelPath);

            // Create directory if it doesn't exist
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`Created directory: ${dir}`);
            }

            const data = {
                agent: this.agent.save(),
                totalEpisodes: this.totalEpisodes,
                episodeRewards: this.episodeRewards.slice(-1000),
                episodeKills: this.episodeKills.slice(-1000),
                episodeLengths: this.episodeLengths.slice(-1000),
                timestamp: new Date().toISOString(),
                config: TRAINING_CONFIG
            };

            fs.writeFileSync(modelPath, JSON.stringify(data, null, 2));
            console.log(`💾 Model saved to ${modelPath} (Episode ${this.totalEpisodes}, ε=${this.agent.epsilon.toFixed(4)})`);

            // Save backup periodically
            if (this.totalEpisodes % (TRAINING_CONFIG.saveInterval * 10) === 0) {
                const backupPath = modelPath.replace('.json', `_ep${this.totalEpisodes}.json`);
                fs.writeFileSync(backupPath, JSON.stringify(data, null, 2));
                console.log(`📦 Backup saved: ${backupPath}`);
            }
        } catch (error) {
            console.error('❌ Error saving model:', error);
        }
    }

    async initialize() {
        console.log('Initializing training system...');

        // Connect operator
        await this.operator.connect();
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Setup arena
        await this.operator.setupArena();
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Create and connect bots
        for (let i = 0; i < TRAINING_CONFIG.numBots; i++) {
            const bot = new PvPBot(`Fighter${i + 1}`, TRAINING_CONFIG, this.agent);
            await bot.connect();
            await new Promise(resolve => setTimeout(resolve, TRAINING_CONFIG.connectDelay));
            this.bots.push(bot);
        }

        // Set opponents
        for (let i = 0; i < this.bots.length; i++) {
            const opponent = this.bots[(i + 1) % this.bots.length];
            this.bots[i].setOpponent(opponent);
        }

        console.log('Training system initialized!');
    }

    async runEpisode() {
        this.episode++;
        this.totalEpisodes++;
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🎮 Episode ${this.totalEpisodes} (Session: ${this.episode}/${TRAINING_CONFIG.episodesPerSession})`);
        console.log(`${'='.repeat(60)}`);

        // Reset bots
        for (const bot of this.bots) {
            bot.resetEpisode();
            await this.operator.healPlayer(bot.name);
            await this.operator.giveEquipment(bot.name);
            const spawnX = ARENA_CONFIG.center.x + (Math.random() - 0.5) * 8;
            const spawnZ = ARENA_CONFIG.center.z + (Math.random() - 0.5) * 8;
            await this.operator.teleportPlayer(bot.name, spawnX, ARENA_CONFIG.respawnY, spawnZ);
        }

        await new Promise(resolve => setTimeout(resolve, 3000));

        // Run episode
        const episodeStartTime = Date.now();
        let steps = 0;
        let winner = null;

        while (steps < TRAINING_CONFIG.maxStepsPerEpisode) {
            for (const bot of this.bots) {
                await bot.step();
            }
            await new Promise(resolve => setTimeout(resolve, 50));
            steps++;

            // Check for winner
            for (const bot of this.bots) {
                if (bot.bot && bot.bot.health <= 0) {
                    winner = this.bots.find(b => b !== bot);
                    break;
                }
            }

            if (winner) break;
        }

        const episodeDuration = (Date.now() - episodeStartTime) / 1000;

        // Collect episode stats
        const totalReward = this.bots.reduce((sum, bot) => sum + bot.episodeReward, 0) / this.bots.length;
        const totalKills = this.bots.reduce((sum, bot) => sum + bot.kills, 0);

        this.episodeRewards.push(totalReward);
        this.episodeKills.push(totalKills);
        this.episodeLengths.push(steps);

        // Update target network periodically
        if (this.totalEpisodes % TRAINING_CONFIG.targetNetworkUpdateInterval === 0) {
            this.agent.updateTargetNetwork();
            console.log('🎯 Target network updated');
        }

        // Save model periodically
        if (this.totalEpisodes % TRAINING_CONFIG.saveInterval === 0) {
            this.saveAgent();
        }

        // Print stats
        if (this.episode % TRAINING_CONFIG.logInterval === 0) {
            this.printDetailedStats(episodeDuration, steps, winner);
        } else {
            console.log(`⏱️  Duration: ${episodeDuration.toFixed(1)}s | Steps: ${steps} | Avg Reward: ${totalReward.toFixed(1)} | ε: ${this.agent.epsilon.toFixed(4)}`);
            if (winner) {
                console.log(`🏆 Winner: ${winner.name}`);
            }
        }
    }

    printDetailedStats(lastEpisodeDuration, steps, winner) {
        console.log('\n' + '='.repeat(60));
        console.log('📊 DETAILED STATISTICS');
        console.log('='.repeat(60));

        const totalTime = (Date.now() - this.startTime) / 1000;
        const avgTime = totalTime / this.episode;

        console.log(`\n⏱️  Time Stats:`);
        console.log(`   Total Training Time: ${(totalTime / 60).toFixed(2)} minutes`);
        console.log(`   Avg Episode Duration: ${avgTime.toFixed(2)}s`);
        console.log(`   Last Episode: ${lastEpisodeDuration.toFixed(2)}s (${steps} steps)`);

        console.log(`\n🤖 Agent Stats:`);
        console.log(`   Total Episodes: ${this.totalEpisodes}`);
        console.log(`   Memory Size: ${this.agent.memory.length} / ${this.agent.maxMemorySize}`);
        console.log(`   Exploration (ε): ${this.agent.epsilon.toFixed(4)}`);
        console.log(`   Learning Rate: ${TRAINING_CONFIG.learningRate}`);
        console.log(`   Batch Size: ${TRAINING_CONFIG.batchSize}`);

        // Recent performance (last 10 episodes)
        const recent = Math.min(10, this.episodeRewards.length);
        const recentRewards = this.episodeRewards.slice(-recent);
        const recentKills = this.episodeKills.slice(-recent);
        const recentLengths = this.episodeLengths.slice(-recent);

        const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
        const avgKills = recentKills.reduce((a, b) => a + b, 0) / recentKills.length;
        const avgLength = recentLengths.reduce((a, b) => a + b, 0) / recentLengths.length;

        console.log(`\n📈 Recent Performance (last ${recent} episodes):`);
        console.log(`   Avg Reward: ${avgReward.toFixed(2)}`);
        console.log(`   Avg Kills per Episode: ${avgKills.toFixed(2)}`);
        console.log(`   Avg Episode Length: ${avgLength.toFixed(0)} steps`);

        console.log(`\n⚔️  Bot Performance:`);
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`   ${stats.name}:`);
            console.log(`      K/D Ratio: ${stats.kd} (${stats.kills}K / ${stats.deaths}D)`);
            console.log(`      Hits: ${stats.hits}`);
            console.log(`      Damage: ${stats.damageDealt} dealt / ${stats.damageTaken} taken`);
            console.log(`      Last Reward: ${stats.lastReward}`);
        }

        if (winner) {
            console.log(`\n🏆 Episode Winner: ${winner.name}`);
        }

        // All-time best
        if (this.episodeRewards.length > 0) {
            const bestReward = Math.max(...this.episodeRewards);
            const bestKills = Math.max(...this.episodeKills);
            const shortestWin = Math.min(...this.episodeLengths.filter((_, i) => this.episodeKills[i] > 0));

            console.log(`\n🏆 All-Time Records:`);
            console.log(`   Best Reward: ${bestReward.toFixed(2)}`);
            console.log(`   Most Kills: ${bestKills}`);
            if (shortestWin !== Infinity) {
                console.log(`   Fastest Kill: ${shortestWin} steps`);
            }
        }

        console.log('\n' + '='.repeat(60) + '\n');
    }

    async train() {
        await this.initialize();

        console.log('\n🚀 Starting training...');
        console.log(`Training for ${TRAINING_CONFIG.episodesPerSession} episodes`);
        console.log(`Saving every ${TRAINING_CONFIG.saveInterval} episodes to ${TRAINING_CONFIG.modelPath}\n`);

        for (let i = 0; i < TRAINING_CONFIG.episodesPerSession; i++) {
            await this.runEpisode();
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Final save
        this.saveAgent();

        console.log('\n' + '='.repeat(60));
        console.log('✅ TRAINING COMPLETE');
        console.log('='.repeat(60));
        console.log(`Total Episodes: ${this.totalEpisodes}`);
        console.log(`Final ε: ${this.agent.epsilon.toFixed(4)}`);
        console.log(`Model saved to: ${TRAINING_CONFIG.modelPath}`);

        console.log('\n⚔️  Final Bot Stats:');
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`${stats.name}: ${stats.kills}K / ${stats.deaths}D (K/D: ${stats.kd})`);
        }

        if (this.episodeRewards.length >= 10) {
            const avgReward = this.episodeRewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
            console.log(`\nFinal 10-episode avg reward: ${avgReward.toFixed(2)}`);
        }
        console.log('='.repeat(60) + '\n');
    }

    cleanup() {
        console.log('Cleaning up...');
        for (const bot of this.bots) {
            if (bot.bot) {
                try {
                    bot.bot.quit();
                } catch (e) { }
            }
        }
        this.operator.disconnect();
    }
}

// ==================== MAIN ====================
async function main() {
    // Parse command line arguments
    const args = process.argv.slice(2);
    const loadModel = args.includes('--load') || args.includes('-l') || args.includes('--continue') || args.includes('-c');

    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
╔════════════════════════════════════════════════════════════╗
║     Minecraft RL PvP Bot Training System                  ║
╚════════════════════════════════════════════════════════════╝

Usage: node bot.js [options]

Options:
  --load, -l         Load existing model and continue training
  --continue, -c     Same as --load
  --help, -h         Show this help message

Examples:
  node bot.js                    # Start fresh training
  node bot.js --continue         # Continue from saved model
  node bot.js --load             # Load and continue training

Configuration:
  Edit TRAINING_CONFIG at the top of the file to adjust:
  
  Training:
    - episodesPerSession: ${TRAINING_CONFIG.episodesPerSession}
    - saveInterval: ${TRAINING_CONFIG.saveInterval} (saves every N episodes)
    - modelPath: ${TRAINING_CONFIG.modelPath}
  
  Learning:
    - learningRate: ${TRAINING_CONFIG.learningRate}
    - explorationRate: ${TRAINING_CONFIG.explorationRate} → ${TRAINING_CONFIG.explorationMin}
    - explorationDecay: ${TRAINING_CONFIG.explorationDecay}
    - batchSize: ${TRAINING_CONFIG.batchSize}
  
  Rewards:
    - Kill: +${TRAINING_CONFIG.rewardKill}
    - Hit: +${TRAINING_CONFIG.rewardHit}
    - Damage Dealt: +${TRAINING_CONFIG.rewardDamageDealt} per HP
    - Death: ${TRAINING_CONFIG.rewardDeath}
    - Damage Taken: ${TRAINING_CONFIG.rewardDamageTaken} per HP

Server:
  - Host: ${SERVER_CONFIG.host}
  - Port: ${SERVER_CONFIG.port}
  - Version: ${SERVER_CONFIG.version}

Note: The Operator bot must have OP permissions on the server!
    `);
        process.exit(0);
    }

    const trainer = new TrainingSystem(loadModel);

    process.on('SIGINT', () => {
        console.log('\n\n⚠️  Interrupt received! Saving model...');
        trainer.saveAgent();
        console.log('✅ Model saved. Shutting down...');
        trainer.cleanup();
        process.exit(0);
    });

    process.on('uncaughtException', (error) => {
        console.error('❌ Uncaught exception:', error);
        console.log('Saving model before exit...');
        trainer.saveAgent();
        trainer.cleanup();
        process.exit(1);
    });

    try {
        await trainer.train();
        trainer.cleanup();
    } catch (error) {
        console.error('❌ Error during training:', error);
        console.log('Saving model before exit...');
        trainer.saveAgent();
        trainer.cleanup();
        process.exit(1);
    }
}

// Run if executed directly
if (require.main === module) {
    main();
}

module.exports = { TrainingSystem, PvPBot, OperatorBot, RLAgent };