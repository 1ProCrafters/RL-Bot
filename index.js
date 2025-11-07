const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock, GoalXZ, GoalY, GoalInvert, GoalFollow } = goals;

// ==================== CONFIGURATION ====================
const SERVER_CONFIG = {
    host: 'localhost',
    port: 25565,
    version: '1.20.1' // Adjust to your server version
};

const TRAINING_CONFIG = {
    numBots: 2,
    episodesPerSession: 100,
    maxStepsPerEpisode: 1000,
    learningRate: 0.001,
    discountFactor: 0.95,
    explorationRate: 1.0,
    explorationDecay: 0.995,
    explorationMin: 0.01,
    rewardKill: 100,
    rewardHit: 10,
    rewardDamage: 5,
    rewardDeath: -50,
    rewardHealthLoss: -2,
    rewardStep: -0.1
};

const ARENA_CONFIG = {
    center: { x: 0, y: 65, z: 0 },
    size: 20,
    respawnY: 70
};

// ==================== NEURAL NETWORK (Simple Q-Network) ====================
class SimpleQNetwork {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        // Simple 2-layer network weights
        this.hiddenSize = 64;
        this.weights1 = this.randomMatrix(stateSize, this.hiddenSize);
        this.bias1 = this.randomArray(this.hiddenSize);
        this.weights2 = this.randomMatrix(this.hiddenSize, actionSize);
        this.bias2 = this.randomArray(actionSize);
    }

    randomMatrix(rows, cols) {
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.1)
        );
    }

    randomArray(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    relu(x) {
        return Math.max(0, x);
    }

    forward(state) {
        // Hidden layer
        const hidden = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        // Output layer
        const output = Array(this.actionSize).fill(0).map((_, i) => {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.weights2[j][i];
            }
            return sum;
        });

        return output;
    }

    update(state, action, target, learningRate) {
        // Forward pass
        const hidden = Array(this.hiddenSize).fill(0).map((_, i) => {
            let sum = this.bias1[i];
            for (let j = 0; j < this.stateSize; j++) {
                sum += state[j] * this.weights1[j][i];
            }
            return this.relu(sum);
        });

        const output = this.forward(state);

        // Backward pass (simplified gradient descent)
        const outputError = output[action] - target;

        // Update output layer
        for (let i = 0; i < this.hiddenSize; i++) {
            this.weights2[i][action] -= learningRate * outputError * hidden[i];
        }
        this.bias2[action] -= learningRate * outputError;

        // Update hidden layer (simplified)
        const hiddenError = outputError * this.weights2.map(w => w[action]);
        for (let i = 0; i < this.stateSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                if (hidden[j] > 0) { // ReLU derivative
                    this.weights1[i][j] -= learningRate * hiddenError[j] * state[i] * 0.01;
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
        return newNetwork;
    }
}

// ==================== RL AGENT ====================
class RLAgent {
    constructor(config) {
        this.config = config;

        // State: [health, food, opponent_distance, opponent_angle, has_weapon, opponent_health]
        this.stateSize = 6;

        // Actions: [attack, move_forward, move_backward, strafe_left, strafe_right, jump, sprint]
        this.actionSize = 7;
        this.actions = ['attack', 'forward', 'backward', 'left', 'right', 'jump', 'sprint'];

        this.qNetwork = new SimpleQNetwork(this.stateSize, this.actionSize);
        this.targetNetwork = this.qNetwork.copy();

        this.epsilon = config.explorationRate;
        this.memory = [];
        this.maxMemorySize = 10000;
    }

    getState(bot, opponent) {
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

        return [
            bot.health / 20, // Normalized health
            bot.food / 20, // Normalized food
            Math.min(distance / 20, 1), // Normalized distance
            relativeAngle / Math.PI, // Normalized angle
            bot.inventory.slots.some(slot => slot && (slot.name.includes('sword') || slot.name.includes('axe'))) ? 1 : 0,
            opponent.health ? opponent.health / 20 : 0
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
        if (this.memory.length < 32) return;

        // Sample random batch
        const batchSize = Math.min(32, this.memory.length);
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            const idx = Math.floor(Math.random() * this.memory.length);
            batch.push(this.memory[idx]);
        }

        // Train on batch
        for (const { state, action, reward, nextState, done } of batch) {
            let target = reward;
            if (!done) {
                const nextQValues = this.targetNetwork.forward(nextState);
                target += this.config.discountFactor * Math.max(...nextQValues);
            }

            this.qNetwork.update(state, action, target, this.config.learningRate);
        }

        // Decay exploration
        this.epsilon = Math.max(this.config.explorationMin, this.epsilon * this.config.explorationDecay);
    }

    updateTargetNetwork() {
        this.targetNetwork = this.qNetwork.copy();
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
        this.kills = 0;
        this.deaths = 0;
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
        this.bot.on('entityHurt', (entity) => {
            if (entity === this.bot.entity) {
                const damage = this.lastHealth - this.bot.health;
                this.lastHealth = this.bot.health;

                if (this.currentState) {
                    const reward = this.config.rewardHealthLoss * damage;
                    this.episodeReward += reward;
                }
            }
        });

        this.bot.on('death', () => {
            console.log(`${this.name} died!`);
            this.deaths++;

            if (this.currentState && this.currentAction !== null) {
                const nextState = Array(this.agent.stateSize).fill(0);
                this.agent.remember(this.currentState, this.currentAction, this.config.rewardDeath, nextState, true);
                this.episodeReward += this.config.rewardDeath;
            }

            setTimeout(() => this.respawn(), 2000);
        });

        this.bot.on('physicsTick', () => {
            if (this.opponent && this.currentAction !== null) {
                this.executeAction(this.currentAction);
            }
        });
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

        // Get current state
        const state = this.agent.getState(this.bot, opponentEntity);

        // Select action
        const action = this.agent.selectAction(state);

        // Calculate reward from previous action
        if (this.currentState !== null && this.currentAction !== null) {
            const reward = this.config.rewardStep;
            this.episodeReward += reward;

            const nextState = state;
            this.agent.remember(this.currentState, this.currentAction, reward, nextState, false);
        }

        this.currentState = state;
        this.currentAction = action;
        this.episodeSteps++;

        // Learn from experience
        if (this.episodeSteps % 10 === 0) {
            this.agent.learn();
        }
    }

    executeAction(action) {
        const actionName = this.agent.actions[action];

        if (!this.opponent || !this.bot.entity) return;

        const opponentEntity = this.bot.players[this.opponent.name]?.entity;
        if (!opponentEntity) return;

        switch (actionName) {
            case 'attack':
                if (this.bot.entity.position.distanceTo(opponentEntity.position) < 4) {
                    this.bot.attack(opponentEntity);

                    // Check if we hit
                    setTimeout(() => {
                        if (opponentEntity.health < 20) {
                            this.episodeReward += this.config.rewardHit;
                        }
                        if (!opponentEntity.isValid) {
                            this.episodeReward += this.config.rewardKill;
                            this.kills++;
                            console.log(`${this.name} killed ${this.opponent.name}!`);
                        }
                    }, 100);
                }
                break;

            case 'forward':
                this.bot.setControlState('forward', true);
                setTimeout(() => this.bot.setControlState('forward', false), 50);
                break;

            case 'backward':
                this.bot.setControlState('back', true);
                setTimeout(() => this.bot.setControlState('back', false), 50);
                break;

            case 'left':
                this.bot.setControlState('left', true);
                setTimeout(() => this.bot.setControlState('left', false), 50);
                break;

            case 'right':
                this.bot.setControlState('right', true);
                setTimeout(() => this.bot.setControlState('right', false), 50);
                break;

            case 'jump':
                this.bot.setControlState('jump', true);
                setTimeout(() => this.bot.setControlState('jump', false), 50);
                break;

            case 'sprint':
                this.bot.setControlState('sprint', true);
                setTimeout(() => this.bot.setControlState('sprint', false), 100);
                break;
        }

        // Always look at opponent
        this.bot.lookAt(opponentEntity.position.offset(0, opponentEntity.height, 0));
    }

    resetEpisode() {
        this.episodeReward = 0;
        this.episodeSteps = 0;
        this.currentState = null;
        this.currentAction = null;
        this.lastHealth = 20;
    }

    getStats() {
        return {
            name: this.name,
            kills: this.kills,
            deaths: this.deaths,
            kd: this.deaths > 0 ? (this.kills / this.deaths).toFixed(2) : this.kills,
            epsilon: this.agent.epsilon.toFixed(3),
            lastReward: this.episodeReward.toFixed(2)
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

        console.log('Arena setup complete!');
    }

    async giveEquipment(playerName) {
        await this.executeCommand(`/give ${playerName} diamond_sword 1`);
        await this.executeCommand(`/give ${playerName} golden_apple 64`);
        await this.executeCommand(`/give ${playerName} diamond_helmet 1`);
        await this.executeCommand(`/give ${playerName} diamond_chestplate 1`);
        await this.executeCommand(`/give ${playerName} diamond_leggings 1`);
        await this.executeCommand(`/give ${playerName} diamond_boots 1`);
    }

    async healPlayer(playerName) {
        await this.executeCommand(`/effect give ${playerName} instant_health 1 10`);
        await this.executeCommand(`/effect give ${playerName} saturation 1 10`);
    }

    async teleportPlayer(playerName, x, y, z) {
        await this.executeCommand(`/tp ${playerName} ${x} ${y} ${z}`);
    }

    async executeCommand(command) {
        return new Promise((resolve) => {
            this.bot.chat(command);
            setTimeout(resolve, 100);
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
    constructor() {
        this.operator = new OperatorBot();
        this.agent = new RLAgent(TRAINING_CONFIG);
        this.bots = [];
        this.episode = 0;
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
            await new Promise(resolve => setTimeout(resolve, 1000));
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
        console.log(`\n=== Episode ${this.episode} ===`);

        // Reset bots
        for (const bot of this.bots) {
            bot.resetEpisode();
            await this.operator.healPlayer(bot.name);
            await this.operator.giveEquipment(bot.name);
            await this.operator.teleportPlayer(
                bot.name,
                ARENA_CONFIG.center.x + (Math.random() - 0.5) * 10,
                ARENA_CONFIG.respawnY,
                ARENA_CONFIG.center.z + (Math.random() - 0.5) * 10
            );
        }

        await new Promise(resolve => setTimeout(resolve, 2000));

        // Run episode
        for (let step = 0; step < TRAINING_CONFIG.maxStepsPerEpisode; step++) {
            for (const bot of this.bots) {
                await bot.step();
            }
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        // Update target network periodically
        if (this.episode % 10 === 0) {
            this.agent.updateTargetNetwork();
            console.log('Target network updated');
        }

        // Print stats
        console.log('\nEpisode Stats:');
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`${stats.name}: K/D=${stats.kd}, Reward=${stats.lastReward}, ε=${stats.epsilon}`);
        }
    }

    async train() {
        await this.initialize();

        for (let i = 0; i < TRAINING_CONFIG.episodesPerSession; i++) {
            await this.runEpisode();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log('\n=== Training Complete ===');
        console.log('Final Stats:');
        for (const bot of this.bots) {
            const stats = bot.getStats();
            console.log(`${stats.name}: Kills=${stats.kills}, Deaths=${stats.deaths}, K/D=${stats.kd}`);
        }
    }

    cleanup() {
        for (const bot of this.bots) {
            if (bot.bot) bot.bot.quit();
        }
        this.operator.disconnect();
    }
}

// ==================== MAIN ====================
async function main() {
    const trainer = new TrainingSystem();

    process.on('SIGINT', () => {
        console.log('\nShutting down...');
        trainer.cleanup();
        process.exit(0);
    });

    try {
        await trainer.train();
    } catch (error) {
        console.error('Error during training:', error);
        trainer.cleanup();
    }
}

// Run if executed directly
if (require.main === module) {
    main();
}

module.exports = { TrainingSystem, PvPBot, OperatorBot, RLAgent };
