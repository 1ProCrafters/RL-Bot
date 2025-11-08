const SimpleQNetwork = require('./SimpleQNetwork');

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

    getState(bot, opponent, lastDistance, opponentReportedHealth) {
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
            opponentReportedHealth / 20, // Opponent health
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

module.exports = RLAgent;