const mineflayer = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');

const { SERVER_CONFIG, TRAINING_CONFIG, ARENA_CONFIG } = require('./config');

// ==================== PVP BOT ====================
class PvPBot {
    constructor(name, config, agent, arenaIndex = 0) {
        this.name = name;
        this.config = config;
        this.agent = agent;
        this.operator = null; // Will be assigned by TrainingSystem
        this.bot = null;
        this.opponent = null;
        this.arenaIndex = arenaIndex;
        this.currentState = null;
        this.currentAction = null;
        this.episodeReward = 0;
        this.episodeSteps = 0;
        this.lastHealth = 20;
        this.lastOpponentHealth = 20;
        this.opponentReportedHealth = 20;
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

    setOperator(operator) {
        this.operator = operator;
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

                // Send health update to opponent
                if (this.opponent) {
                    const healthMessage = `[HEALTH] ${this.bot.health.toFixed(1)}`;
                    this.bot.chat(`/msg ${this.opponent.name} ${healthMessage}`);
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
            // Reset health on spawn/respawn and inform opponent
            this.lastHealth = 20;
            if (this.opponent) {
                const healthMessage = `[HEALTH] 20.0`;
                this.bot.chat(`/msg ${this.opponent.name} ${healthMessage}`);
            }
            setTimeout(() => this.equipBestWeapon(), 1000);
        });

        // Listen for health updates from opponent
        this.bot.on('messagestr', (message, messagePosition, jsonMsg) => {
            // Example message: "Fighter1 whispers to you: [HEALTH] 15.5"
            const opponentWhisper = `${this.opponent?.name} whispers to you: `;
            if (message.startsWith(opponentWhisper)) {
                const content = message.substring(opponentWhisper.length);
                if (content.startsWith('[HEALTH] ')) {
                    const newHealth = parseFloat(content.substring(9));
                    if (!isNaN(newHealth)) {
                        this.opponentReportedHealth = newHealth;
                    }
                }
            }
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
        const arenaCenter = this.getArenaCenter();
        this.bot.chat(`/tp ${this.name} ${arenaCenter.x} ${ARENA_CONFIG.respawnY} ${arenaCenter.z}`);
        this.lastHealth = 20;
        this.lastOpponentHealth = 20;
    }

    setOpponent(opponent) {
        this.opponent = opponent;
    }

    async step() {
        if (!this.opponent || !this.bot.entity) return;

        const opponentEntity = this.bot.players[this.opponent.name]?.entity;
        if (!opponentEntity) return;

        // Track opponent health for damage dealt based on reported health
        if (this.lastOpponentHealth > this.opponentReportedHealth) {
            const damage = this.lastOpponentHealth - this.opponentReportedHealth;
            this.totalDamageDealt += damage;
            const reward = this.config.rewardDamageDealt * damage;
            this.episodeReward += reward;
        }
        this.lastOpponentHealth = this.opponentReportedHealth;

        const state = this.agent.getState(this.bot, opponentEntity, this.lastDistance, this.opponentReportedHealth);
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
        this.opponentReportedHealth = 20;
        this.lastDistance = null;
        this.lastAttackTime = 0;
        this.totalDamageDealt = 0;
    }

    getArenaCenter() {
        return {
            x: ARENA_CONFIG.center.x + this.arenaIndex * TRAINING_CONFIG.arenaSpacing,
            y: ARENA_CONFIG.center.y,
            z: ARENA_CONFIG.center.z
        };
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

module.exports = PvPBot;